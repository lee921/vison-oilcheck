"""动态物料跟踪。

物料盘连续移动时，固定 1~4 号位 ROI 不再适用。本模块在取料区（``pick_roi``）内
对丝杆检测做跨帧关联，为每根物料分配稳定的 ``track_id``；状态机按 track 维护
单根物料状态，而不是按固定位置。

跟踪策略：

- 用“预测位置 + 中心距离”做贪心匹配，速度根据相邻帧位移平滑更新，适应连续移动；
- 新 track 只有在检测中心进入取料区后才创建，需连续命中 ``track_min_hits`` 次才
  被状态机采用，过滤单帧误检；
- 短暂漏检保留 ``track_max_missing_s``，超过则判定丢失；
- 物料中心离开取料区判定为“离开”，与“手取走”区分，避免料盘移动产生取料事件。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .roi import Roi
from .types import BBox, Detection


@dataclass
class Track:
    track_id: int
    box: BBox
    last_ts: float
    last_seen_ts: float
    velocity: tuple[float, float] = (0.0, 0.0)
    hits: int = 1
    missing_s: float = 0.0
    matched: bool = True
    exited: bool = False
    confidence: float = 0.0

    def predicted_box(self, ts: float) -> BBox:
        dt = max(0.0, float(ts) - self.last_ts)
        dx = self.velocity[0] * dt
        dy = self.velocity[1] * dt
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return self.box
        return BBox(
            self.box.x1 + dx,
            self.box.y1 + dy,
            self.box.x2 + dx,
            self.box.y2 + dy,
        )


@dataclass(frozen=True)
class TrackerOutput:
    active: tuple[Track, ...]
    lost: tuple[Track, ...]


class ObjectTracker:
    """通用多目标跟踪（手部/刷子）。

    与 :class:`RodTracker` 相同的贪心关联策略，但不限定 ROI：手和刷子可能
    离开取料区进入人工操作区，轨迹需要继续保留一段时间。
    """

    def __init__(
        self,
        *,
        max_distance: float = 0.15,
        max_missing_s: float = 0.8,
        min_hits: int = 1,
        max_active: int = 4,
        velocity_smoothing: float = 0.5,
    ) -> None:
        self._max_distance = max(0.005, float(max_distance))
        self._max_missing_s = max(0.0, float(max_missing_s))
        self._min_hits = max(1, int(min_hits))
        self._max_active = max(1, int(max_active))
        self._smoothing = max(0.0, min(0.95, float(velocity_smoothing)))
        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def min_hits(self) -> int:
        return self._min_hits

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def get(self, track_id: int | None) -> Track | None:
        if track_id is None:
            return None
        return self._tracks.get(int(track_id))

    def active_tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks.values())

    def update(
        self,
        *,
        ts: float,
        detections: tuple[Detection, ...],
        frame_size: tuple[int, int],
    ) -> TrackerOutput:
        width, height = frame_size
        if width <= 0 or height <= 0:
            return TrackerOutput(active=(), lost=())
        diag = max(1.0, math.hypot(width, height))

        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()
        pairs: list[tuple[float, int, int]] = []
        for track in self._tracks.values():
            predicted = track.predicted_box(ts)
            pcx, pcy = predicted.center
            for index, det in enumerate(detections):
                cx, cy = det.box.center
                distance = math.hypot(cx - pcx, cy - pcy) / diag
                iou = predicted.iou(det.box)
                if distance <= self._max_distance or iou >= 0.15:
                    pairs.append(
                        (distance - min(iou, 0.5) * 0.01, track.track_id, index)
                    )
        pairs.sort(key=lambda item: item[0])
        for _cost, track_id, det_index in pairs:
            if track_id in matched_tracks or det_index in matched_dets:
                continue
            track = self._tracks.get(track_id)
            if track is None:
                continue
            self._update_track(track, detections[det_index], ts)
            matched_tracks.add(track_id)
            matched_dets.add(det_index)

        lost: list[Track] = []
        for track_id in list(self._tracks):
            track = self._tracks[track_id]
            if track_id in matched_tracks:
                track.matched = True
                continue
            track.matched = False
            track.missing_s = max(0.0, ts - track.last_seen_ts)
            if track.missing_s > self._max_missing_s:
                lost.append(track)
                self._tracks.pop(track_id, None)

        for det_index, det in enumerate(detections):
            if det_index in matched_dets:
                continue
            if len(self._tracks) >= self._max_active:
                break
            track = Track(
                track_id=self._next_id,
                box=det.box,
                last_ts=float(ts),
                last_seen_ts=float(ts),
                confidence=det.confidence,
            )
            self._tracks[track.track_id] = track
            self._next_id += 1

        return TrackerOutput(active=tuple(self._tracks.values()), lost=tuple(lost))

    def _update_track(self, track: Track, det: Detection, ts: float) -> None:
        old_cx, old_cy = track.box.center
        new_cx, new_cy = det.box.center
        dt = max(1e-3, float(ts) - track.last_ts)
        inst_vx = (new_cx - old_cx) / dt
        inst_vy = (new_cy - old_cy) / dt
        s = self._smoothing
        track.velocity = (
            s * track.velocity[0] + (1.0 - s) * inst_vx,
            s * track.velocity[1] + (1.0 - s) * inst_vy,
        )
        track.box = det.box
        track.confidence = det.confidence
        track.last_ts = float(ts)
        track.last_seen_ts = float(ts)
        track.hits += 1
        track.missing_s = 0.0
        track.matched = True


@dataclass
class TrayTrack:
    """机器涂油盘的单目标跟踪状态。"""

    track_id: int | None = None
    box: BBox | None = None
    confidence: float = 0.0
    velocity: tuple[float, float] = (0.0, 0.0)
    last_ts: float = 0.0
    last_seen_ts: float = 0.0
    matched: bool = False
    missing_s: float = 0.0
    hits: int = 0
    reliable: bool = False

    def predicted_box(self, ts: float) -> BBox | None:
        if self.box is None:
            return None
        dt = max(0.0, float(ts) - self.last_ts)
        dx = self.velocity[0] * dt
        dy = self.velocity[1] * dt
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return self.box
        return BBox(
            self.box.x1 + dx,
            self.box.y1 + dy,
            self.box.x2 + dx,
            self.box.y2 + dy,
        )


class TrayTracker:
    """跟踪机器涂油盘整体框；短时遮挡保留预测框，但标记为不可靠。"""

    def __init__(
        self,
        *,
        max_distance: float = 0.20,
        max_missing_s: float = 1.0,
        min_confidence: float = 0.30,
        velocity_smoothing: float = 0.4,
    ) -> None:
        self._max_distance = max(0.01, float(max_distance))
        self._max_missing_s = max(0.0, float(max_missing_s))
        self._min_confidence = max(0.0, min(1.0, float(min_confidence)))
        self._smoothing = max(0.0, min(0.95, float(velocity_smoothing)))
        self._track = TrayTrack()
        self._next_id = 1

    def reset(self) -> None:
        self._track = TrayTrack()
        self._next_id = 1

    def update(
        self,
        *,
        ts: float,
        detections: tuple[Detection, ...],
        frame_size: tuple[int, int],
    ) -> TrayTrack:
        width, height = frame_size
        if width <= 0 or height <= 0:
            return self._track
        diag = max(1.0, math.hypot(width, height))
        predicted = self._track.predicted_box(ts)

        best: Detection | None = None
        best_cost = float("inf")
        for det in detections:
            if det.confidence < self._min_confidence:
                continue
            if predicted is not None:
                pcx, pcy = predicted.center
                cx, cy = det.box.center
                distance = math.hypot(cx - pcx, cy - pcy) / diag
                iou = predicted.iou(det.box)
                if distance > self._max_distance and iou < 0.1:
                    continue
                cost = distance - min(iou, 0.5) * 0.01
            else:
                cost = 1.0 - det.confidence
            if cost < best_cost:
                best_cost = cost
                best = det

        if best is not None:
            if self._track.track_id is None:
                self._track.track_id = self._next_id
                self._next_id += 1
            old_ts = self._track.last_ts
            old_box = self._track.box
            if old_box is not None and old_ts > 0:
                dt = max(1e-3, float(ts) - old_ts)
                ocx, ocy = old_box.center
                ncx, ncy = best.box.center
                inst = ((ncx - ocx) / dt, (ncy - ocy) / dt)
                s = self._smoothing
                self._track.velocity = (
                    s * self._track.velocity[0] + (1.0 - s) * inst[0],
                    s * self._track.velocity[1] + (1.0 - s) * inst[1],
                )
            self._track.box = best.box
            self._track.confidence = best.confidence
            self._track.last_ts = float(ts)
            self._track.last_seen_ts = float(ts)
            self._track.matched = True
            self._track.missing_s = 0.0
            self._track.hits += 1
            self._track.reliable = True
            return self._track

        if self._track.track_id is None:
            return self._track
        self._track.matched = False
        self._track.reliable = False
        self._track.missing_s = max(0.0, float(ts) - self._track.last_seen_ts)
        if self._track.missing_s > self._max_missing_s:
            self._track.box = None
            self._track.hits = 0
        return self._track


class RodTracker:
    def __init__(
        self,
        *,
        roi: Roi,
        max_distance: float = 0.08,
        max_missing_s: float = 0.7,
        min_hits: int = 2,
        max_active: int = 8,
        velocity_smoothing: float = 0.6,
    ) -> None:
        self._roi = roi
        self._max_distance = max(0.005, float(max_distance))
        self._max_missing_s = max(0.0, float(max_missing_s))
        self._min_hits = max(1, int(min_hits))
        self._max_active = max(1, int(max_active))
        self._smoothing = max(0.0, min(0.95, float(velocity_smoothing)))

        self._tracks: dict[int, Track] = {}
        self._next_id = 1

    @property
    def min_hits(self) -> int:
        return self._min_hits

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def drop(self, track_id: int) -> None:
        self._tracks.pop(int(track_id), None)

    def active_tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks.values())

    def update(
        self,
        *,
        ts: float,
        detections: tuple[Detection, ...],
        frame_size: tuple[int, int],
    ) -> TrackerOutput:
        width, height = frame_size
        if width <= 0 or height <= 0:
            return TrackerOutput(active=(), lost=())

        frame_diag = max(1.0, math.hypot(width, height))
        rods = list(detections)
        matched_dets: set[int] = set()
        matched_tracks: set[int] = set()

        pairs: list[tuple[float, int, int]] = []
        for track in self._tracks.values():
            predicted = track.predicted_box(ts)
            pcx, pcy = predicted.center
            for index, det in enumerate(rods):
                cx, cy = det.box.center
                distance = math.hypot(cx - pcx, cy - pcy) / frame_diag
                iou = predicted.iou(det.box)
                if distance <= self._max_distance or iou >= 0.15:
                    # 距离优先；IoU 高时给予小幅优势。
                    pairs.append(
                        (distance - min(iou, 0.5) * 0.01, track.track_id, index)
                    )

        pairs.sort(key=lambda item: item[0])
        for _cost, track_id, det_index in pairs:
            if track_id in matched_tracks or det_index in matched_dets:
                continue
            track = self._tracks.get(track_id)
            if track is None:
                continue
            det = rods[det_index]
            self._update_track(track, det, ts)
            matched_tracks.add(track_id)
            matched_dets.add(det_index)

        lost: list[Track] = []
        for track_id in list(self._tracks):
            track = self._tracks[track_id]
            if track_id not in matched_tracks:
                track.matched = False
                track.missing_s = max(0.0, ts - track.last_seen_ts)
                if track.missing_s > self._max_missing_s:
                    track.exited = not self._roi.contains_box_center(
                        track.predicted_box(ts), width, height
                    )
                    lost.append(track)
                    self._tracks.pop(track_id, None)
                continue

            track.matched = True
            if not self._roi.contains_box_center(track.box, width, height):
                track.exited = True
                lost.append(track)
                self._tracks.pop(track_id, None)

        for det_index, det in enumerate(rods):
            if det_index in matched_dets:
                continue
            if not self._roi.contains_box_center(det.box, width, height):
                continue
            if len(self._tracks) >= self._max_active:
                continue
            track = Track(
                track_id=self._next_id,
                box=det.box,
                last_ts=float(ts),
                last_seen_ts=float(ts),
                confidence=det.confidence,
            )
            self._tracks[track.track_id] = track
            self._next_id += 1

        active = tuple(self._tracks.values())
        return TrackerOutput(active=active, lost=tuple(lost))

    def _update_track(self, track: Track, det: Detection, ts: float) -> None:
        old_cx, old_cy = track.box.center
        new_cx, new_cy = det.box.center
        dt = max(1e-3, float(ts) - track.last_ts)
        inst_vx = (new_cx - old_cx) / dt
        inst_vy = (new_cy - old_cy) / dt
        s = self._smoothing
        track.velocity = (
            s * track.velocity[0] + (1.0 - s) * inst_vx,
            s * track.velocity[1] + (1.0 - s) * inst_vy,
        )
        track.box = det.box
        track.confidence = det.confidence
        track.last_ts = float(ts)
        track.last_seen_ts = float(ts)
        track.hits += 1
        track.missing_s = 0.0
        track.matched = True
