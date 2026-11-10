"""短时目标跟踪。

视觉 Track ID 只表示相邻帧的短时视觉身份，遮挡后可能发生 ID Switch；
作业周期（``TASK-xxxxxx``）在 :mod:`app.oilcheck.pipeline` 中独立维护。

跟踪策略：

- 用“预测位置 + 中心距离”做贪心匹配，速度根据相邻帧位移平滑更新；
- 短时漏检保留 ``max_missing_s``，超过则判定丢失；
- 手/刷头走 :mod:`app.oilcheck.bytetrack_adapter`，这里保留通用实现与回退；
- 检测器提供 OBB 角度/关键点时，随轨迹一起保存。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .types import BBox, Detection

# 涂油盘框外推位移上限（按画面对角线归一化）。
MAX_PREDICT_FRACTION = 0.10


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
    angle: float | None = None
    point: tuple[float, float] | None = None
    # 本帧的**原始检测框**（未经过卡尔曼平滑）。``box`` 是滤波后的状态，
    # 快速移动时会滞后（实测 200px 跳变只走到 155px）；业务判定（进盘/离盘、
    # 握持、锚点交叠）应该用原始检测，显示仍用平滑框。
    raw_box: BBox | None = None

    def observation_box(self) -> BBox:
        """本帧用于判定的框：优先原始检测，没有观测时退回滤波框。"""

        return self.box if self.raw_box is None else self.raw_box

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

    def predicted_point(self, ts: float) -> tuple[float, float] | None:
        if self.point is None:
            return None
        dt = max(0.0, float(ts) - self.last_ts)
        return (
            self.point[0] + self.velocity[0] * dt,
            self.point[1] + self.velocity[1] * dt,
        )

    @property
    def axis(self) -> tuple[float, float]:
        """单位主轴方向：优先用 OBB 角度，否则取框的长边方向。"""

        if self.angle is not None:
            return (math.cos(self.angle), math.sin(self.angle))
        return self.box.long_axis()


@dataclass(frozen=True)
class TrackerOutput:
    active: tuple[Track, ...]
    lost: tuple[Track, ...]


class ObjectTracker:
    """通用多目标跟踪（手部/刷头，ByteTrack 不可用时的回退）。

    不限定 ROI：手和刷头会从涂油盘跨到工作追踪区；
    区域过滤由 :class:`StationPipeline` 在送入跟踪器前完成。
    """

    def __init__(
        self,
        *,
        max_distance: float = 0.15,
        max_missing_s: float = 0.8,
        min_hits: int = 1,
        max_active: int = 16,
        velocity_smoothing: float = 0.5,
        id_base: int = 0,
    ) -> None:
        self._max_distance = max(0.005, float(max_distance))
        self._max_missing_s = max(0.0, float(max_missing_s))
        self._min_hits = max(1, int(min_hits))
        self._max_active = max(1, int(max_active))
        self._smoothing = max(0.0, min(0.95, float(velocity_smoothing)))
        # id_base 用于给不同跟踪器分配互不重叠的 id 段（手 / 刷头）。
        self._id_base = max(0, int(id_base))
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
        distance_scale: float = 1.0,
    ) -> TrackerOutput:
        width, height = frame_size
        if width <= 0 or height <= 0:
            return TrackerOutput(active=(), lost=())
        diag = max(1.0, math.hypot(width, height))
        max_distance = self._max_distance * max(1e-3, float(distance_scale))

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
                if distance <= max_distance or iou >= 0.15:
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
            if not self._make_room_for_new_track():
                break
            track = Track(
                track_id=self._id_base + self._next_id,
                box=det.box,
                last_ts=float(ts),
                last_seen_ts=float(ts),
                confidence=det.confidence,
                angle=det.angle,
                point=det.point,
                raw_box=det.box,
            )
            self._tracks[track.track_id] = track
            self._next_id += 1

        return TrackerOutput(active=tuple(self._tracks.values()), lost=tuple(lost))

    def _make_room_for_new_track(self) -> bool:
        """为新轨迹腾名额：满了先淘汰最久未匹配的轨迹。

        漏检轨迹在 ``max_missing_s`` 内仍占名额，直接拒绝新目标会让新手框
        整段建不了轨迹（画面里常有手臂、盘边被误检成 hand）。淘汰顺序：
        先丢未匹配的，再丢匹配过的，同类按漏检时长从长到短。
        """

        if len(self._tracks) < self._max_active:
            return True
        if not self._tracks:
            return False
        victim = max(
            self._tracks.values(),
            key=lambda item: (not item.matched, item.missing_s),
        )
        self._tracks.pop(victim.track_id, None)
        return True

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
        track.angle = det.angle
        track.point = det.point
        track.raw_box = det.box
        track.last_ts = float(ts)
        track.last_seen_ts = float(ts)
        track.hits += 1
        track.missing_s = 0.0
        track.matched = True


@dataclass
class TrayTrack:
    """涂油盘的单目标跟踪状态，只用于稳定取料参考区域。"""

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
    # 画面对角线（像素），用于限制外推位移。由 TrayTracker.update 维护。
    diag: float = 0.0

    def predicted_box(self, ts: float) -> BBox | None:
        """按速度外推的盘体框。

        外推位移有硬上限（``MAX_PREDICT_FRACTION`` × 画面对角线）：盘体漏检
        一段时间后纯速度外推会把框送到画面外，``_tray_region_box`` 取并集后
        「在盘内」恒真，取料判定随之失效。
        """

        if self.box is None:
            return None
        dt = max(0.0, float(ts) - self.last_ts)
        dx = self.velocity[0] * dt
        dy = self.velocity[1] * dt
        limit = MAX_PREDICT_FRACTION * max(1.0, self.diag)
        span = math.hypot(dx, dy)
        if span > limit > 0:
            dx *= limit / span
            dy *= limit / span
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return self.box
        return BBox(
            self.box.x1 + dx,
            self.box.y1 + dy,
            self.box.x2 + dx,
            self.box.y2 + dy,
        )


class TrayTracker:
    """跟踪涂油盘整体框；短时遮挡保留预测框，但标记为不可靠。"""

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
        distance_scale: float = 1.0,
    ) -> TrayTrack:
        width, height = frame_size
        if width <= 0 or height <= 0:
            return self._track
        diag = max(1.0, math.hypot(width, height))
        self._track.diag = diag
        max_distance = self._max_distance * max(1e-3, float(distance_scale))
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
                if distance > max_distance and iou < 0.1:
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
