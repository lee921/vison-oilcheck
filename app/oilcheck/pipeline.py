"""取料事件、手持丝杆跟踪与二次涂油行为识别。

实现《涂油工位视觉检测项目实现文档》第 5~8 节：

- **取料事件**：手部交互 + 丝杆共同运动 + 离开涂油盘 + 进入手持区并持续关联，
  不以“手离开涂油盘”作为取料证据；
- **视觉/业务身份分离**：跟踪器只给短时 Track ID，取料候选/确认时建立业务
  ``task_id``，遮挡后优先按业务层重关联；
- **刷涂行为**：刷头工作端进入丝杆接触邻域后，将刷头投影到丝杆局部轴向
  坐标系，累计有效相对运动与覆盖区段；
- **结果判定**：OK / NG / PENDING_CONFIRMATION / PROCESSING；证据不足时输出
  待确认，视频异常或关键目标长期不可见时不自动判 OK。

本模块不依赖 Qt、相机 SDK 或具体推理框架，测试直接回放
:class:`FrameDetections`。
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from .config import PipelineConfig, StationConfig
from .roi import Roi
from .tracker import ObjectTracker, Track, TrayTrack, TrayTracker
from .types import (
    BBox,
    Detection,
    EventRecord,
    EventResult,
    EventType,
    FrameDetections,
    HoldingStatus,
    OilingStatus,
    PickStatus,
    StationSnapshot,
    TaskResult,
    TaskSnapshot,
    TaskStage,
    TrackSnapshot,
    TraySnapshot,
)

EventCallback = Any  # Callable[[EventRecord], None]，避免循环导入类型噪音


@dataclass
class _PickCandidate:
    """取料事件引擎中的候选丝杆（业务 ID 尚未确认）。"""

    screw_track_id: int
    box: BBox
    axis: tuple[float, float]
    last_seen_ts: float
    confidence: float
    was_in_tray: bool = False
    in_tray: bool = False
    left_tray_ts: float | None = None
    entered_hand_ts: float | None = None
    hand_track_id: int | None = None
    hand_near_since: float | None = None
    last_hand_ts: float | None = None
    comotion_s: float = 0.0
    prev_center: tuple[float, float] | None = None
    prev_hand_center: tuple[float, float] | None = None
    prev_rel: tuple[float, float] | None = None
    missing_since: float | None = None
    task_id: str | None = None
    pending_emitted: bool = False
    reassociated: bool = False
    last_update_ts: float = 0.0

    @property
    def left_tray(self) -> bool:
        return self.left_tray_ts is not None


@dataclass
class _Task:
    """单根丝杆业务任务（对应文档第 10 节的 ScrewTask）。"""

    task_id: str
    created_at: float
    stage: TaskStage
    stage_since: float
    pick_status: PickStatus
    holding_status: HoldingStatus
    oiling_status: OilingStatus
    final_result: TaskResult
    screw_track_id: int | None
    hand_track_id: int | None = None
    brush_track_id: int | None = None
    screw_box: BBox | None = None
    screw_axis: tuple[float, float] = (1.0, 0.0)
    screw_last_seen_ts: float | None = None
    hand_last_seen_ts: float | None = None
    identity_status: str = "ASSOCIATED"
    confidence: float | None = None
    reason_code: str | None = None
    evidence: list[str] = field(default_factory=list)
    contact_intervals: list[list[float]] = field(default_factory=list)
    contact_open_ts: float | None = None
    last_brush_ts: float | None = None
    last_axial: float | None = None
    brush_motion: float = 0.0
    effective_brush_s: float = 0.0
    segments: list[bool] = field(default_factory=list)
    coverage: float = 0.0
    picked_at: float | None = None
    confirmed_at: float | None = None
    completed_at: float | None = None
    last_seen_ts: float | None = None
    timeline: list[tuple[str, float]] = field(default_factory=list)
    events_sent: set[str] = field(default_factory=set)
    missing_since: float | None = None
    finalized: bool = False

    @property
    def active(self) -> bool:
        return self.final_result == TaskResult.PROCESSING


class StationPipeline:
    """单相机/单工位的取料、手持与刷涂状态机。"""

    def __init__(
        self,
        *,
        station: StationConfig,
        pipeline_cfg: PipelineConfig,
        on_event: EventCallback | None = None,
        cycle_id: str | None = None,
        detector_name: str = "null",
        logger: logging.Logger | None = None,
    ) -> None:
        self.station = station
        self.cfg = pipeline_cfg
        self._on_event = on_event
        self.cycle_id = cycle_id
        self.detector_name = detector_name
        self._logger = logger or logging.getLogger(
            f"oilcheck.pipeline.{station.station_id}"
        )

        self._screw_tracker = ObjectTracker(
            max_distance=pipeline_cfg.screw_track_max_distance,
            max_missing_s=pipeline_cfg.screw_track_max_missing_s,
            min_hits=pipeline_cfg.screw_track_min_hits,
            max_active=pipeline_cfg.screw_track_max_active,
        )
        self._hand_tracker = ObjectTracker(
            max_distance=pipeline_cfg.hand_track_max_distance,
            max_missing_s=pipeline_cfg.hand_track_max_missing_s,
            min_hits=pipeline_cfg.hand_track_min_hits,
            max_active=4,
        )
        self._brush_tracker = ObjectTracker(
            max_distance=pipeline_cfg.brush_track_max_distance,
            max_missing_s=pipeline_cfg.brush_track_max_missing_s,
            min_hits=pipeline_cfg.brush_track_min_hits,
            max_active=4,
        )
        self._tray_tracker = TrayTracker(
            max_distance=pipeline_cfg.tray_max_distance,
            max_missing_s=pipeline_cfg.tray_max_missing_s,
            min_confidence=pipeline_cfg.tray_min_confidence,
        )
        self._tray = TrayTrack()

        self._candidates: dict[str, _PickCandidate] = {}
        self._candidate_seq = 0
        self._tasks: dict[str, _Task] = {}
        self._task_seq = 0

        self._last_frame_ts: float | None = None
        self._frame_size: tuple[int, int] = (0, 0)
        self._last_detections: tuple[Detection, ...] = ()
        self._screw_tracks: tuple[Track, ...] = ()
        self._hand_tracks: tuple[Track, ...] = ()
        self._brush_tracks: tuple[Track, ...] = ()
        self._frames_processed = 0
        self._events_emitted = 0
        self._frame_gap_events = 0
        self._last_frame_gap_s = 0.0

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    @property
    def tasks(self) -> tuple[TaskSnapshot, ...]:
        return tuple(
            self._to_snapshot(task)
            for task in sorted(
                self._tasks.values(),
                key=lambda item: (item.created_at, item.task_id),
            )
        )

    @property
    def operations(self) -> tuple[TaskSnapshot, ...]:
        return self.tasks

    @property
    def materials(self) -> tuple[TaskSnapshot, ...]:
        return self.tasks

    def reset(self) -> None:
        now = time.time()
        self._candidates.clear()
        self._tasks.clear()
        self._candidate_seq = 0
        self._task_seq = 0
        self._screw_tracker.reset()
        self._hand_tracker.reset()
        self._brush_tracker.reset()
        self._tray_tracker.reset()
        self._tray = TrayTrack()
        self._last_frame_ts = None
        self._frame_size = (0, 0)
        self._last_detections = ()
        self._screw_tracks = ()
        self._hand_tracks = ()
        self._brush_tracks = ()
        self._frame_gap_events = 0
        self._last_frame_gap_s = 0.0
        del now

    def process(self, *, detections: FrameDetections) -> StationSnapshot:
        ts = float(detections.ts)
        width, height = detections.frame_size
        if width <= 0 or height <= 0:
            return self.snapshot()
        if self.station.detector.raw_mode:
            return self._process_raw(detections)

        dt = 0.0
        if self._last_frame_ts is not None:
            raw_gap = ts - self._last_frame_ts
            self._last_frame_gap_s = raw_gap
            # 视频时间戳跳变/长间隔会破坏时序证据，单独记录，不静默吞掉。
            if raw_gap < 0 or raw_gap > 1.0:
                self._frame_gap_events += 1
                self._logger.warning(
                    "视频时间戳异常：%.3fs（工位=%s），暂停依赖连续时间的判断",
                    raw_gap,
                    self.station.station_id,
                )
            dt = max(0.0, min(1.0, raw_gap))
        self._last_frame_ts = ts
        self._frame_size = (int(width), int(height))
        self._last_detections = detections.detections
        self._frames_processed += 1

        cfg = self.station.detector
        frame_size = (int(width), int(height))

        trays = detections.by_label(cfg.tray_label)
        screws = self._valid_detections(
            detections.by_label(cfg.screw_label), frame_size
        )
        hands = self._valid_detections(detections.by_label(cfg.hand_label), frame_size)
        brushes = self._valid_detections(
            detections.by_label(cfg.brush_label), frame_size
        )

        screw_out = self._screw_tracker.update(
            ts=ts, detections=screws, frame_size=frame_size
        )
        hand_out = self._hand_tracker.update(
            ts=ts, detections=hands, frame_size=frame_size
        )
        brush_out = self._brush_tracker.update(
            ts=ts, detections=brushes, frame_size=frame_size
        )
        self._screw_tracks = screw_out.active
        self._hand_tracks = hand_out.active
        self._brush_tracks = brush_out.active
        self._tray = self._tray_tracker.update(
            ts=ts, detections=trays, frame_size=frame_size
        )

        tray_box = self._tray_region_box(ts, frame_size)
        self._update_candidates(ts=ts, dt=dt, tray_box=tray_box)
        self._update_tasks(ts=ts, dt=dt, tray_box=tray_box)
        self._cleanup(ts)
        return self.snapshot()

    def _process_raw(self, detections: FrameDetections) -> StationSnapshot:
        """旁路模式：只交出这一帧的全部检测结果。

        不做 ROI 过滤、不建轨迹、不判取料/刷涂、不产生事件。用于先确认
        模型对画面的标签覆盖情况，SOP 状态机保持空闲。
        """

        self._last_frame_ts = float(detections.ts)
        width, height = detections.frame_size
        self._frame_size = (int(width), int(height))
        self._last_detections = detections.detections
        self._frames_processed += 1
        self._screw_tracks = ()
        self._hand_tracks = ()
        self._brush_tracks = ()
        self._tray = TrayTrack()
        return self.snapshot()

    def snapshot(self) -> StationSnapshot:
        active = next(
            (task for task in self._tasks.values() if task.active),
            None,
        )
        return StationSnapshot(
            station_id=self.station.station_id,
            station_name=self.station.name,
            camera_id=self.station.camera_id,
            mode=self.detector_name,
            online=False,
            tasks=self.tasks,
            tray=self._tray_snapshot(),
            tracks=self._track_snapshots(),
            detections=self._last_detections,
            frame_size=self._frame_size,
            updated_at=self._last_frame_ts or 0.0,
            stats={
                "frames_processed": self._frames_processed,
                "events_emitted": self._events_emitted,
                "tasks": len(self._tasks),
                "pick_candidates": len(self._candidates),
                "active_task_id": active.task_id if active else "",
                "tray_reliable": self._tray_region_reliable(),
                "frame_gap_events": self._frame_gap_events,
                "last_frame_gap_s": round(self._last_frame_gap_s, 3),
                "raw_mode": self.station.detector.raw_mode,
                "label_counts": _label_counts(self._last_detections),
            },
        )

    # ------------------------------------------------------------------
    # ROI 与检测过滤
    # ------------------------------------------------------------------
    def _valid_detections(
        self,
        detections: tuple[Detection, ...],
        frame_size: tuple[int, int],
    ) -> tuple[Detection, ...]:
        """屏蔽物料盒等排除区域，并限定在有效推理 ROI 内。"""

        width, height = frame_size
        valid_rois = (
            self.station.tray_roi,
            self.station.hand_roi,
            self.station.brush_roi,
        )
        result: list[Detection] = []
        for det in detections:
            if self._in_excluded_roi(det.box, width, height):
                continue
            if any(
                roi.contains_box_center(det.box, width, height)
                for roi in valid_rois
            ):
                result.append(det)
        return tuple(result)

    def _in_excluded_roi(self, box: BBox, width: int, height: int) -> bool:
        return any(
            roi.contains_box_center(box, width, height)
            for roi in self.station.exclude_rois
        )

    def _tray_region_box(
        self, ts: float, frame_size: tuple[int, int]
    ) -> BBox:
        """涂油盘参考区域：检测盘体与固定 ROI 的并集。

        固定 ROI 覆盖盘体边缘缓冲；检测框负责跟随上下移动。二者取并集可避免
        盘体检测框偏小时把仍在盘内的丝杆误判为“已离盘”。
        """

        width, height = frame_size
        fixed = self.station.tray_roi.pixel_bbox(width, height)
        predicted = self._tray.predicted_box(ts)
        if predicted is None or not self._tray.reliable:
            return fixed
        return BBox(
            min(fixed.x1, predicted.x1),
            min(fixed.y1, predicted.y1),
            max(fixed.x2, predicted.x2),
            max(fixed.y2, predicted.y2),
        )

    def _tray_region_reliable(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # 取料事件
    # ------------------------------------------------------------------
    def _update_candidates(
        self, *, ts: float, dt: float, tray_box: BBox
    ) -> None:
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        active_screws = [
            track
            for track in self._screw_tracks
            if track.matched and track.hits >= self._screw_tracker.min_hits
        ]
        hands = [
            track
            for track in self._hand_tracks
            if track.matched and track.hits >= self._hand_tracker.min_hits
        ]
        claimed_screws = {
            task.screw_track_id
            for task in self._tasks.values()
            if task.active
            and task.pick_status == PickStatus.CONFIRMED
            and task.screw_track_id is not None
        }

        assigned: set[int] = set()
        # 1) 延续已有候选：优先沿用原 Track ID，丢失时做业务层重关联。
        for candidate in list(self._candidates.values()):
            track = self._screw_tracker.get(candidate.screw_track_id)
            if track is not None and track.matched and track.track_id not in claimed_screws:
                self._update_candidate_geometry(candidate, track, ts)
                assigned.add(track.track_id)
                continue
            replacement = self._reassociate_screw(
                candidate, active_screws, assigned, claimed_screws, diag
            )
            if replacement is not None:
                candidate.screw_track_id = replacement.track_id
                candidate.reassociated = True
                self._update_candidate_geometry(candidate, replacement, ts)
                assigned.add(replacement.track_id)
                continue
            candidate.missing_since = candidate.missing_since or ts
            self._handle_candidate_occlusion(candidate, ts)

        # 2) 新候选：手部靠近涂油盘内（或手持区内）的丝杆时才建立。
        single_active = self.cfg.single_active_task and any(
            task.active and task.pick_status == PickStatus.CONFIRMED
            for task in self._tasks.values()
        )
        for track in active_screws:
            if single_active:
                break
            if track.track_id in assigned or track.track_id in claimed_screws:
                continue
            hand = self._nearest_hand(track, hands, diag)
            in_tray = _box_center_in(track.box, tray_box)
            if hand is None:
                continue
            # 新候选必须先在涂油盘内存在过；离开涂油盘后的重关联由已有候选处理。
            if not in_tray:
                continue
            candidate = self._new_candidate(track, hand, ts, in_tray)
            assigned.add(track.track_id)

        # 3) 推进每个候选的证据链。
        for candidate in list(self._candidates.values()):
            self._advance_candidate(candidate, ts=ts, dt=dt, tray_box=tray_box)
        self._prune_candidates(ts)

    def _new_candidate(
        self, track: Track, hand: Track, ts: float, in_tray: bool
    ) -> _PickCandidate:
        self._candidate_seq += 1
        candidate = _PickCandidate(
            screw_track_id=track.track_id,
            box=track.box,
            axis=track.axis,
            last_seen_ts=ts,
            confidence=track.confidence,
            was_in_tray=in_tray,
            in_tray=in_tray,
            hand_track_id=hand.track_id,
            hand_near_since=ts,
            last_hand_ts=ts,
            last_update_ts=ts,
        )
        key = f"cand-{self._candidate_seq:04d}"
        self._candidates[key] = candidate
        task = self._ensure_candidate_task(candidate, ts)
        self._set_stage(task, TaskStage.PICK_CANDIDATE, ts, "HAND_INTERACTION")
        task.evidence.append("HAND_INTERACTION")
        if in_tray:
            task.evidence.append("SCREW_IN_TRAY_BEFORE_PICK")
        self._logger.debug(
            "取料候选 %s：丝杆轨迹=%s 手部轨迹=%s",
            task.task_id,
            track.track_id,
            hand.track_id,
        )
        return candidate

    def _update_candidate_geometry(
        self, candidate: _PickCandidate, track: Track, ts: float
    ) -> None:
        candidate.box = track.box
        candidate.axis = track.axis
        candidate.confidence = track.confidence
        candidate.last_seen_ts = ts
        candidate.missing_since = None
        candidate.last_update_ts = ts
        task = self._task_for_candidate(candidate)
        if task is not None:
            task.screw_box = track.box
            task.screw_axis = track.axis
            task.screw_track_id = track.track_id
            task.screw_last_seen_ts = ts
            task.confidence = track.confidence
            task.last_seen_ts = ts
            if candidate.reassociated:
                task.evidence.append("TRACK_REASSOCIATED")
                candidate.reassociated = False

    def _advance_candidate(
        self,
        candidate: _PickCandidate,
        *,
        ts: float,
        dt: float,
        tray_box: BBox,
    ) -> None:
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        hands = [
            track
            for track in self._hand_tracks
            if track.matched and track.hits >= self._hand_tracker.min_hits
        ]
        hand = self._hand_tracker.get(candidate.hand_track_id)
        if hand is None or not hand.matched:
            hand = self._nearest_hand_to_box(candidate.box, hands, diag)
            if hand is not None:
                candidate.hand_track_id = hand.track_id
                candidate.prev_hand_center = None
                candidate.prev_rel = None
        if hand is not None and hand.matched:
            candidate.last_hand_ts = ts

        task = self._task_for_candidate(candidate)
        if task is None:
            return
        if task.pick_status == PickStatus.CONFIRMED:
            return

        # 手部离开且丝杆仍在盘中：取消候选（空手经过不等于取料）。
        if (
            hand is None
            and candidate.last_hand_ts is not None
            and ts - candidate.last_hand_ts >= self.cfg.pick_abort_s
            and not candidate.left_tray
        ):
            self._cancel_candidate(candidate, task, ts, "HAND_LEFT_WITHOUT_PICK")
            return

        if hand is not None:
            in_tray_now = _box_center_in(candidate.box, tray_box)
            candidate.in_tray = in_tray_now
            if in_tray_now:
                candidate.was_in_tray = True

            # 共同运动：丝杆与手位移方向一致，且二者相对位置基本稳定。
            hand_center = hand.box.center
            screw_center = candidate.box.center
            if (
                candidate.prev_center is not None
                and candidate.prev_hand_center is not None
                and candidate.prev_rel is not None
                and dt > 0
            ):
                dsx = screw_center[0] - candidate.prev_center[0]
                dsy = screw_center[1] - candidate.prev_center[1]
                dhx = hand_center[0] - candidate.prev_hand_center[0]
                dhy = hand_center[1] - candidate.prev_hand_center[1]
                ds = math.hypot(dsx, dsy) / diag
                dh = math.hypot(dhx, dhy) / diag
                rel = (
                    screw_center[0] - hand_center[0],
                    screw_center[1] - hand_center[1],
                )
                rel_change = (
                    math.hypot(
                        rel[0] - candidate.prev_rel[0],
                        rel[1] - candidate.prev_rel[1],
                    )
                    / diag
                )
                if (
                    ds >= self.cfg.pick_motion_min
                    and dh >= self.cfg.pick_motion_min
                    and rel_change <= self.cfg.pick_relative_max
                ):
                    norm = math.hypot(dsx, dsy) * math.hypot(dhx, dhy)
                    cosine = 1.0 if norm <= 1e-9 else (dsx * dhx + dsy * dhy) / norm
                    if cosine >= 0.5:
                        candidate.comotion_s += dt
                        if "SCREW_HAND_COMOTION" not in task.evidence:
                            task.evidence.append("SCREW_HAND_COMOTION")
            candidate.prev_center = screw_center
            candidate.prev_hand_center = hand_center
            candidate.prev_rel = (
                screw_center[0] - hand_center[0],
                screw_center[1] - hand_center[1],
            )

            inside_hand = self._inside_roi(
                candidate.box, self.station.hand_roi
            )
            if not candidate.in_tray and candidate.was_in_tray:
                if candidate.left_tray_ts is None:
                    candidate.left_tray_ts = ts
                    self._set_stage(
                        task, TaskStage.LEAVING_TRAY, ts, "SCREW_LEFT_TRAY"
                    )
                    task.evidence.append("SCREW_LEFT_TRAY")
                if inside_hand:
                    if candidate.entered_hand_ts is None:
                        candidate.entered_hand_ts = ts
                        task.evidence.append("SCREW_ENTERED_HAND_AREA")

            # 证据充分：共同运动 + 离盘 + 进入手持区 + 持续关联。
            if (
                candidate.missing_since is None
                and candidate.left_tray
                and candidate.comotion_s >= self.cfg.pick_comotion_min_s
                and inside_hand
                and candidate.entered_hand_ts is not None
                and ts - candidate.entered_hand_ts >= self.cfg.pick_confirm_s
            ):
                if self._pick_is_ambiguous(candidate, hands, diag):
                    self._mark_candidate_pending(
                        candidate, ts, "MULTI_PICK_CANDIDATE", task
                    )
                else:
                    self._confirm_pick(candidate, task, ts)

    def _handle_candidate_occlusion(self, candidate: _PickCandidate, ts: float) -> None:
        task = self._task_for_candidate(candidate)
        if task is None or task.pick_status == PickStatus.CONFIRMED:
            return
        if candidate.missing_since is None:
            return
        hand = self._hand_tracker.get(candidate.hand_track_id)
        hand_in_area = hand is not None and hand.matched and self._inside_roi(
            hand.box, self.station.hand_roi
        )
        if not candidate.left_tray or not hand_in_area:
            return
        if ts - candidate.missing_since >= self.cfg.pick_occlusion_timeout_s:
            self._mark_candidate_pending(
                candidate, ts, "PICK_OCCLUSION_TIMEOUT", task
            )

    def _confirm_pick(
        self, candidate: _PickCandidate, task: _Task, ts: float
    ) -> None:
        task.pick_status = PickStatus.CONFIRMED
        task.holding_status = HoldingStatus.HOLDING
        task.confirmed_at = ts
        task.final_result = TaskResult.PROCESSING
        task.reason_code = "PICK_CONFIRMED"
        task.evidence.append("PICK_CONFIRMED")
        self._set_stage(task, TaskStage.PICK_CONFIRMED, ts, "PICK_CONFIRMED")
        self._set_stage(task, TaskStage.HOLDING, ts, "HOLDING_STARTED")
        self._emit_event(
            task,
            event_type=EventType.PICK,
            result=EventResult.CONFIRMED,
            reason="pick_confirmed",
            extra={
                "screw_track_id": candidate.screw_track_id,
                "hand_track_id": candidate.hand_track_id,
                "comotion_s": round(candidate.comotion_s, 3),
            },
            start_ts=task.created_at,
            end_ts=ts,
        )
        self._logger.info(
            "取料确认 %s（丝杆轨迹=%s 手部轨迹=%s）",
            task.task_id,
            candidate.screw_track_id,
            candidate.hand_track_id,
        )
        # 候选证据链已完成，后续由任务层维护手持关联。
        key = self._candidate_key(candidate)
        if key is not None:
            self._candidates.pop(key, None)
        # 同一只手下的其他候选不再成立。
        for other in list(self._candidates.values()):
            if other is candidate:
                continue
            if other.hand_track_id != candidate.hand_track_id:
                continue
            other_task = self._task_for_candidate(other)
            if other_task is not None and other_task.pick_status == PickStatus.CANDIDATE:
                self._cancel_candidate(other, other_task, ts, "OTHER_PICK_CONFIRMED")

    def _mark_candidate_pending(
        self,
        candidate: _PickCandidate,
        ts: float,
        reason: str,
        task: _Task,
    ) -> None:
        if candidate.pending_emitted:
            return
        candidate.pending_emitted = True
        task.pick_status = PickStatus.UNKNOWN
        task.final_result = TaskResult.PENDING_CONFIRMATION
        task.reason_code = reason
        task.evidence.append(reason)
        self._set_stage(
            task, TaskStage.PENDING_CONFIRMATION, ts, reason
        )
        self._emit_event(
            task,
            event_type=EventType.PICK,
            result=EventResult.NEEDS_REVIEW,
            reason=reason,
            extra={
                "screw_track_id": candidate.screw_track_id,
                "hand_track_id": candidate.hand_track_id,
                "comotion_s": round(candidate.comotion_s, 3),
            },
            start_ts=task.created_at,
            end_ts=ts,
        )

    def _cancel_candidate(
        self, candidate: _PickCandidate, task: _Task, ts: float, reason: str
    ) -> None:
        key = self._candidate_key(candidate)
        if key is not None:
            self._candidates.pop(key, None)
        if task.pick_status == PickStatus.CANDIDATE and not task.finalized:
            self._tasks.pop(task.task_id, None)
        self._logger.debug("取消取料候选 %s（%s）", task.task_id, reason)
        del ts

    def _prune_candidates(self, ts: float) -> None:
        for key, candidate in list(self._candidates.items()):
            task = self._task_for_candidate(candidate)
            if task is None:
                self._candidates.pop(key, None)
                continue
            if task.pick_status != PickStatus.CANDIDATE:
                # 已输出待确认的取料事件：保留一段时间供复核/重关联。
                if (
                    task.pick_status == PickStatus.UNKNOWN
                    and candidate.missing_since is not None
                    and ts - candidate.missing_since
                    >= self.cfg.pick_occlusion_timeout_s
                    + self.cfg.completed_retain_s
                ):
                    self._candidates.pop(key, None)
                continue
            if (
                not candidate.left_tray
                and ts - candidate.last_update_ts >= self.cfg.pick_window_s
            ):
                self._candidates.pop(key, None)
                self._tasks.pop(task.task_id, None)

    def _reassociate_screw(
        self,
        candidate: _PickCandidate,
        active_screws: list[Track],
        assigned: set[int],
        claimed: set[int | None],
        diag: float,
    ) -> Track | None:
        hand = self._hand_tracker.get(candidate.hand_track_id)
        best: Track | None = None
        best_distance = float("inf")
        second_distance = float("inf")
        for track in active_screws:
            if track.track_id in assigned or track.track_id in claimed:
                continue
            distance = _center_distance(track.box, candidate.box) / diag
            if distance > self.cfg.screw_track_max_distance * 2.0:
                continue
            if hand is not None and hand.matched:
                hand_distance = _center_distance(track.box, hand.box) / diag
                if hand_distance > self.cfg.pick_hand_distance * 2.0:
                    continue
            if distance < best_distance:
                second_distance = best_distance
                best_distance = distance
                best = track
            elif distance < second_distance:
                second_distance = distance
        if best is None:
            return None
        if second_distance - best_distance <= self.cfg.identity_ambiguity_margin:
            return None
        return best

    def _pick_is_ambiguous(
        self, candidate: _PickCandidate, hands: list[Track], diag: float
    ) -> bool:
        hand = self._hand_tracker.get(candidate.hand_track_id)
        if hand is None:
            return False
        competitors: list[float] = []
        for other in self._candidates.values():
            if other is candidate:
                continue
            if other.hand_track_id != candidate.hand_track_id:
                continue
            if not other.left_tray:
                continue
            if other.missing_since is not None:
                continue
            competitors.append(_center_distance(other.box, hand.box) / diag)
        if not competitors:
            return False
        best = _center_distance(candidate.box, hand.box) / diag
        return any(abs(value - best) <= self.cfg.pick_ambiguity_margin for value in competitors)

    def _ensure_candidate_task(
        self, candidate: _PickCandidate, ts: float
    ) -> _Task:
        task = self._task_for_candidate(candidate)
        if task is not None:
            return task
        self._task_seq += 1
        task_id = f"TASK-{self._task_seq:06d}"
        task = _Task(
            task_id=task_id,
            created_at=ts,
            stage=TaskStage.PICK_CANDIDATE,
            stage_since=ts,
            pick_status=PickStatus.CANDIDATE,
            holding_status=HoldingStatus.HOLDING,
            oiling_status=OilingStatus.WAITING,
            final_result=TaskResult.PROCESSING,
            screw_track_id=candidate.screw_track_id,
            hand_track_id=candidate.hand_track_id,
            screw_box=candidate.box,
            screw_axis=candidate.axis,
            screw_last_seen_ts=ts,
            confidence=candidate.confidence,
            picked_at=ts,
            last_seen_ts=ts,
            segments=[False] * max(1, self.cfg.coverage_segments),
            timeline=[("PICK_CANDIDATE", ts)],
        )
        self._tasks[task_id] = task
        candidate.task_id = task_id
        self._logger.info(
            "建立取料候选任务 %s（丝杆轨迹=%s）", task_id, candidate.screw_track_id
        )
        return task

    def _task_for_candidate(self, candidate: _PickCandidate) -> _Task | None:
        if candidate.task_id is None:
            return None
        return self._tasks.get(candidate.task_id)

    def _candidate_key(self, target: _PickCandidate) -> str | None:
        for key, candidate in self._candidates.items():
            if candidate is target:
                return key
        return None

    # ------------------------------------------------------------------
    # 手持与刷涂
    # ------------------------------------------------------------------
    def _update_tasks(
        self, *, ts: float, dt: float, tray_box: BBox
    ) -> None:
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        active_tasks = [task for task in self._tasks.values() if task.active]
        if not active_tasks:
            return
        claimed = {
            task.screw_track_id
            for task in active_tasks
            if task.screw_track_id is not None
        }
        for task in active_tasks:
            if task.pick_status == PickStatus.CANDIDATE:
                continue
            self._update_holding(task, ts=ts, diag=diag, claimed=claimed)
            self._update_brush(task, ts=ts, dt=dt, diag=diag)
            self._settle_task(task, ts=ts)
        del tray_box, width, height

    def _update_holding(
        self, task: _Task, *, ts: float, diag: float, claimed: set[int | None]
    ) -> None:
        if task.holding_status == HoldingStatus.RELEASED:
            return
        track = self._screw_tracker.get(task.screw_track_id)
        if track is not None and track.matched:
            task.screw_box = track.box
            task.screw_axis = track.axis
            task.screw_track_id = track.track_id
            task.screw_last_seen_ts = ts
            task.holding_status = HoldingStatus.HOLDING
            task.identity_status = "ASSOCIATED"
            task.last_seen_ts = ts
            task.confidence = track.confidence
            if task.final_result == TaskResult.PENDING_CONFIRMATION and task.pick_status == PickStatus.CONFIRMED:
                task.final_result = TaskResult.PROCESSING
                self._set_stage(task, TaskStage.HOLDING, ts, "IDENTITY_REASSOCIATED")
            return

        # 视觉 Track 丢失：尝试业务层重关联，不确定时暂停累计刷涂证据。
        replacement = self._reassociate_held_screw(task, ts, diag, claimed)
        if replacement is not None:
            task.screw_track_id = replacement.track_id
            task.screw_box = replacement.box
            task.screw_axis = replacement.axis
            task.screw_last_seen_ts = ts
            task.holding_status = HoldingStatus.HOLDING
            task.identity_status = "ASSOCIATED"
            task.missing_since = None
            task.evidence.append("TRACK_REASSOCIATED")
            if task.final_result == TaskResult.PENDING_CONFIRMATION:
                task.final_result = TaskResult.PROCESSING
            self._set_stage(task, TaskStage.HOLDING, ts, "IDENTITY_REASSOCIATED")
            return

        if task.missing_since is None:
            task.missing_since = ts
        task.identity_status = "UNCERTAIN"
        if "SCREW_OCCLUDED" not in task.evidence:
            task.evidence.append("SCREW_OCCLUDED")
        if ts - task.missing_since >= self.cfg.holding_lost_timeout_s:
            task.holding_status = HoldingStatus.LOST
            if task.oiling_status != OilingStatus.COMPLETED:
                task.reason_code = "HOLDING_LOST"
                if "HOLDING_LOST" not in task.evidence:
                    task.evidence.append("HOLDING_LOST")
                # 保留任务，等待业务层重关联；仅暂停刷涂证据累计。
                self._set_stage(
                    task, TaskStage.PENDING_CONFIRMATION, ts, "HOLDING_LOST"
                )

    def _reassociate_held_screw(
        self, task: _Task, ts: float, diag: float, claimed: set[int | None]
    ) -> Track | None:
        del ts
        hand = self._hand_tracker.get(task.hand_track_id)
        candidates: list[tuple[float, Track]] = []
        for track in self._screw_tracks:
            if not track.matched or track.hits < self._screw_tracker.min_hits:
                continue
            if track.track_id in claimed and track.track_id != task.screw_track_id:
                continue
            distance = (
                _center_distance(track.box, task.screw_box)
                if task.screw_box is not None
                else float("inf")
            ) / diag
            if distance > self.cfg.screw_track_max_distance * 2.0:
                continue
            if hand is not None and hand.matched:
                hand_distance = _center_distance(track.box, hand.box) / diag
                if hand_distance > self._track_hand_distance_max:
                    continue
            candidates.append((distance, track))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        best_distance, best = candidates[0]
        if len(candidates) > 1:
            if candidates[1][0] - best_distance <= self.cfg.identity_ambiguity_margin:
                return None
        return best

    def _update_brush(self, task: _Task, *, ts: float, dt: float, diag: float) -> None:
        if task.pick_status != PickStatus.CONFIRMED:
            return
        if task.holding_status != HoldingStatus.HOLDING:
            return
        if task.oiling_status == OilingStatus.COMPLETED:
            return
        screw_box = task.screw_box
        if screw_box is None:
            return
        if self.cfg.brush_require_hand and not self._hand_near_screw(task, diag):
            self._close_contact(task, ts)
            return

        contact = self._best_brush_contact(task, screw_box, ts, diag)
        if contact is None:
            self._close_contact(task, ts)
            if task.oiling_status == OilingStatus.CONTACT:
                task.oiling_status = OilingStatus.WAITING
            return

        brush, work_point, distance = contact
        if task.contact_open_ts is None:
            task.contact_open_ts = ts
            task.contact_intervals.append([ts, ts])
            task.brush_track_id = brush.track_id
            task.evidence.append("BRUSH_CONTACT")
            self._set_stage(task, TaskStage.BRUSH_CONTACT, ts, "BRUSH_CONTACT")
            task.oiling_status = OilingStatus.CONTACT
            self._emit_event(
                task,
                event_type=EventType.BRUSH_CONTACT,
                result=EventResult.CONFIRMED,
                reason="brush_contact",
                extra={
                    "brush_track_id": brush.track_id,
                    "distance": round(distance / diag, 4),
                },
                start_ts=ts,
                end_ts=ts,
            )
        else:
            task.contact_intervals[-1][1] = ts
            if task.last_brush_ts is not None:
                gap = ts - task.last_brush_ts
                if gap <= self.cfg.brush_max_gap_s:
                    task.effective_brush_s += max(dt, 0.0)

        axial, center, axis, half_length = _project_on_screw(
            work_point, screw_box, task.screw_axis
        )
        segment = _segment_index(axial / max(1e-6, half_length), len(task.segments))
        if 0 <= segment < len(task.segments) and distance <= self.cfg.brush_contact_distance * diag:
            task.segments[segment] = True
        if task.last_axial is not None and half_length > 1e-6:
            task.brush_motion += abs(axial - task.last_axial) / (2.0 * half_length)
        task.last_axial = axial
        task.last_brush_ts = ts
        task.brush_track_id = brush.track_id
        task.last_seen_ts = ts
        task.coverage = sum(task.segments) / max(1, len(task.segments))

        if (
            task.effective_brush_s >= self.cfg.brush_min_duration_s
            and task.brush_motion >= self.cfg.brush_min_motion
        ):
            if task.oiling_status != OilingStatus.BRUSHING:
                task.oiling_status = OilingStatus.BRUSHING
                self._set_stage(task, TaskStage.BRUSHING, ts, "BRUSH_RELATIVE_MOTION")
                task.evidence.append("BRUSH_RELATIVE_MOTION")
            if task.coverage + 1e-9 >= self.cfg.coverage_required:
                self._complete_task(task, ts)
        del center, axis

    def _close_contact(self, task: _Task, ts: float) -> None:
        if task.contact_open_ts is None:
            return
        if task.contact_intervals:
            task.contact_intervals[-1][1] = max(
                task.contact_intervals[-1][0], ts
            )
        task.contact_open_ts = None
        task.last_axial = None
        task.last_brush_ts = None

    def _best_brush_contact(
        self, task: _Task, screw_box: BBox, ts: float, diag: float
    ) -> tuple[Track, tuple[float, float], float] | None:
        best: tuple[Track, tuple[float, float], float] | None = None
        for brush in self._brush_tracks:
            if not brush.matched or brush.hits < self._brush_tracker.min_hits:
                continue
            point = self._brush_work_point(brush)
            _axial, _center, _axis, _half = _project_on_screw(
                point, screw_box, task.screw_axis
            )
            distance = _point_to_axis_distance(
                point, screw_box, task.screw_axis
            )
            if distance > self.cfg.brush_contact_distance * diag:
                continue
            if best is None or distance < best[2]:
                best = (brush, point, distance)
        del ts
        return best

    def _brush_work_point(self, brush: Track) -> tuple[float, float]:
        """刷头工作端：优先关键点，否则按标定偏移从检测框推算。"""

        if brush.point is not None:
            return brush.point
        cfg = self.station.detector
        cx, cy = brush.box.center
        return (
            cx + (cfg.brush_point_offset_x - 0.5) * brush.box.width,
            cy + (cfg.brush_point_offset_y - 0.5) * brush.box.height,
        )

    def _hand_near_screw(self, task: _Task, diag: float) -> bool:
        hand = self._hand_tracker.get(task.hand_track_id)
        if hand is None or not hand.matched:
            return False
        if task.screw_box is None:
            return False
        return _center_distance(hand.box, task.screw_box) / diag <= (
            self.cfg.pick_hand_distance * 2.0
        )

    def _complete_task(self, task: _Task, ts: float) -> None:
        self._close_contact(task, ts)
        task.oiling_status = OilingStatus.COMPLETED
        task.holding_status = HoldingStatus.RELEASED
        task.final_result = TaskResult.OK
        task.completed_at = ts
        task.reason_code = "OILING_COMPLETED"
        task.coverage = sum(task.segments) / max(1, len(task.segments))
        task.evidence.append("OILING_COMPLETED")
        self._set_stage(
            task, TaskStage.OILING_COMPLETED, ts, "OILING_COMPLETED"
        )
        evidence = self._brush_evidence(task)
        self._emit_event(
            task,
            event_type=EventType.MANUAL_OILING,
            result=EventResult.CONFIRMED,
            reason="oiling_completed",
            extra=evidence,
            start_ts=task.confirmed_at or task.created_at,
            end_ts=ts,
        )
        self._emit_event(
            task,
            event_type=EventType.TASK_RESULT,
            result=EventResult.CONFIRMED,
            reason="task_ok",
            extra=evidence,
            start_ts=task.created_at,
            end_ts=ts,
        )
        self._logger.info(
            "任务 %s 判定 OK（覆盖 %.0f%%，有效刷涂 %.2fs）",
            task.task_id,
            task.coverage * 100,
            task.effective_brush_s,
        )

    def _settle_task(self, task: _Task, *, ts: float) -> None:
        """任务无法继续时给出终态，避免无限进行中。"""

        if not task.active:
            return
        if task.pick_status != PickStatus.CONFIRMED:
            return
        hand = self._hand_tracker.get(task.hand_track_id)
        hand_visible = hand is not None and hand.matched
        screw_visible = (
            task.screw_last_seen_ts is not None
            and ts - task.screw_last_seen_ts <= self.cfg.identity_timeout_s
        )
        if hand_visible or screw_visible:
            if (
                self.cfg.oiling_wait_timeout_s > 0
                and task.oiling_status == OilingStatus.WAITING
                and task.confirmed_at is not None
                and ts - task.confirmed_at >= self.cfg.oiling_wait_timeout_s
            ):
                self._finalize_ng(task, ts, "OILING_WAIT_TIMEOUT")
            return

        if task.holding_status != HoldingStatus.LOST:
            task.holding_status = HoldingStatus.LOST
            task.reason_code = "HAND_AND_SCREW_LOST"
            task.evidence.append("HAND_AND_SCREW_LOST")
        if task.oiling_status in (OilingStatus.BRUSHING, OilingStatus.CONTACT) or task.coverage > 0:
            task.final_result = TaskResult.PENDING_CONFIRMATION
            self._set_stage(
                task, TaskStage.PENDING_CONFIRMATION, ts, "INSUFFICIENT_BRUSH_EVIDENCE"
            )
            self._emit_event(
                task,
                event_type=EventType.TASK_RESULT,
                result=EventResult.NEEDS_REVIEW,
                reason="insufficient_brush_evidence",
                extra=self._brush_evidence(task),
                start_ts=task.created_at,
                end_ts=ts,
            )
        else:
            self._finalize_ng(task, ts, "NO_BRUSH_EVIDENCE")

    def _finalize_ng(self, task: _Task, ts: float, reason: str) -> None:
        task.final_result = TaskResult.NG
        task.holding_status = HoldingStatus.RELEASED
        task.reason_code = reason
        task.completed_at = ts
        task.evidence.append(reason)
        self._set_stage(task, TaskStage.NG, ts, reason)
        self._emit_event(
            task,
            event_type=EventType.TASK_RESULT,
            result=EventResult.UNCONFIRMED,
            reason=reason,
            extra=self._brush_evidence(task),
            start_ts=task.created_at,
            end_ts=ts,
        )
        self._logger.info("任务 %s 判定 NG（%s）", task.task_id, reason)

    def _brush_evidence(self, task: _Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "screw_track_id": task.screw_track_id,
            "hand_track_id": task.hand_track_id,
            "brush_track_id": task.brush_track_id,
            "contact_intervals": [
                [round(start, 3), round(end, 3)]
                for start, end in task.contact_intervals
            ],
            "axis_segments": list(task.segments),
            "coverage": round(task.coverage, 3),
            "effective_brush_duration_ms": int(round(task.effective_brush_s * 1000)),
            "brush_motion": round(task.brush_motion, 4),
            "min_duration_s": self.cfg.brush_min_duration_s,
            "min_motion": self.cfg.brush_min_motion,
            "coverage_required": self.cfg.coverage_required,
            "final_result": task.final_result.value,
        }

    # ------------------------------------------------------------------
    # 清理与事件
    # ------------------------------------------------------------------
    def _cleanup(self, ts: float) -> None:
        remove: list[str] = []
        for task_id, task in self._tasks.items():
            if not task.active:
                finished = task.completed_at or task.stage_since
                if ts - finished >= self.cfg.completed_retain_s:
                    remove.append(task_id)
        for task_id in remove:
            self._tasks.pop(task_id, None)

        if len(self._tasks) <= self.cfg.task_max_active:
            return
        ordered = sorted(
            self._tasks.values(),
            key=lambda item: (item.active, item.created_at),
        )
        for task in ordered[: len(self._tasks) - self.cfg.task_max_active]:
            self._tasks.pop(task.task_id, None)

    def _set_stage(
        self, task: _Task, stage: TaskStage, ts: float, reason: str
    ) -> None:
        if task.stage == stage:
            return
        task.stage = stage
        task.stage_since = ts
        task.reason_code = reason
        task.timeline.append((stage.value, ts))

    def _emit_event(
        self,
        task: _Task,
        *,
        event_type: EventType,
        result: EventResult,
        reason: str,
        extra: dict[str, Any] | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
    ) -> EventRecord | None:
        key = f"{event_type.value}:{result.value}:{reason}"
        if key in task.events_sent:
            return None
        task.events_sent.add(key)
        evidence: dict[str, Any] = {
            "task_id": task.task_id,
            "reason": reason,
            "stage": task.stage.value,
            "pick_status": task.pick_status.value,
            "holding_status": task.holding_status.value,
            "oiling_status": task.oiling_status.value,
            "final_result": task.final_result.value,
            "evidence": list(task.evidence),
            "screw_track_id": task.screw_track_id,
            "hand_track_id": task.hand_track_id,
            "brush_track_id": task.brush_track_id,
            "detector": self.detector_name,
        }
        if extra:
            evidence.update(extra)
        event = EventRecord.create(
            timestamp_start=float(
                start_ts if start_ts is not None else task.created_at
            ),
            timestamp_end=float(end_ts if end_ts is not None else task.stage_since),
            station_id=self.station.station_id,
            camera_id=self.station.camera_id,
            cycle_id=self.cycle_id,
            task_id=task.task_id,
            event_type=event_type,
            result=result,
            evidence=evidence,
        )
        self._events_emitted += 1
        self._logger.info(
            "事件 %s/%s：工位=%s 任务=%s 原因=%s",
            event.event_type.value,
            event.result.value,
            event.station_id,
            event.task_id,
            reason,
        )
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:  # noqa: BLE001
                self._logger.exception("事件回调失败")
        return event

    # ------------------------------------------------------------------
    # 快照与几何
    # ------------------------------------------------------------------
    def _to_snapshot(self, task: _Task) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=task.task_id,
            created_at=task.created_at,
            stage=task.stage,
            stage_since=task.stage_since,
            pick_status=task.pick_status,
            holding_status=task.holding_status,
            oiling_status=task.oiling_status,
            final_result=task.final_result,
            screw_track_id=task.screw_track_id,
            hand_track_id=task.hand_track_id,
            brush_track_id=task.brush_track_id,
            contact_intervals=tuple(
                (float(start), float(end)) for start, end in task.contact_intervals
            ),
            axis_segments=tuple(task.segments),
            coverage=task.coverage,
            effective_brush_duration_s=task.effective_brush_s,
            brush_motion=task.brush_motion,
            confidence=task.confidence,
            reason_code=task.reason_code,
            evidence=tuple(task.evidence),
            picked_at=task.picked_at,
            confirmed_at=task.confirmed_at,
            completed_at=task.completed_at,
            last_seen_ts=task.last_seen_ts,
            timeline=tuple(task.timeline),
        )

    def _tray_snapshot(self) -> TraySnapshot:
        diag = max(1.0, math.hypot(*self._frame_size))
        speed = math.hypot(*self._tray.velocity) / diag
        if not self._tray.matched:
            motion = "UNKNOWN"
        elif speed >= 0.05:
            motion = "MOVING"
        else:
            motion = "STILL"
        return TraySnapshot(
            track_id=self._tray.track_id,
            box=self._tray.predicted_box(
                self._last_frame_ts or self._tray.last_ts
            ),
            confidence=self._tray.confidence or None,
            motion=motion,
            reliable=self._tray.reliable,
            visible=self._tray.matched,
            missing_s=self._tray.missing_s,
            last_seen_ts=self._tray.last_seen_ts or None,
        )

    def _track_snapshots(self) -> tuple[TrackSnapshot, ...]:
        items: list[TrackSnapshot] = []
        for label, tracks in (
            (self.station.detector.screw_label, self._screw_tracks),
            (self.station.detector.hand_label, self._hand_tracks),
            (self.station.detector.brush_label, self._brush_tracks),
        ):
            for track in tracks:
                items.append(
                    TrackSnapshot(
                        track_id=track.track_id,
                        label=label,
                        box=(
                            track.box
                            if track.matched
                            else track.predicted_box(
                                self._last_frame_ts or track.last_ts
                            )
                        ),
                        confidence=track.confidence,
                        observed=track.matched,
                        missing_s=track.missing_s,
                        last_seen_ts=track.last_seen_ts,
                        angle=track.angle,
                        point=track.point,
                    )
                )
        return tuple(items)

    def _inside_roi(self, box: BBox, roi: Roi) -> bool:
        width, height = self._frame_size
        return roi.contains_box_center(box, width, height)

    def _nearest_hand(
        self, screw: Track, hands: list[Track], diag: float
    ) -> Track | None:
        return self._nearest_hand_to_box(screw.box, hands, diag)

    def _nearest_hand_to_box(
        self, box: BBox, hands: list[Track], diag: float
    ) -> Track | None:
        best: Track | None = None
        best_distance = float("inf")
        for hand in hands:
            distance = _center_distance(hand.box, box) / diag
            if distance <= self.cfg.pick_hand_distance and distance < best_distance:
                best_distance = distance
                best = hand
        return best

    # 兼容旧代码的属性名
    @property
    def cfg_hand_distance(self) -> float:
        return self.cfg.pick_hand_distance

    @property
    def _track_hand_distance_max(self) -> float:
        return self.cfg.pick_hand_distance * 2.0


# ----------------------------------------------------------------------
# 几何工具
# ----------------------------------------------------------------------
def _box_center_in(box: BBox, region: BBox) -> bool:
    cx, cy = box.center
    return region.x1 <= cx <= region.x2 and region.y1 <= cy <= region.y2


def _center_distance(a: BBox, b: BBox) -> float:
    ax, ay = a.center
    bx, by = b.center
    return math.hypot(ax - bx, ay - by)


def _project_on_screw(
    point: tuple[float, float], box: BBox, axis: tuple[float, float]
) -> tuple[float, tuple[float, float], tuple[float, float], float]:
    """把点投影到丝杆主轴坐标系。

    返回 ``(轴向坐标, 中心点, 单位轴, 半长)``；轴向坐标单位为像素，
    相对丝杆中心，正值沿主轴方向。
    """

    cx, cy = box.center
    ax, ay = axis
    norm = math.hypot(ax, ay) or 1.0
    ax, ay = ax / norm, ay / norm
    half_length = _projected_half_length(box, (ax, ay))
    t = (point[0] - cx) * ax + (point[1] - cy) * ay
    return t, (cx, cy), (ax, ay), half_length


def _projected_half_length(box: BBox, axis: tuple[float, float]) -> float:
    ax, ay = axis
    cx, cy = box.center
    projections = []
    for px, py in (
        (box.x1, box.y1),
        (box.x2, box.y1),
        (box.x1, box.y2),
        (box.x2, box.y2),
    ):
        projections.append((px - cx) * ax + (py - cy) * ay)
    if not projections:
        return max(1.0, box.size * 0.5)
    return max(1.0, max(abs(min(projections)), abs(max(projections))))


def _point_to_axis_distance(
    point: tuple[float, float], box: BBox, axis: tuple[float, float]
) -> float:
    t, center, (ax, ay), half_length = _project_on_screw(point, box, axis)
    t = max(-half_length, min(half_length, t))
    px = center[0] + ax * t
    py = center[1] + ay * t
    return math.hypot(point[0] - px, point[1] - py)


def _segment_index(axial: float, count: int) -> int:
    """把轴向坐标映射到 0..count-1 的覆盖区段。"""

    if count <= 0:
        return -1
    # 以 ±1 归一化；覆盖区段按可见长度均分。
    ratio = (axial + 1.0) * 0.5
    ratio = max(0.0, min(0.999999, ratio))
    return int(ratio * count)


def _label_counts(detections: tuple[Detection, ...]) -> dict[str, int]:
    """按类别统计本帧目标数，供 UI 核对“所有标签是否都出来了”。"""

    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return dict(sorted(counts.items()))
