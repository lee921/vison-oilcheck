"""机器涂油盘、4 个动态槽位与逐根物料二次涂油状态机。

对应设计文档《工业视觉检测系统设计文档_手部刷子与丝杆二次涂油》：

- 机器涂油盘（tray）跨帧跟踪，盘体上下移动时 4 个槽位 ROI 随盘体更新；
- 每个槽位独立维护 OCCUPIED / EMPTY / UNKNOWN，遮挡或盘体定位不可靠时
  暂停状态切换，不把“没看到物料”直接解释为 EMPTY；
- 手部进入 OCCUPIED 槽位邻域、随后槽位确认变空，才建立人工取料操作记录；
- 刷子相对被取走物料的运动用于累计二次涂油证据，手部/刷子/物料关联不确定
  时输出 PENDING_CONFIRMATION 或 IDENTITY_UNCERTAIN，不强行判定完成；
- 机器盘槽位状态与人工操作记录完全并行，机器继续处理其他槽位不会阻塞人工
  二次涂油。

本模块不依赖 Qt、相机 SDK 或具体推理框架，测试直接回放
:class:`FrameDetections`。
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import PipelineConfig, SlotLayout, StationConfig, default_slot_layouts
from .tracker import ObjectTracker, Track, TrayTrack, TrayTracker
from .types import (
    BBox,
    Detection,
    EventRecord,
    EventResult,
    EventType,
    FrameDetections,
    OperationSnapshot,
    OperationState,
    SlotOccupancy,
    SlotSnapshot,
    StationSnapshot,
    TrackSnapshot,
    TraySnapshot,
    Visibility,
)

EventCallback = Callable[[EventRecord], None]


@dataclass
class _SlotRuntime:
    layout: SlotLayout
    occupancy: SlotOccupancy = SlotOccupancy.UNKNOWN
    visibility: Visibility = Visibility.UNRELIABLE
    confidence: float | None = None
    state_since: float = 0.0
    last_change_ts: float = 0.0
    reason: str = "INITIAL"
    box: BBox | None = None
    observed_ts: float | None = None

    occupied_since: float | None = None
    empty_since: float | None = None
    occluded_since: float | None = None
    unreliable_since: float | None = None
    last_screw_seen_ts: float | None = None

    hand_near_since: float | None = None
    hand_track_id: int | None = None
    last_hand_near_ts: float | None = None
    hand_left_ts: float | None = None
    pick_candidate_since: float | None = None
    pick_candidate_hand_id: int | None = None
    pick_interaction_started_ts: float | None = None


@dataclass
class _Operation:
    operation_id: str
    logical_part_id: str
    source_slot: int
    slot_name: str
    state: OperationState
    state_since: float
    picked_at: float
    interaction_start_ts: float | None = None
    pickup_status: str = "CONFIRMED"
    identity_status: str = "ASSOCIATED"
    confidence: float | None = None
    reason_code: str | None = None
    evidence: list[str] = field(default_factory=list)
    hand_track_id: int | None = None
    brush_track_id: int | None = None
    last_seen_ts: float | None = None

    part_box: BBox | None = None
    part_last_seen_ts: float | None = None
    brush_last_ts: float | None = None
    brush_accum_s: float = 0.0
    brush_motion: float = 0.0
    brush_last_relative: tuple[float, float] | None = None
    events_sent: set[str] = field(default_factory=set)

    @property
    def active(self) -> bool:
        return self.state not in (OperationState.OILING_CONFIRMED,)


class StationPipeline:
    """单相机/单工位的盘体、槽位与人工操作状态机。"""

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

        now = time.time()
        layouts = station.slot_layout or default_slot_layouts(4)
        self._slots: dict[int, _SlotRuntime] = {
            layout.index: _SlotRuntime(
                layout=layout,
                state_since=now,
                last_change_ts=now,
                reason="INITIAL",
            )
            for layout in layouts
        }
        self._operations: dict[str, _Operation] = {}
        self._operation_seq = 0
        self._part_seq = 0

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

        self._last_frame_ts: float | None = None
        self._last_detections: tuple[Detection, ...] = ()
        self._frame_size: tuple[int, int] = (0, 0)
        self._hand_tracks: tuple[Track, ...] = ()
        self._brush_tracks: tuple[Track, ...] = ()
        self._frames_processed = 0
        self._events_emitted = 0

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    @property
    def operations(self) -> tuple[OperationSnapshot, ...]:
        return tuple(
            self._to_snapshot(op)
            for op in sorted(
                self._operations.values(),
                key=lambda item: (item.picked_at, item.operation_id),
            )
        )

    @property
    def materials(self) -> tuple[OperationSnapshot, ...]:
        return self.operations

    @property
    def slots(self) -> tuple[SlotSnapshot, ...]:
        return tuple(self._slot_snapshot(slot) for slot in self._sorted_slots())

    def reset(self) -> None:
        now = time.time()
        for slot in self._slots.values():
            slot.occupancy = SlotOccupancy.UNKNOWN
            slot.visibility = Visibility.UNRELIABLE
            slot.confidence = None
            slot.state_since = now
            slot.last_change_ts = now
            slot.reason = "RESET"
            slot.box = None
            slot.observed_ts = None
            slot.occupied_since = None
            slot.empty_since = None
            slot.occluded_since = None
            slot.unreliable_since = None
            slot.last_screw_seen_ts = None
            slot.hand_near_since = None
            slot.hand_track_id = None
            slot.last_hand_near_ts = None
            slot.hand_left_ts = None
            slot.pick_candidate_since = None
            slot.pick_candidate_hand_id = None
            slot.pick_interaction_started_ts = None
        self._operations.clear()
        self._operation_seq = 0
        self._part_seq = 0
        self._hand_tracker.reset()
        self._brush_tracker.reset()
        self._tray_tracker.reset()
        self._tray = TrayTrack()
        self._last_frame_ts = None
        self._frame_size = (0, 0)
        self._hand_tracks = ()
        self._brush_tracks = ()
        self._last_detections = ()

    def process(self, *, detections: FrameDetections) -> StationSnapshot:
        ts = float(detections.ts)
        width, height = detections.frame_size
        if width <= 0 or height <= 0:
            return self.snapshot()

        dt = 0.0
        if self._last_frame_ts is not None:
            dt = max(0.0, min(1.0, ts - self._last_frame_ts))
        self._last_frame_ts = ts
        self._frame_size = (int(width), int(height))
        self._last_detections = detections.detections
        self._frames_processed += 1

        cfg = self.station.detector
        raw_hands = detections.by_label(cfg.hand_label)
        raw_brushes = detections.by_label(cfg.brush_label)
        screws = detections.by_label(cfg.rod_label)
        trays = detections.by_label(cfg.tray_label)

        hand_out = self._hand_tracker.update(
            ts=ts,
            detections=raw_hands,
            frame_size=(int(width), int(height)),
        )
        brush_out = self._brush_tracker.update(
            ts=ts,
            detections=raw_brushes,
            frame_size=(int(width), int(height)),
        )
        self._hand_tracks = hand_out.active
        self._brush_tracks = brush_out.active
        self._tray = self._tray_tracker.update(
            ts=ts,
            detections=trays,
            frame_size=(int(width), int(height)),
        )

        tray_box = self._tray.predicted_box(ts)
        tray_reliable = bool(self._tray.reliable and tray_box is not None)
        for slot in self._sorted_slots():
            slot.box = (
                slot.layout.pixel_bbox(tray_box) if tray_box is not None else None
            )
            self._update_slot(
                slot,
                ts=ts,
                dt=dt,
                width=int(width),
                height=int(height),
                tray_reliable=tray_reliable,
                screws=screws,
            )

        self._update_operations(
            ts=ts,
            dt=dt,
            width=int(width),
            height=int(height),
            screws=screws,
        )
        self._cleanup_operations(ts)
        return self.snapshot()

    def snapshot(self) -> StationSnapshot:
        return StationSnapshot(
            station_id=self.station.station_id,
            station_name=self.station.name,
            camera_id=self.station.camera_id,
            mode=self.detector_name,
            online=False,
            operations=self.operations,
            slots=self.slots,
            tray=self._tray_snapshot(),
            tracks=self._track_snapshots(),
            detections=self._last_detections,
            frame_size=self._frame_size,
            updated_at=self._last_frame_ts or 0.0,
            stats={
                "frames_processed": self._frames_processed,
                "events_emitted": self._events_emitted,
                "operations": len(self._operations),
                "tray_reliable": self._tray.reliable,
            },
        )

    # ------------------------------------------------------------------
    # 槽位状态
    # ------------------------------------------------------------------
    def _update_slot(
        self,
        slot: _SlotRuntime,
        *,
        ts: float,
        dt: float,
        width: int,
        height: int,
        tray_reliable: bool,
        screws: tuple[Detection, ...],
    ) -> None:
        del dt
        box = slot.box
        if box is None:
            self._mark_unreliable(slot, ts, "NO_TRAY")
            return
        if not tray_reliable:
            self._mark_unreliable(slot, ts, "TRAY_UNRELIABLE")
            return

        slot.unreliable_since = None
        occluding_hand = self._first_hand_covering(box, width, height)
        hand_near = self._first_hand_near(box, width, height)
        screw = _best_detection_in_box(screws, box)
        if hand_near is not None:
            if slot.hand_near_since is None:
                slot.hand_near_since = ts
            slot.hand_track_id = hand_near.track_id
            slot.last_hand_near_ts = ts
            if slot.occupancy == SlotOccupancy.OCCUPIED:
                if slot.pick_candidate_since is None:
                    slot.pick_candidate_since = ts
                    slot.pick_candidate_hand_id = hand_near.track_id
                    slot.pick_interaction_started_ts = ts
        else:
            if slot.hand_near_since is not None:
                slot.hand_left_ts = ts
            slot.hand_near_since = None
            if (
                slot.pick_candidate_since is not None
                and slot.hand_left_ts is not None
                and slot.occupancy == SlotOccupancy.OCCUPIED
                and ts - slot.hand_left_ts >= self.cfg.pick_abort_s
            ):
                slot.pick_candidate_since = None
                slot.pick_candidate_hand_id = None

        if occluding_hand is not None:
            slot.visibility = Visibility.OCCLUDED
            if screw is not None:
                # 手部遮挡但丝杆仍然可见：有明确占用证据，不因遮挡触发待确认。
                slot.last_screw_seen_ts = ts
                slot.observed_ts = ts
                slot.occluded_since = None
                slot.empty_since = None
                if slot.occupied_since is None:
                    slot.occupied_since = ts
                if (
                    slot.occupancy != SlotOccupancy.OCCUPIED
                    and ts - slot.occupied_since >= self.cfg.slot_occupied_confirm_s
                ):
                    self._set_occupancy(
                        slot,
                        SlotOccupancy.OCCUPIED,
                        ts,
                        "SCREW_STABLE_UNDER_HAND",
                        screw.confidence,
                    )
                return
            slot.observed_ts = None
            slot.occupied_since = None
            slot.empty_since = None
            if slot.occluded_since is None:
                slot.occluded_since = ts
            occluded_s = ts - slot.occluded_since
            if (
                slot.occupancy == SlotOccupancy.OCCUPIED
                and slot.pick_candidate_since is not None
                and occluded_s >= self.cfg.occlusion_timeout_s
            ):
                self._create_pending_pick(slot, ts, "PICK_OCCLUSION_TIMEOUT")
            if (
                occluded_s >= self.cfg.slot_occlusion_unknown_s
                and slot.occupancy != SlotOccupancy.UNKNOWN
            ):
                self._set_occupancy(
                    slot, SlotOccupancy.UNKNOWN, ts, "OCCLUSION_TIMEOUT", None
                )
            return

        slot.visibility = Visibility.VISIBLE
        slot.occluded_since = None
        if screw is not None:
            slot.last_screw_seen_ts = ts
            slot.observed_ts = ts
            slot.empty_since = None
            if slot.occupied_since is None:
                slot.occupied_since = ts
            if (
                slot.occupancy != SlotOccupancy.OCCUPIED
                and ts - slot.occupied_since >= self.cfg.slot_occupied_confirm_s
            ):
                self._set_occupancy(
                    slot,
                    SlotOccupancy.OCCUPIED,
                    ts,
                    "SCREW_STABLE",
                    screw.confidence,
                )
                self._cancel_pending_pick(slot, ts, "SCREW_REAPPEARED")
            return

        slot.occupied_since = None
        slot.observed_ts = ts
        if slot.empty_since is None:
            slot.empty_since = ts
        if (
            slot.occupancy != SlotOccupancy.EMPTY
            and ts - slot.empty_since >= self.cfg.slot_empty_confirm_s
        ):
            previous = slot.occupancy
            confidence = self._tray.confidence or None
            self._set_occupancy(
                slot, SlotOccupancy.EMPTY, ts, "CLEAR_EMPTY_EVIDENCE", confidence
            )
            if previous in (SlotOccupancy.OCCUPIED, SlotOccupancy.UNKNOWN):
                self._on_slot_empty(slot, ts)

    def _mark_unreliable(self, slot: _SlotRuntime, ts: float, reason: str) -> None:
        slot.visibility = Visibility.UNRELIABLE
        slot.observed_ts = None
        slot.occupied_since = None
        slot.empty_since = None
        if slot.unreliable_since is None:
            slot.unreliable_since = ts
        if (
            ts - slot.unreliable_since >= self.cfg.slot_occlusion_unknown_s
            and slot.occupancy != SlotOccupancy.UNKNOWN
        ):
            self._set_occupancy(slot, SlotOccupancy.UNKNOWN, ts, reason, None)

    def _set_occupancy(
        self,
        slot: _SlotRuntime,
        occupancy: SlotOccupancy,
        ts: float,
        reason: str,
        confidence: float | None,
    ) -> None:
        if slot.occupancy == occupancy:
            return
        old = slot.occupancy
        slot.occupancy = occupancy
        slot.state_since = ts
        slot.last_change_ts = ts
        slot.reason = reason
        slot.confidence = confidence
        self._logger.debug(
            "槽位 %s 占用 %s -> %s (%s)",
            slot.layout.index,
            old.value,
            occupancy.value,
            reason,
        )

    def _on_slot_empty(self, slot: _SlotRuntime, ts: float) -> None:
        pending = self._pending_pick_for_slot(slot.layout.index)
        if pending is not None:
            self._confirm_pending_pick(pending, ts)
            return
        if slot.pick_candidate_since is None and slot.last_hand_near_ts is None:
            return
        last_hand = slot.last_hand_near_ts or slot.pick_candidate_since
        if last_hand is None or ts - last_hand > self.cfg.pick_window_s:
            return
        hand_id = slot.pick_candidate_hand_id or slot.hand_track_id
        self._create_operation(
            slot,
            ts=ts,
            hand_track_id=hand_id,
            pickup_status="CONFIRMED",
            state=OperationState.PICKED,
            reason="PICK_CONFIRMED",
            extra_evidence=("SLOT_EMPTY_AFTER_HAND_INTERACTION",),
            interaction_ts=slot.pick_interaction_started_ts,
        )
        slot.pick_candidate_since = None
        slot.pick_candidate_hand_id = None
        slot.pick_interaction_started_ts = None
        slot.last_hand_near_ts = None
        slot.hand_track_id = None

    # ------------------------------------------------------------------
    # 取料候选与待确认
    # ------------------------------------------------------------------
    def _create_pending_pick(
        self, slot: _SlotRuntime, ts: float, reason: str
    ) -> _Operation | None:
        if self._pending_pick_for_slot(slot.layout.index) is not None:
            return None
        operation = self._create_operation(
            slot,
            ts=ts,
            hand_track_id=slot.pick_candidate_hand_id or slot.hand_track_id,
            pickup_status="PENDING",
            state=OperationState.PENDING_CONFIRMATION,
            reason=reason,
            extra_evidence=("HAND_INTERACTION", "SLOT_OCCLUDED"),
            interaction_ts=slot.pick_interaction_started_ts,
        )
        self._emit_event(
            operation,
            ts=ts,
            event_type=EventType.PICK,
            result=EventResult.NEEDS_REVIEW,
            reason=reason,
            extra={"occlusion_timeout_s": self.cfg.occlusion_timeout_s},
            start_ts=operation.interaction_start_ts or operation.picked_at,
        )
        return operation

    def _cancel_pending_pick(self, slot: _SlotRuntime, ts: float, reason: str) -> None:
        pending = self._pending_pick_for_slot(slot.layout.index)
        if pending is None:
            return
        # 取料候选已被否定：运行列表移除该操作，但已发出的待复核事件仍保留。
        self._operations.pop(pending.operation_id, None)
        self._logger.info(
            "取消取料候选 %s（%s，槽位=%s）",
            pending.operation_id,
            reason,
            slot.layout.index,
        )
        del ts
        slot.pick_candidate_since = None
        slot.pick_candidate_hand_id = None
        slot.pick_interaction_started_ts = None
        slot.last_hand_near_ts = None
        slot.hand_track_id = None

    def _confirm_pending_pick(self, operation: _Operation, ts: float) -> None:
        operation.state = OperationState.PICKED
        operation.state_since = ts
        operation.pickup_status = "CONFIRMED"
        operation.reason_code = "PICK_CONFIRMED_AFTER_OCCLUSION"
        operation.evidence.append("SLOT_EMPTY_AFTER_OCCLUSION")
        slot = self._slots.get(operation.source_slot)
        if slot is not None:
            slot.pick_candidate_since = None
            slot.pick_candidate_hand_id = None
            slot.pick_interaction_started_ts = None
            slot.last_hand_near_ts = None
            slot.hand_track_id = None
        self._emit_event(
            operation,
            ts=ts,
            event_type=EventType.PICK,
            result=EventResult.CONFIRMED,
            reason="pick_confirmed_after_occlusion",
            extra={"pickup_status": operation.pickup_status},
            start_ts=operation.interaction_start_ts or operation.picked_at,
        )

    def _pending_pick_for_slot(self, slot_index: int) -> _Operation | None:
        for operation in self._operations.values():
            if (
                operation.source_slot == slot_index
                and operation.pickup_status == "PENDING"
            ):
                return operation
        return None

    def _create_operation(
        self,
        slot: _SlotRuntime,
        *,
        ts: float,
        hand_track_id: int | None,
        pickup_status: str,
        state: OperationState,
        reason: str,
        extra_evidence: tuple[str, ...] = (),
        interaction_ts: float | None = None,
    ) -> _Operation:
        self._operation_seq += 1
        self._part_seq += 1
        operation_id = f"op-{self.station.station_id}-{self._operation_seq:04d}"
        logical_part_id = f"Part-{_part_suffix(self._part_seq)}"
        operation = _Operation(
            operation_id=operation_id,
            logical_part_id=logical_part_id,
            source_slot=slot.layout.index,
            slot_name=slot.layout.name,
            state=state,
            state_since=ts,
            picked_at=ts,
            interaction_start_ts=interaction_ts,
            pickup_status=pickup_status,
            hand_track_id=hand_track_id,
            confidence=slot.confidence,
            reason_code=reason,
            evidence=[
                "HAND_NEAR_SLOT",
                "PICK_CANDIDATE",
                *extra_evidence,
            ],
            last_seen_ts=ts,
        )
        self._operations[operation_id] = operation
        self._logger.info(
            "建立人工操作 %s（物料=%s，槽位=%s，手部轨迹=%s）",
            operation_id,
            logical_part_id,
            slot.layout.index,
            hand_track_id,
        )
        if pickup_status == "CONFIRMED":
            self._emit_event(
                operation,
                ts=ts,
                event_type=EventType.PICK,
                result=EventResult.CONFIRMED,
                reason="pick_confirmed",
                extra={
                    "pickup_status": pickup_status,
                    "hand_track_id": hand_track_id,
                },
                start_ts=interaction_ts or slot.pick_candidate_since or ts,
            )
        return operation

    # ------------------------------------------------------------------
    # 手部/刷子关联与二次涂油
    # ------------------------------------------------------------------
    def _update_operations(
        self,
        *,
        ts: float,
        dt: float,
        width: int,
        height: int,
        screws: tuple[Detection, ...],
    ) -> None:
        diag = max(1.0, math.hypot(width, height))
        active = [op for op in self._operations.values() if op.active]
        if not active:
            return

        for operation in active:
            self._refresh_part_observation(operation, ts=ts, diag=diag, screws=screws)

        assignments = self._assign_brushes(active, ts=ts, diag=diag)
        for operation in active:
            if operation.pickup_status != "CONFIRMED":
                continue
            brush = assignments.get(operation.operation_id)
            if brush is not None:
                self._accumulate_oiling(
                    operation,
                    brush=brush,
                    ts=ts,
                    dt=dt,
                    diag=diag,
                )
            else:
                self._settle_oiling_without_brush(operation, ts=ts)

    def _refresh_part_observation(
        self,
        operation: _Operation,
        *,
        ts: float,
        diag: float,
        screws: tuple[Detection, ...],
    ) -> None:
        hand = self._hand_tracker.get(operation.hand_track_id)
        if (
            hand is None or hand.missing_s > self.cfg.hand_track_max_missing_s
        ) and operation.part_box is not None:
            observed = [track for track in self._hand_tracks if track.matched]
            if len(observed) == 1:
                distance = (
                    _box_center_distance(observed[0].box, operation.part_box) / diag
                )
                if distance <= self.cfg.hand_track_max_distance:
                    hand = observed[0]
                    operation.hand_track_id = hand.track_id
                    operation.identity_status = "ASSOCIATED"
                    if "HAND_REASSOCIATED" not in operation.evidence:
                        operation.evidence.append("HAND_REASSOCIATED")
        hand_box = None
        if hand is not None and hand.missing_s <= self.cfg.hand_track_max_missing_s:
            hand_box = hand.box if hand.matched else hand.predicted_box(ts)
            if hand.matched:
                operation.last_seen_ts = ts

        best: Detection | None = None
        best_distance = float("inf")
        for screw in screws:
            if hand_box is not None:
                distance = _box_center_distance(screw.box, hand_box) / diag
            elif operation.part_box is not None:
                distance = _box_center_distance(screw.box, operation.part_box) / diag
            else:
                continue
            if distance <= max(self.cfg.brush_part_distance * 1.5, 0.12):
                if distance < best_distance:
                    best_distance = distance
                    best = screw

        if best is not None:
            operation.part_box = best.box
            operation.part_last_seen_ts = ts
            operation.confidence = best.confidence
            operation.last_seen_ts = ts
            if "PART_OBSERVED" not in operation.evidence:
                operation.evidence.append("PART_OBSERVED")
        elif hand_box is not None:
            # 手部仍在画面中：用手的框作为物料关联代理，但不作为接触证据。
            operation.part_box = hand_box
        else:
            operation.part_box = (
                hand.predicted_box(ts) if hand is not None else operation.part_box
            )

        hand_observed = hand is not None and hand.matched
        if operation.part_last_seen_ts is None:
            if (
                not hand_observed
                and ts - operation.picked_at >= self.cfg.identity_timeout_s
            ):
                self._mark_identity_uncertain(
                    operation, ts, "PART_NOT_OBSERVED_AFTER_PICK"
                )
            return
        if (
            ts - operation.part_last_seen_ts >= self.cfg.identity_timeout_s
            and not hand_observed
        ):
            self._mark_identity_uncertain(operation, ts, "PART_LOST")

    def _assign_brushes(
        self, operations: list[_Operation], *, ts: float, diag: float
    ) -> dict[str, Track]:
        brushes = [
            track
            for track in self._brush_tracks
            if track.matched and track.hits >= self._brush_tracker.min_hits
        ]
        assigned: dict[str, Track] = {}
        if not brushes:
            return assigned

        for brush in brushes:
            candidates: list[tuple[float, _Operation]] = []
            for operation in operations:
                if operation.pickup_status != "CONFIRMED":
                    continue
                if operation.part_box is None:
                    continue
                distance = _box_center_distance(brush.box, operation.part_box) / diag
                if distance <= self.cfg.brush_part_distance:
                    candidates.append((distance, operation))
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0])
            best_distance, best_operation = candidates[0]
            if (
                len(candidates) > 1
                and candidates[1][0] - best_distance
                <= self.cfg.identity_ambiguity_margin
            ):
                for _distance, operation in candidates:
                    self._mark_identity_uncertain(
                        operation, ts, "MULTI_PART_BRUSH_AMBIGUITY"
                    )
                continue
            previous = assigned.get(best_operation.operation_id)
            if previous is not None and previous.track_id != brush.track_id:
                self._mark_identity_uncertain(
                    best_operation, ts, "MULTI_BRUSH_AMBIGUITY"
                )
                continue
            assigned[best_operation.operation_id] = brush
        return assigned

    def _accumulate_oiling(
        self,
        operation: _Operation,
        *,
        brush: Track,
        ts: float,
        dt: float,
        diag: float,
    ) -> None:
        if operation.state == OperationState.IDENTITY_UNCERTAIN:
            operation.state = OperationState.OILING_CANDIDATE
            operation.state_since = ts
            operation.identity_status = "ASSOCIATED"
            operation.reason_code = "IDENTITY_REASSOCIATED"
            operation.evidence.append("IDENTITY_REASSOCIATED")
            self._reset_brush_evidence(operation)

        if operation.state == OperationState.PENDING_CONFIRMATION:
            if operation.pickup_status == "CONFIRMED":
                operation.state = OperationState.OILING_CANDIDATE
                operation.state_since = ts
                operation.reason_code = "OILING_RETRY"
                operation.evidence.append("OILING_RETRY")
                self._reset_brush_evidence(operation)
            else:
                # 取料本身仍待确认，不能提前累计涂油证据。
                return

        if operation.state == OperationState.PICKED:
            operation.state = OperationState.OILING_CANDIDATE
            operation.state_since = ts
            operation.reason_code = "BRUSH_NEAR_PART"
            operation.evidence.append("BRUSH_NEAR_PART")
            self._reset_brush_evidence(operation)

        if (
            operation.brush_track_id is not None
            and operation.brush_track_id != brush.track_id
        ):
            # 刷子轨迹切换时重新累计，避免跨轨迹跳变被当作相对运动。
            self._reset_brush_evidence(operation)

        if not self._hand_ok(operation, ts):
            return

        if operation.part_box is None:
            return
        bcx, bcy = brush.box.center
        pcx, pcy = operation.part_box.center
        relative = (bcx - pcx, bcy - pcy)
        if operation.brush_last_ts is not None:
            gap = max(0.0, ts - operation.brush_last_ts)
            if gap > self.cfg.brush_max_gap_s:
                self._reset_brush_evidence(operation)
            else:
                operation.brush_accum_s += max(gap, dt)
        if operation.brush_last_relative is not None:
            operation.brush_motion += (
                math.hypot(
                    relative[0] - operation.brush_last_relative[0],
                    relative[1] - operation.brush_last_relative[1],
                )
                / diag
            )
        operation.brush_last_relative = relative
        operation.brush_last_ts = ts
        operation.brush_track_id = brush.track_id
        operation.last_seen_ts = ts
        operation.evidence[:] = [
            item
            for item in operation.evidence
            if item not in ("BRUSH_RELATIVE_MOTION", "BRUSH_DURATION_SATISFIED")
        ]
        if operation.brush_motion > 0:
            operation.evidence.append("BRUSH_RELATIVE_MOTION")
        if operation.brush_accum_s >= self.cfg.brush_min_duration_s:
            operation.evidence.append("BRUSH_DURATION_SATISFIED")

        if (
            operation.brush_accum_s >= self.cfg.brush_min_duration_s
            and operation.brush_motion >= self.cfg.brush_min_motion
        ):
            self._confirm_oiling(operation, ts)

    def _settle_oiling_without_brush(self, operation: _Operation, *, ts: float) -> None:
        if operation.state != OperationState.OILING_CANDIDATE:
            return
        last_ts = operation.brush_last_ts or operation.state_since
        if ts - last_ts < self.cfg.brush_max_gap_s:
            return
        candidate_since = operation.state_since
        has_evidence = (
            operation.brush_accum_s >= self.cfg.brush_min_duration_s * 0.3
            or operation.brush_motion >= self.cfg.brush_min_motion * 0.3
        )
        if has_evidence:
            operation.state = OperationState.PENDING_CONFIRMATION
            operation.state_since = ts
            operation.reason_code = "INSUFFICIENT_BRUSH_EVIDENCE"
            operation.evidence.append("INSUFFICIENT_BRUSH_EVIDENCE")
            self._emit_event(
                operation,
                ts=ts,
                event_type=EventType.MANUAL_OILING,
                result=EventResult.NEEDS_REVIEW,
                reason="insufficient_brush_evidence",
                extra={
                    "brush_duration_s": round(operation.brush_accum_s, 3),
                    "brush_motion": round(operation.brush_motion, 4),
                    "min_duration_s": self.cfg.brush_min_duration_s,
                    "min_motion": self.cfg.brush_min_motion,
                },
                start_ts=candidate_since,
            )
        else:
            operation.state = OperationState.PICKED
            operation.state_since = ts
            operation.reason_code = "BRUSH_LEFT_WITHOUT_MOTION"
        self._reset_brush_evidence(operation)

    def _confirm_oiling(self, operation: _Operation, ts: float) -> None:
        candidate_since = operation.state_since
        operation.state = OperationState.OILING_CONFIRMED
        operation.state_since = ts
        operation.reason_code = "BRUSH_RELATIVE_MOTION_SATISFIED"
        operation.evidence.append("OILING_CONFIRMED")
        self._emit_event(
            operation,
            ts=ts,
            event_type=EventType.MANUAL_OILING,
            result=EventResult.CONFIRMED,
            reason="brush_relative_motion_satisfied",
            extra={
                "brush_duration_s": round(operation.brush_accum_s, 3),
                "brush_motion": round(operation.brush_motion, 4),
                "min_duration_s": self.cfg.brush_min_duration_s,
                "min_motion": self.cfg.brush_min_motion,
                "brush_track_id": operation.brush_track_id,
            },
            start_ts=candidate_since,
        )
        self._reset_brush_evidence(operation)

    def _hand_ok(self, operation: _Operation, ts: float) -> bool:
        if not self.cfg.brush_require_hand:
            return True
        hand = self._hand_tracker.get(operation.hand_track_id)
        if hand is None:
            return False
        if hand.missing_s > self.cfg.hand_track_max_missing_s:
            return False
        if not hand.matched:
            return False
        if operation.part_box is not None:
            diag = max(1.0, math.hypot(*self._frame_size))
            distance = _box_center_distance(hand.box, operation.part_box) / diag
            if distance > self.cfg.hand_track_max_distance:
                return False
        del ts
        return True

    def _mark_identity_uncertain(
        self, operation: _Operation, ts: float, reason: str
    ) -> None:
        if operation.state == OperationState.OILING_CONFIRMED:
            return
        if operation.identity_status == "UNCERTAIN" and operation.reason_code == reason:
            return
        operation.identity_status = "UNCERTAIN"
        operation.state = OperationState.IDENTITY_UNCERTAIN
        operation.state_since = ts
        operation.reason_code = reason
        operation.evidence.append(reason)
        self._emit_event(
            operation,
            ts=ts,
            event_type=EventType.MANUAL_OILING,
            result=EventResult.NEEDS_REVIEW,
            reason=reason,
            extra={"identity_status": "UNCERTAIN"},
            start_ts=operation.picked_at,
        )

    def _reset_brush_evidence(self, operation: _Operation) -> None:
        operation.brush_accum_s = 0.0
        operation.brush_motion = 0.0
        operation.brush_last_relative = None
        operation.brush_last_ts = None
        operation.brush_track_id = None

    # ------------------------------------------------------------------
    # 清理与事件
    # ------------------------------------------------------------------
    def _cleanup_operations(self, ts: float) -> None:
        remove: list[str] = []
        for operation_id, operation in self._operations.items():
            if (
                operation.state == OperationState.OILING_CONFIRMED
                and ts - operation.state_since >= self.cfg.completed_retain_s
            ):
                remove.append(operation_id)
        for operation_id in remove:
            self._operations.pop(operation_id, None)

        if len(self._operations) <= self.cfg.operation_max_active:
            return
        ordered = sorted(
            self._operations.values(),
            key=lambda item: (
                item.state == OperationState.OILING_CONFIRMED,
                item.state_since,
            ),
        )
        for operation in ordered[
            : len(self._operations) - self.cfg.operation_max_active
        ]:
            self._operations.pop(operation.operation_id, None)

    def _emit_event(
        self,
        operation: _Operation,
        *,
        ts: float,
        event_type: EventType,
        result: EventResult,
        reason: str,
        extra: dict[str, Any] | None = None,
        start_ts: float | None = None,
    ) -> EventRecord | None:
        key = f"{event_type.value}:{result.value}:{reason}"
        if key in operation.events_sent:
            return None
        operation.events_sent.add(key)
        evidence: dict[str, Any] = {
            "operation_id": operation.operation_id,
            "source_slot": operation.source_slot,
            "pickup_status": operation.pickup_status,
            "identity_status": operation.identity_status,
            "reason": reason,
            "state": operation.state.value,
            "evidence": list(operation.evidence),
            "hand_track_id": operation.hand_track_id,
            "brush_track_id": operation.brush_track_id,
            "detector": self.detector_name,
        }
        if extra:
            evidence.update(extra)
        event = EventRecord.create(
            timestamp_start=float(
                start_ts if start_ts is not None else operation.picked_at
            ),
            timestamp_end=float(ts),
            station_id=self.station.station_id,
            camera_id=self.station.camera_id,
            cycle_id=self.cycle_id,
            logical_part_id=operation.logical_part_id,
            slot_index=operation.source_slot,
            event_type=event_type,
            result=result,
            evidence=evidence,
        )
        self._events_emitted += 1
        self._logger.info(
            "事件 %s/%s：工位=%s 槽位=%s 物料=%s 原因=%s",
            event.event_type.value,
            event.result.value,
            event.station_id,
            event.slot_index,
            event.logical_part_id,
            reason,
        )
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:  # noqa: BLE001
                self._logger.exception("事件回调失败")
        return event

    # ------------------------------------------------------------------
    # 快照与查询
    # ------------------------------------------------------------------
    def _sorted_slots(self) -> list[_SlotRuntime]:
        return [self._slots[key] for key in sorted(self._slots)]

    def _slot_snapshot(self, slot: _SlotRuntime) -> SlotSnapshot:
        return SlotSnapshot(
            slot_index=slot.layout.index,
            slot_name=slot.layout.name,
            occupancy=slot.occupancy,
            visibility=slot.visibility,
            confidence=slot.confidence,
            last_change_ts=slot.last_change_ts,
            reason=slot.reason,
            box=slot.box,
            last_observed_ts=slot.observed_ts,
        )

    def _to_snapshot(self, operation: _Operation) -> OperationSnapshot:
        return OperationSnapshot(
            operation_id=operation.operation_id,
            logical_part_id=operation.logical_part_id,
            source_slot=operation.source_slot,
            slot_name=operation.slot_name,
            state=operation.state,
            state_since=operation.state_since,
            pickup_status=operation.pickup_status,
            identity_status=operation.identity_status,
            confidence=operation.confidence,
            reason_code=operation.reason_code,
            evidence=tuple(operation.evidence),
            hand_track_id=operation.hand_track_id,
            brush_track_id=operation.brush_track_id,
            picked_at=operation.picked_at,
            last_seen_ts=operation.last_seen_ts,
        )

    def _tray_snapshot(self) -> TraySnapshot:
        diag = max(1.0, math.hypot(*self._frame_size))
        speed = math.hypot(*self._tray.velocity) / diag
        if not self._tray.matched:
            motion = "UNKNOWN"
        elif speed >= self.cfg.tray_motion_speed:
            motion = "MOVING"
        else:
            motion = "STILL"
        return TraySnapshot(
            track_id=self._tray.track_id,
            box=self._tray.predicted_box(self._last_frame_ts or self._tray.last_ts),
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
                    )
                )
        return tuple(items)

    # ------------------------------------------------------------------
    # 手部/刷子几何
    # ------------------------------------------------------------------
    def _first_hand_covering(self, box: BBox, width: int, height: int) -> Track | None:
        for hand in self._hand_tracks:
            # 预测位置不能作为遮挡/接触证据；只接受本帧真实观测到的手部。
            if not hand.matched:
                continue
            if _box_overlap_ratio(hand.box, box) >= self.cfg.slot_hand_occlusion_ratio:
                return hand
            hx, hy = hand.box.center
            if box.x1 <= hx <= box.x2 and box.y1 <= hy <= box.y2:
                return hand
        return None

    def _first_hand_near(self, box: BBox, width: int, height: int) -> Track | None:
        diag = max(1.0, math.hypot(width, height))
        margin = self.cfg.slot_hand_margin * diag
        expanded = BBox(
            max(0.0, box.x1 - margin),
            max(0.0, box.y1 - margin),
            min(float(width), box.x2 + margin),
            min(float(height), box.y2 + margin),
        )
        best: Track | None = None
        best_distance = float("inf")
        for hand in self._hand_tracks:
            if not hand.matched:
                continue
            hx, hy = hand.box.center
            if expanded.x1 <= hx <= expanded.x2 and expanded.y1 <= hy <= expanded.y2:
                distance = math.hypot(
                    hx - (box.x1 + box.x2) * 0.5,
                    hy - (box.y1 + box.y2) * 0.5,
                )
                if distance < best_distance:
                    best_distance = distance
                    best = hand
        return best


def _part_suffix(seq: int) -> str:
    """1 -> A, 26 -> Z, 27 -> AA。"""
    seq = max(1, int(seq))
    chars: list[str] = []
    while seq > 0:
        seq, remainder = divmod(seq - 1, 26)
        chars.append(chr(ord("A") + remainder))
    return "".join(reversed(chars))


def _best_detection_in_box(
    detections: tuple[Detection, ...], box: BBox
) -> Detection | None:
    best: Detection | None = None
    best_score = 0.0
    for det in detections:
        cx, cy = det.box.center
        inside = box.x1 <= cx <= box.x2 and box.y1 <= cy <= box.y2
        overlap = det.box.intersection_area(box) / max(
            1.0, det.box.width * det.box.height
        )
        if not inside and overlap < 0.25:
            continue
        score = det.confidence + (0.05 if inside else 0.0)
        if best is None or score > best_score:
            best_score = score
            best = det
    return best


def _box_center_distance(a: BBox, b: BBox) -> float:
    ax, ay = a.center
    bx, by = b.center
    return math.hypot(ax - bx, ay - by)


def _box_overlap_ratio(hand: BBox, slot: BBox) -> float:
    area = max(1.0, slot.width * slot.height)
    return hand.intersection_area(slot) / area
