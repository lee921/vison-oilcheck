from __future__ import annotations

import math
from dataclasses import replace
from itertools import count

import numpy as np

from app.oilcheck.config import (
    DetectorConfig,
    PipelineConfig,
    StationConfig,
    default_slot_layouts,
)
from app.oilcheck.detector import SimulatedDetector
from app.oilcheck.pipeline import StationPipeline, _Operation, _part_suffix
from app.oilcheck.roi import Roi
from app.oilcheck.tracker import Track
from app.oilcheck.types import (
    BBox,
    Detection,
    EventResult,
    EventType,
    FrameDetections,
    OperationState,
    SlotOccupancy,
    Visibility,
)

WIDTH = 1000
HEIGHT = 1000
TRAY_BOX = BBox(200, 100, 800, 400)


def _rect(x1, y1, x2, y2) -> Roi:
    roi = Roi.from_dict({"kind": "rect", "points": [[x1, y1], [x2, y2]]})
    assert roi is not None
    return roi


def make_station() -> StationConfig:
    station = StationConfig.from_dict(
        {
            "station_id": "ST01",
            "name": "测试工位",
            "camera_id": "cam_test",
            "enabled": True,
            "slot_mode": "tray",
            "pick_roi": {"kind": "rect", "points": [[0.10, 0.05], [0.90, 0.50]]},
            "oiling_roi": {"kind": "rect", "points": [[0.10, 0.65], [0.60, 0.95]]},
            "tray_roi": {"kind": "rect", "points": [[0.10, 0.05], [0.90, 0.50]]},
            "slot_layout": [layout.to_dict() for layout in default_slot_layouts(4)],
            "detector": {
                "kind": "null",
                "hand_label": "hand",
                "rod_label": "rod",
                "brush_label": "brush",
                "tray_label": "tray",
            },
        }
    )
    assert station is not None
    return station


def make_pipeline(
    station: StationConfig | None = None, **overrides
) -> tuple[StationPipeline, list]:
    cfg = PipelineConfig(
        slot_occupied_confirm_s=0.2,
        slot_empty_confirm_s=0.3,
        slot_occlusion_unknown_s=10.0,
        occlusion_timeout_s=1.0,
        pick_abort_s=0.5,
        pick_window_s=2.0,
        brush_min_duration_s=0.5,
        brush_max_gap_s=0.3,
        brush_min_motion=0.03,
        brush_part_distance=0.12,
        brush_require_hand=True,
        hand_track_max_distance=0.2,
        brush_track_max_distance=0.2,
        identity_timeout_s=2.0,
        event_cooldown_s=0.0,
        completed_retain_s=30.0,
    )
    if overrides:
        cfg = replace(cfg, **overrides)
    events: list = []
    pipeline = StationPipeline(
        station=station or make_station(),
        pipeline_cfg=cfg,
        on_event=events.append,
        cycle_id="TEST",
        detector_name="test",
    )
    return pipeline, events


def slot_box(index: int) -> BBox:
    layouts = make_station().slot_layout
    return layouts[index - 1].pixel_bbox(TRAY_BOX)


def tray() -> Detection:
    return Detection(label="tray", confidence=0.95, box=TRAY_BOX)


def screw(index: int) -> Detection:
    box = slot_box(index)
    cx, cy = box.center
    return Detection(
        label="rod",
        confidence=0.90,
        box=BBox(cx - 12, cy - 40, cx + 12, cy + 40),
    )


def hand_over(index: int) -> Detection:
    box = slot_box(index)
    return Detection(
        label="hand",
        confidence=0.90,
        box=BBox(box.x1 - 15, box.y1 - 10, box.x2 + 15, box.y2 + 10),
    )


def hand_at(cx: float, cy: float) -> Detection:
    return Detection(
        label="hand",
        confidence=0.90,
        box=BBox(cx - 45, cy - 45, cx + 45, cy + 45),
    )


def held_screw_at(cx: float, cy: float) -> Detection:
    return Detection(
        label="rod",
        confidence=0.86,
        box=BBox(cx - 40, cy - 12, cx + 40, cy + 12),
    )


def brush_at(cx: float, cy: float) -> Detection:
    return Detection(
        label="brush",
        confidence=0.88,
        box=BBox(cx - 15, cy - 15, cx + 15, cy + 15),
    )


def feed(pipeline: StationPipeline, ts: float, *detections: Detection):
    return pipeline.process(
        detections=FrameDetections(
            ts=ts,
            frame_size=(WIDTH, HEIGHT),
            detections=tuple(detections),
            source="test",
        )
    )


def timestamps(start: float = 1.0, step: float = 0.1):
    for i in count():
        yield start + i * step


def occupy_all(pipeline: StationPipeline, ts_iter, cycles: int = 4) -> float:
    ts = 0.0
    for _ in range(cycles):
        ts = next(ts_iter)
        feed(pipeline, ts, tray(), *(screw(i) for i in range(1, 5)))
    return ts


def do_pick(
    pipeline: StationPipeline,
    ts_iter,
    *,
    slot_index: int = 1,
) -> float:
    occupy_all(pipeline, ts_iter)
    ts = next(ts_iter)
    feed(
        pipeline,
        ts,
        tray(),
        hand_over(slot_index),
        *(screw(i) for i in range(1, 5)),
    )
    for _ in range(3):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            tray(),
            hand_over(slot_index),
            *(screw(i) for i in range(1, 5) if i != slot_index),
        )
    for _ in range(5):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            tray(),
            *(screw(i) for i in range(1, 5) if i != slot_index),
        )
    return ts


def move_hand_to_oiling(
    pipeline: StationPipeline,
    ts_iter,
    *,
    slot_index: int = 1,
    target: tuple[float, float] = (350.0, 800.0),
) -> float:
    start = slot_box(slot_index).center
    ts = 0.0
    other_screws = [screw(i) for i in range(1, 5) if i != slot_index]
    for step in range(1, 8):
        ts = next(ts_iter)
        t = step / 7.0
        cx = start[0] + (target[0] - start[0]) * t
        cy = start[1] + (target[1] - start[1]) * t
        feed(
            pipeline,
            ts,
            tray(),
            hand_at(cx, cy),
            held_screw_at(cx, cy),
            *other_screws,
        )
    return ts


def test_slot_occupancy_needs_temporal_evidence_and_holds_on_unknown():
    pipeline, _events = make_pipeline()
    ts_iter = timestamps()

    feed(pipeline, next(ts_iter), tray(), *(screw(i) for i in range(1, 5)))
    assert pipeline.slots[0].occupancy == SlotOccupancy.UNKNOWN

    ts = 0.0
    for _ in range(3):
        ts = next(ts_iter)
        feed(pipeline, ts, tray(), *(screw(i) for i in range(1, 5)))
    assert all(slot.occupancy == SlotOccupancy.OCCUPIED for slot in pipeline.slots)

    # 槽位 1 短时没有丝杆：仍保持 OCCUPIED。
    ts = next(ts_iter)
    feed(pipeline, ts, tray(), *(screw(i) for i in range(2, 5)))
    assert pipeline.slots[0].occupancy == SlotOccupancy.OCCUPIED

    # 盘体不可靠时暂停切换，不把“没看到”解释为 EMPTY。
    for _ in range(3):
        ts = next(ts_iter)
        feed(pipeline, ts, *(screw(i) for i in range(2, 5)))
    assert pipeline.slots[0].occupancy == SlotOccupancy.OCCUPIED
    assert pipeline.slots[0].visibility in {
        Visibility.UNRELIABLE,
        Visibility.VISIBLE,
    }


def test_pick_confirmed_creates_operation_and_logical_id():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter, slot_index=1)

    operations = pipeline.operations
    assert len(operations) == 1
    operation = operations[0]
    assert operation.logical_part_id == "Part-A"
    assert operation.source_slot == 1
    assert operation.state == OperationState.PICKED
    assert operation.pickup_status == "CONFIRMED"

    pick_events = [e for e in events if e.event_type == EventType.PICK]
    assert len(pick_events) == 1
    assert pick_events[0].result == EventResult.CONFIRMED
    assert pick_events[0].evidence["source_slot"] == 1


def test_empty_slot_without_hand_does_not_create_operation():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    occupy_all(pipeline, ts_iter)

    ts = 0.0
    for _ in range(8):
        ts = next(ts_iter)
        feed(pipeline, ts, tray(), *(screw(i) for i in range(2, 5)))

    assert pipeline.slots[0].occupancy == SlotOccupancy.EMPTY
    assert pipeline.operations == ()
    assert [e for e in events if e.event_type == EventType.PICK] == []


def test_tray_missing_marks_slots_unknown_but_not_empty():
    pipeline, _events = make_pipeline(slot_occlusion_unknown_s=0.5)
    ts_iter = timestamps()
    occupy_all(pipeline, ts_iter)

    ts = 0.0
    for _ in range(8):
        ts = next(ts_iter)
        # 只丢失盘体检测，丝杆检测仍在：不能把定位失效解释为空槽。
        feed(pipeline, ts, *(screw(i) for i in range(1, 5)))

    assert all(slot.occupancy == SlotOccupancy.UNKNOWN for slot in pipeline.slots)
    assert all(slot.reason in {"NO_TRAY", "TRAY_UNRELIABLE"} for slot in pipeline.slots)


def test_occlusion_timeout_pending_then_confirmed_when_slot_visible_empty():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    occupy_all(pipeline, ts_iter)

    ts = 0.0
    for _ in range(14):
        ts = next(ts_iter)
        feed(pipeline, ts, tray(), hand_over(1), *(screw(i) for i in range(2, 5)))
    pending = [e for e in events if e.event_type == EventType.PICK]
    assert pending and pending[0].result == EventResult.NEEDS_REVIEW
    assert pipeline.operations
    assert pipeline.operations[0].pickup_status == "PENDING"

    for _ in range(6):
        ts = next(ts_iter)
        feed(pipeline, ts, tray(), *(screw(i) for i in range(2, 5)))

    assert pipeline.operations
    assert pipeline.operations[0].pickup_status == "CONFIRMED"
    confirmed = [
        e
        for e in events
        if e.event_type == EventType.PICK and e.result == EventResult.CONFIRMED
    ]
    assert confirmed


def test_brush_relative_motion_confirms_secondary_oiling():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    move_hand_to_oiling(pipeline, ts_iter)

    ts = 0.0
    for i in range(10):
        ts = next(ts_iter)
        # 丝杆保持不动，刷子在物料附近往复移动。
        offset = 55.0 if i % 2 == 0 else -55.0
        feed(
            pipeline,
            ts,
            tray(),
            hand_at(350, 800),
            held_screw_at(350, 800),
            brush_at(350 + offset, 800 + (10 if i % 3 else -10)),
            *(screw(index) for index in range(2, 5)),
        )

    operation = pipeline.operations[0]
    assert operation.state == OperationState.OILING_CONFIRMED
    oil_events = [e for e in events if e.event_type == EventType.MANUAL_OILING]
    assert len(oil_events) == 1
    assert oil_events[0].result == EventResult.CONFIRMED
    assert oil_events[0].evidence["reason"] == "brush_relative_motion_satisfied"


def test_static_brush_is_pending_not_confirmed():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    move_hand_to_oiling(pipeline, ts_iter)

    ts = 0.0
    for _ in range(8):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            tray(),
            hand_at(350, 800),
            held_screw_at(350, 800),
            brush_at(350, 800),
            *(screw(index) for index in range(2, 5)),
        )

    for _ in range(6):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            tray(),
            hand_at(350, 800),
            held_screw_at(350, 800),
            *(screw(index) for index in range(2, 5)),
        )

    operation = pipeline.operations[0]
    assert operation.state == OperationState.PENDING_CONFIRMATION
    oil_events = [e for e in events if e.event_type == EventType.MANUAL_OILING]
    assert oil_events and oil_events[0].result == EventResult.NEEDS_REVIEW
    assert oil_events[0].evidence["reason"] == "insufficient_brush_evidence"


def test_other_slots_stay_occupied_while_manual_operation_runs():
    pipeline, _events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter, slot_index=1)
    move_hand_to_oiling(pipeline, ts_iter)

    ts = 0.0
    for i in range(10):
        ts = next(ts_iter)
        offset = 55.0 if i % 2 == 0 else -55.0
        snapshot = feed(
            pipeline,
            ts,
            tray(),
            hand_at(350, 800),
            held_screw_at(350, 800),
            brush_at(350 + offset, 800),
            *(screw(index) for index in range(2, 5)),
        )
        assert all(
            slot.occupancy == SlotOccupancy.OCCUPIED
            for slot in snapshot.slots
            if slot.slot_index != 1
        )

    assert pipeline.operations[0].state == OperationState.OILING_CONFIRMED


def test_one_brush_near_two_parts_becomes_identity_uncertain():
    pipeline, _events = make_pipeline()
    diag = math.hypot(WIDTH, HEIGHT)
    first = _Operation(
        operation_id="op-1",
        logical_part_id="Part-A",
        source_slot=1,
        slot_name="1号槽",
        state=OperationState.PICKED,
        state_since=1.0,
        picked_at=1.0,
        part_box=BBox(400, 700, 440, 720),
        part_last_seen_ts=1.0,
    )
    second = _Operation(
        operation_id="op-2",
        logical_part_id="Part-B",
        source_slot=2,
        slot_name="2号槽",
        state=OperationState.PICKED,
        state_since=1.0,
        picked_at=1.0,
        part_box=BBox(460, 700, 500, 720),
        part_last_seen_ts=1.0,
    )
    pipeline._brush_tracks = (
        Track(
            track_id=1,
            box=BBox(435, 690, 465, 710),
            last_ts=1.0,
            last_seen_ts=1.0,
            hits=2,
            matched=True,
            confidence=0.9,
        ),
    )
    assigned = pipeline._assign_brushes([first, second], ts=1.1, diag=diag)

    assert assigned == {}
    assert first.state == OperationState.IDENTITY_UNCERTAIN
    assert second.state == OperationState.IDENTITY_UNCERTAIN
    assert first.reason_code == "MULTI_PART_BRUSH_AMBIGUITY"
    assert second.reason_code == "MULTI_PART_BRUSH_AMBIGUITY"


def test_simulated_detector_drives_full_flow():
    station = StationConfig.from_dict(
        {
            **make_station().to_dict(),
            "detector": {
                "kind": "simulated",
                "hand_label": "hand",
                "rod_label": "rod",
                "brush_label": "brush",
                "tray_label": "tray",
                "cycle_seconds": 18.0,
            },
        }
    )
    assert station is not None
    pipeline, events = make_pipeline(station)
    detector = SimulatedDetector(station, station.detector)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    ts = 100.0
    for _ in range(260):
        ts += 0.1
        pipeline.process(detections=detector.detect(frame, ts))

    kinds = {(event.event_type, event.result) for event in events}
    assert (EventType.PICK, EventResult.CONFIRMED) in kinds
    assert (EventType.MANUAL_OILING, EventResult.CONFIRMED) in kinds


def test_roi_supports_rect_and_polygon():
    rect = _rect(0.1, 0.2, 0.4, 0.6)
    assert rect.contains_point(200, 400, 1000, 1000)
    assert not rect.contains_point(900, 900, 1000, 1000)

    polygon = Roi.from_dict(
        {"kind": "polygon", "points": [[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]]}
    )
    assert polygon is not None
    assert polygon.contains_point(200, 200, 1000, 1000)
    assert not polygon.contains_point(800, 800, 1000, 1000)


def test_station_config_roundtrip_with_slot_layout():
    station = make_station()
    restored = StationConfig.from_dict(station.to_dict())
    assert restored is not None
    assert len(restored.slot_layout) == 4
    assert restored.tray_roi is not None
    assert restored.slot_mode == "tray"
    assert restored.slot_layout[0].pixel_bbox(TRAY_BOX) == slot_box(1)
    assert restored.detector.brush_label == "brush"


def test_part_suffix():
    assert _part_suffix(1) == "A"
    assert _part_suffix(26) == "Z"
    assert _part_suffix(27) == "AA"
    assert _part_suffix(52) == "AZ"
    assert _part_suffix(53) == "BA"


def make_detector_config() -> DetectorConfig:
    return DetectorConfig()
