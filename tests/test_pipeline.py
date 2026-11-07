from __future__ import annotations

import math
from dataclasses import replace
from itertools import count

import numpy as np

from app.oilcheck.config import PipelineConfig, StationConfig
from app.oilcheck.detector import SimulatedDetector
from app.oilcheck.pipeline import StationPipeline
from app.oilcheck.roi import Roi
from app.oilcheck.types import (
    BBox,
    Detection,
    EventResult,
    EventType,
    FrameDetections,
    TaskResult,
    TaskStage,
)

WIDTH = 1000
HEIGHT = 1000
TRAY_CENTER_Y = 200.0
HAND_TARGET = (350.0, 650.0)
SCREW_X = (300.0, 450.0, 600.0, 750.0)


def _rect(x1, y1, x2, y2) -> Roi:
    roi = Roi.from_dict({"kind": "rect", "points": [[x1, y1], [x2, y2]]})
    assert roi is not None
    return roi


def make_station(exclude: bool = False) -> StationConfig:
    data = {
        "station_id": "ST01",
        "name": "测试工位",
        "camera_id": "cam_test",
        "enabled": True,
        "tray_roi": {"kind": "rect", "points": [[0.2, 0.05], [0.8, 0.35]]},
        "hand_roi": {"kind": "rect", "points": [[0.1, 0.30], [0.6, 0.95]]},
        "brush_roi": {"kind": "rect", "points": [[0.1, 0.30], [0.6, 0.95]]},
        "detector": {
            "kind": "null",
            "screw_label": "screw",
            "hand_label": "hand",
            "brush_head_label": "brush_head",
            "tray_label": "oil_tray",
        },
    }
    if exclude:
        data["exclude_rois"] = [
            {"kind": "rect", "points": [[0.42, 0.15], [0.5, 0.25]]}
        ]
    station = StationConfig.from_dict(data)
    assert station is not None
    return station


def make_pipeline(
    station: StationConfig | None = None, **overrides
) -> tuple[StationPipeline, list]:
    cfg = PipelineConfig(
        pick_hand_distance=0.05,
        pick_comotion_min_s=0.25,
        pick_confirm_s=0.3,
        pick_abort_s=0.5,
        pick_occlusion_timeout_s=0.4,
        pick_window_s=2.0,
        screw_track_min_hits=2,
        brush_min_duration_s=0.5,
        brush_max_gap_s=0.3,
        brush_min_motion=0.15,
        brush_contact_distance=0.06,
        coverage_segments=4,
        coverage_required=0.75,
        identity_timeout_s=1.0,
        holding_lost_timeout_s=0.6,
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


def tray_screw(index: int, *, cy: float = TRAY_CENTER_Y) -> Detection:
    cx = SCREW_X[index]
    return Detection(
        label="screw",
        confidence=0.9,
        box=BBox(cx - 10, cy - 40, cx + 10, cy + 40),
        angle=math.pi * 0.5,
    )


def hand_at(cx: float, cy: float) -> Detection:
    return Detection(
        label="hand",
        confidence=0.9,
        box=BBox(cx - 45, cy - 45, cx + 45, cy + 45),
    )


def held_screw_at(cx: float, cy: float) -> Detection:
    return Detection(
        label="screw",
        confidence=0.88,
        box=BBox(cx - 45, cy - 7, cx + 45, cy + 7),
        angle=0.0,
    )


def brush_at(cx: float, cy: float) -> Detection:
    return Detection(
        label="brush_head",
        confidence=0.88,
        box=BBox(cx - 10, cy - 10, cx + 10, cy + 10),
        point=(cx, cy),
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


def timestamps(start: float = 1.0, step: float = 0.2):
    for i in count():
        yield start + i * step


def occupy(pipeline: StationPipeline, ts_iter, cycles: int = 3) -> float:
    ts = 0.0
    for _ in range(cycles):
        ts = next(ts_iter)
        feed(pipeline, ts, *(tray_screw(i) for i in range(4)))
    return ts


def do_pick(
    pipeline: StationPipeline,
    ts_iter,
    *,
    index: int = 0,
    move_steps: int = 8,
    hold_steps: int = 4,
) -> float:
    occupy(pipeline, ts_iter)
    others = [tray_screw(i) for i in range(4) if i != index]
    start = (SCREW_X[index], TRAY_CENTER_Y)

    # 手靠近丝杆：还不构成取料。
    for _ in range(2):
        ts = next(ts_iter)
        feed(pipeline, ts, hand_at(*start), *(tray_screw(i) for i in range(4)))

    # 手与丝杆共同移动离开涂油盘，进入手持操作区。
    ts = 0.0
    for step in range(1, move_steps + 1):
        ts = next(ts_iter)
        t = step / move_steps
        cx = start[0] + (HAND_TARGET[0] - start[0]) * t
        cy = start[1] + (HAND_TARGET[1] - start[1]) * t
        feed(
            pipeline,
            ts,
            hand_at(cx, cy),
            held_screw_at(cx, cy),
            *others,
        )

    # 进入手持区后持续关联，满足确认时长。
    for _ in range(hold_steps):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            hand_at(*HAND_TARGET),
            held_screw_at(*HAND_TARGET),
            *others,
        )
    return ts


def stroke_brush(
    pipeline: StationPipeline,
    ts_iter,
    *,
    steps: int = 10,
    offsets: tuple[float, ...] = (-40, -20, 0, 20, 40, 20, 0, -20),
) -> float:
    ts = 0.0
    for step in range(steps):
        ts = next(ts_iter)
        offset = offsets[step % len(offsets)]
        feed(
            pipeline,
            ts,
            hand_at(*HAND_TARGET),
            held_screw_at(*HAND_TARGET),
            brush_at(HAND_TARGET[0] + offset, HAND_TARGET[1]),
            *(tray_screw(i) for i in range(1, 4)),
        )
    return ts


def test_pick_requires_hand_comotion_and_leaving_tray():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)

    tasks = pipeline.tasks
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id.startswith("TASK-")
    assert task.pick_status.value == "CONFIRMED"
    assert task.holding_status.value == "HOLDING"
    assert task.stage in {TaskStage.HOLDING, TaskStage.PICK_CONFIRMED}
    assert task.final_result == TaskResult.PROCESSING
    assert task.screw_track_id is not None
    assert task.hand_track_id is not None

    pick_events = [e for e in events if e.event_type == EventType.PICK]
    assert len(pick_events) == 1
    assert pick_events[0].result == EventResult.CONFIRMED
    assert pick_events[0].task_id == task.task_id
    assert "SCREW_LEFT_TRAY" in pick_events[0].evidence["evidence"]


def test_hand_only_never_creates_pick_event():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    occupy(pipeline, ts_iter)

    ts = 0.0
    # 手经过所有丝杆但不带走任何一根，然后离开。
    for step in range(12):
        ts = next(ts_iter)
        cx = 250 + step * 45
        feed(pipeline, ts, hand_at(cx, 250), *(tray_screw(i) for i in range(4)))
    for _ in range(10):
        ts = next(ts_iter)
        feed(pipeline, ts, *(tray_screw(i) for i in range(4)))

    assert [e for e in events if e.event_type == EventType.PICK] == []
    assert pipeline.tasks == ()


def test_occluded_screw_after_leaving_tray_is_pending_not_confirmed():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    occupy(pipeline, ts_iter)
    others = [tray_screw(i) for i in range(1, 4)]

    # 共同移动到刚离开涂油盘的位置。
    for step in range(1, 5):
        ts = next(ts_iter)
        t = step / 8
        cx = SCREW_X[0] + (HAND_TARGET[0] - SCREW_X[0]) * t
        cy = TRAY_CENTER_Y + (HAND_TARGET[1] - TRAY_CENTER_Y) * t
        feed(pipeline, ts, hand_at(cx, cy), held_screw_at(cx, cy), *others)

    # 丝杆完全被遮挡，手仍在手持区。
    ts = 0.0
    for _ in range(5):
        ts = next(ts_iter)
        feed(pipeline, ts, hand_at(300, 450), *others)

    pick_events = [e for e in events if e.event_type == EventType.PICK]
    assert pick_events
    assert all(e.result != EventResult.CONFIRMED for e in pick_events)
    assert pipeline.tasks
    assert pipeline.tasks[0].final_result == TaskResult.PENDING_CONFIRMATION


def test_brush_contact_and_relative_motion_complete_task():
    pipeline, events = make_pipeline(coverage_required=1.0)
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    stroke_brush(pipeline, ts_iter)

    task = pipeline.tasks[0]
    assert task.oiling_status.value == "COMPLETED"
    assert task.final_result == TaskResult.OK
    assert task.stage == TaskStage.OILING_COMPLETED
    assert task.coverage >= pipeline.cfg.coverage_required
    assert all(task.axis_segments)
    assert task.effective_brush_duration_s >= pipeline.cfg.brush_min_duration_s

    kinds = {(e.event_type, e.result) for e in events}
    assert (EventType.PICK, EventResult.CONFIRMED) in kinds
    assert (EventType.BRUSH_CONTACT, EventResult.CONFIRMED) in kinds
    assert (EventType.MANUAL_OILING, EventResult.CONFIRMED) in kinds
    assert (EventType.TASK_RESULT, EventResult.CONFIRMED) in kinds


def test_static_brush_does_not_complete_and_ends_pending():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)

    # 刷头接触但不动。
    ts = 0.0
    for _ in range(8):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            hand_at(*HAND_TARGET),
            held_screw_at(*HAND_TARGET),
            brush_at(HAND_TARGET[0], HAND_TARGET[1]),
            *(tray_screw(i) for i in range(1, 4)),
        )

    task = pipeline.tasks[0]
    assert task.final_result == TaskResult.PROCESSING
    assert task.oiling_status.value in {"CONTACT", "WAITING"}

    # 刷头离开，手和丝杆随后离开画面：证据不足，输出待确认。
    for _ in range(12):
        ts = next(ts_iter)
        feed(pipeline, ts, *(tray_screw(i) for i in range(1, 4)))

    task = pipeline.tasks[0]
    assert task.final_result == TaskResult.PENDING_CONFIRMATION
    results = [e for e in events if e.event_type == EventType.TASK_RESULT]
    assert results and results[-1].result == EventResult.NEEDS_REVIEW


def test_material_box_detections_are_ignored():
    pipeline, events = make_pipeline(make_station(exclude=True))
    ts_iter = timestamps()
    occupy(pipeline, ts_iter, cycles=4)

    # 排除区覆盖 450,200 附近的丝杆：它不应进入跟踪。
    screw_ids = [track.track_id for track in pipeline._screw_tracks]
    assert len(screw_ids) == 3

    ts = 0.0
    for _ in range(6):
        ts = next(ts_iter)
        feed(pipeline, ts, hand_at(450, 200), *(tray_screw(i) for i in range(4)))

    assert all(
        candidate.box.center[0] != 450 for candidate in pipeline._candidates.values()
    )
    assert [e for e in events if e.event_type == EventType.PICK] == []


def test_tray_motion_does_not_break_screw_tracks():
    pipeline, _events = make_pipeline()
    ts_iter = timestamps()
    occupy(pipeline, ts_iter)
    before = {track.track_id for track in pipeline._screw_tracks}
    assert len(before) == 4

    for offset in (20, 40, 60, 60):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            *(
                tray_screw(i, cy=TRAY_CENTER_Y + offset)
                for i in range(4)
            ),
        )

    after = {track.track_id for track in pipeline._screw_tracks}
    assert after == before
    assert pipeline.tasks == ()


def test_simulated_detector_drives_full_flow():
    station = make_station()
    station = replace(station, detector=replace(station.detector, kind="simulated"))
    pipeline, events = make_pipeline(station)
    detector = SimulatedDetector(station, station.detector)
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    ts = 100.0
    for _ in range(400):
        ts += 0.1
        pipeline.process(detections=detector.detect(frame, ts))

    kinds = {(event.event_type, event.result) for event in events}
    assert (EventType.PICK, EventResult.CONFIRMED) in kinds
    assert (EventType.BRUSH_CONTACT, EventResult.CONFIRMED) in kinds
    assert (EventType.MANUAL_OILING, EventResult.CONFIRMED) in kinds
    assert (EventType.TASK_RESULT, EventResult.CONFIRMED) in kinds


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


def test_station_config_roundtrip_without_slots():
    station = make_station()
    restored = StationConfig.from_dict(station.to_dict())
    assert restored is not None
    assert restored.tray_roi == station.tray_roi
    assert restored.hand_roi == station.hand_roi
    assert restored.brush_roi == station.brush_roi
    assert restored.detector.brush_head_label == "brush_head"
    assert not hasattr(restored, "slot_layout")
