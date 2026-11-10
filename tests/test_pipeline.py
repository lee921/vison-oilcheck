"""取料与人工二次涂油判定的端到端测试。

判定只有两条：

1. 手**空手进盘**、**带料离盘** → ``PICK/CONFIRMED``；
2. 取料后手持丝杆、刷头靠近累计 ≥ ``brush_hold_s`` → ``MANUAL_OILING/CONFIRMED``。

测试直接回放 :class:`FrameDetections`，不依赖模型或相机。
"""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import count

import pytest

from app.oilcheck.config import PipelineConfig, StationConfig
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

# 涂油盘区：x 400–600、y 20–520；工作追踪区：x 100–600、y 300–950。
TRAY_SCREW = (500.0, 200.0)
# 涂油盘区现在由模型检出的 oil_station 框决定，测试默认喂这个框。
TRAY_BOX = BBox(400.0, 20.0, 600.0, 520.0)
# 空手进盘位置：离盘内丝杆 270px（归一化 0.19）> hand_screw_max_distance。
ENTRY_HAND = (500.0, 470.0)
# 离盘后的手持位置：仍在工作追踪区内。
HAND_TARGET = (350.0, 650.0)


def make_station(**station_overrides) -> StationConfig:
    data = {
        "station_id": "ST01",
        "name": "测试工位",
        "camera_id": "cam_test",
        "enabled": True,
        "tray_roi": {"kind": "rect", "points": [[0.40, 0.02], [0.60, 0.52]]},
        "work_roi": {"kind": "rect", "points": [[0.10, 0.30], [0.60, 0.95]]},
        "detector": {
            "kind": "null",
            "screw_label": "screw",
            "hand_label": "hand",
            "brush_label": "brush",
            "tray_label": "oil_station",
        },
    }
    data.update(station_overrides)
    station = StationConfig.from_dict(data)
    assert station is not None
    return station


def make_pipeline(
    station: StationConfig | None = None, **overrides
) -> tuple[StationPipeline, list]:
    cfg = PipelineConfig(
        pick_hand_distance=0.05,
        pick_confirm_s=0.3,
        hand_screw_max_distance=0.10,
        held_infer_distance=0.10,
        brush_contact_distance=0.06,
        brush_hold_s=0.5,
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


def hand_at(cx: float, cy: float) -> Detection:
    return Detection(
        label="hand",
        confidence=0.9,
        box=BBox(cx - 45, cy - 45, cx + 45, cy + 45),
    )


def screw_at(cx: float, cy: float) -> Detection:
    return Detection(
        label="screw",
        confidence=0.9,
        box=BBox(cx - 10, cy - 40, cx + 10, cy + 40),
        angle=math.pi * 0.5,
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
        label="brush",
        confidence=0.88,
        box=BBox(cx - 10, cy - 10, cx + 10, cy + 10),
        point=(cx, cy),
    )


def tray_detection(box: BBox) -> Detection:
    return Detection(label="oil_station", confidence=0.9, box=box)


def feed(
    pipeline: StationPipeline,
    ts: float,
    *detections: Detection,
    tray: bool = False,
):
    """喂一帧。``tray=True`` 时附带涂油盘区的 ``oil_station`` 检测。

    盘区默认由手工 ``tray_roi`` 决定（夹具里已配置），这里不额外注入，
    需要验证“检测框扩大盘区”的用例可以显式打开。
    """

    items = list(detections)
    if tray and not any(det.label == "oil_station" for det in items):
        items.append(tray_detection(TRAY_BOX))
    return pipeline.process(
        detections=FrameDetections(
            ts=ts,
            frame_size=(WIDTH, HEIGHT),
            detections=tuple(items),
            source="test",
        )
    )


def timestamps(start: float = 1.0, step: float = 0.2):
    for i in count():
        yield start + i * step


def enter_tray_empty(
    pipeline: StationPipeline, ts_iter, *, screw: Detection, cycles: int = 3
) -> float:
    """手空手进入涂油盘并在盘内停留到满足确认时长。"""

    ts = 0.0
    for step in range(cycles):
        ts = next(ts_iter)
        cx = ENTRY_HAND[0] - step * 20
        feed(
            pipeline,
            ts,
            hand_at(cx, ENTRY_HAND[1] - step * 10),
            screw_at(*TRAY_SCREW),
        )
    return ts


def leave_tray_holding(
    pipeline: StationPipeline,
    ts_iter,
    *,
    tray_screw: Detection,
    target: tuple[float, float] = HAND_TARGET,
    steps: int = 3,
) -> float:
    """手带着丝杆从盘内走到目标位置。"""

    start = (ENTRY_HAND[0] - 40, ENTRY_HAND[1] - 20)
    ts = 0.0
    for step in range(1, steps + 1):
        ts = next(ts_iter)
        t = step / steps
        cx = start[0] + (target[0] - start[0]) * t
        cy = start[1] + (target[1] - start[1]) * t
        feed(pipeline, ts, hand_at(cx, cy), held_screw_at(cx, cy), tray_screw)
    return ts


def do_pick(pipeline: StationPipeline, ts_iter) -> float:
    """完整一次「空手进盘 → 带料离盘」。"""

    tray_screw = screw_at(*TRAY_SCREW)
    enter_tray_empty(pipeline, ts_iter, screw=tray_screw)
    return leave_tray_holding(pipeline, ts_iter, tray_screw=tray_screw)


def feed_brush(
    pipeline: StationPipeline, ts_iter, *, frames: int, near: bool = True
) -> float:
    """手持丝杆，刷头靠近（或远离）若干帧。"""

    ts = 0.0
    for _ in range(frames):
        ts = next(ts_iter)
        brush = (
            brush_at(HAND_TARGET[0] + 10, HAND_TARGET[1])
            if near
            else brush_at(HAND_TARGET[0] + 300, HAND_TARGET[1] - 300)
        )
        feed(
            pipeline,
            ts,
            hand_at(*HAND_TARGET),
            held_screw_at(*HAND_TARGET),
            brush,
        )
    return ts


# ----------------------------------------------------------------------
# 取料
# ----------------------------------------------------------------------
def test_empty_hand_in_then_leaving_with_screw_emits_pick_once():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)

    pick_events = [e for e in events if e.event_type == EventType.PICK]
    assert len(pick_events) == 1
    assert pick_events[0].result == EventResult.CONFIRMED
    assert pick_events[0].evidence["reason"] == "pick_confirmed"
    assert pick_events[0].evidence["hand_track_id"] == pipeline.tasks[0].hand_track_id
    assert pick_events[0].evidence["dwell_in_tray_s"] >= 0.3

    task = pipeline.tasks[0]
    assert task.task_id.startswith("TASK-")
    assert task.stage == TaskStage.PICKED
    assert task.final_result == TaskResult.PROCESSING
    assert task.picked_at is not None
    assert "SCREW_LEFT_TRAY_WITH_HAND" in task.evidence


def test_entering_and_leaving_with_screw_does_not_pick():
    """带料进盘又带料离盘（把丝杆放回去再拿走）不算取料。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    # 手带着丝杆进盘。
    for _ in range(3):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            hand_at(*ENTRY_HAND),
            held_screw_at(*ENTRY_HAND),
            screw_at(*TRAY_SCREW),
        )
    # 带着丝杆离盘。
    for step in range(1, 4):
        ts = next(ts_iter)
        cx = ENTRY_HAND[0] + (HAND_TARGET[0] - ENTRY_HAND[0]) * step / 3
        cy = ENTRY_HAND[1] + (HAND_TARGET[1] - ENTRY_HAND[1]) * step / 3
        feed(
            pipeline,
            ts,
            hand_at(cx, cy),
            held_screw_at(cx, cy),
            screw_at(*TRAY_SCREW),
        )

    assert [e for e in events if e.event_type == EventType.PICK] == []
    assert pipeline.tasks == ()


def test_empty_hand_in_and_empty_hand_out_does_not_pick():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    for _ in range(3):
        ts = next(ts_iter)
        feed(pipeline, ts, hand_at(*ENTRY_HAND), screw_at(*TRAY_SCREW))
    for step in range(1, 4):
        ts = next(ts_iter)
        cx = ENTRY_HAND[0] + (HAND_TARGET[0] - ENTRY_HAND[0]) * step / 3
        cy = ENTRY_HAND[1] + (HAND_TARGET[1] - ENTRY_HAND[1]) * step / 3
        feed(pipeline, ts, hand_at(cx, cy), screw_at(*TRAY_SCREW))

    assert [e for e in events if e.event_type == EventType.PICK] == []
    assert pipeline.tasks == ()


def test_too_short_dwell_in_tray_does_not_pick():
    """手从盘边划过（停留不足 pick_confirm_s）不判取料。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    ts = next(ts_iter)
    feed(pipeline, ts, hand_at(*ENTRY_HAND), screw_at(*TRAY_SCREW))
    ts = next(ts_iter)
    feed(
        pipeline,
        ts,
        hand_at(*HAND_TARGET),
        held_screw_at(*HAND_TARGET),
        screw_at(*TRAY_SCREW),
    )

    assert [e for e in events if e.event_type == EventType.PICK] == []


def test_raw_mode_passes_all_detections_without_events():
    """raw_mode：全量交出检测结果，不建轨迹、不产生事件。"""

    station = make_station()
    station = replace(station, detector=replace(station.detector, raw_mode=True))
    pipeline, events = make_pipeline(station)

    detections = (
        Detection(label="hand", confidence=0.9, box=BBox(10, 10, 60, 60)),
        Detection(label="brush", confidence=0.8, box=BBox(70, 70, 120, 120)),
        tray_detection(BBox(130, 130, 200, 200)),
        screw_at(210, 210),
    )
    snapshot = feed(pipeline, 1.0, *detections)

    assert snapshot.detections == detections
    assert snapshot.tracks == ()
    assert snapshot.tasks == ()
    assert events == []
    assert snapshot.stats["raw_mode"] is True
    assert snapshot.stats["label_counts"] == {
        "brush": 1,
        "hand": 1,
        "oil_station": 1,
        "screw": 1,
    }


def test_raw_mode_skips_roi_filtering():
    """raw_mode 不做 ROI 过滤，画面外的框也照样交出。"""

    station = make_station()
    station = replace(station, detector=replace(station.detector, raw_mode=True))
    pipeline, _ = make_pipeline(station)

    outside = screw_at(10, 10)
    snapshot = feed(pipeline, 1.0, outside, tray=False)
    assert snapshot.detections == (outside,)


def test_targets_outside_work_area_are_not_tracked():
    """工作追踪区外的手不建轨迹，也不参与取料判定。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    for _ in range(6):
        ts = next(ts_iter)
        feed(pipeline, ts, hand_at(50, 50), screw_at(*TRAY_SCREW))

    assert pipeline._hand_tracks == ()
    assert pipeline._visits == {}
    assert [e for e in events if e.event_type == EventType.PICK] == []


def test_tray_detection_extends_tray_region():
    """模型检出的 oil_station 框扩大涂油盘区，框内手仍按“进盘”处理。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    # 盘体检测框放在固定 tray_roi 之外的下方，手只落在检测框内。
    tray_box = BBox(200, 600, 800, 900)
    hand_pos = (500.0, 750.0)
    screw_pos = (300.0, 320.0)
    for _ in range(3):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            tray_detection(tray_box),
            hand_at(*hand_pos),
            screw_at(*screw_pos),
        )

    assert pipeline._visits, "oil_station 框内的手应被当作已进盘"


def test_tray_motion_does_not_break_hand_tracking():
    """盘体上下移动时手部轨迹保持稳定，不产生误取料。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    screw = screw_at(*TRAY_SCREW)

    for offset in (0, 10, 20, 30):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            hand_at(ENTRY_HAND[0], ENTRY_HAND[1] + offset),
            screw,
        )

    assert len(pipeline._hand_tracks) == 1
    assert pipeline._visits
    assert [e for e in events if e.event_type == EventType.PICK] == []


def test_frame_detections_contract_drives_full_flow():
    """按检测器契约（FrameDetections）喂帧即可跑完整流程。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    feed_brush(pipeline, ts_iter, frames=4)

    kinds = {(event.event_type, event.result) for event in events}
    assert kinds == {
        (EventType.PICK, EventResult.CONFIRMED),
        (EventType.MANUAL_OILING, EventResult.CONFIRMED),
    }


# ----------------------------------------------------------------------
# 人工二次涂油
# ----------------------------------------------------------------------
def test_brush_near_long_enough_completes_oiling_and_resets():
    pipeline, events = make_pipeline(brush_hold_s=0.5)
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    feed_brush(pipeline, ts_iter, frames=4)

    oil_events = [e for e in events if e.event_type == EventType.MANUAL_OILING]
    assert len(oil_events) == 1
    assert oil_events[0].result == EventResult.CONFIRMED
    assert oil_events[0].evidence["reason"] == "manual_oiling_completed"
    assert oil_events[0].evidence["brush_hold_s"] >= 0.5
    assert oil_events[0].evidence["required_s"] == 0.5

    task = pipeline.tasks[0]
    assert task.stage == TaskStage.OILED
    assert task.final_result == TaskResult.COMPLETED
    assert task.completed_at is not None

    # 周期结束后没有活动周期，等待下一次取料。
    assert not any(
        t.final_result == TaskResult.PROCESSING for t in pipeline.tasks
    )


def test_brush_briefly_near_does_not_complete_oiling():
    pipeline, events = make_pipeline(brush_hold_s=1.0)
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    # 两帧 0.2s/帧 ≈ 0.4s < 1.0s，然后刷子离开。
    feed_brush(pipeline, ts_iter, frames=2)
    feed_brush(pipeline, ts_iter, frames=3, near=False)

    assert [e for e in events if e.event_type == EventType.MANUAL_OILING] == []
    task = pipeline.tasks[0]
    assert task.stage == TaskStage.PICKED
    assert task.final_result == TaskResult.PROCESSING
    assert 0.0 < task.brush_hold_s < 1.0


def test_second_cycle_is_detected_after_first_completes():
    pipeline, events = make_pipeline(brush_hold_s=0.5)
    ts_iter = timestamps()

    do_pick(pipeline, ts_iter)
    feed_brush(pipeline, ts_iter, frames=4)
    # 第一周期结束，手回到盘边，空手再次进盘取第二根。
    do_pick(pipeline, ts_iter)
    feed_brush(pipeline, ts_iter, frames=4)

    picks = [e for e in events if e.event_type == EventType.PICK]
    oils = [e for e in events if e.event_type == EventType.MANUAL_OILING]
    assert len(picks) == 2
    assert len(oils) == 2
    assert len({e.task_id for e in picks}) == 2
    completed = [t for t in pipeline.tasks if t.final_result == TaskResult.COMPLETED]
    assert len(completed) == 2


def test_oiling_not_started_before_pick():
    """没有取料就没有涂油判定，即使手与刷子一直在一起。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    feed_brush(pipeline, ts_iter, frames=6)

    assert events == []
    assert pipeline.tasks == ()


def test_only_pick_and_manual_oiling_events_are_emitted():
    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    feed_brush(pipeline, ts_iter, frames=4)

    assert {e.event_type for e in events} <= {
        EventType.PICK,
        EventType.MANUAL_OILING,
    }
    for event in events:
        assert "slot" not in str(event.evidence)
        assert "PUT" not in str(event.evidence)


# ----------------------------------------------------------------------
# 配置与 ROI
# ----------------------------------------------------------------------
def test_roi_supports_rect_and_polygon():
    rect = Roi.from_dict({"kind": "rect", "points": [[0.1, 0.2], [0.4, 0.6]]})
    assert rect is not None
    assert rect.contains_point(200, 400, 1000, 1000)
    assert not rect.contains_point(900, 900, 1000, 1000)

    polygon = Roi.from_dict(
        {"kind": "polygon", "points": [[0.1, 0.1], [0.5, 0.1], [0.3, 0.5]]}
    )
    assert polygon is not None
    assert polygon.contains_point(200, 200, 1000, 1000)
    assert not polygon.contains_point(800, 800, 1000, 1000)


def test_station_config_roundtrip_keeps_three_regions():
    station = make_station()
    restored = StationConfig.from_dict(station.to_dict())
    assert restored is not None
    assert restored.tray_roi == station.tray_roi
    assert restored.work_roi == station.work_roi
    assert restored.oiling_roi == station.oiling_roi
    assert restored.detector.brush_label == "brush"
    assert not hasattr(restored, "slot_layout")


def test_screw_briefly_occluded_still_completes_oiling():
    """手持丝杆被手挡住 1~2 帧时用偏移预测续上接触，不打断累计。"""

    pipeline, events = make_pipeline(brush_hold_s=0.6, held_predict_hold_s=0.5)
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)

    ts = 0.0
    # 2 帧可见的接触
    for _ in range(2):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            hand_at(*HAND_TARGET),
            held_screw_at(*HAND_TARGET),
            brush_at(HAND_TARGET[0] + 10, HAND_TARGET[1]),
        )
    # 2 帧丝杆被完全挡住，但手与刷子还在：用偏移预测续上接触。
    for _ in range(2):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            hand_at(*HAND_TARGET),
            brush_at(HAND_TARGET[0] + 10, HAND_TARGET[1]),
        )

    oil_events = [e for e in events if e.event_type == EventType.MANUAL_OILING]
    assert len(oil_events) == 1
    assert pipeline.tasks[0].stage == TaskStage.OILED


def test_screw_occluded_beyond_predict_hold_pauses_accumulation():
    """丝杆被挡超过预测保持窗口后停止累计，周期不结束。

    窗口是 ``held_predict_hold_s × 帧率倍数``，等价于「恒定多少个漏检帧」：
    0.2s/帧时放大 3 倍（0.3s → 0.9s）。阈值设高，避免在窗口内就判完成，
    这样才能观察到「窗口过后累计停住」。
    """

    pipeline, events = make_pipeline(brush_hold_s=5.0, held_predict_hold_s=0.3)
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)

    ts = 0.0
    for _ in range(2):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            hand_at(*HAND_TARGET),
            held_screw_at(*HAND_TARGET),
            brush_at(HAND_TARGET[0] + 10, HAND_TARGET[1]),
        )
    # 丝杆消失（只剩手和刷子）：前几帧靠预测窗口继续累计，之后停住。
    for _ in range(12):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            hand_at(*HAND_TARGET),
            brush_at(HAND_TARGET[0] + 10, HAND_TARGET[1]),
        )
    settled = pipeline.tasks[0].brush_hold_s
    for _ in range(6):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            hand_at(*HAND_TARGET),
            brush_at(HAND_TARGET[0] + 10, HAND_TARGET[1]),
        )

    task = pipeline.tasks[0]
    # 窗口内累计过一段，窗口外不再增长。
    assert settled > 0.0
    assert task.brush_hold_s == pytest.approx(settled, abs=1e-6)
    assert task.stage == TaskStage.PICKED
    assert task.final_result == TaskResult.PROCESSING
    assert [e for e in events if e.event_type == EventType.MANUAL_OILING] == []


def test_hand_outside_tray_carrying_screw_blocks_pick():
    """手先在盘外带着丝杆，再走进盘里又走出来：不算取料。

    这是“带料进盘”的真实形态——盘内静置丝杆会让当帧持有判定恒为真，
    所以必须用进盘前的状态判定。
    """

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    # 手在盘外（涂油盘区上方外的位置）带着丝杆。
    outside = (500.0, 620.0)
    for _ in range(2):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            hand_at(*outside),
            held_screw_at(*outside),
            screw_at(*TRAY_SCREW),
        )
    # 走进盘里。
    for step in range(1, 4):
        ts = next(ts_iter)
        cx = outside[0] + (ENTRY_HAND[0] - outside[0]) * step / 3
        cy = outside[1] + (ENTRY_HAND[1] - outside[1]) * step / 3
        feed(
            pipeline,
            ts,
            hand_at(cx, cy),
            held_screw_at(cx, cy),
            screw_at(*TRAY_SCREW),
        )
    # 再走出来。
    for step in range(1, 4):
        ts = next(ts_iter)
        cx = ENTRY_HAND[0] + (HAND_TARGET[0] - ENTRY_HAND[0]) * step / 3
        cy = ENTRY_HAND[1] + (HAND_TARGET[1] - ENTRY_HAND[1]) * step / 3
        feed(
            pipeline,
            ts,
            hand_at(cx, cy),
            held_screw_at(cx, cy),
            screw_at(*TRAY_SCREW),
        )

    assert [e for e in events if e.event_type == EventType.PICK] == []


# ----------------------------------------------------------------------
# 漏涂（NG）
# ----------------------------------------------------------------------
def test_pick_without_oiling_then_next_pick_marks_previous_cycle_ng():
    """只取料、二次涂油没完成又来下一次取料 ⇒ 上一周期判 NG。"""

    pipeline, events = make_pipeline(brush_hold_s=1.0)
    ts_iter = timestamps()

    do_pick(pipeline, ts_iter)
    # 刷子只短暂靠近，不够 brush_hold_s，然后直接去取下一根。
    feed_brush(pipeline, ts_iter, frames=2)
    do_pick(pipeline, ts_iter)

    picks = [e for e in events if e.event_type == EventType.PICK]
    ng = [e for e in events if e.event_type == EventType.OILING_MISSING]
    assert len(picks) == 2
    assert len(ng) == 1
    assert ng[0].result == EventResult.NEEDS_REVIEW
    assert ng[0].evidence["reason"] == "oiling_missing_superseded_by_pick"
    assert ng[0].evidence["required_s"] == 1.0
    assert 0.0 < ng[0].evidence["brush_hold_s"] < 1.0
    assert "NEXT_PICK_BEFORE_MANUAL_OILING" in ng[0].evidence["evidence"]

    ng_task = pipeline.tasks[0]
    assert ng_task.task_id == ng[0].task_id
    assert ng_task.final_result == TaskResult.NG
    assert ng_task.completed_at is not None
    # 新取料是独立周期，不会被并进上一件。
    assert len(pipeline.tasks) == 2
    assert pipeline.tasks[1].task_id != ng_task.task_id
    assert picks[1].task_id == pipeline.tasks[1].task_id


def test_completed_oiling_before_next_pick_is_not_ng():
    pipeline, events = make_pipeline(brush_hold_s=0.5)
    ts_iter = timestamps()

    do_pick(pipeline, ts_iter)
    feed_brush(pipeline, ts_iter, frames=4)
    do_pick(pipeline, ts_iter)

    assert [e for e in events if e.event_type == EventType.OILING_MISSING] == []
    assert all(t.final_result != TaskResult.NG for t in pipeline.tasks)


def test_ng_cycle_ends_and_new_cycle_can_complete():
    """NG 之后新周期照常能完成涂油，不影响后续判定。"""

    pipeline, events = make_pipeline(brush_hold_s=0.5)
    ts_iter = timestamps()

    do_pick(pipeline, ts_iter)
    do_pick(pipeline, ts_iter)
    feed_brush(pipeline, ts_iter, frames=4)

    ng = [e for e in events if e.event_type == EventType.OILING_MISSING]
    oils = [e for e in events if e.event_type == EventType.MANUAL_OILING]
    assert len(ng) == 1
    assert len(oils) == 1
    assert oils[0].task_id != ng[0].task_id
    assert pipeline.tasks[1].final_result == TaskResult.COMPLETED


def test_hand_track_max_active_is_configurable():
    """手部轨迹名额可配：名额太小会让新手检测框建不了轨迹。"""

    pipeline, _ = make_pipeline(hand_track_max_active=12)
    assert pipeline._hand_tracker._max_active == 12

    pipeline, _ = make_pipeline()
    assert pipeline._hand_tracker._max_active == 8
    assert pipeline._brush_tracker._max_active == 4


# ----------------------------------------------------------------------
# 丝杆锚点跟踪 / 帧率自适应（示例①现场问题回归）
# ----------------------------------------------------------------------
def test_oiling_survives_hand_track_id_loss():
    """手部视觉 ID 被删除（ID Switch）后，涂油仍按丝杆锚点累计。

    现场示例①的真实故障：一个 12 秒周期里手部 ID 换了 6 次，绑定 ID 一旦
    删除，旧实现就再也拿不到丝杆位置，涂油累计停在 0 并误判漏涂。
    """

    pipeline, events = make_pipeline(brush_hold_s=0.5)
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)

    # 刷子靠近（首帧只建轨迹，第二帧才累计），累计成功。
    feed_brush(pipeline, ts_iter, frames=3)
    assert pipeline.tasks[0].brush_hold_s > 0

    # 把绑定的手轨迹从跟踪器里删掉，模拟 ByteTrack 回收旧 ID。
    task = pipeline._cycles[pipeline.tasks[0].task_id]
    pipeline._hand_tracker._tracks.pop(task.hand_track_id, None)

    feed_brush(pipeline, ts_iter, frames=4)

    oil = [e for e in events if e.event_type == EventType.MANUAL_OILING]
    assert len(oil) == 1
    assert pipeline.tasks[0].stage == TaskStage.OILED


def test_screw_anchor_prefers_screw_gripped_by_hand():
    """锚点从「与手框交叠的杆」初始化，而不是盘里任意一根。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    task = pipeline._cycles[pipeline.tasks[0].task_id]

    # 手持位置有一根杆与手框交叠，远处盘内还有一根静置杆。
    ts = next(ts_iter)
    feed(
        pipeline,
        ts,
        hand_at(*HAND_TARGET),
        held_screw_at(*HAND_TARGET),
        screw_at(200.0, 200.0),
    )
    anchor = task.screw_anchor
    assert anchor is not None
    ax, ay = anchor
    assert math.hypot(ax - HAND_TARGET[0], ay - HAND_TARGET[1]) < 60


def test_frame_scale_adapts_to_actual_frame_interval():
    """帧间隔变大时门限按倍率放大，基准帧率下倍率为 1。"""

    pipeline, _ = make_pipeline(max_fps=15)
    ts_iter = timestamps(step=1.0 / 15.0)
    feed(pipeline, next(ts_iter))
    feed(pipeline, next(ts_iter))
    assert pipeline._frame_scale == pytest.approx(1.0, abs=0.05)

    # 0.2s/帧（5fps）→ 3 倍，但不超过 frame_scale_max。
    pipeline, _ = make_pipeline(max_fps=15)
    ts_iter = timestamps(step=0.2)
    feed(pipeline, next(ts_iter))
    feed(pipeline, next(ts_iter))
    assert pipeline._frame_scale == pytest.approx(3.0, abs=0.05)

    # 0.5s/帧（2fps）→ 理论 7.5 倍，被 frame_scale_max 夹到 4。
    pipeline, _ = make_pipeline(max_fps=15, frame_scale_max=4.0)
    ts_iter = timestamps(step=0.5)
    feed(pipeline, next(ts_iter))
    feed(pipeline, next(ts_iter))
    assert pipeline._frame_scale == pytest.approx(4.0, abs=0.05)


def test_brush_gap_bridged_only_when_brush_is_missing():
    """刷子漏检的帧补回时间；刷子检测到但离得远则不补。"""

    pipeline, _ = make_pipeline(brush_hold_s=10.0)
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    task = pipeline._cycles[pipeline.tasks[0].task_id]

    feed_brush(pipeline, ts_iter, frames=3)
    first = task.brush_hold_s
    assert first > 0

    # 刷子整帧漏检（只有手和丝杆）→ 下一次靠近把空档补回来。
    ts = next(ts_iter)
    feed(pipeline, ts, hand_at(*HAND_TARGET), held_screw_at(*HAND_TARGET))
    feed_brush(pipeline, ts_iter, frames=1)
    bridged = task.brush_hold_s
    assert bridged > first + 0.3

    # 刷子检测到但离得远 → 不累计，并清空桥接点。
    before = task.brush_hold_s
    ts = next(ts_iter)
    feed(
        pipeline,
        ts,
        hand_at(*HAND_TARGET),
        held_screw_at(*HAND_TARGET),
        brush_at(HAND_TARGET[0] + 300, HAND_TARGET[1] - 300),
    )
    assert task.brush_hold_s == pytest.approx(before)
    # 再靠近时只能补回一个被上限截断的空档，不会把「刷子在远处」的那段算进来。
    feed_brush(pipeline, ts_iter, frames=1)
    assert task.brush_hold_s <= before + 0.5 * 3.0 + 1e-6


# ----------------------------------------------------------------------
# 显示范围 / 区外继续跟踪
# ----------------------------------------------------------------------
def test_screws_outside_work_area_are_not_displayed():
    """工作追踪区外的丝杆不出现在画面检测结果里。"""

    pipeline, _ = make_pipeline()
    ts_iter = timestamps()
    feed(pipeline, next(ts_iter))  # 建立盘区

    inside = screw_at(*TRAY_SCREW)
    outside = screw_at(50.0, 50.0)  # 工作追踪区与涂油盘区之外
    snapshot = feed(pipeline, next(ts_iter), inside, outside)

    shown = [d for d in snapshot.detections if d.label == "screw"]
    assert len(shown) == 1
    assert shown[0].box == inside.box


def test_active_screw_stays_visible_outside_area():
    """正在判定的那根丝杆即使移到区外也要画出来。

    手把杆举起涂油时杆会跑出工作追踪区，按区过滤会让画面上的杆消失。
    """

    pipeline, _ = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)

    # 把锚点挪到工作追踪区外（模拟手举出区）。
    task = pipeline._cycles[pipeline.tasks[0].task_id]
    task.screw_anchor = (50.0, 50.0)
    task.screw_anchor_ts = pipeline._last_frame_ts

    snapshot = feed(
        pipeline,
        next(ts_iter),
        hand_at(50.0, 50.0),
        screw_at(50.0, 50.0),
    )
    assert any(
        d.label == "screw" and math.hypot(*d.box.center) < 100
        for d in snapshot.detections
    )


def test_target_leaving_work_area_keeps_track():
    """目标从工作追踪区内移出后，轨迹继续存在（不再一出去就断轨）。"""

    pipeline, _ = make_pipeline()
    ts_iter = timestamps()

    # 区内的手，移动到区外（work_roi 是 x 100–600、y 300–950）。
    for cy in (700.0, 700.0, 650.0, 550.0, 420.0):
        ts = next(ts_iter)
        feed(pipeline, ts, hand_at(300.0, cy))
    tracked_inside = len(pipeline._hand_tracks)
    assert tracked_inside >= 1

    # 连续走出工作追踪区（y < 300）。
    for cy in (250.0, 180.0, 120.0, 60.0):
        ts = next(ts_iter)
        feed(pipeline, ts, hand_at(300.0, cy))

    assert len(pipeline._hand_tracks) >= 1
    assert any(track.matched for track in pipeline._hand_tracks)


def test_outside_area_does_not_create_new_tracks():
    """区外目标不会新建轨迹：跟踪范围仍然受控。"""

    pipeline, _ = make_pipeline()
    ts_iter = timestamps()

    # 手只在区外，从没进过区。
    for _ in range(5):
        feed(pipeline, next(ts_iter), hand_at(50.0, 50.0))

    assert pipeline._hand_tracks == ()
