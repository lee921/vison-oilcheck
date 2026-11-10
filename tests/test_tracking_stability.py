"""跟踪轨迹稳定性的症状级回归。

每个用例对应一次现场/回放复现出来的误判：掉帧伪造涂油时长、锚点被别人
的手劫持、盘区漏检后外推到画面外、首次进盘被误判成带料进盘、视觉 ID
复活、有效关联门限被 α 截断、Kalman 用 XYAH 把框宽拉回、min_hits 对手部
无效。夹具与 ``tests/test_pipeline.py`` 一致，直接回放 ``FrameDetections``。
"""

from __future__ import annotations

import pytest

from app.oilcheck.bytetrack_adapter import ByteTrackObjectTracker
from app.oilcheck.types import (
    BBox,
    Detection,
    EventType,
    FrameDetections,
    TaskStage,
)
from tests.test_pipeline import (
    ENTRY_HAND,
    HAND_TARGET,
    TRAY_BOX,
    TRAY_SCREW,
    brush_at,
    do_pick,
    feed,
    feed_brush,
    hand_at,
    held_screw_at,
    make_pipeline,
    screw_at,
    timestamps,
)

WIDTH = HEIGHT = 1000


def _near_frame(pipeline, ts: float) -> None:
    """手持丝杆、刷头靠近的一帧（带明确时间戳）。"""

    pipeline.process(
        detections=FrameDetections(
            ts=ts,
            frame_size=(WIDTH, HEIGHT),
            detections=(
                hand_at(*HAND_TARGET),
                held_screw_at(*HAND_TARGET),
                brush_at(HAND_TARGET[0] + 10, HAND_TARGET[1]),
            ),
        )
    )


# ----------------------------------------------------------------------
# A1 / A2：掉帧不能凭空放大累计时长
# ----------------------------------------------------------------------
@pytest.mark.parametrize("gap", [0.95, 2.1])
def test_dropped_frame_does_not_complete_oiling(gap):
    """刷头只看过 3 帧（≈0.2s）时，一帧长间隔不能凑够 brush_hold_s。"""

    pipeline, events = make_pipeline(brush_hold_s=1.0)
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    feed_brush(pipeline, ts_iter, frames=3)

    _near_frame(pipeline, next(ts_iter) + gap)

    # 掉帧那一帧最多补一个可信帧间隔（≈0.4s），不能把 1.0s 一次补齐。
    assert pipeline.tasks[0].brush_hold_s < 1.0
    assert pipeline.tasks[0].stage.value == "PICKED"
    assert EventType.MANUAL_OILING not in [e.event_type for e in events]


def test_long_gap_is_reported_as_unreliable_timing():
    pipeline, _ = make_pipeline()
    ts_iter = timestamps()
    feed(pipeline, next(ts_iter))
    feed(pipeline, next(ts_iter))
    assert pipeline.snapshot().stats["timing_reliable"] is True

    pipeline.process(
        detections=FrameDetections(
            ts=next(ts_iter) + 2.0, frame_size=(WIDTH, HEIGHT), detections=()
        )
    )
    assert pipeline.snapshot().stats["timing_reliable"] is False


def test_low_fps_stream_is_still_trusted_and_accumulates():
    """真正的低帧率不能被当成掉帧：2fps 连续靠近仍应判完成。"""

    pipeline, events = make_pipeline(brush_hold_s=1.0)
    ts_iter = timestamps(step=0.5)
    do_pick(pipeline, ts_iter)
    for _ in range(8):
        feed_brush(pipeline, ts_iter, frames=1)

    assert pipeline.snapshot().stats["timing_reliable"] is True
    assert pipeline.tasks[0].stage.value == "OILED"
    assert [e.event_type for e in events].count(EventType.MANUAL_OILING) == 1


# ----------------------------------------------------------------------
# A3：单帧长间隔不能把门限撑大
# ----------------------------------------------------------------------
def test_single_long_gap_does_not_inflate_frame_scale():
    pipeline, _ = make_pipeline()
    ts_iter = timestamps()
    for _ in range(6):
        feed(pipeline, next(ts_iter))
    before = pipeline._frame_scale

    pipeline.process(
        detections=FrameDetections(
            ts=next(ts_iter) + 0.9, frame_size=(WIDTH, HEIGHT), detections=()
        )
    )

    assert pipeline._frame_scale == pytest.approx(before, abs=0.5)
    step = min(
        pipeline.cfg.screw_anchor_max_step * pipeline._frame_scale,
        pipeline.cfg.screw_anchor_max_step_abs,
    )
    assert step <= pipeline.cfg.screw_anchor_max_step_abs


# ----------------------------------------------------------------------
# A4：锚点不能被别人的手劫持
# ----------------------------------------------------------------------
def test_anchor_not_hijacked_by_another_hand():
    """本周期那只手还在画面里时，另一只手握的杆不能被接成锚点。"""

    pipeline, events = make_pipeline(brush_hold_s=0.5)
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    task = pipeline._cycles[pipeline.tasks[0].task_id]
    anchor_before = task.screw_anchor
    cycle_hand = task.hand_track_id

    # 两只手同时在画面里：本周期那只手停在原处（杆被手遮住），
    # 另一只手在旁边握着另一根杆，刷子刷的是另一根。
    others = (150.0, 700.0)
    for _ in range(6):
        feed(
            pipeline,
            next(ts_iter),
            hand_at(*HAND_TARGET),
            hand_at(*others),
            held_screw_at(*others),
            brush_at(others[0] + 10, others[1]),
        )

    assert task.hand_track_id == cycle_hand
    assert task.screw_anchor is not None
    assert abs(task.screw_anchor[0] - anchor_before[0]) < 60
    assert task.stage.value == "PICKED"
    assert EventType.MANUAL_OILING not in [e.event_type for e in events]


def test_anchor_rebinds_when_cycle_hand_track_is_retired():
    """本周期那只手的轨迹被跟踪器回收后，换绑到新出现的手（同一工人）。"""

    pipeline, events = make_pipeline(brush_hold_s=0.5, max_fps=1)
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    task = pipeline._cycles[pipeline.tasks[0].task_id]

    for _ in range(30):
        feed(pipeline, next(ts_iter))
    assert pipeline._hand_tracker.get(task.hand_track_id) is None

    spot = (180.0, 800.0)
    for _ in range(10):
        feed(
            pipeline,
            next(ts_iter),
            hand_at(*spot),
            held_screw_at(*spot),
            brush_at(spot[0] + 10, spot[1]),
        )

    assert EventType.MANUAL_OILING in [e.event_type for e in events]
    assert task.hand_track_id != 100001 or task.stage.value == "OILED"


# ----------------------------------------------------------------------
# A5：盘体漏检时盘区退回手工标定区
# ----------------------------------------------------------------------
def test_tray_region_falls_back_when_tray_missing():
    pipeline, _ = make_pipeline()
    ts_iter = timestamps(step=1 / 15)
    base = TRAY_BOX
    for index in range(4):
        box = BBox(base.x1, base.y1 + index * 40, base.x2, base.y2 + index * 40)
        pipeline.process(
            detections=FrameDetections(
                ts=next(ts_iter),
                frame_size=(WIDTH, HEIGHT),
                detections=(Detection(label="oil_station", confidence=0.95, box=box),),
            )
        )

    for _ in range(14):
        pipeline.process(
            detections=FrameDetections(
                ts=next(ts_iter), frame_size=(WIDTH, HEIGHT), detections=()
            )
        )

    expected = pipeline.station.tray_roi.pixel_bbox(WIDTH, HEIGHT)
    region = pipeline._tray_region_box(pipeline._last_frame_ts, (WIDTH, HEIGHT))
    assert region is not None
    assert region == expected
    assert pipeline.snapshot().stats["tray_reliable"] is False


# ----------------------------------------------------------------------
# A6：首次被跟踪到就在盘内，仍按「盘外见过空手」判定
# ----------------------------------------------------------------------
def test_first_tracked_frame_inside_tray_still_picks():
    pipeline, events = make_pipeline()
    ts_iter = timestamps(step=0.2)

    # 盘外空手两帧（第一帧还没建轨迹），进盘当帧已握住盘内那根杆。
    feed(pipeline, next(ts_iter), hand_at(500, 560), screw_at(500, 200))
    feed(pipeline, next(ts_iter), hand_at(500, 560), screw_at(500, 200))
    for step in range(3):
        feed(
            pipeline,
            next(ts_iter),
            hand_at(500, 250 - step * 10),
            screw_at(500, 250),
        )

    for cx, cy in ((470, 340), (430, 440), (390, 540), (350, 640), (330, 660)):
        feed(pipeline, next(ts_iter), hand_at(cx, cy), screw_at(cx, cy))
    # 离盘后继续手持几帧，让盘外握持窗口攒够 k×n 个采样点。
    for _ in range(4):
        feed(pipeline, next(ts_iter), hand_at(300.0, 700.0), held_screw_at(300.0, 700.0))

    assert [e.event_type for e in events].count(EventType.PICK) == 1


# ----------------------------------------------------------------------
# A8：视觉 ID 宣布丢失后不得复活
# ----------------------------------------------------------------------
def test_retired_track_id_never_revives():
    tracker = ByteTrackObjectTracker(
        max_distance=0.15, max_missing_s=0.8, buffer_frames=45
    )

    def det(cx: float) -> Detection:
        return Detection(
            label="hand", confidence=0.9, box=BBox(cx, 300, cx + 100, 400)
        )

    for index in range(2):
        tracker.update(
            ts=1.0 + index * 0.066,
            detections=(det(500),),
            frame_size=(2560, 1440),
        )
    retired = tracker.active_tracks()[0].track_id

    for step in range(1, 24):
        tracker.update(
            ts=1.0 + step * 0.066, detections=(), frame_size=(2560, 1440)
        )
    assert tracker.get(retired) is None

    out = tracker.update(ts=2.5, detections=(det(500),), frame_size=(2560, 1440))
    assert all(track.track_id != retired for track in out.active)


# ----------------------------------------------------------------------
# A9：有效关联门限与配置一致
# ----------------------------------------------------------------------
def test_distance_gate_matches_configured_value():
    """max_distance=0.15 时应能在 0.14 归一化处接上，而不是 0.10。"""

    tracker = ByteTrackObjectTracker(max_distance=0.15, match_thresh=1.0)
    diag = 2937.0  # 2560×1440 的画面

    def det(cx: float) -> Detection:
        return Detection(
            label="hand", confidence=0.9, box=BBox(cx, 300, cx + 100, 400)
        )

    cx = 300.0
    matched = True
    for index in range(6):
        out = tracker.update(
            ts=1.0 + index / 15, detections=(det(cx),), frame_size=(2560, 1440)
        )
        if index:
            track = next((t for t in out.active if t.track_id == 1), None)
            matched = matched and bool(track and track.matched)
        cx += 0.14 * diag

    assert matched


# ----------------------------------------------------------------------
# A10：低置信度检测能维持已有轨迹（第二阶段）
# ----------------------------------------------------------------------
def test_low_confidence_fast_target_keeps_id():
    tracker = ByteTrackObjectTracker(
        max_distance=0.15, match_thresh=1.0, high_thresh=0.3, low_thresh=0.1
    )

    def det(cx: float, conf: float) -> Detection:
        return Detection(
            label="hand", confidence=conf, box=BBox(cx, 300, cx + 100, 400)
        )

    ids: set[int] = set()
    cx = 300.0
    for index, conf in enumerate([0.9, 0.9, 0.15, 0.15, 0.15, 0.9]):
        out = tracker.update(
            ts=1.0 + index / 15, detections=(det(cx, conf),), frame_size=(2560, 1440)
        )
        ids.update(track.track_id for track in out.active if track.matched)
        cx += 120.0

    assert ids == {1}


# ----------------------------------------------------------------------
# A11：Kalman 不能把框宽拉回
# ----------------------------------------------------------------------
def test_xywh_kalman_tracks_box_growth():
    tracker = ByteTrackObjectTracker(max_distance=0.3, match_thresh=1.0)
    out = None
    for index, width in enumerate([40, 40, 80, 160, 200, 200, 200]):
        out = tracker.update(
            ts=1.0 + index / 15,
            detections=(
                Detection(
                    label="brush",
                    confidence=0.9,
                    box=BBox(300 - width / 2, 300 - width / 2, 300 + width / 2, 300 + width / 2),
                ),
            ),
            frame_size=(2560, 1440),
        )
    assert out is not None
    assert out.active[0].box.width >= 180.0


# ----------------------------------------------------------------------
# A12：min_hits 对手部生效；名额满时先淘汰漏检轨迹
# ----------------------------------------------------------------------
def test_single_frame_hand_does_not_create_visit():
    pipeline, _ = make_pipeline(hand_track_min_hits=2)
    ts_iter = timestamps()

    feed(pipeline, next(ts_iter), hand_at(500, 470), screw_at(*TRAY_SCREW))

    assert pipeline._visits == {}
    assert pipeline.snapshot().stats["hand_visits"] == 0


def test_full_slot_evicts_longest_missing_track():
    tracker = ByteTrackObjectTracker(
        max_distance=0.15, max_active=1, max_missing_s=0.3
    )

    def det(cx: float) -> Detection:
        return Detection(
            label="hand", confidence=0.9, box=BBox(cx, 300, cx + 100, 400)
        )

    for index in range(2):
        tracker.update(
            ts=1.0 + index * 0.066, detections=(det(100),), frame_size=(2560, 1440)
        )

    ids: set[int] = set()
    for step in range(1, 4):
        out = tracker.update(
            ts=1.0 + step * 0.066,
            detections=(det(2400),),
            frame_size=(2560, 1440),
        )
        ids.update(track.track_id for track in out.active if track.matched)

    assert 1 not in ids
    assert ids


# ----------------------------------------------------------------------
# P1-15：判定用原始检测框，不用滞后的滤波框
# ----------------------------------------------------------------------
def test_judgement_uses_raw_detection_box():
    """业务判定读的是本帧原始检测框，不是滞后的滤波框。

    快速移动时 ``track.box``（卡尔曼状态）只走到一半；用它做边界判定会晚
    一帧。这里直接验证两个框确实不同、且 ``observation_box()`` 取的是原始
    检测框。
    """

    pipeline, _ = make_pipeline()
    ts_iter = timestamps()
    for _ in range(3):
        feed(pipeline, next(ts_iter), hand_at(*ENTRY_HAND))

    jump = 120.0
    feed(pipeline, next(ts_iter), hand_at(ENTRY_HAND[0] + jump, 470))

    hand = next(
        (t for t in pipeline._hand_tracks if t.track_id == 100001), None
    )
    assert hand is not None and hand.matched and hand.raw_box is not None
    # 滤波框明显滞后于本帧检测（只走了一部分），判定必须用后者。
    assert hand.raw_box.center == pytest.approx(
        (ENTRY_HAND[0] + jump, 470.0), abs=1.0
    )
    assert hand.box.center[0] < hand.raw_box.center[0] - 20.0
    assert hand.observation_box() == hand.raw_box


# ----------------------------------------------------------------------
# 取料判定：单帧异常不决定结论（历史 n 帧投票）
# ----------------------------------------------------------------------
def test_pick_survives_single_frame_holding_dropout():
    """离盘那一刻杆检测闪断一帧（置信度掉到 0.5 附近）：仍应判取料。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    # 空手走近 → 进盘（5 帧，窗口内多数空手）。
    for _ in range(5):
        feed(pipeline, next(ts_iter), hand_at(350.0, 650.0))
    for _ in range(5):
        feed(pipeline, next(ts_iter), hand_at(*ENTRY_HAND), screw_at(*TRAY_SCREW))

    # 带杆离盘：中间一帧杆检测消失（手把杆完全遮住）。
    spots = ((470, 520), (430, 570), (390, 610), (360, 640), (330, 660))
    for index, (cx, cy) in enumerate(spots):
        if index == 2:
            feed(pipeline, next(ts_iter), hand_at(cx, cy))
        else:
            feed(pipeline, next(ts_iter), hand_at(cx, cy), held_screw_at(cx, cy))
    # 离盘后继续手持几帧，让盘外握持窗口攒够 k×n 个采样点。
    for _ in range(4):
        feed(
            pipeline,
            next(ts_iter),
            hand_at(300.0, 700.0),
            held_screw_at(300.0, 700.0),
        )

    assert [e.event_type for e in events].count(EventType.PICK) == 1


def test_pick_not_split_by_single_frame_boundary_flicker():
    """盘区边线抖一帧：同一件取料不能被拆成两段。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    for _ in range(5):
        feed(pipeline, next(ts_iter), hand_at(350.0, 650.0))
    # 盘内 4 帧，第 3 帧中心刚越界 5px。
    for index in range(4):
        cx = 605.0 if index == 2 else ENTRY_HAND[0]
        feed(
            pipeline,
            next(ts_iter),
            hand_at(cx, ENTRY_HAND[1]),
            screw_at(*TRAY_SCREW),
        )
    for cx, cy in ((470, 520), (430, 570), (390, 610), (360, 640), (330, 660)):
        feed(pipeline, next(ts_iter), hand_at(cx, cy), held_screw_at(cx, cy))
    # 离盘后继续手持几帧，让盘外握持窗口攒够 k×n 个采样点。
    for _ in range(4):
        feed(
            pipeline,
            next(ts_iter),
            hand_at(300.0, 700.0),
            held_screw_at(300.0, 700.0),
        )

    assert [e.event_type for e in events].count(EventType.PICK) == 1


def test_history_window_is_configurable():
    """k / n 来自配置，且默认值能装下现场节拍。"""

    pipeline, _ = make_pipeline()
    assert pipeline.cfg.pick_sample_every_frames >= 1
    assert pipeline.cfg.pick_history_samples >= 3


# ----------------------------------------------------------------------
# A9：离盘后手很快出画面 / 视频结束，取料不能被「采样点不足」吞掉
# ----------------------------------------------------------------------
def _enter_and_leave(pipeline, ts_iter, *, exit_frames: int) -> None:
    """空手进盘 → 带杆离盘 exit_frames 帧（走完就断，不额外补帧）。"""

    tray_screw = screw_at(*TRAY_SCREW)
    for _ in range(3):
        feed(pipeline, next(ts_iter), hand_at(350.0, 650.0))
    for step in range(6):
        feed(
            pipeline,
            next(ts_iter),
            hand_at(ENTRY_HAND[0] - step * 15, ENTRY_HAND[1] - step * 10),
            tray_screw,
        )
    for step in range(1, exit_frames + 1):
        t = step / 6
        cx = (ENTRY_HAND[0] - 40) + (HAND_TARGET[0] - (ENTRY_HAND[0] - 40)) * t
        cy = (ENTRY_HAND[1] - 20) + (HAND_TARGET[1] - (ENTRY_HAND[1] - 20)) * t
        feed(pipeline, next(ts_iter), hand_at(cx, cy), held_screw_at(cx, cy), tray_screw)


def test_pick_survives_hand_leaving_frame_right_after_exit():
    """离盘后手立刻出画面（轨迹被回收），盘外采样一个点都没攒到：仍判取料。

    取料判定要求离盘后攒满 k×n 个**盘外**采样点。手带着丝杆走出画面时
    这些点根本采不到，之前「点数不足按空手处理」会把整件吞掉——连作业
    周期都建不出来，后面的涂油判定自然无从谈起（现场表现为取料没跑完
    就停了）。
    """

    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    _enter_and_leave(pipeline, ts_iter, exit_frames=3)

    # 画面里只剩盘体，手一直不再出现。
    for _ in range(20):
        feed(pipeline, next(ts_iter), screw_at(*TRAY_SCREW))

    assert [e.event_type for e in events].count(EventType.PICK) == 1
    task = pipeline.tasks[0]
    assert task.stage == TaskStage.PICKED
    assert "SCREW_LEFT_TRAY_WITH_HAND" in task.evidence


def test_empty_hand_after_exit_still_does_not_pick():
    """离盘后手里没有杆：点数不足也不能靠兜底造出取料事件。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    tray_screw = screw_at(*TRAY_SCREW)
    for _ in range(3):
        feed(pipeline, next(ts_iter), hand_at(350.0, 650.0))
    for step in range(6):
        feed(
            pipeline,
            next(ts_iter),
            hand_at(ENTRY_HAND[0] - step * 15, ENTRY_HAND[1] - step * 10),
            tray_screw,
        )
    # 空手走出盘区，随后消失。
    for step in range(1, 4):
        t = step / 6
        cx = (ENTRY_HAND[0] - 40) + (HAND_TARGET[0] - (ENTRY_HAND[0] - 40)) * t
        cy = (ENTRY_HAND[1] - 20) + (HAND_TARGET[1] - (ENTRY_HAND[1] - 20)) * t
        feed(pipeline, next(ts_iter), hand_at(cx, cy), tray_screw)
    for _ in range(20):
        feed(pipeline, next(ts_iter), tray_screw)

    assert [e.event_type for e in events].count(EventType.PICK) == 0
