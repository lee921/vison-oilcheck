"""第二条取料判据：丝杆自己从盘区内移到盘区外。

现场手被遮挡、漏检、或整段没被检出时，「空手进盘→带料离盘」这条判据会把
一次真实取料整件丢掉。丝杆是刚体、检测稳定得多，所以另有一条只看杆的
判据并行运行。它的前提是**丝杆确实在油盘里**——在盘区内放稳过（位置不动）
够 ``screw_exit_pick_inside_s``，然后被检到移到盘区外。

夹具与 ``tests/test_pipeline.py`` 一致，直接回放 ``FrameDetections``。
"""

from __future__ import annotations

from app.oilcheck.types import BBox, Detection, EventType, TaskStage
from tests.test_pipeline import (
    ENTRY_HAND,
    HAND_TARGET,
    feed,
    hand_at,
    make_pipeline,
    screw_at,
    timestamps,
)

# 盘区是 x 400–600、y 20–520。杆先放在盘内远离边线的位置，再移到盘区下方。
PARKED = (500.0, 300.0)
OUTSIDE = (500.0, 700.0)


def _picks(events: list) -> list:
    return [e for e in events if e.event_type == EventType.PICK]


def _park(pipeline, ts_iter, *, frames: int = 14, ts=None) -> float:
    """杆静止在盘区内若干帧（让它「确实在盘里」）。"""

    ts = 0.0
    for _ in range(frames):
        ts = next(ts_iter)
        feed(pipeline, ts, screw_at(*PARKED))
    return ts


def _move_out(pipeline, ts_iter, *, frames: int = 8, start=PARKED, end=OUTSIDE):
    """杆从盘内连续移动到盘区外，并在外停留几帧。"""

    ts = 0.0
    for step in range(1, frames + 1):
        t = step / frames
        cx = start[0] + (end[0] - start[0]) * t
        cy = start[1] + (end[1] - start[1]) * t
        ts = next(ts_iter)
        feed(pipeline, ts, screw_at(cx, cy))
    for _ in range(4):
        ts = next(ts_iter)
        feed(pipeline, ts, screw_at(*end))
    return ts


def test_screw_moved_out_of_tray_counts_as_pick_without_any_hand():
    """画面里一根手都没有，杆自己从盘内移到盘外 ⇒ 取料。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    _park(pipeline, ts_iter)
    _move_out(pipeline, ts_iter)

    picks = _picks(events)
    assert len(picks) == 1
    assert picks[0].evidence["pick_source"] == "screw_exit"
    assert "SCREW_MOVED_OUT_OF_TRAY" in picks[0].evidence["evidence"]
    task = pipeline.tasks[0]
    assert task.stage == TaskStage.PICKED
    assert task.hand_track_id is None


def test_screw_never_inside_tray_is_not_a_pick():
    """杆一直在盘区外移动（零件盒、台面上的待刷杆）：不判取料。

    「确实在油盘里」是这条规则的前提。去掉它的话，工人举着杆从盘前走过、
    或画面里的待刷杆被摆来摆去，都会被判成从盘里取料。
    """

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    for step in range(20):
        feed(pipeline, next(ts_iter), screw_at(200.0 + step * 20, 800.0))

    assert _picks(events) == []


def test_screw_passing_through_tray_quickly_is_not_a_pick():
    """杆只是从盘区上方划过、没在盘里停稳 ⇒ 不判取料。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    # 从盘区左上角快速斜穿到右下角，途中一直在动（不是放稳）。
    for step in range(8):
        t = step / 7
        feed(
            pipeline,
            next(ts_iter),
            screw_at(420.0 + 160.0 * t, 60.0 + 440.0 * t),
        )
    for _ in range(6):
        feed(pipeline, next(ts_iter), screw_at(600.0, 560.0))

    assert _picks(events) == []


def test_screw_touching_tray_edge_flicker_is_not_a_pick():
    """杆心贴着盘区边线来回抖（±margin 以内）：不算出盘。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    _park(pipeline, ts_iter)
    # 盘区下沿 y=520。在 525 / 515 之间抖，越界量 5px < margin(0.01×1414≈14px)。
    for index in range(30):
        cy = 525.0 if index % 2 else 515.0
        feed(pipeline, next(ts_iter), screw_at(500.0, cy))

    assert _picks(events) == []


def test_screw_lifted_and_put_back_is_not_a_pick():
    """杆被举出盘区又放回原处（比如重新摆料）：不判取料。

    杆在盘外停留的时间短于确认时长就先回盘了，这一趟出盘作废。
    """

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    _park(pipeline, ts_iter)
    # 只短暂出盘一帧，立刻回到盘内原位。
    feed(pipeline, next(ts_iter), screw_at(500.0, 600.0))
    for _ in range(10):
        feed(pipeline, next(ts_iter), screw_at(*PARKED))

    assert _picks(events) == []

    # 之后真的带走，才判这一次取料。
    _move_out(pipeline, ts_iter)
    assert len(_picks(events)) == 1


def test_second_screw_leaving_after_first_counts_separately():
    """盘里两根杆先后被带走：各判一次。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    second = (560.0, 240.0)
    for _ in range(14):
        feed(pipeline, next(ts_iter), screw_at(*PARKED), screw_at(*second))

    # 第一根被带走。
    for step in range(1, 9):
        t = step / 8
        feed(
            pipeline,
            next(ts_iter),
            screw_at(PARKED[0], PARKED[1] + (720.0 - PARKED[1]) * t),
            screw_at(*second),
        )
    for _ in range(4):
        feed(pipeline, next(ts_iter), screw_at(PARKED[0], 720.0), screw_at(*second))

    assert len(_picks(events)) == 1

    # 第二根随后也被带走。
    for step in range(1, 9):
        t = step / 8
        feed(pipeline, next(ts_iter), screw_at(second[0], second[1] + (720.0 - second[1]) * t))
    for _ in range(4):
        feed(pipeline, next(ts_iter), screw_at(second[0], 720.0))

    assert len(_picks(events)) == 2


def test_screw_exit_disabled_restores_old_behaviour():
    """``screw_exit_pick_enabled=false`` 时整条关闭，只看手，与旧版本一致。"""

    pipeline, events = make_pipeline(screw_exit_pick_enabled=False)
    ts_iter = timestamps()

    _park(pipeline, ts_iter)
    _move_out(pipeline, ts_iter)

    assert _picks(events) == []


def test_hand_and_screw_rules_do_not_double_report_one_pick():
    """手部判据与丝杆判据判的是同一件料：只报一次取料。

    手空手进盘、带料离盘的同时，杆也从盘内移到盘外——两条判据都会确认。
    去重窗口内同一位置只应留下一条 PICK。
    """

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    # 盘内放一根杆（手取走的就是它）。
    for _ in range(14):
        feed(pipeline, next(ts_iter), screw_at(*PARKED))
    # 空手进盘停留。
    for step in range(5):
        feed(
            pipeline,
            next(ts_iter),
            hand_at(ENTRY_HAND[0], ENTRY_HAND[1] - step * 8),
            screw_at(*PARKED),
        )
    # 手带着这根杆离盘：杆心与手同步走。
    for step in range(1, 9):
        t = step / 8
        cx = ENTRY_HAND[0] + (HAND_TARGET[0] - ENTRY_HAND[0]) * t
        cy = ENTRY_HAND[1] + (HAND_TARGET[1] - ENTRY_HAND[1]) * t
        feed(
            pipeline,
            next(ts_iter),
            hand_at(cx, cy),
            screw_at(cx, cy),
        )
    for _ in range(6):
        feed(pipeline, next(ts_iter), hand_at(*HAND_TARGET), screw_at(*HAND_TARGET))

    assert len(_picks(events)) == 1, "同一件料被两条判据各报了一次"


def test_screw_exit_pick_sets_oiling_anchor():
    """丝杆判据判出的取料同样记锚点，后续涂油判定照常工作。"""

    pipeline, events = make_pipeline(brush_hold_s=0.4)
    ts_iter = timestamps()

    _park(pipeline, ts_iter)
    _move_out(pipeline, ts_iter)

    picks = _picks(events)
    assert len(picks) == 1
    task = pipeline._cycles[picks[0].task_id]
    assert task.screw_anchor is not None
    # 锚点应在杆离盘后的位置附近（已越过盘区下沿 y=520）。
    assert task.screw_anchor[1] > 520.0

    # 刷头贴着手里的杆刷够时长 ⇒ 涂油完成。
    from tests.test_pipeline import brush_at, held_screw_at

    spot = (520.0, 720.0)
    for _ in range(8):
        feed(
            pipeline,
            next(ts_iter),
            hand_at(*spot),
            held_screw_at(*spot),
            brush_at(spot[0] + 10, spot[1]),
        )

    assert [e for e in events if e.event_type == EventType.MANUAL_OILING]


def test_pick_summary_reports_screw_source():
    """复核摘要要说明这次取料是丝杆判据判出的，不是「手空手进盘」。"""

    from app.review import summary_from_event

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    _park(pipeline, ts_iter)
    _move_out(pipeline, ts_iter)

    picks = _picks(events)
    assert len(picks) == 1
    summary = summary_from_event(picks[0].evidence)
    assert summary is not None
    assert "丝杆" in summary.reason
    assert summary.metrics[0].label == "丝杆在盘内停留"


def test_square_blob_in_tray_never_becomes_a_pick():
    """盘区里出现近方形的「丝杆」误检（146×154）时，不能建记录、更不能判取料。

    盘区下沿/手边常被误检出一坨近方形目标。它随手动、静止一会儿再被带走，
    会完整走完「在盘里放稳 → 出盘」，凭空报一次取料（示例 092817
    8.6~9.3s）。丝杆框的长宽比实测 p50≈4，建记录时就要求 ≥2.0。
    """

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    blob = BBox(PARKED[0] - 73, PARKED[1] - 77, PARKED[0] + 73, PARKED[1] + 77)
    for _ in range(14):
        feed(
            pipeline,
            next(ts_iter),
            Detection(label="screw", confidence=0.9, box=blob),
        )
    assert pipeline._screw_exits == {}, "近方形误检不该建丝杆记录"

    for step in range(1, 9):
        t = step / 8
        cy = PARKED[1] + (OUTSIDE[1] - PARKED[1]) * t
        feed(
            pipeline,
            next(ts_iter),
            Detection(
                label="screw",
                confidence=0.9,
                box=BBox(
                    PARKED[0] - 73, cy - 77, PARKED[0] + 73, cy + 77
                ),
            ),
        )

    assert _picks(events) == []


def test_rod_moved_above_tray_is_not_a_pick():
    """杆从盘里被搬到**盘区上方**（整理料盘）：不算取料。

    取料是把杆取出、拿到盘体下方的工作台上二次涂油。示例 091519
    15.7s 工人把杆从盘内 (1537,227) 搬到画面顶部 (1574,40) 并放稳，
    按「杆心出了盘区」会误报。
    """

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    _park(pipeline, ts_iter)
    # 盘区是 y 20~520；搬到 y=40，仍在画面内但在盘区上半部。
    for step in range(1, 9):
        t = step / 8
        cy = PARKED[1] + (40.0 - PARKED[1]) * t
        feed(pipeline, next(ts_iter), screw_at(PARKED[0], cy))
    for _ in range(6):
        feed(pipeline, next(ts_iter), screw_at(PARKED[0], 40.0))

    assert _picks(events) == []


def test_pick_survives_noisy_second_rod_in_tray():
    """盘区里同时有另一根静止杆时，被带走那根仍要判出来。

    盘里那排杆两两间距很近，身份不能靠「离上一帧最近」硬配——必须靠
    位置 + 尺寸双门限（两根杆的框尺寸几乎一样，所以尺寸只是排除盘外方块，
    真正的区分仍靠位置连续性）。
    """

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    neighbor = (PARKED[0], PARKED[1] - 100.0)
    for _ in range(14):
        feed(
            pipeline,
            next(ts_iter),
            screw_at(*PARKED),
            screw_at(*neighbor),
        )

    for step in range(1, 9):
        t = step / 8
        cy = PARKED[1] + (OUTSIDE[1] - PARKED[1]) * t
        feed(
            pipeline,
            next(ts_iter),
            screw_at(PARKED[0], cy),
            screw_at(*neighbor),
        )
    for _ in range(4):
        feed(pipeline, next(ts_iter), screw_at(*OUTSIDE), screw_at(*neighbor))

    picks = _picks(events)
    assert len(picks) == 1
    assert picks[0].evidence["pick_source"] == "screw_exit"


def test_loading_a_rod_into_tray_is_not_a_pick():
    """手伸进盘里**放下**一根杆（装料）：不判取料。

    手框比杆长，放下时大面积盖住躺在盘里那根杆，落手面积能到 0.90、
    连续多帧达标——但杆自己几乎没动。按「落手面积」会被判成「带杆离盘」，
    凭空多一件取料，还会顶掉后面正在等涂油的周期、多报一次漏涂 NG
    （示例 091916 0.47s 即如此，真取料是 14.5s 之后才开始）。
    """

    pipeline, events = make_pipeline()
    ts_iter = timestamps()

    parked = screw_at(*PARKED)
    # 进盘前先把这根杆放进盘里（杆已经在盘内、此后一直不动）。
    for _ in range(6):
        feed(pipeline, next(ts_iter), parked)

    # 手伸进盘、盖住这根杆停留几帧——就是「放下手」的姿态。
    for step in range(5):
        feed(
            pipeline,
            next(ts_iter),
            hand_at(PARKED[0], PARKED[1] + 60 - step * 6),
            parked,
        )
    # 空手退到盘外：手走了，杆留在原处不动。
    for step in range(1, 7):
        feed(
            pipeline,
            next(ts_iter),
            hand_at(PARKED[0] - step * 12, PARKED[1] + 90 + step * 10),
            parked,
        )
    for _ in range(4):
        feed(pipeline, next(ts_iter), hand_at(PARKED[0] - 80, OUTSIDE[1]), parked)

    assert _picks(events) == [], "往盘里放杆不是取料"
