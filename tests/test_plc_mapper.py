"""PLC 4 工步映射的单元测试（纯逻辑，不碰网络）。"""

from __future__ import annotations

import pytest

from app.oilcheck.plc_io import (
    PlcEvidence,
    PlcFrame,
    PlcStepMapper,
    ProcessStep,
)
from app.oilcheck.types import TaskResult, TaskStage


def _ev(
    *,
    task_id: str | None = "TASK-000001",
    stage: TaskStage | None = TaskStage.PICKED,
    result: TaskResult | None = TaskResult.PROCESSING,
    brush: float = 0.0,
    online: bool = True,
) -> PlcEvidence:
    return PlcEvidence(
        online=online,
        task_id=task_id,
        stage=stage,
        final_result=result,
        brush_hold_s=brush,
    )


def _mapper(
    hold_s: float = 1.0,
    *,
    steps: dict[str, int] | None = None,
) -> PlcStepMapper:
    return PlcStepMapper(
        hold_s=hold_s,
        steps=steps or {"pick": 2, "oiling": 4, "result": 0},
    )


def _mapper_with_result(hold_s: float = 1.0):
    """带结果灯（默认灯5）的映射器：多数结果灯用例用它。"""

    return PlcStepMapper(
        hold_s=hold_s,
        steps={"pick": 2, "oiling": 4, "result": 5},
    )


# ----------------------------------------------------------------------
# 各工步的字节。编码由现场逐位实测得出（docs/12 §3.4）：
#   VB0(HR0 低字节) bitN = 灯 N+1 绿
#   VB1(HR0 高字节) bitN = 灯 N+1 黄
#   同一位 VB0+VB1 同时置 = 灯 N+1 红
#   HR1 与灯完全无关，恒 0
# 灯1 只能红，所以工步从灯2 起用（pick=2, oiling=4）。
# ----------------------------------------------------------------------
def test_idle_writes_step1_and_clears_records():
    frame = _mapper().update(_ev(task_id=None, stage=None, result=None), 0.0)
    assert frame.step == ProcessStep.IDLE
    # 等待取料：灯2 黄（进行中），其余灭
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x00, 0x02, 0x00)
    assert frame.registers == (0x0200, 0x0000)
    assert frame.lamps[:4] == (None, "yellow", None, None)


def test_picked_writes_step2_and_pick_bit():
    frame = _mapper().update(_ev(), 0.0)
    assert frame.step == ProcessStep.PICKED
    # 取料完成：灯2、灯3 绿，灯1 不碰
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x06, 0x00, 0x00)
    # HR0 = VB0 | VB1<<8：低字节是 VB0
    assert frame.hr0 == 0x0006
    assert frame.lamps[:4] == (None, "green", "green", None)


def test_oiling_starts_when_brush_hold_positive():
    mapper = _mapper()
    mapper.update(_ev(brush=0.0), 0.0)
    frame = mapper.update(_ev(brush=0.42), 0.1)
    assert frame.step == ProcessStep.OILING
    # 涂油中：灯2、灯3 绿，灯4 黄
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x06, 0x08, 0x00)
    assert frame.lamps[:4] == (None, "green", "green", "yellow")


def test_completed_writes_step3_and_oiling_good_bit():
    mapper = _mapper()
    frame = mapper.update(
        _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1), 0.0
    )
    assert frame.step == ProcessStep.FINISHED
    # 良品：灯2、灯3、灯4 全绿
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x0E, 0x00, 0x00)
    assert frame.lamps[:4] == (None, "green", "green", "green")


def test_ng_shows_red_by_setting_green_and_yellow_together():
    """漏涂：灯4 **同时**给绿和黄 → PLC 显示红。

    红不在 HR1，这是现场逐位实测的结论（写 VB2 任何位灯都不动）。
    """

    mapper = _mapper()
    frame = mapper.update(
        _ev(stage=TaskStage.OILED, result=TaskResult.NG, brush=0.2), 0.0
    )
    assert frame.step == ProcessStep.FINISHED
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x0E, 0x08, 0x00)
    assert frame.registers == (0x080E, 0x0000)
    assert frame.lamps[:4] == (None, "green", "green", "red")


def test_hr1_is_always_zero():
    """HR1（VB2/VB3）与灯无关：实测写任何位灯都不动，所以恒写 0。"""

    mapper = _mapper()
    frames = [
        mapper.update(_ev(), 0.0),
        mapper.update(_ev(brush=0.5), 0.1),
        mapper.update(
            _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1),
            0.2,
        ),
        mapper.update(_ev(stage=TaskStage.OILED, result=TaskResult.NG, brush=0.1), 0.3),
    ]
    for frame in frames:
        assert frame.vb2 == 0
        assert frame.vb3 == 0
        assert frame.hr1 == 0


def test_red_wins_when_one_lamp_gets_multiple_signals():
    """同一位同时置 VB0 和 VB1 显示红；只置其一是绿或黄。"""

    frame = PlcFrame(vb0=0b001, vb1=0b001, vb2=0, step=ProcessStep.IDLE)
    assert frame.lamps[0] == "red"
    frame = PlcFrame(vb0=0b010, vb1=0, vb2=0, step=ProcessStep.IDLE)
    assert frame.lamps[1] == "green"
    frame = PlcFrame(vb0=0, vb1=0b100, vb2=0, step=ProcessStep.IDLE)
    assert frame.lamps[2] == "yellow"
    # HR1 里的位不产生任何颜色
    frame = PlcFrame(vb0=0, vb1=0, vb2=0xFF, step=ProcessStep.IDLE)
    assert all(color is None for color in frame.lamps)


def test_prime_idle_returns_idle_frame():
    """启动压残留值用的帧：灯2 黄，其余全灭，并重置内部状态。"""

    mapper = _mapper(hold_s=1.0)
    # 先跑出一个 NG 保持窗内的状态，prime_idle 必须把它一起清掉
    mapper.update(
        _ev(stage=TaskStage.OILED, result=TaskResult.NG, brush=0.2), 0.0
    )
    frame = mapper.prime_idle()
    assert frame.step == ProcessStep.IDLE
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x00, 0x02, 0x00)
    assert frame.registers == (0x0200, 0x0000)
    assert frame.task_id is None


def test_vb3_is_always_zero():
    mapper = _mapper()
    frames = [
        mapper.update(_ev(), 0.0),
        mapper.update(_ev(brush=0.5), 0.1),
        mapper.update(
            _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1),
            0.2,
        ),
        mapper.update(_ev(stage=TaskStage.OILED, result=TaskResult.NG, brush=0.1), 0.3),
    ]
    for frame in frames:
        assert frame.vb3 == 0
        assert frame.hr1 >> 8 == 0


# ----------------------------------------------------------------------
# 保持窗
# ----------------------------------------------------------------------
def test_hold_window_keeps_step4_then_returns_to_idle():
    mapper = _mapper(hold_s=1.0)
    done = _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1)
    mapper.update(done, 0.0)
    assert mapper.update(done, 0.5).step == ProcessStep.FINISHED
    assert mapper.update(done, 0.99).step == ProcessStep.FINISHED
    frame = mapper.update(done, 1.0)
    assert frame.step == ProcessStep.IDLE
    # 回工步 1：绿/红清零，只有灯1 黄（等取料）
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x00, 0x02, 0x00)
    assert frame.lamps[:4] == (None, "yellow", None, None)


def test_hold_expired_task_does_not_retrigger_step4():
    """已报完的件还会在快照里留一阵，不能再报一次。

    周期的保留策略是按条数（``completed_keep_count``），不是按时间；保持窗
    结束后同一件仍在快照里。不防这一下，工步会在 1 与 4 之间来回跳。
    """

    mapper = _mapper(hold_s=1.0)
    done = _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1)
    mapper.update(done, 0.0)
    mapper.update(done, 1.5)  # 保持窗结束 → 工步 1
    for tick in (2.0, 3.0, 10.0):
        frame = mapper.update(done, tick)
        assert frame.step == ProcessStep.IDLE


def test_hold_window_is_absolute_even_if_next_task_arrives():
    """保持窗内来了下一件也要先把工步 4 走完。

    PLC 靠这段窗口读走结论；提前切到工步 2 会让上一件的良品/不良位被
    新件的清零覆盖掉。
    """

    mapper = _mapper(hold_s=1.0)
    mapper.update(
        _ev(task_id="T1", stage=TaskStage.OILED, result=TaskResult.COMPLETED), 0.0
    )
    inside = mapper.update(_ev(task_id="T2", brush=0.0), 0.2)
    assert inside.step == ProcessStep.FINISHED
    assert inside.vb0 == 0x0E  # 上一件的三盏绿还在
    # 保持窗结束后先清位（新件清零），再按证据进入工步 2
    cleared = mapper.update(_ev(task_id="T2", brush=0.0), 1.0)
    assert cleared.step == ProcessStep.IDLE
    assert cleared.vb0 == 0x00
    picked = mapper.update(_ev(task_id="T2", brush=0.0), 1.2)
    assert picked.step == ProcessStep.PICKED
    assert picked.vb0 == 0x06


# ----------------------------------------------------------------------
# 单调性与快照抖动
# ----------------------------------------------------------------------
def test_step_never_goes_backwards_within_one_task():
    """涂油累计时长可能因刷子短暂离开而变 0，工步不能退回「取料完成」。"""

    mapper = _mapper()
    mapper.update(_ev(brush=0.0), 0.0)
    mapper.update(_ev(brush=0.42), 0.1)
    frame = mapper.update(_ev(brush=0.0), 0.2)
    assert frame.step == ProcessStep.OILING


def test_missing_task_keeps_previous_step():
    """活动周期结束后快照暂时看不到它：不能跳回工步 1。"""

    mapper = _mapper()
    mapper.update(_ev(brush=0.42), 0.0)
    frame = mapper.update(_ev(task_id=None, stage=None, result=None), 0.1)
    assert frame.step == ProcessStep.OILING


def test_missing_task_falls_back_to_idle_after_timeout():
    """长时间没有任务（重置状态/示例回绕）必须回落工步 1。

    否则点一次「重置状态」后工步永久冻结在「涂油中」，警示灯整夜亮着。
    """

    mapper = _mapper()
    mapper.update(_ev(brush=0.42), 0.0)
    # 短时间保持：清理可能让活动周期短暂不可见
    assert mapper.update(_ev(task_id=None), 0.1).step == ProcessStep.OILING
    # 超过超时：回落 IDLE，并清掉同件的结论记忆
    frame = mapper.update(_ev(task_id=None), 100.0)
    assert frame.step == ProcessStep.IDLE
    assert frame.vb1 == 0x02  # 灯2 黄：等取料
    assert frame.vb0 == 0x00 and frame.vb2 == 0x00
    assert "超时" in frame.reason


def test_new_task_after_missing_resets_timeout():
    mapper = _mapper()
    mapper.update(_ev(brush=0.42), 0.0)
    mapper.update(_ev(task_id=None), 0.1)
    mapper.update(_ev(task_id="TASK-000002", brush=0.0), 0.2)
    # 新件到来后再短暂消失，不应沿用上一次的 no_task_since
    frame = mapper.update(_ev(task_id=None), 0.3)
    assert frame.step in (ProcessStep.IDLE, ProcessStep.PICKED)


def test_offline_keeps_red_but_clears_pure_yellow():
    """掉线只灭「纯黄」（进行中），红灯（绿+黄）必须保留。

    红灯是绿+黄同时亮，所以不能把黄位一刀切清零——那会把 NG 显示成良品。
    """

    mapper = _mapper()
    mapper.update(_ev(stage=TaskStage.OILED, result=TaskResult.NG, brush=0.1), 0.0)
    frame = mapper.update(PlcEvidence.offline(), 0.1)
    assert frame.step == ProcessStep.OFFLINE
    # NG 是 绿=0x0E 黄=0x08，掉线后原样保留（灯4 仍红）
    assert (frame.vb0, frame.vb1) == (0x0E, 0x08)
    assert frame.lamps[:4] == (None, "green", "green", "red")
    assert frame.hr1 == 0


def test_offline_clears_yellow_while_oiling():
    """涂油中掉线：进行中的黄灯灭，已通过的两盏绿保留。"""

    mapper = _mapper()
    mapper.update(_ev(brush=0.5), 0.0)  # 涂油中：绿=0x06 黄=0x08
    frame = mapper.update(PlcEvidence.offline(), 0.1)
    assert (frame.vb0, frame.vb1) == (0x06, 0x00)
    assert frame.lamps[:4] == (None, "green", "green", None)


def test_repeated_evidence_is_idempotent():
    mapper = _mapper()
    first = mapper.update(_ev(brush=0.42), 0.0)
    second = mapper.update(_ev(brush=0.42), 0.05)
    assert first.registers == second.registers
    assert first.step == second.step


def test_waiting_pick_stage_maps_to_step1():
    """``WAITING_PICK`` 在运行期不会停留，见到它按工步 1 处理。"""

    frame = _mapper().update(
        _ev(stage=TaskStage.WAITING_PICK, result=TaskResult.PROCESSING), 0.0
    )
    assert frame.step == ProcessStep.IDLE


# ----------------------------------------------------------------------
# 工步编号可配
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("pick", "oiling", "expect_green"),
    [
        # 绿从第 pick 位累计到第 oiling 位；进度灯只占前 4 盏
        (1, 3, 0b00000111),
        (2, 4, 0b00001110),
    ],
)
def test_step_numbers_are_configurable(
    pick: int, oiling: int, expect_green: int
):
    mapper = PlcStepMapper(
        hold_s=1.0, steps={"pick": pick, "oiling": oiling, "result": 0}
    )
    frame = mapper.update(
        _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1), 0.0
    )
    assert frame.vb0 == expect_green
    assert frame.vb1 == 0
    assert frame.vb2 == 0


def test_invalid_step_numbers_fall_back_to_default():
    """灯1 不能用（只能红）、oiling 必须 = pick+2 且不超过灯数。"""

    for steps in (
        {"pick": 0, "oiling": 2},  # pick 从 1 起
        {"pick": 4, "oiling": 6},  # 进度灯只占前 4 盏
        {"pick": 3, "oiling": 4},  # oiling != pick+2
        {"pick": 8, "oiling": 10},  # 超出灯数
        {"pick": 5, "oiling": 2},  # 倒序
        {"pick": 2, "oiling": 4, "result": 3},  # 结果灯与进度灯重叠
        {"pick": 2, "oiling": 4, "result": 9},  # 结果灯超出灯数
    ):
        mapper = PlcStepMapper(hold_s=1.0, steps=steps)
        frame = mapper.update(
            _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1), 0.0
        )
        # 兜底 pick=2/oiling=4 ⇒ 灯2、灯3、灯4 绿
        assert frame.vb0 == 0b00001110, steps


# ----------------------------------------------------------------------
# 被下一次取料顶掉时的 NG 补报
# ----------------------------------------------------------------------
def test_superseded_ng_is_reported_before_next_task():
    """漏涂的主要路径：上一件没刷完就被下一次取料顶掉。

    200ms 的轮询很容易跨过那一帧终态。等到看见新周期时如果直接切工步 2，
    上一件的 NG 就永远写不进 PLC。
    """

    mapper = _mapper(hold_s=1.0)
    mapper.update(_ev(task_id="T1", brush=0.3), 0.0)  # 工步 3
    # 这一拍：当前已是 T2（新取料），但 T1 刚被判 NG
    frame = mapper.update(
        PlcEvidence(
            online=True,
            task_id="T2",
            stage=TaskStage.PICKED,
            final_result=TaskResult.PROCESSING,
            brush_hold_s=0.0,
            superseded_task_id="T1",
            superseded_result=TaskResult.NG,
        ),
        0.2,
    )
    assert frame.step == ProcessStep.FINISHED
    assert frame.task_id == "T1"
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x0E, 0x08, 0x00)
    assert frame.lamps[:4] == (None, "green", "green", "red")


def test_hold_window_applies_to_superseded_ng():
    """补报的 NG 同样要走完成保持窗，之后才轮到新件。"""

    mapper = _mapper(hold_s=1.0)
    mapper.update(_ev(task_id="T1", brush=0.3), 0.0)
    ng_evidence = PlcEvidence(
        online=True,
        task_id="T2",
        stage=TaskStage.PICKED,
        final_result=TaskResult.PROCESSING,
        superseded_task_id="T1",
        superseded_result=TaskResult.NG,
    )
    assert mapper.update(ng_evidence, 0.2).step == ProcessStep.FINISHED
    assert mapper.update(ng_evidence, 0.9).step == ProcessStep.FINISHED
    # 保持窗结束：清位回工步 1（只剩灯1 黄）
    cleared = mapper.update(ng_evidence, 1.3)
    assert cleared.step == ProcessStep.IDLE
    assert (cleared.vb0, cleared.vb1, cleared.vb2) == (0x00, 0x02, 0x00)
    # 下一拍才按证据进入新件的工步
    picked = mapper.update(ng_evidence, 1.5)
    assert picked.step == ProcessStep.PICKED
    assert picked.task_id == "T2"


def test_superseded_already_held_is_not_reported_twice():
    """同一件顶掉后只补报一次；快照里它继续留着也不再触发。"""

    mapper = _mapper(hold_s=1.0)
    mapper.update(_ev(task_id="T1", brush=0.3), 0.0)
    evidence = PlcEvidence(
        online=True,
        task_id="T2",
        stage=TaskStage.PICKED,
        final_result=TaskResult.PROCESSING,
        superseded_task_id="T1",
        superseded_result=TaskResult.NG,
    )
    mapper.update(evidence, 0.2)  # 补报 NG
    mapper.update(evidence, 1.5)  # 保持窗结束 → 工步 1
    for tick in (1.7, 2.0, 5.0):
        frame = mapper.update(evidence, tick)
        assert frame.step != ProcessStep.FINISHED


# ----------------------------------------------------------------------
# 当前件本身已终态时不能再补报「上一件」
# ----------------------------------------------------------------------
def test_finished_current_task_does_not_replay_previous_task():
    """当前件已判终态、快照里还留着上一件时，不能每轮都补报上一件。

    原实现里 ``_pick_plc_task`` 无条件扫描当前件之前的已终态条目：当前件
    自己也是终态（两件都已完成、都在快照里）时，每一拍的证据都是
    「当前=T2（终态），superseded=T1（终态）」，mapper 的补报分支就把
    T1 重新报一遍，保持窗结束后又回到 T2，工步在 4 与 1 之间来回跳。
    """

    from app.oilcheck.pipeline import TaskSnapshot
    from app.oilcheck.service import _pick_plc_task

    def snap(tid: str, created: float, result: TaskResult) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=tid,
            created_at=created,
            stage=TaskStage.OILED,
            stage_since=created,
            final_result=result,
        )

    tasks = (
        snap("TASK-1", 1.0, TaskResult.COMPLETED),
        snap("TASK-2", 2.0, TaskResult.COMPLETED),
    )
    current, superseded = _pick_plc_task(tasks)
    assert current is not None and current.task_id == "TASK-2"
    assert superseded is None, "当前件已是终态，没有「没报出去的上一件」"

    mapper = _mapper(hold_s=1.0)
    steps = []
    now = 0.0
    for _ in range(20):
        now += 0.2
        cur, sup = _pick_plc_task(tasks)
        ev = PlcEvidence(
            online=True,
            task_id=cur.task_id,
            stage=cur.stage,
            final_result=cur.final_result,
            superseded_task_id=sup.task_id if sup else None,
            superseded_result=sup.final_result if sup else None,
        )
        steps.append(mapper.update(ev, now).step)

    # 前 1 秒保持工步 4，之后回工步 1 并**一直**停在那里
    assert steps[0] == ProcessStep.FINISHED
    assert steps[-1] == ProcessStep.IDLE
    assert steps.count(ProcessStep.FINISHED) <= 6, steps


def test_superseded_reported_when_new_task_is_processing():
    """新件仍在进行中时，上一件的终态才需要补报。"""

    from app.oilcheck.pipeline import TaskSnapshot
    from app.oilcheck.service import _pick_plc_task

    def snap(tid, created, result):
        return TaskSnapshot(
            task_id=tid,
            created_at=created,
            stage=TaskStage.PICKED,
            stage_since=created,
            final_result=result,
        )

    current, superseded = _pick_plc_task(
        (
            snap("TASK-1", 1.0, TaskResult.NG),
            snap("TASK-2", 2.0, TaskResult.PROCESSING),
        )
    )
    assert current is not None and current.task_id == "TASK-2"
    assert superseded is not None and superseded.task_id == "TASK-1"


# ----------------------------------------------------------------------
# 结果灯（默认灯5）：显示上一件检测结果
#
# 现场要求：切换到一个新的检测结果时**先闪一下黄**，然后常亮（良品绿、
# 漏涂红），**一直亮到下一个结果到来**，不按时间自动灭。同一件不会被重复
# 闪——工步 4 的保持窗里每拍都在重算帧，不判重的话闪烁会随节拍反复重放。
# ----------------------------------------------------------------------
def test_result_lamp_flashes_yellow_then_green_for_ok():
    mapper = _mapper_with_result(hold_s=1.0)
    done = _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1)

    # 新结论到来的第一拍：黄闪，灯5 只亮黄（绿位要清掉）
    flash = mapper.update(done, 0.0)
    assert flash.lamps[4] == "yellow"
    assert flash.vb0 & 0b10000 == 0
    assert flash.vb1 & 0b10000

    # 闪完转常亮：灯5 绿
    steady = mapper.update(done, 0.5)
    assert steady.lamps[4] == "green"

    # 一直常亮，不按时间灭（远超过旧的 5 秒窗口）
    assert mapper.update(done, 4.0).lamps[4] == "green"
    assert mapper.update(done, 5.5).lamps[4] == "green"
    assert mapper.update(done, 3600.0).lamps[4] == "green"


def test_result_lamp_flashes_yellow_then_red_for_ng():
    mapper = _mapper_with_result(hold_s=1.0)
    done = _ev(stage=TaskStage.OILED, result=TaskResult.NG, brush=0.2)

    assert mapper.update(done, 0.0).lamps[4] == "yellow"
    # 红 = 同一位绿+黄
    red = mapper.update(done, 0.5)
    assert red.lamps[4] == "red"
    assert red.vb0 & 0b10000 and red.vb1 & 0b10000
    # 常亮到下一个结论
    assert mapper.update(done, 60.0).lamps[4] == "red"


def test_result_lamp_flash_is_not_replayed_within_the_same_task():
    """保持窗里每拍都重算帧：同一件不能反复重闪黄。"""

    mapper = _mapper_with_result(hold_s=3.0)
    done = _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1)
    mapper.update(done, 0.0)  # 闪
    for tick in (0.2, 0.4, 0.6, 0.8, 1.0):
        assert mapper.update(done, tick).lamps[4] in ("yellow", "green")
    # 闪烁阶段过后必须一直是绿，不再回到黄
    for tick in (1.5, 2.0, 2.5):
        assert mapper.update(done, tick).lamps[4] == "green"


def test_new_task_result_flashes_again():
    """下一件判完时结果灯重新闪一次，显示的是新的结论。"""

    mapper = _mapper_with_result(hold_s=0.5)
    ok = _ev(task_id="T1", stage=TaskStage.OILED, result=TaskResult.COMPLETED)
    ng = _ev(task_id="T2", stage=TaskStage.OILED, result=TaskResult.NG)
    mapper.update(ok, 0.0)
    assert mapper.update(ok, 0.5).lamps[4] == "green"

    # 下一件在半秒后判完：先闪黄
    assert mapper.update(ng, 1.0).lamps[4] == "yellow"
    # 再显示红
    assert mapper.update(ng, 1.5).lamps[4] == "red"


def test_result_lamp_survives_hold_window_and_next_pick():
    """结果灯与工步 4 的保持窗是两条时间线。

    工步 4 只亮 1 秒（hold_s）；保持窗结束后回到工步 1、下一件正在取料时，
    灯5 仍要一直显示上一件的结论，直到下一个结果到来。
    """

    mapper = _mapper_with_result(hold_s=1.0)
    done = _ev(task_id="T1", stage=TaskStage.OILED, result=TaskResult.NG)
    mapper.update(done, 0.0)
    mapper.update(done, 0.5)  # 常亮红

    # 保持窗结束时回到工步 1，但结果灯还在
    idle = mapper.update(done, 1.2)
    assert idle.step == ProcessStep.IDLE
    assert idle.lamps[4] == "red"

    # 下一件已取料：进度灯换成工步 2，结果灯仍是上一件的红
    picked = mapper.update(_ev(task_id="T2", brush=0.0), 1.5)
    assert picked.step == ProcessStep.PICKED
    assert picked.lamps[4] == "red"

    # 很久之后仍亮着（不再自动灭）
    later = mapper.update(_ev(task_id="T2", brush=0.0), 600.0)
    assert later.lamps[4] == "red"


def test_result_lamp_off_by_zero():
    """``steps.result = 0`` 关闭结果灯：灯5 永远不亮。"""

    mapper = _mapper()
    done = _ev(stage=TaskStage.OILED, result=TaskResult.NG, brush=0.2)
    for tick in (0.0, 0.5, 2.0, 6.0):
        assert mapper.update(done, tick).lamps[4] is None

