"""PLC 4 工步映射的单元测试（纯逻辑，不碰网络）。"""

from __future__ import annotations

import pytest

from app.oilcheck.plc_io import (
    PlcEvidence,
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


def _mapper(hold_s: float = 1.0) -> PlcStepMapper:
    return PlcStepMapper(hold_s=hold_s, steps={"pick": 1, "oiling": 3})


# ----------------------------------------------------------------------
# 4 个工步的字节
# ----------------------------------------------------------------------
def test_idle_writes_step1_and_clears_records():
    frame = _mapper().update(_ev(task_id=None, stage=None, result=None), 0.0)
    assert frame.step == ProcessStep.IDLE
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x01, 0x00, 0x00)
    assert frame.registers == (0x0001, 0x0000)


def test_picked_writes_step2_and_pick_bit():
    frame = _mapper().update(_ev(), 0.0)
    assert frame.step == ProcessStep.PICKED
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x02, 0x01, 0x00)
    # HR0 = VB0 | VB1<<8：低字节是 VB0
    assert frame.hr0 == 0x0102


def test_oiling_starts_when_brush_hold_positive():
    mapper = _mapper()
    mapper.update(_ev(brush=0.0), 0.0)
    frame = mapper.update(_ev(brush=0.42), 0.1)
    assert frame.step == ProcessStep.OILING
    # 涂油中：VB0 指第 3 步；VB1 已通过前两步 = 0b0011
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x04, 0x03, 0x00)


def test_completed_writes_step4_and_oiling_good_bit():
    mapper = _mapper()
    frame = mapper.update(
        _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1), 0.0
    )
    assert frame.step == ProcessStep.FINISHED
    # 做完的件不再是任何一步：VB0=0；VB1 累计三步全过 = 0b0111
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x00, 0x07, 0x00)


def test_ng_writes_step4_and_oiling_bad_bit():
    mapper = _mapper()
    frame = mapper.update(
        _ev(stage=TaskStage.OILED, result=TaskResult.NG, brush=0.2), 0.0
    )
    assert frame.step == ProcessStep.FINISHED
    # 漏涂：VB1 只留取料位，VB2 记涂油位为不良
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x00, 0x03, 0b0100)


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
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x01, 0x00, 0x00)


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
    assert inside.vb1 == 0x07  # 上一件的涂油良品位还在
    # 保持窗结束后先清位（新件清零），再按证据进入工步 2
    cleared = mapper.update(_ev(task_id="T2", brush=0.0), 1.0)
    assert cleared.step == ProcessStep.IDLE
    assert cleared.vb1 == 0x00
    picked = mapper.update(_ev(task_id="T2", brush=0.0), 1.2)
    assert picked.step == ProcessStep.PICKED
    assert picked.vb1 == 0x01


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
    assert frame.vb0 == 0x01  # 工步 1
    assert frame.vb1 == 0x00 and frame.vb2 == 0x00
    assert "超时" in frame.reason


def test_new_task_after_missing_resets_timeout():
    mapper = _mapper()
    mapper.update(_ev(brush=0.42), 0.0)
    mapper.update(_ev(task_id=None), 0.1)
    mapper.update(_ev(task_id="TASK-000002", brush=0.0), 0.2)
    # 新件到来后再短暂消失，不应沿用上一次的 no_task_since
    frame = mapper.update(_ev(task_id=None), 0.3)
    assert frame.step in (ProcessStep.IDLE, ProcessStep.PICKED)


def test_offline_clears_vb0_but_keeps_records():
    mapper = _mapper()
    mapper.update(_ev(stage=TaskStage.OILED, result=TaskResult.NG, brush=0.1), 0.0)
    frame = mapper.update(PlcEvidence.offline(), 0.1)
    assert frame.step == ProcessStep.OFFLINE
    assert frame.vb0 == 0x00
    # 已写出的良品/不良记录是这一件的历史，掉线不该抹掉
    assert (frame.vb1, frame.vb2) == (0x03, 0b0100)


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
    ("pick", "oiling", "expect_vb1", "expect_vb2"),
    [
        # 完成：VB1 从第 1 位连续到涂油位
        (1, 3, 0b00000111, 0),
        (1, 4, 0b00001111, 0),
        (2, 5, 0b00011111, 0),
    ],
)
def test_step_numbers_are_configurable(
    pick: int, oiling: int, expect_vb1: int, expect_vb2: int
):
    mapper = PlcStepMapper(hold_s=1.0, steps={"pick": pick, "oiling": oiling})
    frame = mapper.update(
        _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1), 0.0
    )
    assert (frame.vb1, frame.vb2) == (expect_vb1, expect_vb2)


def test_out_of_range_step_numbers_fall_back_to_defaults():
    mapper = PlcStepMapper(hold_s=1.0, steps={"pick": 99, "oiling": 0})
    frame = mapper.update(
        _ev(stage=TaskStage.OILED, result=TaskResult.COMPLETED, brush=1.1), 0.0
    )
    # 兜底：pick=1、oiling=3 ⇒ VB1 累计 0b0111
    assert frame.vb1 == 0b0111


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
    assert (frame.vb0, frame.vb1, frame.vb2) == (0x00, 0x03, 0b0100)


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
    # 保持窗结束：清位回工步 1
    cleared = mapper.update(ng_evidence, 1.3)
    assert cleared.step == ProcessStep.IDLE
    assert (cleared.vb1, cleared.vb2) == (0x00, 0x00)
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
