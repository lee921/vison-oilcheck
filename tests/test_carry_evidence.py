"""「手边有盘内杆」误判成取料的现场形态（示例⑧⑨），以及逻辑回归的接入点。

真取料的 carry 形态是「杆跟着手走」或「杆持续落在手里」；手按在夹具上的
杆、或从盘边划过时，杆本身一动不动，只有手在动。
"""

from __future__ import annotations

from app.oilcheck.judge import LogisticJudge
from app.oilcheck.types import EventType
from tests.test_pipeline import (
    HAND_TARGET,
    TRAY_SCREW,
    do_pick,
    enter_tray_empty,
    feed,
    hand_at,
    make_pipeline,
    screw_at,
    timestamps,
)


def _picks(events: list) -> list:
    return [e for e in events if e.event_type == EventType.PICK]


def test_hand_pressing_parked_screw_does_not_pick():
    """手按在夹具的杆上、随后空手离开：杆没被带走，不能判取料。

    示例⑧ 的形态：手在盘区里按着杆，退到盘外时杆留在夹具上。「手在盘外 +
    手边有盘内杆」帧帧成立，但杆一动不动。
    """

    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    parked = screw_at(*TRAY_SCREW)

    # 盘外空手（进盘定性）。
    for step in range(3):
        feed(pipeline, next(ts_iter), hand_at(HAND_TARGET[0] - step * 10,
                                             HAND_TARGET[1] + step * 10))
    enter_tray_empty(pipeline, ts_iter, screw=parked)
    # 退到盘外：手走了，杆留在原处不动（离 TRUE 目标近，仍在 hand_screw_max_distance 内）。
    for step in range(12):
        ts = next(ts_iter)
        feed(
            pipeline,
            ts,
            hand_at(470.0 - step * 4, 560.0 + step * 6),
            parked,
        )

    assert _picks(events) == [], "夹具上的杆没被带走，不该判取料"


def test_hand_brushing_past_parked_screw_does_not_pick():
    """手从盘边划过、旁边那根杆压在夹具上：杆只随检测框抖动，不判取料。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    for step in range(3):
        feed(pipeline, next(ts_iter), hand_at(HAND_TARGET[0] - step * 10,
                                             HAND_TARGET[1] + step * 10))
    enter_tray_empty(pipeline, ts_iter, screw=screw_at(*TRAY_SCREW))
    for step in range(14):
        jitter = 1.0 if step % 3 == 0 else 0.0
        feed(
            pipeline,
            next(ts_iter),
            hand_at(470.0 - step * 4, 560.0 + step * 5),
            screw_at(TRAY_SCREW[0] + jitter, TRAY_SCREW[1] + jitter),
        )

    assert _picks(events) == [], "杆没跟着手走，不该判取料"


def test_real_pick_still_works():
    """常规「空手进盘、带杆离盘」仍然判得出来（守住修复没吞掉真取料）。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    assert len(_picks(events)) == 1


def test_pick_carries_episode_when_judge_attached():
    """挂上逻辑回归后，取料事件带整段动作的起止（``judge_episode``）。"""

    pipeline, events = make_pipeline()
    # 与线上训练出的权重同向：seen / 握持采样为真越像取料。
    pipeline.attach_judges(
        LogisticJudge(
            features=["seen", "hs_true", "carry_inhand_frames"],
            mean=[0.0, 0.0, 0.0],
            scale=[1.0, 1.0, 1.0],
            weights=[3.0, 2.0, 1.0],
            bias=-1.0,
        ),
        None,
    )
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)

    picks = _picks(events)
    assert len(picks) == 1
    evidence = picks[0].evidence
    assert "judge" in evidence
    assert evidence["judge"]["probability"] > 0.5
    episode = evidence.get("judge_episode")
    assert episode is not None, "取料事件没带整段动作区间"
    assert episode["start_ts"] <= episode["end_ts"]
    assert picks[0].timestamp_start <= picks[0].timestamp_end


def test_rule_only_when_no_judge_attached():
    """不挂模型时判定退化成纯规则，事件不带模型字段。"""

    pipeline, events = make_pipeline()
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)

    picks = _picks(events)
    assert len(picks) == 1
    assert "judge" not in picks[0].evidence
    assert "judge_episode" not in picks[0].evidence


def _reverse_judge() -> LogisticJudge:
    return LogisticJudge(
        features=["seen", "hs_true", "carry_inhand_frames"],
        mean=[0.0, 0.0, 0.0],
        scale=[1.0, 1.0, 1.0],
        weights=[-3.0, -2.0, -1.0],  # 反向权重：越像取料越判「否」
        bias=0.0,
    )


def test_fusion_gate_vetoes_a_rule_true_that_the_model_rejects():
    """规则投「是」但模型强烈反对 ⇒ 改判「否」。

    原先规则投真就让模型完全不参与。现场 2026-11-27 10:40 的整理料盘被
    判成取料（模型 p=0.146，规则投真），就是这条被绕过：手在盘里挪杆
    6.75 秒后空手离开，还连带报了一次漏涂 NG。
    """

    pipeline, _ = make_pipeline()
    pipeline.attach_judges(_reverse_judge(), None)
    strong = {"seen": 1.0, "hs_true": 3.0, "carry_inhand_frames": 2.0}

    # 规则明确判「是」，但模型概率很低 ⇒ 以模型为准，改判否。
    result = pipeline._judge_pick(strong, rule=True)
    assert result is not None and result.used is True
    assert result.decision is False
    assert result.probability < 0.5


def test_fusion_gate_keeps_a_rule_true_the_model_does_not_strongly_reject():
    """规则投真、模型概率在门限之上 ⇒ 仍按规则判「是」。"""

    pipeline, _ = make_pipeline()
    pipeline.attach_judges(_fixed_probability_judge(0.75), None)
    result = pipeline._judge_pick({"seen": 1.0, "hs_true": 3.0}, rule=True)
    assert result is not None and result.used is True
    assert result.decision is True


def test_fusion_gate_lets_model_decide_in_the_middle_band():
    """规则判「否」但见过握持 ⇒ 中间带，模型说了算。"""

    pipeline, _ = make_pipeline()
    pipeline.attach_judges(_reverse_judge(), None)
    # 规则判「否」，但见过握持（hs_true>0）⇒ 模型定。
    result = pipeline._judge_pick({"seen": 0.0, "hs_true": 1.0}, rule=False)
    assert result is not None and result.used is True
    assert result.decision is False

    # 一帧握持都没有 ⇒ 规则明确否定，不进模型。
    result = pipeline._judge_pick({"seen": 0.0, "hs_true": 0.0}, rule=False)
    assert result is not None and result.used is False
    assert result.decision is False


def _fixed_probability_judge(value: float):
    """测试替身：无论输入如何都给出固定概率。"""

    class _Fixed:
        def probability(self, values):  # noqa: ANN001, ARG002
            return value

        def decide(self, values, *, rule=None):  # noqa: ANN001, ANN003
            from app.oilcheck.judge import JudgeResult

            probability = value
            if rule is None:
                return JudgeResult(probability, probability >= 0.5, True, dict(values))
            return JudgeResult(probability, bool(rule), False, dict(values))

    return _Fixed()


def test_no_judge_means_gate_is_inert():
    pipeline, _ = make_pipeline()
    assert pipeline._judge_pick({"hs_true": 1.0}, rule=False) is None


def test_judge_veto_on_rule_true_blocks_the_pick_end_to_end():
    """规则投「是」、模型强烈否决 ⇒ 端到端不出取料事件。

    规则这一路投真不等于可以跳过模型：现场 122925 第 1 秒工人把夹具摆到
    工位上，手在盘区里停 3.1 秒，规则凭「杆落在手框里 + 杆动过」投真，
    模型给 p≈0.14 明确判否。否决门（``_RULE_TRUE_VETO_P``）在
    ``_judge_pick`` 里已经把 ``decision`` 改成 False，``_settle_pick``
    必须照它走——否则否决只是记了个概率，取料照样上报，还连带一次漏涂 NG。

    模型概率在门限之上的情形由
    ``test_fusion_gate_keeps_a_rule_true_the_model_does_not_strongly_reject``
    覆盖（规则成立、模型不反对 ⇒ 照常判取料）。
    """

    pipeline, events = make_pipeline()
    pipeline.attach_judges(
        LogisticJudge(
            features=["seen", "hs_true", "carry_inhand_frames"],
            mean=[0.0, 0.0, 0.0],
            scale=[1.0, 1.0, 1.0],
            weights=[-50.0, -50.0, -50.0],
            bias=100.0,
        ),
        None,
    )
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    # 这个模型对任何输入都给极低概率（权重全负、特征非负），等于强烈否决。
    assert len(_picks(events)) == 0
