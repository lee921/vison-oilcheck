"""「手边有盘内杆」误判成取料的现场形态（示例⑧⑨），以及逻辑回归的接入点。

真取料的 carry 形态是「杆跟着手走」或「杆持续落在手里」；手按在夹具上的
杆、或从盘边划过时，杆本身一动不动，只有手在动。
"""

from __future__ import annotations

import pytest

from app.oilcheck.judge import LogisticJudge
from app.oilcheck.types import EventType

from tests.test_pipeline import (
    HAND_TARGET,
    TRAY_SCREW,
    do_pick,
    enter_tray_empty,
    feed,
    hand_at,
    held_screw_at,
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


def test_fusion_gate_lets_model_decide_only_in_the_middle_band():
    """融合策略：规则明确时不经过模型，规则拿不准时由模型定。

    直接测 ``_judge_pick`` 这一层门限——用真实取料流造「规则拿不准」的
    场景很难稳定复现，而门限本身就是策略所在。
    """

    pipeline, _ = make_pipeline()
    pipeline.attach_judges(_reverse_judge(), None)
    strong = {"seen": 1.0, "hs_true": 3.0, "carry_inhand_frames": 2.0}
    weak = {"seen": 0.0, "hs_true": 1.0, "carry_inhand_frames": 0.0}

    # 规则明确判「是」⇒ 模型只记录概率，不改结论。
    result = pipeline._judge_pick(strong, rule=True)
    assert result is not None and result.used is False
    assert result.decision is True
    assert result.probability < 0.5  # 反向模型的概率确实很低

    # 规则判「否」，但见过握持（hs_true>0）⇒ 中间带，模型说了算。
    result = pipeline._judge_pick(weak, rule=False)
    assert result is not None and result.used is True
    assert result.decision is False

    # 一帧握持都没有 ⇒ 规则明确否定，不进模型。
    result = pipeline._judge_pick({"seen": 0.0, "hs_true": 0.0,
                                   "carry_inhand_frames": 0.0}, rule=False)
    assert result is not None and result.used is False
    assert result.decision is False


def test_no_judge_means_gate_is_inert():
    pipeline, _ = make_pipeline()
    assert pipeline._judge_pick({"hs_true": 1.0}, rule=False) is None


def test_judge_cannot_override_confident_rule():
    """规则明确成立时模型不参与（避免样本有限时搅乱稳定判定）。"""

    pipeline, events = make_pipeline()
    pipeline.attach_judges(
        LogisticJudge(
            features=["seen", "hs_true", "carry_inhand_frames"],
            mean=[0.0, 0.0, 0.0],
            scale=[1.0, 1.0, 1.0],
            weights=[-50.0, -50.0, -50.0],
            bias=100.0,  # 逻辑回归仍判「是」？不——见下面断言
        ),
        None,
    )
    ts_iter = timestamps()
    do_pick(pipeline, ts_iter)
    # 这里只验证「挂上模型不影响明确成立的判定」的结果。
    assert len(_picks(events)) == 1
