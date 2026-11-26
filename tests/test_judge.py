"""逻辑回归二级判定与分段器的单元测试。"""

from __future__ import annotations

import json
import math

import pytest

from app.oilcheck.judge import (
    EpisodeSegmenter,
    JudgeResult,
    LogisticJudge,
    save_judge,
    train_logistic,
)


def _make_judge(features: list[str], weights: list[float], bias: float) -> LogisticJudge:
    return LogisticJudge(
        features=features,
        mean=[0.0] * len(features),
        scale=[1.0] * len(features),
        weights=weights,
        bias=bias,
    )


def test_probability_is_monotonic_and_sigmoid_shaped():
    judge = _make_judge(["x"], [1.0], 0.0)
    assert judge.probability({"x": 0.0}) == pytest.approx(0.5)
    assert judge.probability({"x": 3.0}) > judge.probability({"x": 1.0}) > 0.5
    assert judge.probability({"x": -3.0}) < 0.5
    # 数值稳定：极大/极小都不溢出。
    assert judge.probability({"x": 1e6}) == pytest.approx(1.0)
    assert judge.probability({"x": -1e6}) == pytest.approx(0.0)


def test_missing_feature_falls_back_to_mean():
    """缺的特征按均值处理（标准化后为 0），不会静默变成 0 值特征。"""

    judge = LogisticJudge(
        features=["x", "y"], mean=[10.0, 0.0], scale=[2.0, 1.0],
        weights=[1.0, 0.0], bias=0.0,
    )
    assert judge.probability({"x": 10.0}) == pytest.approx(0.5)
    assert judge.probability({"y": 99.0}) == pytest.approx(0.5)


def test_decide_respects_rule_when_rule_is_confident():
    judge = _make_judge(["x"], [10.0], 0.0)  # 模型强烈判「是」
    result = judge.decide({"x": 1.0}, rule=False)
    assert isinstance(result, JudgeResult)
    assert result.decision is False      # 规则明确否定时模型不改结论
    assert result.used is False
    assert result.probability > 0.9      # 概率仍记录下来供复核

    result = judge.decide({"x": 1.0}, rule=None)
    assert result.decision is True
    assert result.used is True


def test_judge_roundtrip(tmp_path):
    judge = _make_judge(["a", "b"], [0.5, -0.25], 0.1)
    path = save_judge(
        tmp_path / "j.json",
        (judge.features, judge.mean.tolist(), judge.scale.tolist(),
         judge.weights.tolist(), judge.bias),
        low=0.3, high=0.7, note="unit test",
    )
    loaded = LogisticJudge.load(path)
    assert loaded is not None
    assert loaded.features == ["a", "b"]
    assert loaded.low == pytest.approx(0.3)
    assert loaded.probability({"a": 2.0, "b": 1.0}) == pytest.approx(
        judge.probability({"a": 2.0, "b": 1.0})
    )
    assert json.loads(path.read_text())["note"] == "unit test"


def test_load_missing_file_returns_none(tmp_path):
    assert LogisticJudge.load(tmp_path / "nope.json") is None


def test_load_malformed_file_returns_none(tmp_path):
    """坏权重不能让监控进程起不来，宁可退回纯规则判定。"""

    path = tmp_path / "bad.json"
    path.write_text("{not json")
    assert LogisticJudge.load(path) is None

    path.write_text(json.dumps({"features": ["a"], "mean": [0], "scale": [1],
                                "weights": [1, 2], "bias": 0}))
    assert LogisticJudge.load(path) is None  # 维度不一致


def test_train_separates_two_clusters():
    samples = [
        {"a": 0.0, "b": 0.0}, {"a": 0.1, "b": 0.05}, {"a": 0.05, "b": 0.0},
        {"a": 1.0, "b": 1.0}, {"a": 0.9, "b": 1.1}, {"a": 1.05, "b": 0.95},
    ]
    labels = [0, 0, 0, 1, 1, 1]
    trained = train_logistic(samples, labels)
    features, mean, scale, weights, bias = trained
    judge = LogisticJudge(features, mean, scale, weights, bias)
    for sample, label in zip(samples, labels):
        assert int(judge.probability(sample) >= 0.5) == label


def test_train_regularization_keeps_weights_bounded():
    """样本能被完全分开时权重也不能无限膨胀（换数据就崩）。"""

    samples = [{"x": float(i)} for i in range(6)]
    labels = [0, 0, 0, 1, 1, 1]
    _, _, _, weights, _ = train_logistic(samples, labels)
    assert all(abs(w) < 50 for w in weights), weights


# ----------------------------------------------------------------------
# 分段器：把逐帧概率切成带起止时刻的动作
# ----------------------------------------------------------------------
def _feed(segmenter: EpisodeSegmenter, probs: list[float], dt: float = 0.1) -> list:
    episodes = []
    for index, prob in enumerate(probs):
        segmenter.update(index * dt, prob)
        episodes.extend(segmenter.drain())
    segmenter.close()
    episodes.extend(segmenter.drain())
    return episodes


def test_segmenter_finds_start_before_threshold_crossing():
    """动作起点要退回到概率刚抬头的帧，而不是「攒够证据」的那帧。"""

    probs = [0.0, 0.05, 0.30, 0.55, 0.80, 0.90, 0.85, 0.10, 0.02, 0.0]
    episodes = _feed(EpisodeSegmenter(on=0.6, off=0.35, low=0.2, bridge_s=0.5), probs)
    assert len(episodes) == 1
    assert episodes[0].start_ts == pytest.approx(0.2)   # 0.30 那一帧
    assert episodes[0].end_ts == pytest.approx(0.6)     # 最后一个高概率帧
    assert episodes[0].peak == pytest.approx(0.90)


def test_segmenter_bridges_short_gaps():
    """刷手抬一下再放回来仍算同一次（空档 < bridge_s）。"""

    probs = [0.9, 0.9, 0.1, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
    episodes = _feed(EpisodeSegmenter(on=0.6, off=0.35, low=0.2, bridge_s=0.5), probs)
    assert len(episodes) == 1


def test_segmenter_splits_long_gaps():
    probs = [0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0,
             0.0, 0.0, 0.0]
    episodes = _feed(EpisodeSegmenter(on=0.6, off=0.35, low=0.2, bridge_s=0.3), probs)
    assert len(episodes) == 2


def test_segmenter_ignores_faint_noise():
    """从没越过 on 的一段不算动作。"""

    probs = [0.25, 0.30, 0.28, 0.0, 0.0]
    assert _feed(EpisodeSegmenter(on=0.6, off=0.35, low=0.2), probs) == []


def test_segmenter_take_active_returns_current_start():
    segmenter = EpisodeSegmenter(on=0.6, off=0.35, low=0.2, bridge_s=0.5)
    for index, prob in enumerate([0.3, 0.9, 0.9]):
        segmenter.update(index * 0.1, prob)
    episode = segmenter.take_active()
    assert episode is not None
    assert episode.start_ts == pytest.approx(0.0)
    assert episode.end_ts == pytest.approx(0.2)
    # 已经封口，再取是空的。
    assert segmenter.take_active() is None


def test_episode_duration_never_negative():
    from app.oilcheck.judge import Episode

    assert Episode(start_ts=2.0, end_ts=1.0, peak=0.9, frames=3).duration == 0.0
    assert Episode(start_ts=1.0, end_ts=2.5, peak=0.9, frames=3).duration == pytest.approx(1.5)
