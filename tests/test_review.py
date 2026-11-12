"""通用复核摘要结构的测试。

:mod:`app.review` 是 SOP 无关的中间层：判定侧产出、复核页渲染。这里只验证
结构本身（序列化、渲染、旧数据兼容），不涉及任何具体检测逻辑。
"""

from __future__ import annotations

from app.review import (
    SUMMARY_KEY,
    ReviewMetric,
    ReviewSummary,
    fallback_lines,
    format_clock,
    summary_from_event,
)


def _summary() -> ReviewSummary:
    return ReviewSummary(
        check_name="人工二次涂油",
        verdict="漏涂 NG",
        reason="取料后 10.40 s 未达到 1.00 s 接触要求（超时）",
        metrics=(
            ReviewMetric(
                key="brush_hold_s",
                label="刷头接触累计",
                value="0.42 s",
                required="≥ 1.00 s",
                passed=False,
                hint="差 0.58 s",
            ),
            ReviewMetric(
                key="brush_min_distance",
                label="刷头离杆最近距离",
                value="0.031",
                required="≤ 0.060",
                passed=True,
            ),
            ReviewMetric(
                key="screw_visible_ratio",
                label="手持丝杆可见比例",
                value="88%",
                passed=None,
            ),
        ),
        timeline=(("PICKED", 1_700_000_000.5), ("NG", 1_700_000_010.9)),
        hints=("接触时长已达要求的 42%…",),
    )


def test_summary_roundtrip():
    summary = _summary()
    restored = ReviewSummary.from_dict(summary.to_dict())
    assert restored == summary


def test_summary_dict_contains_stable_key():
    payload = _summary().to_dict()
    assert payload["check_name"] == "人工二次涂油"
    assert payload["verdict"] == "漏涂 NG"
    assert len(payload["metrics"]) == 3
    assert payload["metrics"][0]["key"] == "brush_hold_s"
    assert payload["metrics"][0]["passed"] is False
    assert payload["timeline"][0] == ["PICKED", 1_700_000_000.5]


def test_summary_lines_show_verdict_metrics_and_hints():
    lines = _summary().lines()
    text = "\n".join(lines)
    assert "【人工二次涂油】漏涂 NG" in text
    assert "刷头接触累计：0.42 s（要求 ≥ 1.00 s）  ✗" in text
    assert "刷头离杆最近距离：0.031（要求 ≤ 0.060）  ✓" in text
    assert "复核建议" in text
    assert "时间线" in text


def test_summary_lines_can_omit_timeline():
    lines = _summary().lines(with_timeline=False)
    assert not any("时间线" in line for line in lines)


def test_summary_from_event_reads_nested_summary():
    summary = _summary()
    evidence = {"reason": "oiling_missing_timeout", SUMMARY_KEY: summary.to_dict()}
    assert summary_from_event(evidence) == summary


def test_summary_from_event_returns_none_for_old_events():
    """历史事件没有摘要键：必须返回 None，让复核页走回退渲染。"""

    assert summary_from_event({"reason": "oiling_missing_timeout"}) is None
    assert summary_from_event({}) is None
    assert summary_from_event(None) is None
    assert summary_from_event("not-a-dict") is None


def test_summary_from_dict_tolerates_broken_payload():
    """手工改过的 JSONL / 别的 SOP 写的旧结构不能让复核页崩掉。"""

    assert ReviewSummary.from_dict(None) is None
    assert ReviewSummary.from_dict("x") is None
    assert ReviewSummary.from_dict({}) is None
    # 只有一个字段也要能读出来，坏掉的 metric 直接跳过
    partial = ReviewSummary.from_dict(
        {
            "check_name": "某项检查",
            "metrics": ["bad", {"label": "只有标签"}, {"key": "k", "value": 1}],
            "timeline": [["PICKED", "123.5"], ["bad"], [None, "nan"]],
            "hints": ["提示", ""],
        }
    )
    assert partial is not None
    assert partial.check_name == "某项检查"
    assert [m.key for m in partial.metrics] == ["只有标签", "k"]
    assert partial.metrics[1].value == "1"
    assert len(partial.timeline) == 1
    assert partial.hints == ("提示",)


def test_metric_line_without_required_or_state():
    metric = ReviewMetric(key="k", label="某项", value="3 帧")
    assert metric.line() == "某项：3 帧"


def test_fallback_lines_render_legacy_evidence():
    lines = fallback_lines(
        {"reason": "oiling_missing_timeout", "brush_hold_s": 0.42, "tags": ["A", "B"]}
    )
    text = "\n".join(lines)
    assert "原始证据" in text
    assert "reason：oiling_missing_timeout" in text
    assert "tags：A、B" in text


def test_fallback_lines_handles_empty_evidence():
    assert fallback_lines({}) == ["（该事件没有记录证据）"]
    assert fallback_lines(None) == ["（该事件没有记录证据）"]
    # 带摘要的事件在回退渲染里不应重复打出摘要 dict
    lines = fallback_lines({SUMMARY_KEY: {"check_name": "x"}, "reason": "r"})
    assert not any(SUMMARY_KEY in line for line in lines)


def test_summary_supports_a_completely_different_sop():
    """复用性底线：另一个检测项只写自己的 builder，通用层无需改动。

    这里模拟「螺丝拧紧扭矩」检测，不使用任何 oilcheck 的字段或枚举。
    """

    def build_torque_summary(measured: float, required: float) -> ReviewSummary:
        ok = measured >= required
        return ReviewSummary(
            check_name="螺丝拧紧扭矩",
            verdict="扭矩合格" if ok else "扭矩不足 NG",
            metrics=(
                ReviewMetric(
                    key="torque",
                    label="实测扭矩",
                    value=f"{measured:.1f} N·m",
                    required=f"≥ {required:.1f} N·m",
                    passed=ok,
                ),
            ),
            hints=() if ok else ("检查电批设定或螺纹是否滑牙。",),
        )

    summary = build_torque_summary(8.4, 12.0)
    # 往返序列化与渲染都只依赖通用结构
    restored = ReviewSummary.from_dict(summary.to_dict())
    assert restored == summary
    text = "\n".join(restored.lines())
    assert "【螺丝拧紧扭矩】扭矩不足 NG" in text
    assert "实测扭矩：8.4 N·m（要求 ≥ 12.0 N·m）  ✗" in text


def test_format_clock_formats_and_tolerates_bad_input():
    assert format_clock(1_700_000_000.5).endswith(".500")
    assert format_clock("bad") == "-"
    assert format_clock(None) == "-"
