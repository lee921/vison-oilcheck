"""通用复核摘要结构。

判定侧产出、复核页渲染的统一中间层。设计目标是 **SOP 无关**：任何检测项
（二次涂油、螺丝有无、涂油范围……）只要产出一个 :class:`ReviewSummary`，
复核页就能按同一套结构显示，不需要为每种检测各写一份 UI。

本模块不依赖 Qt、相机 SDK 或任何具体检测逻辑，只做数据与渲染。

存放位置：``EventRecord.evidence["review_summary"]``（``to_dict`` 的结果）。
旧事件没有这个键时 :func:`summary_from_event` 返回 ``None``，复核页回退到
通用的键值渲染——因此事件 JSONL 完全向后兼容。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# 摘要存进事件 evidence 时用的键名。判定侧与复核页共用这一处常量，
# 避免两边各写一遍字符串。``SUMMARY_KEY`` 一旦写进历史事件就不要改。
SUMMARY_KEY = "review_summary"


@dataclass(frozen=True)
class ReviewMetric:
    """一项可核对的实测指标。

    ``value`` / ``required`` 都是**已格式化**的字符串：格式化属于检测项的
    口径（秒、百分比、像素各有各的单位），复核页不做单位换算，只按
    ``passed`` 着色。这样换一种 SOP 时复核页一行都不用改。
    """

    key: str
    label: str
    value: str
    required: str | None = None
    passed: bool | None = None
    hint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "value": self.value,
            "required": self.required,
            "passed": self.passed,
            "hint": self.hint,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ReviewMetric | None:
        if not isinstance(data, dict):
            return None
        key = str(data.get("key", "")).strip()
        label = str(data.get("label", "")).strip()
        if not key and not label:
            return None
        passed = data.get("passed")
        if passed is not None:
            passed = bool(passed)
        required = data.get("required")
        return cls(
            key=key or label,
            label=label or key,
            value=str(data.get("value", "-")),
            required=None if required in (None, "") else str(required),
            passed=passed,
            hint=str(data.get("hint", "")),
        )

    def line(self) -> str:
        """单行文本：``标签：值（要求 ≥ x）✓/✗``。"""

        mark = ""
        if self.passed is True:
            mark = "  ✓"
        elif self.passed is False:
            mark = "  ✗"
        required = f"（要求 {self.required}）" if self.required else ""
        return f"{self.label}：{self.value}{required}{mark}"


@dataclass(frozen=True)
class ReviewSummary:
    """一次判定的复核摘要。

    - ``check_name``：检查项名称，如「人工二次涂油」；
    - ``verdict``：结论短语，如「漏涂 NG」「已完成」；
    - ``reason``：一句判定依据（可直接对应事件的 reason 文案）；
    - ``metrics``：实测指标，复核员逐条核对；
    - ``timeline``：关键时刻 ``(阶段名, 时刻)``，与 ``_Cycle.timeline`` 同构；
    - ``hints``：复核建议，按条件生成（例如「接触时长接近阈值」）。
    """

    check_name: str
    verdict: str
    reason: str = ""
    metrics: tuple[ReviewMetric, ...] = field(default_factory=tuple)
    timeline: tuple[tuple[str, float], ...] = field(default_factory=tuple)
    hints: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_name": self.check_name,
            "verdict": self.verdict,
            "reason": self.reason,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "timeline": [[name, float(ts)] for name, ts in self.timeline],
            "hints": list(self.hints),
        }

    @classmethod
    def from_dict(cls, data: Any) -> ReviewSummary | None:
        """从事件里读回摘要；结构不对时返回 ``None`` 而不是抛异常。

        复核页面对的可能是历史事件、手工改过的 JSONL 或别的 SOP 写的
        旧结构，任何一条都不该让整个复核页崩掉。
        """

        if not isinstance(data, dict):
            return None
        check_name = str(data.get("check_name", "")).strip()
        verdict = str(data.get("verdict", "")).strip()
        if not check_name and not verdict:
            return None
        metrics: list[ReviewMetric] = []
        for item in data.get("metrics") or ():
            metric = ReviewMetric.from_dict(item)
            if metric is not None:
                metrics.append(metric)
        timeline: list[tuple[str, float]] = []
        for item in data.get("timeline") or ():
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            name, raw_ts = item[0], item[1]
            if name in (None, ""):
                continue
            try:
                value = float(raw_ts)
            except (TypeError, ValueError):
                continue
            # "nan"/"inf" 能被 float() 接受，但拿去格式化时间会炸或显示乱码。
            if not math.isfinite(value):
                continue
            timeline.append((str(name), value))
        hints = tuple(str(h) for h in (data.get("hints") or ()) if str(h).strip())
        return cls(
            check_name=check_name or verdict,
            verdict=verdict or check_name,
            reason=str(data.get("reason", "")),
            metrics=tuple(metrics),
            timeline=tuple(timeline),
            hints=hints,
        )

    def lines(self, *, with_timeline: bool = True) -> list[str]:
        """渲染成逐行文本，供 ``PlainTextEdit`` 直接显示。

        段落顺序固定为「结论 → 指标 → 建议 → 时间线」：复核员先看结论，
        再逐项核对数字，最后看建议与时间。
        """

        lines: list[str] = [f"【{self.check_name}】{self.verdict}"]
        if self.reason:
            lines.append(f"判定依据：{self.reason}")
        if self.metrics:
            lines.append("")
            lines.append("实测指标：")
            for metric in self.metrics:
                lines.append(f"  {metric.line()}")
                if metric.hint:
                    lines.append(f"      {metric.hint}")
        if self.hints:
            lines.append("")
            lines.append("复核建议：")
            for hint in self.hints:
                lines.append(f"  · {hint}")
        if with_timeline and self.timeline:
            lines.append("")
            lines.append("时间线：")
            for name, ts in self.timeline:
                lines.append(f"  {format_clock(ts)}  {name}")
        return lines


def summary_from_event(evidence: Any) -> ReviewSummary | None:
    """从事件 ``evidence`` 里取出摘要；没有或是旧结构时返回 ``None``。

    复核页用这个函数决定走「结构化渲染」还是「通用键值回退」。
    """

    if not isinstance(evidence, dict):
        return None
    return ReviewSummary.from_dict(evidence.get(SUMMARY_KEY))


def format_clock(ts: float) -> str:
    """把时间戳格式化成 ``HH:MM:SS.mmm``，时间线里用。"""

    try:
        value = float(ts)
    except (TypeError, ValueError):
        return "-"
    return datetime.fromtimestamp(value).strftime("%H:%M:%S.%f")[:-3]


def fallback_lines(evidence: Any) -> list[str]:
    """通用回退渲染：没有 ``review_summary`` 的事件（历史数据、别的 SOP）。

    只做「键 → 值」平铺，不做任何 SOP 专有解释。这样旧事件在复核页里
    至少能看到全部原始字段，而不是一片空白。
    """

    if not isinstance(evidence, dict) or not evidence:
        return ["（该事件没有记录证据）"]

    lines: list[str] = ["原始证据（无结构化摘要）："]
    for key, value in evidence.items():
        if key == SUMMARY_KEY:
            continue
        if isinstance(value, (list, tuple)):
            rendered = "、".join(str(item) for item in value)
        elif isinstance(value, dict):
            rendered = "、".join(f"{k}={v}" for k, v in value.items())
        else:
            rendered = str(value)
        lines.append(f"  {key}：{rendered}")
    return lines
