"""油检 SOP 的复核摘要构造。

本模块是「丝杆取料 + 二次涂油」这一检测项与通用复核结构
(:mod:`app.review`) 之间的唯一适配层。以后新增别的 SOP，只需另建一个
同名的 builder 模块（例如 ``app/<sop>/review.py``），复核页不用改。

指标口径与判定逻辑严格对应：

- ``brush_hold_s`` —— ``_update_oiling`` 累加的那个值，判定完成与否的唯一依据；
- ``brush_min_distance`` —— 本周期内刷头工作点到杆心的最小归一化距离，
  与 ``_nearest_brush`` 同一算法（``brush_work_point``）；
- ``brush_near_frames`` / ``screw_visible_frames`` —— 让复核员看出「偶尔蹭到」
  与「一直在刷」、以及判定本身受漏检影响的程度。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.review import ReviewMetric, ReviewSummary

if TYPE_CHECKING:  # 只为类型标注，避免运行时循环导入
    from .config import PipelineConfig
    from .pipeline import _Cycle

# 复核建议里用的阈值：接触累计达到要求的这个比例时，提示「可能只是动作偏快」。
# 取 0.5 而不是 0.7：现场最常见的临界情形是「刷到一半被打断」（0.5~0.7），
# 用 0.7 会把恰好 0.67 这种最该提示的情况漏掉。
_NEAR_THRESHOLD_HINT_RATIO = 0.5
# 手持丝杆可见比例低于该值时，提示判定可能受漏检影响。
_LOW_VISIBLE_RATIO = 0.5
# 刷头最近距离超过接触门限的这个倍数，视为「全程没靠近过」。
_FAR_DISTANCE_RATIO = 1.5


def _fmt_s(value: float) -> str:
    return f"{value:.2f} s"


def _fmt_ratio(value: float) -> str:
    return f"{value * 100:.0f}%"


def _stages(task: _Cycle) -> tuple[tuple[str, float], ...]:
    """周期时间线，去掉重复项（同一阶段可能被追加多次）。"""

    seen: list[tuple[str, float]] = []
    for name, ts in task.timeline:
        if seen and seen[-1][0] == name:
            continue
        seen.append((name, ts))
    return tuple(seen)


def build_pick_summary(task: _Cycle, *, dwell_s: float) -> ReviewSummary:
    """取料完成事件的摘要。"""

    metrics = (
        ReviewMetric(
            key="dwell_in_tray_s",
            label="手在盘内停留",
            value=_fmt_s(dwell_s),
            required="非取料动作通常 < 0.30 s",
            passed=True,
            hint="空手进盘、带料离盘，取料本身成立。",
        ),
        ReviewMetric(
            key="screw_confidence",
            label="离盘时丝杆置信度",
            value=(
                "-"
                if task.confidence is None
                else f"{float(task.confidence):.2f}"
            ),
            passed=None,
            hint=(
                "0.00 表示定论那一帧恰好没检出杆，用的是占位证据（杆被手遮住）。"
                if task.confidence is not None and float(task.confidence) <= 0.0
                else ""
            ),
        ),
    )
    return ReviewSummary(
        check_name="取料",
        verdict="取料确认",
        reason="手空手进盘、随后带着丝杆离开",
        metrics=metrics,
        timeline=_stages(task),
    )


def build_oiling_summary(task: _Cycle, *, required_s: float) -> ReviewSummary:
    """人工二次涂油完成事件的摘要。"""

    metrics = (
        ReviewMetric(
            key="brush_hold_s",
            label="刷头接触累计",
            value=_fmt_s(task.brush_hold_s),
            required=f"≥ {required_s:.2f} s",
            passed=task.brush_hold_s >= required_s,
        ),
        _min_distance_metric(task, None),
    )
    return ReviewSummary(
        check_name="人工二次涂油",
        verdict="涂油完成",
        reason=f"刷头持续靠近手持丝杆累计达到 {required_s:.2f} s",
        metrics=metrics,
        timeline=_stages(task),
    )


def build_oiling_missing_summary(
    task: _Cycle,
    *,
    trigger: str,
    ts: float,
    cfg: PipelineConfig,
) -> ReviewSummary:
    """漏涂 NG 事件的摘要。

    ``trigger`` 区分发现方式：超时（``oiling_timeout``）还是被下一次取料
    打断（``superseded_by_pick``）。两者判的是同一件事，但复核时的处置不同，
    所以分开给建议。
    """

    required_s = float(cfg.brush_hold_s)
    contact_s = float(cfg.brush_contact_distance)
    elapsed = (
        ts - task.picked_at if task.picked_at is not None else 0.0
    )
    # 分母是「取料后进入涂油判定循环的帧数」，分子是「手持丝杆被定位到的
    # 帧数」。两者之比即杆在判定窗口内的可见比例。
    visible_ratio = (
        task.tap_frames / task.oiling_frames if task.oiling_frames > 0 else 0.0
    )
    timed_out = trigger == "oiling_timeout"

    metrics: list[ReviewMetric] = [
        ReviewMetric(
            key="brush_hold_s",
            label="刷头接触累计",
            value=_fmt_s(task.brush_hold_s),
            required=f"≥ {required_s:.2f} s",
            passed=task.brush_hold_s >= required_s,
            hint=(
                "差 "
                f"{max(0.0, required_s - task.brush_hold_s):.2f} s"
                if task.brush_hold_s < required_s
                else ""
            ),
        ),
        _min_distance_metric(task, contact_s),
        ReviewMetric(
            key="brush_near_frames",
            label="刷头落入接触邻域的帧数",
            value=f"{task.brush_near_frames} 帧",
            passed=None,
            hint=(
                "0 帧 ⇒ 本周期刷头从未进入接触距离"
                if task.brush_near_frames == 0
                else ""
            ),
        ),
        ReviewMetric(
            key="brush_longest_run_s",
            label="最长连续接触",
            value=_fmt_s(task.brush_longest_run_s),
            required=f"≥ {required_s:.2f} s",
            passed=task.brush_longest_run_s >= required_s,
            hint=(
                "断续靠近（累计接近阈值但连续段很短）通常是随手点几下，不是涂油。"
                if 0.0 < task.brush_hold_s < required_s
                and task.brush_longest_run_s < task.brush_hold_s * 0.8
                else ""
            ),
        ),
        ReviewMetric(
            key="screw_visible_ratio",
            label="手持丝杆可见比例",
            value=_fmt_ratio(visible_ratio),
            passed=None,
            hint=(
                "判定窗口内丝杆大半时间不可见，接触累计可能被漏检压低。"
                if visible_ratio < _LOW_VISIBLE_RATIO
                else ""
            ),
        ),
        ReviewMetric(
            key="elapsed_s",
            label="取料后经过",
            value=_fmt_s(elapsed),
            required=(
                f"超时 {float(cfg.oiling_timeout_s):.2f} s"
                if timed_out
                else None
            ),
            passed=None,
        ),
    ]

    hints: list[str] = []
    if task.brush_near_frames == 0 or (
        task.brush_min_distance is not None
        and task.brush_min_distance > contact_s * _FAR_DISTANCE_RATIO
    ):
        hints.append(
            "刷头全程没有靠近过丝杆：可能确实没刷，也可能是刷头工作点估算偏移"
            "（检查 detector.brush_point_offset_x / brush_point_offset_y）。"
        )
    elif task.brush_hold_s >= required_s * _NEAR_THRESHOLD_HINT_RATIO:
        hints.append(
            f"接触时长已达要求的 {task.brush_hold_s / required_s * 100:.0f}%，"
            "可能是涂油动作偏快或阈值偏严——建议调出录像人工确认这一件。"
        )
    if visible_ratio < _LOW_VISIBLE_RATIO:
        hints.append(
            "手持丝杆在判定窗口内大量缺失：确认工人是否拿着杆走出了工作追踪区。"
        )
    if timed_out:
        hints.append(
            f"取料后 {elapsed:.1f} s 内未完成涂油。现场若允许工人离岗取料再回来，"
            f"调大 pipeline.oiling_timeout_s（当前 {float(cfg.oiling_timeout_s):.1f} s）。"
        )
    else:
        hints.append("被下一次取料打断：上一件还没刷完就取了下一件。")
    if not hints:
        hints.append("各项指标均未接近阈值，可判为漏涂。")

    return ReviewSummary(
        check_name="人工二次涂油",
        verdict="漏涂 NG",
        reason=(
            f"取料后 {elapsed:.2f} s 未达到 {required_s:.2f} s 接触要求"
            f"（{'超时' if timed_out else '被下一次取料打断'}）"
        ),
        metrics=tuple(metrics),
        timeline=_stages(task),
        hints=tuple(hints),
    )


def _min_distance_metric(
    task: _Cycle, contact_s: float | None
) -> ReviewMetric:
    """刷头离杆最近的距离。没靠近过时 ``value`` 显示 ``-``。"""

    distance = task.brush_min_distance
    if distance is None:
        return ReviewMetric(
            key="brush_min_distance",
            label="刷头离杆最近距离",
            value="-",
            required=f"≤ {contact_s:.3f}" if contact_s is not None else None,
            passed=False if contact_s is not None else None,
            hint="本周期判定窗口内没有成功定位到手持丝杆。",
        )
    passed = None if contact_s is None else distance <= contact_s
    return ReviewMetric(
        key="brush_min_distance",
        label="刷头离杆最近距离",
        value=f"{distance:.3f}",
        required=f"≤ {contact_s:.3f}" if contact_s is not None else None,
        passed=passed,
    )
