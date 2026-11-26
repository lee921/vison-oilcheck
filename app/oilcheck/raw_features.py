"""从**原始检测框**提炼二级判定特征（含数据预处理与序列聚合）。

与 ``pipeline._pick_features`` / ``_oiling_features`` 的区别：那两个函数拿
的是规则已经算好的结论（"有没有见过手握杆"、"carry 累积了多少帧"），
模型学到的是规则的影子。这里只吃原始素材：

- 这一帧的全部 YOLO 检测框（手 / 丝杆 / 刷头）及其置信度；
- 手工标定的三个区域（``tray_roi`` / ``work_roi`` / ``oiling_roi``）；
- 最近若干帧的丝杆框位置。

## 预处理的三条规矩

1. **坐标相对区域归一化**，不是相对画面。「手在盘区左边缘外 20%」在任何
   机位下含义一致；除以画面宽高只会得到一个随相机位置漂移的绝对值。
2. **缺失值单独标记**。归一化距离的理论上限是 1.0（对角线），所以不能用
   1.0 当「没找到」。统一用 ``*_found`` 布尔 + 距离填 ``2.0``（超出理论
   最大值，不会与真实近距离混淆）。
3. **丢掉常量列**。``tray_w`` / ``tray_h`` 在全部样本上是同一个值（9 段
   视频同分辨率、同标定），留着只会让树模型切出无意义的分支。

## 为什么按事件聚合而不是取结算那一帧

单帧几何噪声大：杆框会在相邻帧之间跳 0.02~0.03，手里那根还常被拳头完全
挡住。只看结算帧等于拿一次抛硬币的结果去训练。这里把整个动作期间每帧的
原始测量攒起来，输出**持续性统计量**（占比、最大值、累计值），对单帧
漏检和抖动天然不敏感——这与规则内部累积握持采样的道理相同，但累积的是
原始几何，不是规则结论。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .association import brush_work_point
from .config import DetectorConfig
from .types import BBox, Detection

# 手边候选杆的距离门限（按画面对角线归一化）。
NEAR_HAND_DISTANCE = 0.15
# 「杆落在手里」的面积比例门限。
IN_HAND_FRAC = 0.30
# 「这根杆动过」的位移门限（归一化）。
ROD_MOVE_EPS = 0.010
# 「这根杆一直没动」的 IoU 门限。
PARKED_IOU = 0.50
# 与多少帧前对照判断「动没动」。
MOVE_FRAMES_BACK = 8
# 没找到候选时距离填这个值（超过归一化对角线 1.0，不会与真实距离混淆）。
MISSING_DISTANCE = 2.0


def _iou(a: BBox, b: BBox) -> float:
    inter = a.intersection_area(b)
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _contains(box: BBox, point: tuple[float, float]) -> bool:
    return box.x1 <= point[0] <= box.x2 and box.y1 <= point[1] <= box.y2


def _in_hand_fraction(hand_box: BBox, rod_box: BBox) -> float:
    area = rod_box.width * rod_box.height
    return hand_box.intersection_area(rod_box) / area if area > 0 else 0.0


def _dist_to_long_axis(box: BBox, point: tuple[float, float]) -> float:
    """点到杆身中线的垂距（像素），投影落在线段外时按端点计。"""

    cx, cy = box.center
    if box.width >= box.height:
        over = max(box.x1 - point[0], 0.0, point[0] - box.x2)
        perp = point[1] - cy
    else:
        over = max(box.y1 - point[1], 0.0, point[1] - box.y2)
        perp = point[0] - cx
    return math.hypot(over, perp)


@dataclass
class RawFrame:
    """一帧的原始素材（全部来自检测器输出与手工标定）。"""

    ts: float
    width: int
    height: int
    screws: list[Detection] = field(default_factory=list)
    hands: list[tuple[int, BBox]] = field(default_factory=list)
    brushes: list[tuple[int, BBox]] = field(default_factory=list)
    tray: BBox | None = None
    work: BBox | None = None
    oiling: BBox | None = None
    screw_history: list[list[BBox]] = field(default_factory=list)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    @property
    def diag(self) -> float:
        return max(1.0, math.hypot(self.width, self.height))

    # -- 预处理：相对区域归一化 ------------------------------------------
    def rel_x(self, x: float) -> float:
        """横坐标相对**盘区**归一化：0 = 盘区左沿，1 = 右沿。"""

        if self.tray is None or self.tray.width <= 0:
            return x / max(1, self.width)
        return (x - self.tray.x1) / self.tray.width

    def rel_y(self, y: float) -> float:
        """纵坐标相对**盘区**归一化：0 = 上沿，1 = 下沿。"""

        if self.tray is None or self.tray.height <= 0:
            return y / max(1, self.height)
        return (y - self.tray.y1) / self.tray.height

    def in_tray(self, box: BBox) -> bool:
        return self.tray is not None and _contains(self.tray, box.center)

    # -- 预处理：杆「动没动」 --------------------------------------------
    def moved_since(self, box: BBox, frames_back: int = MOVE_FRAMES_BACK) -> float:
        """这根杆相对 ``frames_back`` 帧前的位置移动了多少（归一化）。

        历史不足时返回 0——开局没有对照，不能凭它下结论。
        """

        if len(self.screw_history) < frames_back:
            return 0.0
        past = self.screw_history[-frames_back]
        if not past:
            return 0.0
        cx, cy = box.center
        return min(
            (math.hypot(cx - other.center[0], cy - other.center[1]) for other in past),
            default=0.0,
        ) / self.diag

    def parked_iou(self, box: BBox, frames_back: int = MOVE_FRAMES_BACK) -> float:
        """这根杆与 ``frames_back`` 帧前同位框的最大 IoU。"""

        if len(self.screw_history) < frames_back:
            return 0.0
        past = self.screw_history[-frames_back]
        return max((_iou(box, other) for other in past), default=0.0)


# ----------------------------------------------------------------------
# 逐帧原始测量（供聚合器收集）
# ----------------------------------------------------------------------
@dataclass
class PickSample:
    """一帧里关于「这只手 + 它手边的杆」的原始测量。"""

    hand_rel_x: float
    hand_rel_y: float
    hand_w: float
    hand_h: float
    hand_in_tray: bool
    hand_below_tray: float     # 相对盘区高度，>0 表示在下沿之外
    found: bool                # 手边找到候选杆
    rod_dist: float
    rod_iou: float
    rod_in_hand: float
    rod_moved: float
    rod_parked: float
    rod_conf: float
    rod_in_tray: bool
    tray_rod_n: int
    other_hand_dist: float
    hand_n: int


def measure_pick(frame: RawFrame, hand_id: int, hand_box: BBox) -> PickSample:
    """从这一帧的原始检测算出测量值（不做任何判定）。"""

    hx, hy = hand_box.center
    tray_rod_n = sum(1 for det in frame.screws if frame.in_tray(det.box))

    # 手边候选杆：离手最近、且重叠最大
    best: tuple[float, Detection] | None = None
    for det in frame.screws:
        cx, cy = det.box.center
        distance = math.hypot(cx - hx, cy - hy) / frame.diag
        if distance > NEAR_HAND_DISTANCE:
            continue
        score = (1.0 - distance / NEAR_HAND_DISTANCE) + _in_hand_fraction(hand_box, det.box)
        if best is None or score > best[0]:
            best = (score, det)

    if best is None:
        rod_dist = MISSING_DISTANCE
        rod_iou = 0.0
        rod_in_hand = 0.0
        rod_moved = 0.0
        rod_parked = 0.0
        rod_conf = 0.0
        rod_in_tray = False
    else:
        rod_box = best[1].box
        cx, cy = rod_box.center
        rod_dist = math.hypot(cx - hx, cy - hy) / frame.diag
        rod_iou = _iou(hand_box, rod_box)
        rod_in_hand = _in_hand_fraction(hand_box, rod_box)
        rod_moved = frame.moved_since(rod_box)
        rod_parked = frame.parked_iou(rod_box)
        rod_conf = float(best[1].confidence)
        rod_in_tray = frame.in_tray(rod_box)

    others = [box for hid, box in frame.hands if hid != hand_id]
    other_dist = min(
        (math.hypot(b.center[0] - hx, b.center[1] - hy) / frame.diag for b in others),
        default=MISSING_DISTANCE,
    )
    return PickSample(
        hand_rel_x=frame.rel_x(hx),
        hand_rel_y=frame.rel_y(hy),
        hand_w=hand_box.width / max(1, frame.width),
        hand_h=hand_box.height / max(1, frame.height),
        hand_in_tray=frame.in_tray(hand_box),
        hand_below_tray=(
            0.0 if frame.tray is None or frame.tray.height <= 0
            else (hy - frame.tray.y2) / frame.tray.height
        ),
        found=best is not None,
        rod_dist=rod_dist,
        rod_iou=rod_iou,
        rod_in_hand=rod_in_hand,
        rod_moved=rod_moved,
        rod_parked=rod_parked,
        rod_conf=rod_conf,
        rod_in_tray=rod_in_tray,
        tray_rod_n=tray_rod_n,
        other_hand_dist=other_dist,
        hand_n=len(frame.hands),
    )


class PickAggregator:
    """把一趟进盘期间每帧的原始测量聚合成鲁棒特征。

    输出的是**占比与极值**，不是某一帧的快照：

    - ``rod_in_hand_ratio``：手里握着杆的时间占比（对单帧漏检不敏感）；
    - ``rod_moved_ratio``：杆「跟着动」的时间占比；
    - ``rod_parked_ratio``：杆「一直没动」的时间占比（夹具形态）；
    - ``tray_rod_delta``：进盘到离盘盘内杆数的变化。

    真取料：``rod_in_hand_ratio`` / ``rod_moved_ratio`` 高、``rod_parked_ratio``
    低；示例⑧⑨ 的夹具杆则相反。
    """

    def __init__(self) -> None:
        self.samples: list[PickSample] = []

    def add(self, sample: PickSample) -> None:
        self.samples.append(sample)

    def features(self, *, dwell: float, start_tray_rods: int | None = None) -> dict[str, float]:
        total = len(self.samples)
        if total == 0:
            return {"dwell": float(dwell), "samples": 0.0}

        found = [s for s in self.samples if s.found]
        n = max(1, len(found))

        def ratio(predicate) -> float:
            return sum(1 for s in found if predicate(s)) / n

        first, last = self.samples[0], self.samples[-1]
        tray_delta = (
            (first.tray_rod_n - last.tray_rod_n)
            if start_tray_rods is None
            else (start_tray_rods - last.tray_rod_n)
        )
        return {
            "dwell": float(dwell),
            "samples": float(total),
            # ---- 手的运动（相对动作本身，不用绝对位置）----
            # 绝对位置（``hand_rel_x_first``）是个陷阱：同一段视频里，工人
            # 从盘区左侧伸进去就是取料、从右侧蹭一下就不是，模型学会的是
            # 「这只手习惯从哪边来」，换个人/换个工位立刻失效。这里只留
            # **运动量**：进了盘区多少、横向走了多远、手框大小。
            "hand_travel": math.hypot(
                last.hand_rel_x - first.hand_rel_x,
                last.hand_rel_y - first.hand_rel_y,
            ),
            "hand_travel_x": abs(last.hand_rel_x - first.hand_rel_x),
            "hand_below_tray_last": last.hand_below_tray,
            "hand_w": last.hand_w,
            "hand_h": last.hand_h,
            # ---- 持续性（核心）----
            "rod_in_hand_ratio": ratio(lambda s: s.rod_in_hand >= IN_HAND_FRAC),
            "rod_moved_ratio": ratio(lambda s: s.rod_moved >= ROD_MOVE_EPS),
            "rod_parked_ratio": ratio(lambda s: s.rod_parked >= PARKED_IOU),
            "rod_in_tray_ratio": ratio(lambda s: s.rod_in_tray),
            "found_ratio": len(found) / total,
            # ---- 幅值 ----
            "rod_in_hand_max": max((s.rod_in_hand for s in found), default=0.0),
            "rod_moved_max": max((s.rod_moved for s in found), default=0.0),
            "rod_parked_max": max((s.rod_parked for s in found), default=0.0),
            "rod_iou_max": max((s.rod_iou for s in found), default=0.0),
            "rod_dist_min": min((s.rod_dist for s in found), default=MISSING_DISTANCE),
            "rod_conf_mean": (
                sum(s.rod_conf for s in found) / n if found else 0.0
            ),
            # ---- 盘内杆数变化 ----
            # 注意：这两个是**计数**，不是归一化距离，别和 MISSING_DISTANCE 混。
            "tray_rod_delta": float(tray_delta),
            "tray_rod_n_last": float(last.tray_rod_n),
            # ---- 手边那根杆的出入盘（取料的物理定义）----
            # 盘里常年有存货，只看「盘里还剩几根」分不开取料与按压：
            # 取料的定义是**手边这一根**从盘内到了盘外。这三个量就按这个
            # 定义取，对存货多少不敏感。
            "rod_tray_first": float(first.found and first.rod_in_tray),
            "rod_tray_last": float(last.found and last.rod_in_tray),
            "rod_exited": float(
                first.found and first.rod_in_tray
                and last.found and not last.rod_in_tray
            ),
            # 出盘那一刻还握在手里（重叠够大）——「带出去」而不是「掉出去」。
            "rod_exited_in_hand": float(
                first.found and first.rod_in_tray
                and last.found and not last.rod_in_tray
                and last.rod_in_hand >= IN_HAND_FRAC
            ),
            # 手边杆在盘内的帧占比（存货一直跟着手 → 接近 1，取料会降）
            "rod_tray_ratio": ratio(lambda s: s.found and s.rod_in_tray),
            # ---- 环境 ----
            # 其他手：没有别的候选时填 MISSING_DISTANCE；有则用最远可及距离，
            # 不用 1.0（1.0 是归一化对角线的理论上限，会和真实距离混淆）。
            "other_hand_dist_min": min(
                (s.other_hand_dist for s in self.samples), default=MISSING_DISTANCE
            ),
            "hand_n_max": float(max(s.hand_n for s in self.samples)),
        }


# ----------------------------------------------------------------------
# 涂油
# ----------------------------------------------------------------------
@dataclass
class OilingSample:
    brush_rod_dist: float
    brush_hand_dist: float
    rod_in_hand: float
    rod_moved: float
    rod_parked: float
    held: bool
    brush_found: bool
    rod_found: bool


def measure_oiling(frame: RawFrame, hand_box: BBox | None) -> OilingSample:
    """刷头工作点与手持杆的原始几何关系。"""

    if hand_box is None:
        return OilingSample(
            brush_rod_dist=MISSING_DISTANCE, brush_hand_dist=MISSING_DISTANCE,
            rod_in_hand=0.0, rod_moved=0.0, rod_parked=0.0,
            held=False, brush_found=bool(frame.brushes), rod_found=False,
        )
    hx, hy = hand_box.center
    held = [
        det for det in frame.screws
        if _contains(hand_box, det.box.center)
        or _in_hand_fraction(hand_box, det.box) >= IN_HAND_FRAC
    ]
    candidates = held or frame.screws

    best_dist = MISSING_DISTANCE
    best_hand_dist = MISSING_DISTANCE
    best_in_hand = 0.0
    best_moved = 0.0
    best_parked = 0.0
    for _, brush_box in frame.brushes:
        point = brush_work_point(
            Detection(label="brush", confidence=0.0, box=brush_box), frame.detector
        )
        best_hand_dist = min(
            best_hand_dist, math.hypot(point[0] - hx, point[1] - hy) / frame.diag
        )
        for det in candidates:
            distance = _dist_to_long_axis(det.box, point) / frame.diag
            if distance < best_dist:
                best_dist = distance
                best_in_hand = _in_hand_fraction(hand_box, det.box)
                best_moved = frame.moved_since(det.box)
                best_parked = frame.parked_iou(det.box)
    return OilingSample(
        brush_rod_dist=best_dist,
        brush_hand_dist=best_hand_dist,
        rod_in_hand=best_in_hand,
        rod_moved=best_moved,
        rod_parked=best_parked,
        held=bool(held),
        brush_found=bool(frame.brushes),
        rod_found=bool(candidates),
    )


class OilingAggregator:
    """把一段刷涂期间每帧的原始测量聚合。"""

    def __init__(self) -> None:
        self.samples: list[OilingSample] = []

    def add(self, sample: OilingSample) -> None:
        self.samples.append(sample)

    def features(self, *, cum_s: float, frames: int) -> dict[str, float]:
        total = len(self.samples)
        if total == 0:
            return {"cum_s": float(cum_s), "frames": float(frames), "samples": 0.0}
        with_brush = [s for s in self.samples if s.brush_found]
        n = max(1, len(with_brush))

        def ratio(predicate) -> float:
            return sum(1 for s in with_brush if predicate(s)) / n

        return {
            "cum_s": float(cum_s),
            "frames": float(frames),
            "samples": float(total),
            # 刷头真的贴近杆身的时间占比（核心：短接触 vs 真刷）
            "contact_ratio": ratio(
                lambda s: s.brush_rod_dist <= 0.06
            ),
            "held_ratio": ratio(lambda s: s.held),
            "rod_moved_ratio": ratio(lambda s: s.rod_moved >= ROD_MOVE_EPS),
            "rod_parked_ratio": ratio(lambda s: s.rod_parked >= PARKED_IOU),
            "brush_found_ratio": len(with_brush) / total,
            # 幅值
            "brush_rod_dist_min": min(
                (s.brush_rod_dist for s in with_brush), default=MISSING_DISTANCE
            ),
            "brush_hand_dist_min": min(
                (s.brush_hand_dist for s in with_brush), default=MISSING_DISTANCE
            ),
            "rod_in_hand_max": max((s.rod_in_hand for s in self.samples), default=0.0),
            "rod_moved_max": max((s.rod_moved for s in self.samples), default=0.0),
        }


PICK_FEATURE_NAMES = [
    "dwell", "samples",
    "hand_travel", "hand_travel_x",
    "hand_below_tray_last", "hand_w", "hand_h",
    "rod_in_hand_ratio", "rod_moved_ratio", "rod_parked_ratio",
    "rod_in_tray_ratio", "found_ratio",
    "rod_in_hand_max", "rod_moved_max", "rod_parked_max",
    "rod_iou_max", "rod_dist_min", "rod_conf_mean",
    "tray_rod_delta", "tray_rod_n_last",
    "rod_tray_first", "rod_tray_last", "rod_exited", "rod_exited_in_hand",
    "rod_tray_ratio",
    "other_hand_dist_min", "hand_n_max",
]

OILING_FEATURE_NAMES = [
    "cum_s", "frames", "samples",
    "contact_ratio", "held_ratio", "rod_moved_ratio", "rod_parked_ratio",
    "brush_found_ratio",
    "brush_rod_dist_min", "brush_hand_dist_min",
    "rod_in_hand_max", "rod_moved_max",
]
