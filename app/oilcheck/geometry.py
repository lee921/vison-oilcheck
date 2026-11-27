"""判定用到的纯几何/统计小工具。

全部是无状态的自由函数：输入检测框与阈值，输出数值或布尔。放在单独模块
里，既方便单测，也让 ``pipeline.py`` 只剩判定流程本身。
"""

from __future__ import annotations

import math
from collections import deque

from .types import BBox, Detection


def _push(samples: deque[bool], value: bool, keep: int) -> None:
    """把一次观测推进滑动窗口，窗口只保留最近 ``keep`` 个。"""

    while len(samples) >= max(1, keep):
        samples.popleft()
    samples.append(value)


def _majority(samples: deque[bool] | list[bool]) -> bool:
    """多数票（严格过半）。空窗口按「否」处理。"""

    if not samples:
        return False
    return sum(1 for value in samples if value) * 2 > len(samples)


# 手框「虎口」区域：以手框中心为原点、右上那一块的相对范围。
_GRIP_CORNER_RX = 0.5
_GRIP_CORNER_RY = 0.5
# 位置没动过的杆要落进虎口多大比例才算握着。
_GRIP_CORNER_MIN_FRAC = 0.30


def _grip_corner_fraction(hand_box: BBox, screw_box: BBox) -> float:
    """杆有多大比例的面积落在手框右上角区域里（0~1）。"""

    x1 = hand_box.x2 - hand_box.width * _GRIP_CORNER_RX
    y2 = hand_box.y1 + hand_box.height * _GRIP_CORNER_RY
    inter = BBox(x1, hand_box.y1, hand_box.x2, y2).intersection_area(screw_box)
    area = screw_box.width * screw_box.height
    return inter / area if area > 0 else 0.0


def _gripped_by_corner(
    hand_box: BBox,
    screw_box: BBox,
    *,
    min_frac: float = _GRIP_CORNER_MIN_FRAC,
) -> bool:
    """杆框与手框右上角区域相交，且杆有足够比例落在这块区域里。

    这是「拳头握住杆身、杆从虎口伸出」的几何特征：杆心已经跑到手框外，
    ``_box_center_in`` 判不出来；丝杆面积只有手框的 17~26%，IoU 也不够。

    ``min_frac`` 由调用方按杆的状态选择：确认静置的盘内杆要求更高的重叠
    （盘边蹭到的静置杆也能交到 0.25~0.42），已经脱离原位的杆取 0 即可。
    """

    if screw_box.width <= 0 or screw_box.height <= 0:
        return False
    frac = _grip_corner_fraction(hand_box, screw_box)
    return frac > 0 and frac >= min_frac


def _distance_to_long_axis(box: BBox, point: tuple[float, float]) -> float:
    """点到杆身中线的垂直距离（像素）；投影落到杆身之外时按端点计。

    丝杆框又细又长（现场框 60×300 px，长宽比 p50≈4、p90 以上 6~10），
    「到框中心」的距离随杆的朝向与长短剧烈变化：刷子贴着手持杆的中下部
    刷，工作点离杆心天然就有 0.05~0.07 对角线，而门限只有 0.06——同一段
    涂油里一半的帧会被判成「没靠近」，累计时长因此偏短，整件被误判漏涂
    （示例⑤ 第 2 次取料：刷了 1.3 s，累计只到 0.93 s）。

    接触的物理含义是「刷子碰到杆身」，不是「刷子碰到杆的中点」，所以取
    到中线的垂距；沿杆身方向的偏移不算距离。杆的朝向按框的较长边取
    （检测器不输出 OBB 角度，见 :meth:`BBox.long_axis`）。
    """

    cx, cy = box.center
    if box.width >= box.height:
        over = max(box.x1 - point[0], 0.0, point[0] - box.x2)
        perp = point[1] - cy
    else:
        over = max(box.y1 - point[1], 0.0, point[1] - box.y2)
        perp = point[0] - cx
    return math.hypot(over, perp)


def _overlaps(a: BBox, b: BBox) -> bool:
    """两个框是否有实际交叠（不含仅贴边）。"""

    return (
        min(a.x2, b.x2) - max(a.x1, b.x1) > 0
        and min(a.y2, b.y2) - max(a.y1, b.y1) > 0
    )


def _distance_to(box: BBox, point: tuple[float, float], diag: float) -> float:
    """框心到点的归一化距离（除以画面对角线）。"""

    cx, cy = box.center
    return math.hypot(cx - point[0], cy - point[1]) / diag


def _box_center_in(box: BBox, region: BBox) -> bool:
    """框心是否落在区域内。"""

    cx, cy = box.center
    return region.x1 <= cx <= region.x2 and region.y1 <= cy <= region.y2


def _exit_depth(box: BBox, tray: BBox) -> float:
    """杆心离盘区边界的**有符号**像素距离。

    正值 = 杆心在盘区内，值是它到最近边线的距离；负值 = 已出盘，绝对值是
    越出边线的距离。取「到最近的边」而不是到某一条边，盘区四边任一方向
    出盘都一致处理；杆心在盘区外时绝对值就是最近的越界量。

    「越过边线多少才算真的出盘」由 ``screw_exit_pick_margin`` 控制
    （调用方按画面对角线归一化后比较），这里只算原始像素量。
    """

    cx, cy = box.center
    return min(cx - tray.x1, tray.x2 - cx, cy - tray.y1, tray.y2 - cy)


def _label_counts(detections: tuple[Detection, ...]) -> dict[str, int]:
    """按类别统计本帧目标数，供 UI 核对“所有标签是否都出来了”。"""

    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return dict(sorted(counts.items()))
