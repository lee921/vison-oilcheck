"""ROI 定义与几何判断。

ROI 坐标统一使用归一化坐标（0~1），相对原始视频帧，现场标定后即可复用，
不受预览窗口大小或解码分辨率变化影响。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .types import BBox, Detection


@dataclass(frozen=True)
class Roi:
    """支持矩形和多边形的归一化 ROI。"""

    kind: str
    points: tuple[tuple[float, float], ...]

    @classmethod
    def from_dict(cls, data: Any) -> Roi | None:
        if data is None:
            return None
        if isinstance(data, dict):
            kind = str(data.get("kind", "rect")).strip().lower()
            raw_points = data.get("points", ())
        else:
            # 兼容 [[x1,y1],[x2,y2]] 的简写
            kind = "rect"
            raw_points = data

        points: list[tuple[float, float]] = []
        try:
            for point in raw_points:
                x, y = float(point[0]), float(point[1])
                points.append((_clamp01(x), _clamp01(y)))
        except (TypeError, ValueError, IndexError):
            return None

        if kind == "polygon":
            if len(points) < 3:
                return None
            return cls(kind="polygon", points=tuple(points))

        if len(points) < 2:
            return None
        (x1, y1), (x2, y2) = points[0], points[1]
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left <= 1e-6 or bottom - top <= 1e-6:
            return None
        return cls(kind="rect", points=((left, top), (right, bottom)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "points": [[round(x, 6), round(y, 6)] for x, y in self.points],
        }

    def pixel_points(self, width: int, height: int) -> tuple[tuple[float, float], ...]:
        w = max(1, int(width))
        h = max(1, int(height))
        return tuple((x * w, y * h) for x, y in self.points)

    def pixel_bbox(self, width: int, height: int) -> BBox:
        points = self.pixel_points(width, height)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return BBox(min(xs), min(ys), max(xs), max(ys))

    def expanded_pixel_bbox(
        self,
        width: int,
        height: int,
        *,
        margin: float,
    ) -> BBox:
        """按归一化边距扩张 ROI，用于“手部接近某工位”的判断。"""
        box = self.pixel_bbox(width, height)
        dx = float(margin) * max(1, width)
        dy = float(margin) * max(1, height)
        return BBox(
            max(0.0, box.x1 - dx),
            max(0.0, box.y1 - dy),
            min(float(width), box.x2 + dx),
            min(float(height), box.y2 + dy),
        )

    def contains_point(self, x: float, y: float, width: int, height: int) -> bool:
        nx = x / max(1, int(width))
        ny = y / max(1, int(height))
        nx = _clamp01(nx)
        ny = _clamp01(ny)
        if self.kind == "rect":
            (x1, y1), (x2, y2) = self.points
            return x1 <= nx <= x2 and y1 <= ny <= y2
        return _point_in_polygon(nx, ny, self.points)

    def contains_box_center(self, box: BBox, width: int, height: int) -> bool:
        cx, cy = box.center
        return self.contains_point(cx, cy, width, height)

    def intersects_box(self, box: BBox, width: int, height: int) -> bool:
        if self.kind == "rect":
            return self.pixel_bbox(width, height).intersects(box)
        # 多边形：中心点在内，或任一 corner 在内，或用采样点近似。
        if self.contains_box_center(box, width, height):
            return True
        for px, py in (
            (box.x1, box.y1),
            (box.x2, box.y1),
            (box.x1, box.y2),
            (box.x2, box.y2),
        ):
            if self.contains_point(px, py, width, height):
                return True
        # 采样边中点，覆盖 ROI 完全落在检测框内部的情况。
        for t in (0.25, 0.5, 0.75):
            if self.contains_point(
                box.x1 + box.width * t,
                box.y1 + box.height * t,
                width,
                height,
            ):
                return True
        return False

    def union_bbox(self, others: Iterable[Roi], width: int, height: int) -> BBox:
        boxes = [self.pixel_bbox(width, height)]
        boxes.extend(r.pixel_bbox(width, height) for r in others)
        return BBox(
            min(b.x1 for b in boxes),
            min(b.y1 for b in boxes),
            max(b.x2 for b in boxes),
            max(b.y2 for b in boxes),
        )


def best_detection_in_roi(
    detections: Iterable[Detection],
    roi: Roi,
    width: int,
    height: int,
    *,
    label: str,
) -> Detection | None:
    """返回中心点落在 ROI 内、指定类别中置信度最高的检测。"""
    best: Detection | None = None
    for det in detections:
        if det.label != label:
            continue
        if not roi.contains_box_center(det.box, width, height):
            continue
        if best is None or det.confidence > best.confidence:
            best = det
    return best


def detections_in_roi(
    detections: Iterable[Detection],
    roi: Roi,
    width: int,
    height: int,
    *,
    label: str,
) -> tuple[Detection, ...]:
    return tuple(
        det
        for det in detections
        if det.label == label and roi.contains_box_center(det.box, width, height)
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _point_in_polygon(
    x: float, y: float, points: tuple[tuple[float, float], ...]
) -> bool:
    inside = False
    n = len(points)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = points[i]
        xj, yj = points[j]
        if (yi > y) != (yj > y):
            ratio = (y - yi) / ((yj - yi) or 1e-12)
            if x < xi + ratio * (xj - xi):
                inside = not inside
        j = i
    return inside
