"""手-丝杆空间关联。

:class:`HandScrewAssociator` 用**空间距离 + 交互关系**判断每只手这一帧正在操作
哪根丝杆，输出 ``hand_track_id -> 丝杆检测``。多手多丝杆时做全局一对一分配
（带 UNMATCHED 虚拟节点），避免两只手抢同一根丝杆。

本模块同时提供 :func:`brush_work_point`：刷头工作端位置的统一算法。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import DetectorConfig, PipelineConfig
from .types import BBox


@dataclass(frozen=True)
class HandScrewLink:
    """一只手与一根丝杆的关联结果。"""

    identity: str
    hand_track_id: int
    distance: float = 1.0
    overlap: float = 0.0


@dataclass(frozen=True)
class AssociationResult:
    links: tuple[HandScrewLink, ...]

    def by_identity(self, identity: str) -> HandScrewLink | None:
        for link in self.links:
            if link.identity == identity:
                return link
        return None

    def by_hand(self, hand_track_id: int | None) -> HandScrewLink | None:
        if hand_track_id is None:
            return None
        for link in self.links:
            if link.hand_track_id == int(hand_track_id):
                return link
        return None

    def identity_to_hand(self) -> dict[str, int]:
        return {link.identity: link.hand_track_id for link in self.links}


@dataclass(frozen=True)
class ScrewCandidate:
    """关联计算时的一根丝杆。"""

    identity: str
    box: BBox


class HandScrewAssociator:
    """基于距离 + 交互关系的手-丝杆一对一关联。

    关联是逐帧计算的：现场只需要知道“这一刻手是否拿着丝杆”，
    跨帧身份由业务层（作业周期）维护，这里不做锁与交接。
    """

    def __init__(self, *, cfg: PipelineConfig) -> None:
        self._cfg = cfg

    def assign(
        self,
        *,
        hands: tuple[object, ...],
        screws: tuple[ScrewCandidate, ...],
        frame_size: tuple[int, int],
    ) -> AssociationResult:
        """计算本帧的手-丝杆分配。

        ``hands`` 只需提供 ``track_id`` / ``box`` / ``matched`` 属性
        （:class:`app.oilcheck.tracker.Track` 满足）。
        """

        width, height = frame_size
        diag = max(1.0, float(np.hypot(width, height)))
        live_hands = [h for h in hands if getattr(h, "matched", False)]
        if not live_hands or not screws:
            return AssociationResult(links=())

        matched = self._global_assign(live_hands, list(screws), diag)
        return AssociationResult(
            links=tuple(
                HandScrewLink(
                    identity=screw.identity,
                    hand_track_id=int(getattr(hand, "track_id", 0)),
                    distance=distance,
                    overlap=overlap,
                )
                for hand, screw, distance, overlap in matched
            )
        )

    # ------------------------------------------------------------------
    def _global_assign(
        self,
        hands: list[object],
        screws: list[ScrewCandidate],
        diag: float,
    ) -> list[tuple[object, ScrewCandidate, float, float]]:
        rows, cols = len(hands), len(screws)
        unmatched = 1.0
        cost = np.full((rows, cols), unmatched, dtype=np.float64)
        distance = np.ones((rows, cols), dtype=np.float64)
        overlap = np.zeros((rows, cols), dtype=np.float64)

        for ri, hand in enumerate(hands):
            hand_box = hand.box
            for ci, screw in enumerate(screws):
                dist = (
                    float(
                        np.hypot(
                            hand_box.center[0] - screw.box.center[0],
                            hand_box.center[1] - screw.box.center[1],
                        )
                    )
                    / diag
                )
                iou = hand_box.iou(screw.box)
                contains = _contains(hand_box, screw.box.center)
                interacts = (
                    dist <= self._cfg.pick_hand_distance
                    or iou > 0.0
                    or contains
                )
                if not (interacts and dist <= self._cfg.hand_screw_max_distance):
                    continue
                distance[ri, ci] = dist
                overlap[ri, ci] = iou if not contains else max(iou, 1.0)
                # 交叠越大、距离越近代价越低；距离权重占主导。
                cost[ri, ci] = min(unmatched, max(0.0, dist - 0.02 * min(iou, 0.5)))

        size = rows + cols
        square = np.full((size, size), unmatched, dtype=np.float64)
        square[:rows, :cols] = cost
        square[rows:, cols:] = 0.0
        square[:rows, cols:] = unmatched
        square[rows:, :cols] = unmatched

        try:
            from scipy.optimize import linear_sum_assignment

            row_ind, col_ind = linear_sum_assignment(square)
        except Exception:  # noqa: BLE001
            row_ind, col_ind = _greedy(square)

        matched: list[tuple[object, ScrewCandidate, float, float]] = []
        for row, col in zip(row_ind, col_ind, strict=False):
            if row >= rows or col >= cols:
                continue
            if square[row, col] >= unmatched - 1e-9:
                continue
            matched.append(
                (
                    hands[row],
                    screws[col],
                    float(distance[row, col]),
                    float(overlap[row, col]),
                )
            )
        return matched


def brush_work_point(brush: object, detector: DetectorConfig) -> tuple[float, float]:
    """刷头工作端：优先关键点，否则按标定偏移从检测框推算。"""

    point = getattr(brush, "point", None)
    if point is not None:
        return point
    box = brush.box
    cx, cy = box.center
    return (
        cx + (detector.brush_point_offset_x - 0.5) * box.width,
        cy + (detector.brush_point_offset_y - 0.5) * box.height,
    )


def _contains(box: BBox, point: tuple[float, float]) -> bool:
    return box.x1 <= point[0] <= box.x2 and box.y1 <= point[1] <= box.y2


def _greedy(square: np.ndarray):
    order = np.dstack(np.unravel_index(np.argsort(square, axis=None), square.shape))[0]
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    row_ind: list[int] = []
    col_ind: list[int] = []
    for row, col in order:
        row, col = int(row), int(col)
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        row_ind.append(row)
        col_ind.append(col)
    return np.asarray(row_ind), np.asarray(col_ind)
