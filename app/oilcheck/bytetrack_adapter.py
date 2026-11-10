"""ByteTrack 适配器：用手部/刷头的多目标跟踪。

ultralytics 自带的 ``BYTETracker`` 只按 IoU 关联，实测 20×20 的小目标每帧移动
120px 时第 2 帧起就丢 ID，无法用于快速移动的手和刷头。这里子类化它并改写
``get_dists``：

1. 类别门控：类别不一致时判为不可匹配；
2. 距离门控：归一化中心距离超过上限时判为不可匹配；
3. 加权代价：``α·(1-IoU) + (1-α)·(中心距离/上限)``。

不用 ``min(iou, distance)``——那会让一个指标的优秀表现完全抵消另一个指标的异常。

两点已核实：
- ``get_dists`` 在卡尔曼 ``multi_predict`` 之后调用，``tlwh`` 已是预测状态，
  中心距离天然基于预测位置；
- 第二阶段的低置信度关联在 ultralytics 源码里是硬编码的纯 IoU，不经过
  ``get_dists``，所以改写它不会破坏 ByteTrack 原始的两阶段策略。

对外保持与 :class:`app.oilcheck.tracker.ObjectTracker` 相同的调用契约，
pipeline 与测试可以原样使用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .tracker import ObjectTracker, Track, TrackerOutput
from .types import BBox, Detection

logger = logging.getLogger("oilcheck.bytetrack")

_LOGGED_FORMAT_WARNING = False


@dataclass(frozen=True)
class CostDebug:
    """一次匹配的代价明细，供现场视频回放标定参数。"""

    ts: float
    iou_cost: list[list[float]]
    center_cost: list[list[float]]
    final_cost: list[list[float]]
    gated: list[list[bool]]


class _DetArray:
    """BYTETracker 需要的 results-like 数组：``conf``/``cls``/``xywh`` + 布尔索引。"""

    __slots__ = ("xywh", "conf", "classes")

    def __init__(
        self, xywh: np.ndarray, conf: np.ndarray, classes: np.ndarray
    ) -> None:
        self.xywh = xywh
        self.conf = conf
        self.classes = classes

    # ultralytics 按 ``.cls`` 读取类别列，这里用 property 暴露。
    @property
    def cls(self) -> np.ndarray:
        return self.classes

    @classmethod
    def from_detections(cls, detections: tuple[Detection, ...]) -> _DetArray:
        count = len(detections)
        xywh = np.zeros((count, 4), dtype=np.float32)
        conf = np.zeros((count,), dtype=np.float32)
        classes = np.zeros((count,), dtype=np.float32)
        for index, det in enumerate(detections):
            box = det.box
            xywh[index] = (box.center[0], box.center[1], box.width, box.height)
            conf[index] = float(det.confidence)
        return cls(xywh=xywh, conf=conf, classes=classes)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, mask: Any) -> _DetArray:
        return _DetArray(self.xywh[mask], self.conf[mask], self.classes[mask])


def _build_impl(
    *,
    frame_diag: float,
    max_center_distance: float,
    alpha: float,
    args: Any,
) -> Any:
    """构造改写过的 BYTETracker；ultralytics 不可用时返回 None。"""

    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.utils import matching
    except Exception as exc:  # noqa: BLE001 - 任何导入失败都回退
        logger.warning("ByteTrack 不可用，回退 ObjectTracker：%s", exc)
        return None

    class _CostByteTracker(BYTETracker):  # type: ignore[misc, valid-type]
        """带类别/距离门控与加权代价的 ByteTrack。"""

        def __init__(self, tracker_args: Any) -> None:
            super().__init__(tracker_args)
            self._diag = max(1.0, float(frame_diag))
            self._max_center = max(1e-3, float(max_center_distance))
            self._alpha = min(0.99, max(0.01, float(alpha)))
            self.last_debug: CostDebug | None = None

        @staticmethod
        def _center(strack: Any) -> tuple[float, float]:
            # STrack.tlwh 已经是卡尔曼预测后的状态。
            x, y, w, h = strack.tlwh
            return (float(x) + float(w) * 0.5, float(y) + float(h) * 0.5)

        def get_dists(self, tracks: list[Any], detections: list[Any]) -> np.ndarray:
            iou_cost = matching.iou_distance(tracks, detections)
            if not tracks or not detections:
                return iou_cost

            track_centers = np.array(
                [self._center(t) for t in tracks], dtype=np.float32
            )
            det_centers = np.array(
                [self._center(d) for d in detections], dtype=np.float32
            )
            distance = (
                np.linalg.norm(
                    track_centers[:, None, :] - det_centers[None, :, :], axis=2
                )
                / self._diag
            )
            center_cost = distance / self._max_center
            gated = distance > self._max_center

            # 类别门控：同一实例内类别不一致时不匹配。
            for ti, track in enumerate(tracks):
                for di, det in enumerate(detections):
                    if getattr(track, "cls", None) != getattr(det, "cls", None):
                        gated[ti, di] = True

            cost = self._alpha * iou_cost + (1.0 - self._alpha) * center_cost
            cost = np.where(gated, 1.0, cost)
            if self.args.fuse_score:
                # 默认关闭；若显式开启，按 ultralytics 语义再融合一次检测分数。
                cost = matching.fuse_score(cost, detections)
            return cost

    from ultralytics.utils import IterableSimpleNamespace

    return _CostByteTracker(IterableSimpleNamespace(**args))


class ByteTrackObjectTracker:
    """以 ByteTrack 为后端的多目标跟踪，保持 ObjectTracker 的调用契约。

    契约要点（pipeline 与测试依赖）：

    - ``update(ts=, detections=, frame_size=) -> TrackerOutput(active, lost)``；
    - 新轨迹首帧 ``hits=1`` 且立即出现在 ``active`` 里；
    - 未匹配轨迹保留到 ``max_missing_s``，期间仍在 ``active``、``matched=False``；
    - ``get(track_id)`` 即使未匹配也返回轨迹，由调用方检查 ``matched``；
    - ``min_hits`` 只作属性暴露，由调用方过滤。

    ByteTrack 本身按帧计数、输出数组，这里做三件事：把输出行映射回
    :class:`Track`（保留 angle/point 等检测字段）、用时间戳维护漏检时长、
    给 track id 加 ``id_base`` 命名空间（手与刷头各用一个实例）。
    """

    def __init__(
        self,
        *,
        id_base: int = 0,
        max_distance: float = 0.15,
        max_missing_s: float = 0.8,
        min_hits: int = 1,
        max_active: int = 16,
        cost_alpha: float = 0.40,
        match_thresh: float = 0.80,
        high_thresh: float = 0.30,
        low_thresh: float = 0.10,
        new_thresh: float = 0.30,
        buffer_frames: int = 30,
        fuse_score: bool = False,
        backend: str = "auto",
    ) -> None:
        self._id_base = max(0, int(id_base))
        self._min_hits = max(1, int(min_hits))
        self._max_missing_s = max(0.0, float(max_missing_s))
        self._max_active = max(1, int(max_active))
        self._max_distance = max(0.005, float(max_distance))
        self._tracks: dict[int, Track] = {}
        self._lost: list[Track] = []

        self._fallback: ObjectTracker | None = None
        self._impl: Any = None
        self._backend = "object"

        if backend != "object":
            # 帧对角线在 update() 里才知道；用 1.0 建实例，首次 update 校准。
            self._impl = _build_impl(
                frame_diag=1.0,
                max_center_distance=self._max_distance,
                alpha=cost_alpha,
                args={
                    "tracker_type": "bytetrack",
                    "track_high_thresh": float(high_thresh),
                    "track_low_thresh": float(low_thresh),
                    "new_track_thresh": float(new_thresh),
                    "track_buffer": max(1, int(buffer_frames)),
                    "match_thresh": max(0.01, float(match_thresh)),
                    "fuse_score": bool(fuse_score),
                },
            )
        if self._impl is None:
            self._fallback = ObjectTracker(
                max_distance=self._max_distance,
                max_missing_s=self._max_missing_s,
                min_hits=self._min_hits,
                max_active=self._max_active,
                id_base=self._id_base,
            )
        else:
            self._backend = "bytetrack"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def min_hits(self) -> int:
        return self._min_hits

    @property
    def max_distance(self) -> float:
        return self._max_distance

    def debug_cost(self) -> CostDebug | None:
        """最近一次匹配的代价明细（仅 ByteTrack 后端）。"""

        return getattr(self._impl, "last_debug", None)

    def reset(self) -> None:
        self._tracks.clear()
        self._lost.clear()
        if self._impl is not None:
            self._impl.reset()
        if self._fallback is not None:
            self._fallback.reset()

    def get(self, track_id: int | None) -> Track | None:
        if track_id is None:
            return None
        return self._tracks.get(int(track_id))

    def active_tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks.values())

    def update(
        self,
        *,
        ts: float,
        detections: tuple[Detection, ...],
        frame_size: tuple[int, int],
        distance_scale: float = 1.0,
    ) -> TrackerOutput:
        """``distance_scale`` 按帧间隔放开门限（低帧率时单帧位移更大）。"""

        scale = max(1e-3, float(distance_scale))
        if self._fallback is not None:
            output = self._fallback.update(
                ts=ts,
                detections=detections,
                frame_size=frame_size,
                distance_scale=scale,
            )
            # 回退后端已自带 id 命名空间，这里只同步注册表供 get() 使用。
            self._tracks = {track.track_id: track for track in output.active}
            return output

        width, height = frame_size
        if width <= 0 or height <= 0:
            return TrackerOutput(active=tuple(self._tracks.values()), lost=())

        diag = max(1.0, float(np.hypot(width, height)))
        self._impl._diag = diag
        # 门限按帧间隔放大；匹配代价阈值同步放宽，否则门限放开了也进不了关联。
        self._impl._max_center = self._max_distance * scale

        rows = self._impl.update(_DetArray.from_detections(detections))
        seen: set[int] = set()
        for row in np.atleast_2d(rows) if len(rows) else ():
            parsed = self._parse_row(row, detections)
            if parsed is None:
                continue
            track_id, det = parsed
            seen.add(track_id)
            existing = self._tracks.get(track_id)
            if existing is None:
                if len(self._tracks) >= self._max_active:
                    continue
                self._tracks[track_id] = Track(
                    track_id=track_id,
                    box=det.box,
                    last_ts=float(ts),
                    last_seen_ts=float(ts),
                    confidence=det.confidence,
                    angle=det.angle,
                    point=det.point,
                )
            else:
                self._merge(existing, det, ts)

        lost: list[Track] = []
        for track_id, track in list(self._tracks.items()):
            if track_id in seen:
                continue
            track.matched = False
            track.missing_s = max(0.0, float(ts) - track.last_seen_ts)
            if track.missing_s > self._max_missing_s:
                lost.append(track)
                self._tracks.pop(track_id, None)
        self._lost = lost

        return TrackerOutput(
            active=tuple(self._tracks.values()), lost=tuple(lost)
        )

    # ------------------------------------------------------------------
    def _parse_row(
        self, row: Any, detections: tuple[Detection, ...]
    ) -> tuple[int, Detection] | None:
        global _LOGGED_FORMAT_WARNING
        width = int(np.asarray(row).shape[0])
        values = np.asarray(row, dtype=np.float64).ravel()
        if width == 8:
            x1, y1, x2, y2, track_id, score, _cls, index = values
            box = BBox(x1, y1, x2, y2)
        elif width == 9:
            # OBB 行：x, y, w, h, angle, track_id, ...
            x, y, w, h, _angle, track_id, _score, _cls, index = values
            box = BBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        else:
            if not _LOGGED_FORMAT_WARNING:
                _LOGGED_FORMAT_WARNING = True
                logger.error(
                    "ByteTrack 输出行宽不支持：%s（预期 8 或 9），已跳过", width
                )
            return None

        index_int = int(index)
        if 0 <= index_int < len(detections):
            det = detections[index_int]
            # 用映射回的原始检测，保留 angle/point/confidence。
            det = Detection(
                label=det.label,
                confidence=det.confidence,
                box=box,
                track_id=int(track_id),
                angle=det.angle,
                point=det.point,
            )
        else:
            det = Detection(
                label="", confidence=float(_score_of(values, width)), box=box
            )
        return int(track_id) + self._id_base, det

    @staticmethod
    def _merge(track: Track, det: Detection, ts: float) -> None:
        old_cx, old_cy = track.box.center
        new_cx, new_cy = det.box.center
        dt = max(1e-3, float(ts) - track.last_ts)
        smoothing = 0.5
        inst = ((new_cx - old_cx) / dt, (new_cy - old_cy) / dt)
        track.velocity = (
            smoothing * track.velocity[0] + (1 - smoothing) * inst[0],
            smoothing * track.velocity[1] + (1 - smoothing) * inst[1],
        )
        track.box = det.box
        track.confidence = det.confidence
        track.angle = det.angle
        track.point = det.point
        track.last_ts = float(ts)
        track.last_seen_ts = float(ts)
        track.hits += 1
        track.missing_s = 0.0
        track.matched = True


def _score_of(values: np.ndarray, width: int) -> float:
    return float(values[5]) if width == 8 else float(values[6])
