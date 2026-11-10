"""ByteTrack 适配器：用手部/刷头的多目标跟踪。

ultralytics 自带的 ``BYTETracker`` 只按 IoU 关联，实测 20×20 的小目标每帧移动
120px 时第 2 帧起就丢 ID，无法用于快速移动的手和刷头。这里子类化它并改写
``get_dists``：

1. 类别门控：类别不一致时判为不可匹配；
2. 距离门控：归一化中心距离超过上限时判为不可匹配；
3. 加权代价：``α·(1-IoU) + (1-α)·(中心距离/上限)``。

不用 ``min(iou, distance)``——那会让一个指标的优秀表现完全抵消另一个指标的异常。

已核实与改写：
- ``get_dists`` 在卡尔曼 ``multi_predict`` 之后调用，``tlwh`` 已是预测状态，
  中心距离天然基于预测位置；
- **第二阶段**（低置信度补救）在 ultralytics 源码里是硬编码的纯 IoU，
  小目标快速移动时必然失配——这正是上面记录的原始故障。这里覆写
  ``_second_association``，让它复用同一套加权代价；
- 卡尔曼改用 ``KalmanFilterXYWH``：手/刷框近似方形且尺寸抖动大，XYAH 的
  长宽比状态会把框宽拉回（实测量测 5.0 被滤到 1.04）；
- 门控未命中的代价固定为 ``GATE_COST=1.5``，使 ``match_thresh=1.0`` 表示
  「只看距离/IoU 门限」，避免 α 把有效距离暗中截到 0.667 倍。

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

# 门控未命中的代价。必须高于任何可配置的 match_thresh（默认 1.0），
# 否则「距离门限」会被代价阈值悄悄截断。
GATE_COST = 1.5


def _ensure_lap_available() -> None:
    """``lap`` 缺失时注入等价实现，让 ultralytics 的 trackers 能正常导入。

    ultralytics 的 ``trackers.utils.matching`` 在**模块导入时**就
    ``import lap``，缺了会触发 ``check_requirements`` 联网安装——服务器
    网络受限时这一段会打印整屏重试日志并失败，ByteTrack 随之不可用，
    只能回退到自研 ``ObjectTracker``（快速小目标的 ID 保持明显更差）。

    纯 Python 分派只需一个 ``lapjv(cost, extend_cost, cost_limit)``。这里用
    ultralytics 自带的 ``linear_sum_assignment``（scipy 优先，否则纯 numpy）
    实现同样的语义：最小代价一一匹配，代价超过 ``cost_limit`` 的视为未匹配。

    只有在真实 ``lap`` 不可用时才注入，装了 ``lap`` 的机器行为完全不变。
    """

    import sys
    import types

    if "lap" in sys.modules:
        return
    try:
        import lap  # noqa: F401  # 已安装：什么都不做

        return
    except ImportError:
        pass

    lap_mod = types.ModuleType("lap")
    lap_mod.__version__ = "0.0.0+oilcheck-fallback"

    def lapjv(
        cost: Any, extend_cost: bool = False, cost_limit: float = float("inf")
    ) -> tuple[float, np.ndarray, np.ndarray]:
        from ultralytics.utils.ops import linear_sum_assignment

        matrix = np.asarray(cost, dtype=np.float64)
        rows, cols = matrix.shape
        # lapjv 要求方阵；用大代价补齐，避免把补齐的虚拟行列选为有效匹配。
        big = (
            float(np.abs(matrix).max()) * 2.0 + 1.0
            if matrix.size
            else 1.0
        )
        size = max(rows, cols)
        padded = np.full((size, size), big, dtype=np.float64)
        padded[:rows, :cols] = matrix
        row_ind, col_ind = linear_sum_assignment(padded)

        x = np.full(rows, -1, dtype=int)
        y = np.full(cols, -1, dtype=int)
        total = 0.0
        for r, c in zip(row_ind, col_ind, strict=False):
            if r >= rows or c >= cols:
                continue  # 补齐的虚拟节点
            cost_value = matrix[r, c]
            if not np.isfinite(cost_value) or cost_value > cost_limit:
                continue  # 超过门限：留作未匹配
            x[r] = c
            y[c] = r
            total += float(cost_value)
        return total, x, y

    lap_mod.lapjv = lapjv  # type: ignore[attr-defined]
    sys.modules["lap"] = lap_mod
    logger.info("未安装 lap，已注入 scipy/numpy 版分配实现供 ByteTrack 使用")

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

    # 必须在导入 trackers 之前补上分配实现：matching 在模块导入时就 import lap。
    _ensure_lap_available()
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.trackers.utils import matching
    except Exception as exc:  # noqa: BLE001 - 任何导入失败都回退
        logger.warning("ByteTrack 不可用，回退 ObjectTracker：%s", exc)
        return None

    from ultralytics.trackers.byte_tracker import STrack
    from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYWH

    class _XYWHSTrack(STrack):  # type: ignore[misc, valid-type]
        """状态用 (cx, cy, w, h)：手/刷框近似方形，长宽比不必单独建模。"""

        def convert_coords(self, tlwh: np.ndarray) -> np.ndarray:
            ret = np.asarray(tlwh).copy()
            ret[:2] += ret[2:] / 2
            return ret

        @property
        def tlwh(self) -> np.ndarray:
            if self.mean is None:
                return self._tlwh.copy()
            ret = self.mean[:4].copy()
            ret[:2] -= ret[2:] / 2
            return ret

    class _CostByteTracker(BYTETracker):  # type: ignore[misc, valid-type]
        """带类别/距离门控与加权代价的 ByteTrack。"""

        track_class = _XYWHSTrack

        def __init__(self, tracker_args: Any) -> None:
            super().__init__(tracker_args)
            self._diag = max(1.0, float(frame_diag))
            self._max_center = max(1e-3, float(max_center_distance))
            self._alpha = min(0.99, max(0.01, float(alpha)))
            self.last_debug: CostDebug | None = None

        def get_kalmanfilter(self) -> Any:
            return KalmanFilterXYWH()

        @staticmethod
        def _center(strack: Any) -> tuple[float, float]:
            # STrack.tlwh 已经是卡尔曼预测后的状态。
            x, y, w, h = strack.tlwh
            return (float(x) + float(w) * 0.5, float(y) + float(h) * 0.5)

        def get_dists(self, tracks: list[Any], detections: list[Any]) -> np.ndarray:
            if not tracks or not detections:
                return matching.iou_distance(tracks, detections)
            return self._cost_matrix(tracks, detections)

        def _cost_matrix(
            self, tracks: list[Any], detections: list[Any]
        ) -> np.ndarray:
            """类别/距离门控 + 加权代价；门控未命中给 ``GATE_COST``。"""

            iou_cost = matching.iou_distance(tracks, detections)
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

            # 类别门控：同一实例内类别不一致时不匹配（向量化，避免 Python 双循环）。
            track_cls = np.array(
                [float(getattr(t, "cls", 0.0)) for t in tracks], dtype=np.float32
            )
            det_cls = np.array(
                [float(getattr(d, "cls", 0.0)) for d in detections],
                dtype=np.float32,
            )
            gated |= track_cls[:, None] != det_cls[None, :]

            cost = self._alpha * iou_cost + (1.0 - self._alpha) * center_cost
            cost = np.where(gated, GATE_COST, cost)
            if self.args.fuse_score:
                # 默认关闭；若显式开启，按 ultralytics 语义再融合一次检测分数。
                cost = matching.fuse_score(cost, detections)
            return cost

        def _second_association(
            self,
            strack_pool: list[Any],
            u_track: list[int],
            detections_second: list[Any],
            activated: list[Any],
            refind: list[Any],
            lost: list[Any],
        ) -> None:
            """第二阶段：用同一套加权代价接回低置信度检测。

            ultralytics 原生这里是硬编码纯 IoU + 阈值 0.5，小目标每帧移动
            上百像素时必然接不上，低分框就白喂进来了。这里沿用第一阶段的
            距离门控与加权代价，代价阈值仍用 ``match_thresh``。
            """

            from ultralytics.trackers.basetrack import TrackState

            r_tracked = [
                strack_pool[i]
                for i in u_track
                if strack_pool[i].state == TrackState.Tracked
            ]
            if r_tracked and detections_second:
                dists = self._cost_matrix(r_tracked, detections_second)
                matches, u_track, _ = matching.linear_assignment(
                    dists, thresh=self.args.match_thresh
                )
                self._apply_matches(
                    matches, r_tracked, detections_second, activated, refind
                )
            else:
                u_track = list(range(len(r_tracked)))

            for it in u_track:
                track = r_tracked[it]
                track.mark_lost()
                lost.append(track)

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
        # 已宣布丢失的轨迹 id：ByteTrack 内部的 lost_stracks 仍会保留一段时间
        # （按帧数计），不加这道闸门的话同一个 id 会在几百帧后「复活」，
        # 业务层会把「同名不同目标」当成同一只手/同一根杆。
        self._retired: set[int] = set()

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
        self._retired.clear()
        self._local_seq = 0
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
            track_id, det, raw_box = parsed
            if track_id in self._retired:
                # ByteTrack 让已退役的 id 重新出现了：换一个新 id，
                # 保证「宣布丢失 = 永不复活」。
                track_id = self._fresh_id()
            seen.add(track_id)
            existing = self._tracks.get(track_id)
            if existing is None:
                if not self._make_room_for_new_track():
                    continue
                self._tracks[track_id] = Track(
                    track_id=track_id,
                    box=det.box,
                    last_ts=float(ts),
                    last_seen_ts=float(ts),
                    confidence=det.confidence,
                    angle=det.angle,
                    point=det.point,
                    raw_box=raw_box if raw_box is not None else det.box,
                )
            else:
                self._merge(existing, det, ts, raw_box)

        lost: list[Track] = []
        for track_id, track in list(self._tracks.items()):
            if track_id in seen:
                continue
            track.matched = False
            track.missing_s = max(0.0, float(ts) - track.last_seen_ts)
            if track.missing_s > self._max_missing_s:
                lost.append(track)
                self._tracks.pop(track_id, None)
                self._retire(track_id)
        self._lost = lost

        return TrackerOutput(
            active=tuple(self._tracks.values()), lost=tuple(lost)
        )

    # ------------------------------------------------------------------
    def _fresh_id(self) -> int:
        """分配一个适配器自己管理的 id（用于已退役 id 的复活情形）。"""

        self._local_seq = getattr(self, "_local_seq", 0) + 1
        return self._id_base + 10_000_000 + self._local_seq

    def _retire(self, track_id: int) -> None:
        """把 id 从 ByteTrack 内部池里永久移除，并记入退役集合。"""

        self._retired.add(int(track_id))
        local_id = int(track_id) - self._id_base
        impl = self._impl
        if impl is None:
            return
        for pool_name in ("tracked_stracks", "lost_stracks", "removed_stracks"):
            pool = getattr(impl, pool_name, None)
            if not pool:
                continue
            setattr(
                impl,
                pool_name,
                [
                    item
                    for item in pool
                    if int(getattr(item, "track_id", -1)) != local_id
                ],
            )

    def _make_room_for_new_track(self) -> bool:
        """名额满时先淘汰最久未匹配的轨迹，再考虑拒绝新目标。"""

        if len(self._tracks) < self._max_active:
            return True
        if not self._tracks:
            return False
        victim = max(
            self._tracks.values(),
            key=lambda item: (not item.matched, item.missing_s),
        )
        self._tracks.pop(victim.track_id, None)
        self._retire(victim.track_id)
        return True

    def _parse_row(
        self, row: Any, detections: tuple[Detection, ...]
    ) -> tuple[int, Detection, BBox | None] | None:
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
            raw = detections[index_int]
            # ``box`` 是卡尔曼状态（显示用），``raw_box`` 是本帧真实检测
            # （业务判定用，快速移动时不滞后）。
            det = Detection(
                label=raw.label,
                confidence=raw.confidence,
                box=box,
                track_id=int(track_id),
                angle=raw.angle,
                point=raw.point,
            )
            return int(track_id) + self._id_base, det, raw.box
        else:
            det = Detection(
                label="", confidence=float(_score_of(values, width)), box=box
            )
            return int(track_id) + self._id_base, det, None

    @staticmethod
    def _merge(
        track: Track, det: Detection, ts: float, raw_box: BBox | None = None
    ) -> None:
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
        track.raw_box = raw_box if raw_box is not None else det.box
        track.last_ts = float(ts)
        track.last_seen_ts = float(ts)
        track.hits += 1
        track.missing_s = 0.0
        track.matched = True


def _score_of(values: np.ndarray, width: int) -> float:
    return float(values[5]) if width == 8 else float(values[6])
