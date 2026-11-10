"""取料与人工二次涂油事件判定（简化版）。

只做两条判定：

1. **取料完成**：跟踪到一只手在**未持有丝杆**的状态下进入涂油盘（``oil_station``
   检测框），随后同一只手**带着丝杆离开**涂油盘；
2. **人工二次涂油完成**：取料之后手持丝杆，刷头工作端持续在丝杆接触邻域内累计
   达到 ``brush_hold_s``。

两条判定各自发一条事件（``PICK`` / ``MANUAL_OILING``），一个周期结束后立即复位，
等待下一次取料。不做 OK/NG 分级、不做槽位跟踪、不做放料判定。

区域：

- **涂油盘区**：手工标定的 ``tray_roi``（覆盖盘体下方的手臂活动范围）与模型检出
  的 ``oil_station`` 框（平滑跟踪、短时漏检保持）的并集，随盘体移动；
- **工作追踪区**：``work_roi``，跟踪与判定的有效范围；
- **涂油区**：``oiling_roi``，标定与显示用，预留给后续涂油范围约束。

本模块不依赖 Qt、相机 SDK 或具体推理框架，测试直接回放 :class:`FrameDetections`。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

from .association import brush_work_point
from .bytetrack_adapter import ByteTrackObjectTracker
from .config import PipelineConfig, StationConfig
from .tracker import ObjectTracker, Track, TrayTrack, TrayTracker
from .types import (
    BBox,
    Detection,
    EventRecord,
    EventResult,
    EventType,
    FrameDetections,
    StationSnapshot,
    TaskResult,
    TaskSnapshot,
    TaskStage,
    TrackSnapshot,
    TraySnapshot,
)

EventCallback = Any  # Callable[[EventRecord], None]，避免循环导入

# 手/刷头的 track id 命名空间，方便日志与 UI 区分。
_HAND_ID_BASE = 100_000
_BRUSH_ID_BASE = 200_000


@dataclass
class _HandVisit:
    """一只手的「进盘 — 离盘」过程。"""

    hand_track_id: int
    entered_tray_ts: float
    # 进盘**那一刻上一帧**手里是否已经拿着丝杆。取料只认“空手进、带料出”。
    #
    # 不能在进盘当帧判：手一伸进盘，手框就与盘内静置的丝杆交叠，
    # 当帧判定永远是“持有”，取料会被全部吞掉（现场视频 140911 即如此）。
    # 进盘前手在盘外，能自由拿着的只可能是从盘外带来的丝杆。
    holding_on_entry: bool
    inside_tray: bool = True
    last_seen_ts: float = 0.0
    # 进盘前一帧（手在盘外）的持有状态；初始沿用进盘帧之前观察到的手部状态。
    holding_before_entry: bool = False


@dataclass
class _Cycle:
    """一个作业周期：取料 → 涂油 → 复位。"""

    task_id: str
    created_at: float
    stage: TaskStage
    stage_since: float
    final_result: TaskResult
    hand_track_id: int | None = None
    screw_track_id: int | None = None
    brush_track_id: int | None = None
    screw_observed: bool = False
    brush_hold_s: float = 0.0
    confidence: float | None = None
    reason_code: str | None = None
    evidence: list[str] = field(default_factory=list)
    picked_at: float | None = None
    completed_at: float | None = None
    last_seen_ts: float | None = None
    # 最近一次观测到锚点的时刻（丝杆锚点跟踪用）。
    last_screw_ts: float | None = None
    # 丝杆锚点：取料时锁定那根杆，之后按最近邻连续性跟它，不再依赖手 ID。
    screw_anchor: tuple[float, float] | None = None
    screw_anchor_ts: float | None = None
    # 最近一次「刷子靠近」的时刻，用于跨过一两帧漏检把空档补回来。
    last_brush_ts: float | None = None
    timeline: list[tuple[str, float]] = field(default_factory=list)
    events_sent: set[str] = field(default_factory=set)

    @property
    def active(self) -> bool:
        return self.final_result == TaskResult.PROCESSING


class StationPipeline:
    """单相机/单工位的取料与人工涂油判定。"""

    def __init__(
        self,
        *,
        station: StationConfig,
        pipeline_cfg: PipelineConfig,
        on_event: EventCallback | None = None,
        cycle_id: str | None = None,
        detector_name: str = "null",
        logger: logging.Logger | None = None,
    ) -> None:
        self.station = station
        self.cfg = pipeline_cfg
        self._on_event = on_event
        self.cycle_id = cycle_id
        self.detector_name = detector_name
        self._logger = logger or logging.getLogger(
            f"oilcheck.pipeline.{station.station_id}"
        )

        self._hand_tracker = ByteTrackObjectTracker(
            id_base=_HAND_ID_BASE,
            max_distance=pipeline_cfg.hand_track_max_distance,
            max_missing_s=pipeline_cfg.hand_track_max_missing_s,
            min_hits=pipeline_cfg.hand_track_min_hits,
            # 漏检轨迹在 max_missing_s 内仍占名额；名额太小会让新手检测框
            # 建不了轨迹，而“手”是全画面最多的目标之一（另有盘边手臂等）。
            max_active=pipeline_cfg.hand_track_max_active,
            cost_alpha=pipeline_cfg.bytetrack_cost_alpha,
            match_thresh=pipeline_cfg.bytetrack_match_thresh,
            high_thresh=pipeline_cfg.bytetrack_high_thresh,
            low_thresh=pipeline_cfg.bytetrack_low_thresh,
            new_thresh=pipeline_cfg.bytetrack_new_thresh,
            buffer_frames=max(
                1, int(round(pipeline_cfg.max_fps * pipeline_cfg.bytetrack_buffer_s))
            ),
            backend=pipeline_cfg.hand_tracker_backend,
        )
        self._brush_tracker = ByteTrackObjectTracker(
            id_base=_BRUSH_ID_BASE,
            max_distance=pipeline_cfg.brush_track_max_distance,
            max_missing_s=pipeline_cfg.brush_track_max_missing_s,
            min_hits=pipeline_cfg.brush_track_min_hits,
            max_active=pipeline_cfg.brush_track_max_active,
            cost_alpha=pipeline_cfg.bytetrack_cost_alpha,
            match_thresh=pipeline_cfg.bytetrack_match_thresh,
            high_thresh=pipeline_cfg.bytetrack_high_thresh,
            low_thresh=pipeline_cfg.bytetrack_low_thresh,
            new_thresh=pipeline_cfg.bytetrack_new_thresh,
            buffer_frames=max(
                1, int(round(pipeline_cfg.max_fps * pipeline_cfg.bytetrack_buffer_s))
            ),
            backend=pipeline_cfg.brush_tracker_backend,
        )
        self._tray_tracker = TrayTracker(
            max_distance=pipeline_cfg.tray_max_distance,
            max_missing_s=pipeline_cfg.tray_max_missing_s,
            min_confidence=pipeline_cfg.tray_min_confidence,
        )
        self._tray = TrayTrack()

        # 每只手的进盘过程；一只手下同一时间只会有一段。
        self._visits: dict[int, _HandVisit] = {}
        # 手在盘外时手里是否拿着丝杆（进盘判定的参照）。
        self._outside_holding: dict[int, bool] = {}
        self._cycles: dict[str, _Cycle] = {}
        self._task_seq = 0

        self._last_frame_ts: float | None = None
        self._frame_size: tuple[int, int] = (0, 0)
        self._last_detections: tuple[Detection, ...] = ()
        self._hand_tracks: tuple[Track, ...] = ()
        self._brush_tracks: tuple[Track, ...] = ()
        self._frames_processed = 0
        self._events_emitted = 0
        self._frame_gap_events = 0
        self._last_frame_gap_s = 0.0
        # 帧率自适应：实测帧间隔 / 配置基准帧间隔。低帧率下按这个倍率放大
        # 距离门限与容差窗口，否则同一动作跨的位移一大就断开。
        self._frame_scale = 1.0

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    @property
    def tasks(self) -> tuple[TaskSnapshot, ...]:
        return tuple(
            self._to_snapshot(task)
            for task in sorted(
                self._cycles.values(),
                key=lambda item: (item.created_at, item.task_id),
            )
        )

    @property
    def operations(self) -> tuple[TaskSnapshot, ...]:
        return self.tasks

    @property
    def materials(self) -> tuple[TaskSnapshot, ...]:
        return self.tasks

    def reset(self) -> None:
        self._visits.clear()
        self._outside_holding.clear()
        self._cycles.clear()
        self._task_seq = 0
        self._hand_tracker.reset()
        self._brush_tracker.reset()
        self._tray_tracker.reset()
        self._tray = TrayTrack()
        self._last_frame_ts = None
        self._frame_size = (0, 0)
        self._last_detections = ()
        self._hand_tracks = ()
        self._brush_tracks = ()
        self._frame_gap_events = 0
        self._last_frame_gap_s = 0.0
        self._frame_scale = 1.0

    def process(self, *, detections: FrameDetections) -> StationSnapshot:
        ts = float(detections.ts)
        width, height = detections.frame_size
        if width <= 0 or height <= 0:
            return self.snapshot()
        if self.station.detector.raw_mode:
            return self._process_raw(detections)

        dt = 0.0
        if self._last_frame_ts is not None:
            raw_gap = ts - self._last_frame_ts
            self._last_frame_gap_s = raw_gap
            # 视频时间戳跳变/长间隔会破坏时序证据，单独记录，不静默吞掉。
            if raw_gap < 0 or raw_gap > 1.0:
                self._frame_gap_events += 1
                self._logger.warning(
                    "视频时间戳异常：%.3fs（工位=%s），暂停依赖连续时间的判断",
                    raw_gap,
                    self.station.station_id,
                )
            dt = max(0.0, min(1.0, raw_gap))
            # 实际帧间隔 / 配置基准帧间隔。dt 已经被夹到 [0,1]，
            # 倍率再夹一次上限，避免门限放大到失去区分度。
            base_dt = 1.0 / max(1, self.cfg.max_fps)
            self._frame_scale = max(
                1.0, min(self.cfg.frame_scale_max, dt / base_dt)
            )
        self._last_frame_ts = ts
        self._frame_size = (int(width), int(height))
        self._last_detections = detections.detections
        self._frames_processed += 1

        cfg = self.station.detector
        frame_size = (int(width), int(height))

        # 盘体先更新：本帧的 oil_station 框直接参与区域过滤，避免目标因盘体
        # 上下移动在一个帧周期内被误判为“区外”。
        self._tray = self._tray_tracker.update(
            ts=ts,
            detections=detections.by_label(cfg.tray_label),
            frame_size=frame_size,
            distance_scale=self._frame_scale,
        )
        tray_box = self._tray_region_box(ts, frame_size)
        hands = self._valid_detections(
            detections.by_label(cfg.hand_label), frame_size, tray_box
        )
        brushes = self._valid_detections(
            detections.by_label(cfg.brush_label), frame_size, tray_box
        )
        # 区外目标默认不进跟踪器，但**已被跟踪的目标移到区外后要继续跟**：
        # 手一伸出去就断轨，取料/涂油判定会直接失去目标。
        hands = self._detections_for_tracking(
            label=cfg.hand_label,
            in_roi=hands,
            frame_size=frame_size,
            tray_box=tray_box,
            tracker=self._hand_tracker,
            ts=ts,
        )
        brushes = self._detections_for_tracking(
            label=cfg.brush_label,
            in_roi=brushes,
            frame_size=frame_size,
            tray_box=tray_box,
            tracker=self._brush_tracker,
            ts=ts,
        )

        # 距离门限按帧间隔放大：同一个动作在低帧率下单帧位移更大，
        # 不放大就会把「还是这只手/这个刷子」判成新目标。
        scale = self._frame_scale
        hand_out = self._hand_tracker.update(
            ts=ts, detections=hands, frame_size=frame_size, distance_scale=scale
        )
        brush_out = self._brush_tracker.update(
            ts=ts, detections=brushes, frame_size=frame_size, distance_scale=scale
        )
        self._hand_tracks = hand_out.active
        self._brush_tracks = brush_out.active

        self._update_pick(ts=ts, tray_box=tray_box)
        self._update_oiling(ts=ts, dt=dt)
        self._cleanup(ts)
        return self.snapshot()

    def _process_raw(self, detections: FrameDetections) -> StationSnapshot:
        """旁路模式：只交出这一帧的全部检测结果。

        不做 ROI 过滤、不建轨迹、不判取料/涂油、不产生事件。用于先确认模型
        对画面的标签覆盖情况。
        """

        self._last_frame_ts = float(detections.ts)
        width, height = detections.frame_size
        self._frame_size = (int(width), int(height))
        self._last_detections = detections.detections
        self._frames_processed += 1
        self._hand_tracks = ()
        self._brush_tracks = ()
        self._tray = TrayTrack()
        return self.snapshot()

    def snapshot(self) -> StationSnapshot:
        active = next(
            (task for task in self._cycles.values() if task.active),
            None,
        )
        # 画面只画工作追踪区/涂油盘区内的检测结果（与跟踪范围一致）。
        # 全量检测仍然保留在 _last_detections 里供判定使用——手持丝杆会走到
        # 工作追踪区边缘外，按区过滤会把正在判定的那根杆从画面上藏掉。
        return StationSnapshot(
            station_id=self.station.station_id,
            station_name=self.station.name,
            camera_id=self.station.camera_id,
            mode=self.detector_name,
            online=False,
            tasks=self.tasks,
            tray=self._tray_snapshot(),
            tracks=self._track_snapshots(),
            detections=self._display_detections(),
            frame_size=self._frame_size,
            updated_at=self._last_frame_ts or 0.0,
            stats={
                "frames_processed": self._frames_processed,
                "events_emitted": self._events_emitted,
                "tasks": len(self._cycles),
                "active_task_id": active.task_id if active else "",
                "tray_reliable": self._tray_region_reliable(),
                "frame_gap_events": self._frame_gap_events,
                "last_frame_gap_s": round(self._last_frame_gap_s, 3),
                "raw_mode": self.station.detector.raw_mode,
                "label_counts": _label_counts(self._last_detections),
                "hand_visits": len(self._visits),
            },
        )

    # ------------------------------------------------------------------
    # ROI 与检测过滤
    # ------------------------------------------------------------------
    def _display_detections(self) -> tuple[Detection, ...]:
        """画面上要显示的检测结果：只保留工作追踪区/涂油盘区内的目标。

        两个例外，否则现场会看到「目标突然消失」：

        - **当前判定中的丝杆**：手把杆举到区外时仍要画出来，否则正在涂油的
          那根杆从画面上没了；
        - **raw_mode**：旁路排查模式保持全量输出，便于核对模型覆盖情况。
        """

        if self.station.detector.raw_mode:
            return self._last_detections
        width, height = self._frame_size
        if width <= 0 or height <= 0:
            return ()
        tray_box = self._tray_region_box(
            self._last_frame_ts or 0.0, self._frame_size
        )
        kept = list(
            self._valid_detections(self._last_detections, self._frame_size, tray_box)
        )
        tracked = self._active_screw_box()
        if tracked is not None and not any(
            det.label == self.station.detector.screw_label
            and _overlaps(det.box, tracked)
            for det in kept
        ):
            for det in self._last_detections:
                if det.label == self.station.detector.screw_label and _overlaps(
                    det.box, tracked
                ):
                    kept.append(det)
                    break
        return tuple(kept)

    def _active_screw_box(self) -> BBox | None:
        """当前正在判定涂油的那根丝杆位置（没有则 None）。"""

        task = next(
            (
                item
                for item in self._cycles.values()
                if item.active and item.stage == TaskStage.PICKED
            ),
            None,
        )
        if task is None or task.screw_anchor is None:
            return None
        cx, cy = task.screw_anchor
        return BBox(cx - 8.0, cy - 8.0, cx + 8.0, cy + 8.0)

    def _valid_detections(
        self,
        detections: tuple[Detection, ...],
        frame_size: tuple[int, int],
        tray_box: BBox | None,
    ) -> tuple[Detection, ...]:
        """只保留中心落在工作追踪区或涂油盘区内的检测。

        区外的手、丝杆、刷头不建轨迹、不参与判定，也不出现在跟踪结果里。
        盘体检测缺失（``tray_box is None``）时只按工作追踪区过滤。
        """

        width, height = frame_size
        work_roi = self.station.work_roi
        return tuple(
            det
            for det in detections
            if work_roi.contains_box_center(det.box, width, height)
            or (tray_box is not None and _box_center_in(det.box, tray_box))
        )

    def _detections_for_tracking(
        self,
        *,
        label: str,
        in_roi: tuple[Detection, ...],
        frame_size: tuple[int, int],
        tray_box: BBox | None,
        tracker: ByteTrackObjectTracker | ObjectTracker,
        ts: float,
    ) -> tuple[Detection, ...]:
        """区外目标丢弃，但**已有轨迹延续到区外时继续喂给跟踪器**。

        原实现把工作追踪区外的检测在进跟踪器之前就删掉了，手或刷头一移出区
        就断轨——取料/涂油判定随即失去目标（现场「到区外就不跟了」）。
        这里改成：区内检测照常保留（区外目标不会新建轨迹，跟踪范围仍受控），
        区外检测只有在「离某条已存在轨迹足够近」时才保留，用来维持这条轨迹。
        """

        width, height = frame_size
        work_roi = self.station.work_roi
        diag = max(1.0, math.hypot(width, height))
        tracks = tuple(tracker.active_tracks())
        if not tracks:
            return in_roi

        # 延续门限：与跟踪器同一套距离门限，再乘 roi_track_margin 留一点余量。
        limit = self.cfg.roi_track_margin * self._frame_scale
        in_roi_ids = {id(det) for det in in_roi}
        extra: list[Detection] = []
        for det in self._last_detections:
            if det.label != label or id(det) in in_roi_ids:
                continue
            if work_roi.contains_box_center(
                det.box, width, height
            ) or (tray_box is not None and _box_center_in(det.box, tray_box)):
                continue
            cx, cy = det.box.center
            for track in tracks:
                px, py = track.predicted_box(ts).center
                if math.hypot(cx - px, cy - py) / diag <= limit:
                    extra.append(det)
                    break
        if not extra:
            return in_roi
        return in_roi + tuple(extra)

    def _tray_region_box(self, ts: float, frame_size: tuple[int, int]) -> BBox | None:
        """涂油盘区：手工 ``tray_roi`` 与模型检出 ``oil_station`` 框的并集。

        两个来源各管一件事，缺一不可：

        - **手工 ``tray_roi``**：覆盖盘体**下方的手臂活动范围**。模型只框得出
          盘体本身（示例①里框底 y≈372），而工人取料时手在盘体下方
          100~280px（y≈422~654）——只用检测框，取料会被全部漏掉。
        - **``oil_station`` 检测框**：随盘体上下移动（示例里跨度约 250px），
          手工区是固定的，不跟随就会在盘体升起时把盘内目标判成“已离盘”。

        检测框由 :class:`TrayTracker` 平滑并做短时保持：平滑避免框抖动让
        “是否在盘内”在边界反复跳变，保持避免一帧漏检就把手判成离盘。
        盘体检测不可用时退回手工区，取料判定照常进行。
        """

        width, height = frame_size
        if width <= 0 or height <= 0:
            return None
        fixed = self.station.tray_roi.pixel_bbox(width, height)
        predicted = self._tray.predicted_box(ts)
        if predicted is None:
            return fixed
        return BBox(
            min(fixed.x1, predicted.x1),
            min(fixed.y1, predicted.y1),
            max(fixed.x2, predicted.x2),
            max(fixed.y2, predicted.y2),
        )

    def _tray_region_reliable(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # 取料判定
    # ------------------------------------------------------------------
    def _update_pick(self, *, ts: float, tray_box: BBox | None) -> None:
        """空手进盘 → 带料离盘 ⇒ 取料完成。

        逐只手跟踪「进盘—离盘」过程：

        - 进盘那一帧记录手里是否已有丝杆（``holding_on_entry``）；
        - 离盘那一帧，如果进盘时是空手、离开时手里有丝杆，就判取料完成。

        带料进盘又带料离开（例如工人把丝杆放回去再拿走）不算取料；空手进、
        空手出也不算。
        """

        live_hands: set[int] = set()
        for hand in self._hand_tracks:
            if not hand.matched:
                continue
            live_hands.add(hand.track_id)
            cx, cy = hand.box.center
            inside = (
                tray_box.x1 <= cx <= tray_box.x2
                and tray_box.y1 <= cy <= tray_box.y2
            )
            holding = self._holding_screw(hand.box)
            visit = self._visits.get(hand.track_id)

            if visit is None:
                if inside:
                    # 手第一次出现就在盘内：没有进盘前的观察，只能按当帧状态取
                    # 保守值——手里有丝杆就不判取料，避免“把丝杆放回去又拿走”
                    # 被算成取料；真正从盘外走进来的手会走下面的分支。
                    self._visits[hand.track_id] = _HandVisit(
                        hand_track_id=hand.track_id,
                        entered_tray_ts=ts,
                        holding_on_entry=holding is not None,
                        inside_tray=True,
                        last_seen_ts=ts,
                    )
                else:
                    self._outside_holding[hand.track_id] = holding is not None
                continue

            visit.last_seen_ts = ts
            was_inside = visit.inside_tray
            visit.inside_tray = inside
            if not inside:
                self._outside_holding[hand.track_id] = holding is not None

            if not was_inside and inside:
                # 从盘外走进盘内：用进盘前一帧（盘外）的持有状态。
                visit.holding_on_entry = self._outside_holding.get(
                    hand.track_id, False
                )
                continue
            if not was_inside or inside:
                # 仍在盘内，或本来就不在盘内：只更新状态。
                continue

            # 刚离开涂油盘。
            self._visits.pop(hand.track_id, None)
            if visit.holding_on_entry:
                self._logger.debug(
                    "手 %s 带着丝杆进盘又离开，不判取料", hand.track_id
                )
                continue
            if holding is None:
                self._logger.debug("手 %s 空手离开涂油盘，不判取料", hand.track_id)
                continue
            dwell = ts - visit.entered_tray_ts
            if dwell < self.cfg.pick_confirm_s:
                # 进盘到离盘太快，多半是手从盘边划过。
                self._logger.debug(
                    "手 %s 在盘内停留过短（%.2fs），不判取料",
                    hand.track_id,
                    dwell,
                )
                continue
            self._confirm_pick(ts=ts, hand=hand, screw=holding, dwell_s=dwell)

        # 手消失：结束它的进盘过程。
        for hand_id in list(self._visits):
            if hand_id in live_hands:
                continue
            visit = self._visits[hand_id]
            if ts - visit.last_seen_ts >= self.cfg.holding_lost_timeout_s:
                self._visits.pop(hand_id, None)
        for hand_id in list(self._outside_holding):
            if hand_id not in live_hands and hand_id not in self._visits:
                self._outside_holding.pop(hand_id, None)

    def _holding_screw(self, hand_box: BBox) -> Detection | None:
        """手里拿着的那根丝杆：与手框**真实交叠或落在手框内**的丝杆检测。

        只按中心距离判定是不够的：手停在涂油盘边、尚未伸进去时，盘内丝杆的
        中心也可能落在 ``hand_screw_max_distance`` 内，于是“空手进盘”被误判成
        “带料进盘”，取料事件整个漏掉（现场视频 140911 就是这样）。手握丝杆时
        丝杆一定会被手框遮住，因此要求交叠/包含才认作持有。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        label = self.station.detector.screw_label
        hx, hy = hand_box.center
        best: tuple[float, Detection] | None = None
        for det in self._last_detections:
            if det.label != label:
                continue
            cx, cy = det.box.center
            distance = math.hypot(cx - hx, cy - hy) / diag
            if distance > self.cfg.hand_screw_max_distance:
                continue
            # 手握丝杆时丝杆中心落在手框内；只是指尖蹭到盘边丝杆时，
            # 两者仅有百分之几的交叠，不能算持有。
            gripped = (
                _box_center_in(det.box, hand_box)
                or hand_box.iou(det.box) >= 0.15
            )
            if not gripped:
                continue
            if best is None or distance < best[0]:
                best = (distance, det)
        return None if best is None else best[1]

    def _confirm_pick(
        self,
        *,
        ts: float,
        hand: Track,
        screw: Detection,
        dwell_s: float,
    ) -> None:
        task = self._ensure_cycle(ts, hand_track_id=hand.track_id)
        task.stage = TaskStage.PICKED
        task.stage_since = ts
        task.picked_at = ts
        task.hand_track_id = hand.track_id
        task.screw_observed = True
        # 锚点留空：离盘那一刻手里可能还压着盘边的杆，直接拿来初始化会锁错
        # 目标（锁成留在盘里的那根）。交给 _current_screw_box 用「跟手框交叠」
        # 的杆初始化，那才是真正被带走的这根。
        task.screw_anchor = None
        task.screw_anchor_ts = None
        task.confidence = screw.confidence
        task.reason_code = "PICK_CONFIRMED"
        task.last_seen_ts = ts
        task.evidence.append("SCREW_LEFT_TRAY_WITH_HAND")
        task.timeline.append((TaskStage.PICKED.value, ts))
        self._emit_event(
            task,
            event_type=EventType.PICK,
            reason="pick_confirmed",
            extra={
                "hand_track_id": hand.track_id,
                "dwell_in_tray_s": round(dwell_s, 3),
                "screw_confidence": round(float(screw.confidence), 3),
            },
            start_ts=ts - dwell_s,
            end_ts=ts,
        )
        self._logger.info(
            "取料完成 %s（手部轨迹=%s，盘内停留 %.2fs）",
            task.task_id,
            hand.track_id,
            dwell_s,
        )

    # ------------------------------------------------------------------
    # 人工二次涂油判定
    # ------------------------------------------------------------------
    def _update_oiling(self, *, ts: float, dt: float) -> None:
        """手持丝杆期间刷头持续靠近 ⇒ 人工二次涂油完成。"""

        task = next(
            (
                item
                for item in self._cycles.values()
                if item.active and item.stage == TaskStage.PICKED
            ),
            None,
        )
        if task is None:
            return

        screw_box = self._current_screw_box(task, ts)
        if screw_box is None:
            # 手持丝杆当帧不可见、也超出预测保持时长：不累计接触时长，
            # 但不结束周期，等丝杆重新出现。
            return
        task.last_seen_ts = ts

        brush = self._nearest_brush(screw_box)
        if brush is None:
            # 区分两种「没靠近」：
            # - 刷子当帧根本没被检测到 → 属于漏检，保留 last_brush_ts，
            #   下一帧靠近时把空档补回来；
            # - 刷子检测到了但离得远 → 确实没在刷，累计中断，清空桥接点。
            if self._has_live_brush():
                task.last_brush_ts = None
            return
        task.brush_track_id = brush.track_id
        if dt <= 0:
            # 时间没前进（同时间戳重复帧）：不累计，但要记住靠近状态。
            task.last_brush_ts = ts
            return
        # 低帧率下刷子常有一两帧漏检，那几帧的 dt 若直接丢弃，累计会明显
        # 偏短（3~5fps 时每漏一帧就少 0.2~0.3s）。这里把「上一次靠近」到
        # 本帧的空档一并计入，上限按实际帧率放大后的 brush_hold_max_gap_s。
        add = dt
        if task.last_brush_ts is not None:
            gap = ts - task.last_brush_ts
            max_gap = self.cfg.brush_hold_max_gap_s * self._frame_scale
            if 0 < gap <= max_gap:
                add = gap
        task.last_brush_ts = ts
        task.brush_hold_s += add
        # 逐帧累加 dt 会有浮点误差（10 个 0.1s 累加小于 1.0），
        # 留一点容差，避免刚好等于阈值时永远差一帧。
        if task.brush_hold_s >= self.cfg.brush_hold_s - 1e-9:
            self._complete_oiling(task, ts)

    def _current_screw_box(self, task: _Cycle, ts: float) -> BBox | None:
        """当前手持丝杆的位置：**丝杆锚点跟踪**，不依赖手部视觉 ID。

        取料那一刻把手里那根杆记成锚点，之后每帧更新：

        1. 优先取「与本帧某个手框交叠」的杆——握着的时候丝杆一定被手框遮住，
           这条约束把候选从满盘丝杆缩到手上那一根；
        2. 交叠候选里取离锚点最近的（``screw_anchor_max_step``，按实际帧率缩放）；
        3. 手上没有交叠杆（手被身体挡住/框偏了）才退化为纯位置连续性；
        4. 全都找不到时，在 ``screw_anchor_hold_s`` 内按锚点位置继续，
           让刷子靠近的累计不因一两帧漏检中断。

        为什么不跟手：手部视觉 ID 在遮挡/快速移动后会重建——示例①一个 12 秒
        周期里手部 ID 换了 6 次，其中某个周期 80% 的帧绑定的 ID 都未匹配，
        涂油累计因此停在 0 并误判漏涂（实测该周期 19.5~22.5s 刷子距杆
        0.011~0.031，远低于阈值 0.06，实际是刷了的）。丝杆是静置物、位移小。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        label = self.station.detector.screw_label
        step = self.cfg.screw_anchor_max_step * self._frame_scale
        anchor = task.screw_anchor
        hand_center = self._hand_center(task.hand_track_id)

        # 1) 握在手里的杆：与手框交叠的那些。
        gripped: list[tuple[float, BBox]] = []
        for det in self._last_detections:
            if det.label != label:
                continue
            if not any(_overlaps(det.box, hand.box) for hand in self._hand_tracks):
                continue
            cx, cy = det.box.center
            distance = (
                0.0
                if anchor is None
                else math.hypot(cx - anchor[0], cy - anchor[1]) / diag
            )
            if anchor is not None and distance > step:
                continue
            gripped.append((distance, det.box))

        if gripped and anchor is None:
            # 初始化：取交叠杆中离手中心最近的那根（握着的一定贴着这只手）。
            best = min(
                gripped, key=lambda item: _distance_to(item[1], hand_center, diag)
            )[1] if hand_center is not None else gripped[0][1]
        else:
            best = min(gripped, key=lambda item: item[0])[1] if gripped else None

        # 2) 手上没有可见的杆：按位置连续性找。
        if best is None and anchor is not None:
            nearest: tuple[float, BBox] | None = None
            for det in self._last_detections:
                if det.label != label:
                    continue
                cx, cy = det.box.center
                distance = math.hypot(cx - anchor[0], cy - anchor[1]) / diag
                if distance > step:
                    continue
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, det.box)
            best = None if nearest is None else nearest[1]

        if best is not None:
            task.screw_anchor = best.center
            task.screw_anchor_ts = ts
            task.screw_observed = True
            task.last_screw_ts = ts
            return best

        # 3) 都没有：hold 窗口内按锚点位置继续，窗口外放弃。
        if (
            anchor is not None
            and task.screw_anchor_ts is not None
            and ts - task.screw_anchor_ts
            <= self.cfg.screw_anchor_hold_s * self._frame_scale
        ):
            task.screw_observed = False
            cx, cy = anchor
            return BBox(cx - 8.0, cy - 8.0, cx + 8.0, cy + 8.0)

        task.screw_observed = False
        return None

    def _hand_center(self, hand_track_id: int | None) -> tuple[float, float] | None:
        if hand_track_id is None:
            return None
        hand = self._hand_tracker.get(hand_track_id)
        if hand is None or not hand.matched:
            return None
        return hand.box.center

    def _has_live_brush(self) -> bool:
        """本帧是否有任一已匹配的刷头轨迹（用于区分漏检和远离）。"""

        return any(
            brush.matched and brush.hits >= self._brush_tracker.min_hits
            for brush in self._brush_tracks
        )

    def _nearest_brush(self, screw_box: BBox) -> Track | None:
        """离丝杆接触邻域最近、且已被本帧检测到的刷头。"""

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        sx, sy = screw_box.center
        best: tuple[float, Track] | None = None
        for brush in self._brush_tracks:
            if not brush.matched or brush.hits < self._brush_tracker.min_hits:
                continue
            point = brush_work_point(brush, self.station.detector)
            distance = math.hypot(point[0] - sx, point[1] - sy) / diag
            if distance > self.cfg.brush_contact_distance:
                continue
            if best is None or distance < best[0]:
                best = (distance, brush)
        return None if best is None else best[1]

    def _complete_oiling(self, task: _Cycle, ts: float) -> None:
        task.stage = TaskStage.OILED
        task.stage_since = ts
        task.completed_at = ts
        task.final_result = TaskResult.COMPLETED
        task.reason_code = "MANUAL_OILING_COMPLETED"
        task.evidence.append("BRUSH_NEAR_SCREW_HELD")
        task.timeline.append((TaskStage.OILED.value, ts))
        self._emit_event(
            task,
            event_type=EventType.MANUAL_OILING,
            reason="manual_oiling_completed",
            extra={
                "hand_track_id": task.hand_track_id,
                "brush_track_id": task.brush_track_id,
                "brush_hold_s": round(task.brush_hold_s, 3),
                "required_s": self.cfg.brush_hold_s,
            },
            start_ts=task.picked_at or task.created_at,
            end_ts=ts,
        )
        self._logger.info(
            "人工二次涂油完成 %s（手持丝杆，刷头靠近 %.2fs）",
            task.task_id,
            task.brush_hold_s,
        )

    # ------------------------------------------------------------------
    # 周期管理
    # ------------------------------------------------------------------
    def _supersede_unfinished(self, task: _Cycle, ts: float) -> None:
        """上一周期只取料、没完成二次涂油 ⇒ ``NG``。

        判定只在「新的取料被确认」这一刻触发，不等超时：现场工艺里一件丝杆
        取走后没刷完就取下一件，就是漏涂。周期立即结束，事件类型
        ``OILING_MISSING``，结果 ``NEEDS_REVIEW`` 交给人工复核。
        """

        if task.stage != TaskStage.PICKED or task.picked_at is None:
            # 还没取料的周期不会漏涂，直接丢弃（当前流程不会出现，留作防御）。
            self._cycles.pop(task.task_id, None)
            return
        task.stage_since = ts
        task.completed_at = ts
        task.final_result = TaskResult.NG
        task.reason_code = "OILING_MISSING_SUPERSEDED"
        task.evidence.append("NEXT_PICK_BEFORE_MANUAL_OILING")
        task.timeline.append(("NG", ts))
        self._emit_event(
            task,
            event_type=EventType.OILING_MISSING,
            reason="oiling_missing_superseded_by_pick",
            result=EventResult.NEEDS_REVIEW,
            extra={
                "hand_track_id": task.hand_track_id,
                "brush_hold_s": round(task.brush_hold_s, 3),
                "required_s": self.cfg.brush_hold_s,
                "picked_at": task.picked_at,
                "superseded_at": ts,
            },
            start_ts=task.picked_at,
            end_ts=ts,
        )
        self._logger.info(
            "周期 %s 取料后未完成二次涂油（累计 %.2fs/%.2fs），判 NG",
            task.task_id,
            task.brush_hold_s,
            self.cfg.brush_hold_s,
        )

    def _ensure_cycle(self, ts: float, *, hand_track_id: int | None) -> _Cycle:
        """新建当前周期；若上一周期只取料未涂油，先判它 NG。"""

        for task in list(self._cycles.values()):
            if not task.active:
                continue
            # 已取料、二次涂油未完成就来了新的取料：上一件漏涂，判 NG 并
            # 结束该周期，新取料不能并进同一周期（否则第二次取料会被
            # ``events_sent`` 去重吞掉，现场看不出漏涂）。
            self._supersede_unfinished(task, ts)
            break
        self._task_seq += 1
        task_id = f"TASK-{self._task_seq:06d}"
        task = _Cycle(
            task_id=task_id,
            created_at=ts,
            stage=TaskStage.WAITING_PICK,
            stage_since=ts,
            final_result=TaskResult.PROCESSING,
            hand_track_id=hand_track_id,
            timeline=[(TaskStage.WAITING_PICK.value, ts)],
        )
        self._cycles[task_id] = task
        self._logger.info("新建作业周期 %s", task_id)
        return task

    def _cleanup(self, ts: float) -> None:
        remove: list[str] = []
        for task_id, task in self._cycles.items():
            if task.active:
                continue
            finished = task.completed_at or task.stage_since
            if ts - finished >= self.cfg.completed_retain_s:
                remove.append(task_id)
        for task_id in remove:
            self._cycles.pop(task_id, None)

        if len(self._cycles) <= self.cfg.task_max_active:
            return
        ordered = sorted(
            self._cycles.values(),
            key=lambda item: (item.active, item.created_at),
        )
        for task in ordered[: len(self._cycles) - self.cfg.task_max_active]:
            self._cycles.pop(task.task_id, None)

    # ------------------------------------------------------------------
    # 事件
    # ------------------------------------------------------------------
    def _emit_event(
        self,
        task: _Cycle,
        *,
        event_type: EventType,
        reason: str,
        extra: dict[str, Any] | None = None,
        start_ts: float | None = None,
        end_ts: float | None = None,
        result: EventResult = EventResult.CONFIRMED,
    ) -> EventRecord | None:
        key = f"{event_type.value}:{reason}"
        if key in task.events_sent:
            return None
        task.events_sent.add(key)
        evidence: dict[str, Any] = {
            "task_id": task.task_id,
            "reason": reason,
            "stage": task.stage.value,
            "final_result": task.final_result.value,
            "evidence": list(task.evidence),
            "detector": self.detector_name,
        }
        if extra:
            evidence.update(extra)
        event = EventRecord.create(
            timestamp_start=float(
                start_ts if start_ts is not None else task.created_at
            ),
            timestamp_end=float(end_ts if end_ts is not None else task.stage_since),
            station_id=self.station.station_id,
            camera_id=self.station.camera_id,
            cycle_id=self.cycle_id,
            task_id=task.task_id,
            event_type=event_type,
            result=result,
            evidence=evidence,
        )
        self._events_emitted += 1
        self._logger.info(
            "事件 %s：工位=%s 周期=%s 原因=%s",
            event.event_type.value,
            event.station_id,
            event.task_id,
            reason,
        )
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:  # noqa: BLE001
                self._logger.exception("事件回调失败")
        return event

    # ------------------------------------------------------------------
    # 快照
    # ------------------------------------------------------------------
    def _to_snapshot(self, task: _Cycle) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=task.task_id,
            created_at=task.created_at,
            stage=task.stage,
            stage_since=task.stage_since,
            final_result=task.final_result,
            hand_track_id=task.hand_track_id,
            screw_track_id=task.screw_track_id,
            brush_track_id=task.brush_track_id,
            screw_observed=task.screw_observed,
            brush_hold_s=task.brush_hold_s,
            brush_hold_target_s=self.cfg.brush_hold_s,
            confidence=task.confidence,
            reason_code=task.reason_code,
            evidence=tuple(task.evidence),
            picked_at=task.picked_at,
            completed_at=task.completed_at,
            last_seen_ts=task.last_seen_ts,
            timeline=tuple(task.timeline),
        )

    def _tray_snapshot(self) -> TraySnapshot:
        diag = max(1.0, math.hypot(*self._frame_size))
        speed = math.hypot(*self._tray.velocity) / diag
        if not self._tray.matched:
            motion = "UNKNOWN"
        elif speed >= 0.05:
            motion = "MOVING"
        else:
            motion = "STILL"
        return TraySnapshot(
            track_id=self._tray.track_id,
            box=self._tray.predicted_box(self._last_frame_ts or self._tray.last_ts),
            confidence=self._tray.confidence or None,
            motion=motion,
            reliable=self._tray.reliable,
            visible=self._tray.matched,
            missing_s=self._tray.missing_s,
            last_seen_ts=self._tray.last_seen_ts or None,
        )

    def _track_snapshots(self) -> tuple[TrackSnapshot, ...]:
        items: list[TrackSnapshot] = []
        for label, tracks in (
            (self.station.detector.hand_label, self._hand_tracks),
            (self.station.detector.brush_label, self._brush_tracks),
        ):
            for track in tracks:
                items.append(
                    TrackSnapshot(
                        track_id=track.track_id,
                        label=label,
                        box=(
                            track.box
                            if track.matched
                            else track.predicted_box(
                                self._last_frame_ts or track.last_ts
                            )
                        ),
                        confidence=track.confidence,
                        observed=track.matched,
                        missing_s=track.missing_s,
                        last_seen_ts=track.last_seen_ts,
                        angle=track.angle,
                        point=track.point,
                    )
                )
        return tuple(items)

# ----------------------------------------------------------------------
# 几何与辅助
# ----------------------------------------------------------------------
def _overlaps(a: BBox, b: BBox) -> bool:
    """两个框是否有实际交叠（不含仅贴边）。"""

    return (
        min(a.x2, b.x2) - max(a.x1, b.x1) > 0
        and min(a.y2, b.y2) - max(a.y1, b.y1) > 0
    )


def _distance_to(box: BBox, point: tuple[float, float], diag: float) -> float:
    cx, cy = box.center
    return math.hypot(cx - point[0], cy - point[1]) / diag


def _box_center_in(box: BBox, region: BBox) -> bool:
    cx, cy = box.center
    return region.x1 <= cx <= region.x2 and region.y1 <= cy <= region.y2


def _label_counts(detections: tuple[Detection, ...]) -> dict[str, int]:
    """按类别统计本帧目标数，供 UI 核对“所有标签是否都出来了”。"""

    counts: dict[str, int] = {}
    for det in detections:
        counts[det.label] = counts.get(det.label, 0) + 1
    return dict(sorted(counts.items()))
