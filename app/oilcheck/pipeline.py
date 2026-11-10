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
from collections import deque
from dataclasses import dataclass, field
from statistics import median
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

# 帧间隔鲁棒估计的样本数。单帧长间隔（视频跳帧、推理卡顿）不应该改变
# 距离门限，所以用中位数而不是「上一帧的间隔」。
_FRAME_GAP_WINDOW = 15

# 区外延续跟踪的硬上限（归一化），不随 frame_scale 放大。
_ROI_TRACK_LIMIT_ABS = 0.10

# 盘体漏检超过该时长，涂油盘区退回手工标定区（不再用外推框）。
_TRAY_PREDICT_TRUST_S = 0.30

# 「纯位置连续性」跟进手持丝杆的单帧位移上限（不随帧率放大）。
#
# 与「杆与本周期那只手交叠」的证据强度不同：握持是强证据，取料瞬间手带杆
# 快速划出，单帧位移可达 0.11；纯位置连续性只是「上一帧在这、这一帧还在
# 附近」，允许的跨度必须小得多，否则锚点会一步跳到旁边另一根杆上（实测
# 在 0.15 的跨度下会被另一只手的杆抢走）。
_CONTINUITY_STEP_ABS = 0.08


@dataclass
class _HandVisit:
    """一只手的「进盘 — 离盘」过程。

    「在不在盘内」「手里有没有杆」都不看单帧：每隔
    ``pick_sample_every_frames`` 帧采一个样，保留最近
    ``pick_history_samples`` 个采样点，按多数投票决定。

    两路采样的口径不同：

    - **在不在盘内**：每帧都参与采样，确认进盘与离盘；
    - **手里有没有杆**：只用**盘外**的采样。手伸在盘上方时必然蹭到盘里的
      静置杆，盘内的握持读数与实际有没有握着无关（实测盘内读数交替闪烁，
      而盘外携带阶段检出率 96~99%）。盘内噪声帧一旦混进窗口，多数票就
      永远投不出正确结论。
    """

    hand_track_id: int
    entered_tray_ts: float
    # 进盘时手里是否已经拿着丝杆。取料只认“空手进、带料出”。
    #
    # 不能在进盘当帧判：手一伸进盘，手框就与盘内静置的丝杆交叠，当帧判定
    # 永远是“持有”，取料会被全部吞掉（现场视频 140911 即如此）。因此按
    # **盘外观测历史**投票判定。
    holding_on_entry: bool
    # 已确认「在盘内」。
    inside_tray: bool = True
    last_seen_ts: float = 0.0
    frames_since_sample: int = 0
    inside_samples: deque[bool] = field(default_factory=deque)
    # **盘外**握持采样（进盘前后与离盘后的「手里有没有杆」）。
    holding_samples: deque[bool] = field(default_factory=deque)
    # 离盘后任意一帧见过「手握杆」。采样点按 k 帧一个，离盘后手很快就
    # 出画面时窗口里一个点都没有；这个标记把那些没落在采样点上的正面
    # 证据留下来，免得点数不足被默认按空手处理。
    holding_seen: bool = False
    # 最近一帧在盘外时手里有没有杆；最近一帧在盘内时为 None。
    # 手在原地消失、离盘还没按采样窗口确认时，这是唯一的依据。
    last_outside_holding: bool | None = None
    # 最近一帧的手框。手消失时跟踪器会先把轨迹删掉（``_hand_tracker.get``
    # 返回 None），那时只能靠这份快照做出占位证据。
    last_hand_box: BBox | None = None
    # 离盘的时刻；此后继续攒盘外握持采样，攒够 n 个再定论。
    exited_at_ts: float | None = None
    # 是否已确认「进过盘」。一旦置真，这一段进盘过程就成立，直到离盘判定。
    entered_confirmed: bool = False

    def tick(self, *, every_frames: int) -> bool:
        """推进一步并返回本帧是否到采样点（每 k 帧一个）。"""

        self.frames_since_sample += 1
        if self.frames_since_sample < max(1, every_frames):
            return False
        self.frames_since_sample = 0
        return True

    def push_inside(self, inside: bool, keep: int) -> None:
        _push(self.inside_samples, bool(inside), keep)

    def push_holding(self, holding: bool, keep: int) -> None:
        """记录盘外握持采样；离盘后重新开始累积（离盘前的属于取料前）。"""

        _push(self.holding_samples, bool(holding), keep)

    def restart_holding(self) -> None:
        """离盘后重置握持窗口：从此刻起重新累积。"""

        self.holding_samples.clear()
        self.frames_since_sample = 0

    @property
    def inside_majority(self) -> bool:
        return _majority(self.inside_samples)

    @property
    def left_majority(self) -> bool:
        return bool(self.inside_samples) and not _majority(self.inside_samples)

    def holding_verdict(self, keep: int) -> bool:
        """盘外握持窗口的结论。

        窗口攒满 ``keep`` 个采样点后按多数投票：单帧闪断（手把杆完全遮住）
        不会把一次取料判丢。

        窗口没攒满就得定论（手离开画面、视频结束、等到超时）时，只要见过
        一帧「手框里/与手框交叠的杆」就认作握持：盘外手持杆的检出率实测
        96~99%，见过一次基本就是真的拿着；反过来，单帧「没杆」多半只是
        杆被手遮住，不足以否定。一帧都没见过时没有正面证据，按空手处理。

        「点数不足 ⇒ 按空手处理」是之前的行为，代价是整件任务停在这里：
        没有取料就没有周期，后面的涂油判定也就无从谈起。
        """

        if len(self.holding_samples) >= max(1, keep):
            return _majority(self.holding_samples)
        return any(self.holding_samples) or self.holding_seen


def _push(samples: deque[bool], value: bool, keep: int) -> None:
    """往采样窗口尾append，超过 keep 个就丢最旧的。"""

    while len(samples) >= max(1, keep):
        samples.popleft()
    samples.append(value)


def _majority(samples: deque[bool] | list[bool]) -> bool:
    """过半即真；没有样本时按假处理。"""

    if not samples:
        return False
    return sum(1 for value in samples if value) * 2 > len(samples)


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
    # 最近一次「锚点来自手里确实握着的杆」的时刻，以及那一刻的位置。
    # 纯位置连续性只在离这个位置 ``hand_screw_max_distance`` 以内有效：
    # 不设这道约束时，锚点会在密集的静置丝杆之间逐帧漂移（实测漂进零件盒
    # 里的整排静置杆，手里那根反而累计不上）。
    screw_gripped_ts: float | None = None
    screw_gripped_at: tuple[float, float] | None = None
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
        # 手在盘外时的持握**历史**，进盘判定用：值为 (是否空手, 时刻) 的
        # 队列，只保留时间窗口内（``_entry_grace_s``）的观测。进盘时看空手
        # 帧占比，而不是「最近一帧」——单帧既可能是手刚伸到盘边与静置杆
        # 交叠，也可能是手正把杆递进来，都不能定性。
        self._outside_holding: dict[int, deque[tuple[bool, float]]] = {}
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
        # 用最近若干帧间隔的中位数估计，避免单帧跳变把门限瞬间撑大。
        self._gap_samples: deque[float] = deque(maxlen=_FRAME_GAP_WINDOW)
        self._frame_scale = 1.0
        # 本帧的时间是否可信。长间隔/时间倒退时不可信：不桥接空档、
        # 不把掉帧的时间算进停留/涂油时长。
        self._timing_reliable = True

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
        self._gap_samples.clear()
        self._frame_scale = 1.0
        self._timing_reliable = True

    def process(self, *, detections: FrameDetections) -> StationSnapshot:
        ts = float(detections.ts)
        width, height = detections.frame_size
        if width <= 0 or height <= 0:
            return self.snapshot()
        if self.station.detector.raw_mode:
            return self._process_raw(detections)

        if self._last_frame_ts is not None and ts <= self._last_frame_ts:
            # 同一时间戳重复帧 / 时间倒退：不推进任何状态。重复处理会把
            # 停留时长、轨迹命中次数都算重，而画面其实是同一帧。
            if ts < self._last_frame_ts:
                self._frame_gap_events += 1
                self._logger.warning(
                    "视频时间戳倒退：%.3fs（工位=%s），本帧跳过",
                    ts - self._last_frame_ts,
                    self.station.station_id,
                )
            return self.snapshot()

        dt = 0.0
        base_dt = 1.0 / max(1, self.cfg.max_fps)
        if self._last_frame_ts is not None:
            raw_gap = ts - self._last_frame_ts
            self._last_frame_gap_s = raw_gap
            # 先看上一帧的节拍，再判断本帧间隔是否可信：真正的低帧率流
            # （每帧都慢）与偶发掉帧（夹一两个长间隔）必须分开处理，否则
            # 2fps 现场会被永久判成「时间不可靠」，判定彻底停摆。
            plausible = self._max_plausible_dt(raw_gap)
            self._timing_reliable = raw_gap <= plausible
            if not self._timing_reliable:
                self._frame_gap_events += 1
                self._logger.warning(
                    "视频时间戳异常：%.3fs（工位=%s），本帧不按连续时间计时",
                    raw_gap,
                    self.station.station_id,
                )
            dt = max(0.0, min(plausible, raw_gap))
            self._gap_samples.append(raw_gap)
            # 节拍用最近帧间隔的中位数：偶发的长间隔不改变它，持续的低帧率
            # 会在几帧内把估计拉到真实值。
            typical = median(self._gap_samples)
            self._frame_scale = max(
                1.0, min(self.cfg.frame_scale_max, typical / base_dt)
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
                "timing_reliable": self._timing_reliable,
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
        # 加绝对上限：帧率倍率放开后这个门限会大到整幅画面，区外检测全被
        # 喂进跟踪器，轨迹碎片随之变多。
        limit = min(
            self.cfg.roi_track_margin * self._frame_scale, _ROI_TRACK_LIMIT_ABS
        )
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
                    extra.append(_demote_for_continuation(det, self.cfg))
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
        if not self._tray_prediction_trustworthy(ts):
            return fixed
        predicted = self._tray.predicted_box(ts)
        if predicted is None:
            return fixed
        return BBox(
            min(fixed.x1, predicted.x1),
            min(fixed.y1, predicted.y1),
            max(fixed.x2, predicted.x2),
            max(fixed.y2, predicted.y2),
        )

    def _tray_prediction_trustworthy(self, ts: float) -> bool:
        """盘体预测框还能不能参与盘区判定。

        漏检时间一长，纯速度外推会把框送到画面外；取并集后整个画面都成了
        「盘内」，取料/离盘判定随即失效（实测 0.9s 漏检即出画面）。外部
        条件：本帧有观测，或漏检在 ``_TRAY_PREDICT_TRUST_S`` 内且预测框没
        明显越界/膨胀。
        """

        if self._tray.box is None:
            return False
        if self._tray.matched:
            return True
        if self._tray.missing_s > _TRAY_PREDICT_TRUST_S:
            return False
        predicted = self._tray.predicted_box(ts)
        if predicted is None:
            return False
        width, height = self._frame_size
        if width <= 0 or height <= 0:
            return False
        if (
            predicted.x1 < -width
            or predicted.y1 < -height
            or predicted.x2 > 2 * width
            or predicted.y2 > 2 * height
        ):
            return False
        area = predicted.width * predicted.height
        base = self._tray.box.width * self._tray.box.height
        return not (base > 0 and area > 2.0 * base)

    def _tray_region_reliable(self) -> bool:
        """盘体定位是否可用（供 UI 状态条）。

        只报告真实情况：漏检超过可信窗口即 ``False``，不再恒为 ``True``。
        """

        return self._tray.box is not None and (
            self._tray.matched
            or self._tray.missing_s <= _TRAY_PREDICT_TRUST_S
        )

    # ------------------------------------------------------------------
    # 取料判定
    # ------------------------------------------------------------------
    def _update_pick(self, *, ts: float, tray_box: BBox | None) -> None:
        """空手进盘 → 带料离盘 ⇒ 取料完成。

        所有子判定都用最近 n 个采样点投票（``pick_sample_every_frames`` /
        ``pick_history_samples``），不拿单帧当结论：

        - **是否在盘内**：窗口内出现过盘内采样即可确认「进过盘」（事件性，
          用 any——短时取料不会被后来的盘外采样淹没）；窗口多数在盘外才
          确认离盘（用多数——避免盘边抖动劈段）；
        - **离盘后手里有没有杆**：只采**盘外**的读数。攒满 n 个采样点就
          按多数投票；点数不足就得定论（手出画面、视频结束、等到超时）
          时，见过一帧手握杆就算握持。离盘那一帧的杆检测往往还没稳（杆心
          刚离开盘区、置信度 0.5 上下），盘内读数又被静置杆污染，两者都
          不能混进窗口；
        - **进盘时手里有没有杆**：同样按盘外观测历史投票。

        带料进盘又带料离开（例如工人把丝杆放回去再拿走）不算取料，空手进、
        空手出也不算。点数不足时**不能**一律按空手处理：那样一次真实取料
        会连周期都建不出来，后面的涂油判定无从谈起（现场「取料没跑完就
        停了」）。只有一帧正面证据都没有时才按空手处理。
        """

        every = max(1, int(self.cfg.pick_sample_every_frames))
        keep = max(1, int(self.cfg.pick_history_samples))
        # 离盘后攒够 n 个采样的时限 + 一点余量；超时仍不足则按空手处理。
        settle_s = (
            (every * keep + 1)
            * (1.0 / max(1, self.cfg.max_fps))
            * self._frame_scale
        )
        live_hands: set[int] = set()
        for hand in self._hand_tracks:
            if not hand.matched or hand.hits < max(
                1, self._hand_tracker.min_hits
            ):
                # 单帧误检不参与进盘判定（min_hits 之前对手部无效）。
                continue
            live_hands.add(hand.track_id)
            # 判定用本帧原始检测框：滤波框在快速移动时滞后（200px 跳变只
            # 走到 155px），边界判定会晚一帧。
            hand_box = hand.observation_box()
            cx, cy = hand_box.center
            inside_raw = (
                tray_box.x1 <= cx <= tray_box.x2
                and tray_box.y1 <= cy <= tray_box.y2
            )
            holding = self._holding_screw(hand_box)
            holding_now = holding is not None
            visit = self._visits.get(hand.track_id)

            if visit is None:
                if not inside_raw:
                    self._record_outside_holding(hand.track_id, holding, ts)
                empty_outside = self._mostly_empty_outside(
                    hand.track_id, ts, default_empty=True
                )
                visit = _HandVisit(
                    hand_track_id=hand.track_id,
                    entered_tray_ts=ts,
                    holding_on_entry=holding_now and not empty_outside,
                    inside_tray=inside_raw,
                    last_seen_ts=ts,
                    frames_since_sample=every,
                )
                self._visits[hand.track_id] = visit

            visit.last_seen_ts = ts
            visit.last_hand_box = hand_box
            visit.last_outside_holding = None if inside_raw else holding_now
            due = visit.tick(every_frames=every)
            if due:
                visit.push_inside(inside_raw, keep)
                # 盘内读数不可用（会蹭到静置杆），只在盘外采握持。
                if not inside_raw and visit.exited_at_ts is None:
                    self._record_outside_holding(hand.track_id, holding, ts)

            # ---- 离盘后：攒盘外握持采样，攒够再定论 ----
            if visit.exited_at_ts is not None:
                if inside_raw:
                    # 又回到盘内：视为同一段取料的中途抖动，取消离盘待定。
                    visit.exited_at_ts = None
                    visit.restart_holding()
                    visit.holding_seen = False
                    continue
                # 正面证据不挑采样点：采样按 k 帧一个，手离盘后很快就
                # 出画面时窗口可能一个点都攒不到。
                if holding_now:
                    visit.holding_seen = True
                if due:
                    visit.push_holding(holding_now, keep)
                # 攒满窗口才定论（和之前一致）。提前定论会改变后面几段的
                # 「进盘时手上有没有杆」：实测 105708 在 7.5fps 下从 4 取
                # 掉到 3 取。点数不足的情形由 holding_verdict 和「手消失」
                # 分支兜底，不需要抢跑。
                ready = len(visit.holding_samples) >= keep
                timed_out = ts - visit.exited_at_ts > settle_s
                if not (ready or timed_out):
                    continue
                self._settle_pick(
                    visit=visit,
                    hand_track_id=hand.track_id,
                    hand_box=hand_box,
                    ts=ts,
                    holding=holding,
                    keep=keep,
                )
                self._visits.pop(hand.track_id, None)
                continue

            # ---- 尚未离盘：判断进盘 / 仍在盘内 ----
            if not visit.entered_confirmed:
                if any(visit.inside_samples):
                    visit.entered_confirmed = True
                    # 进盘定性：看进盘前的盘外历史。
                    visit.holding_on_entry = not self._mostly_empty_outside(
                        hand.track_id, ts, default_empty=True
                    )
                    visit.inside_tray = True
                    visit.restart_holding()
                elif not inside_raw:
                    self._record_outside_holding(hand.track_id, holding, ts)
                continue

            if not visit.left_majority:
                visit.inside_tray = True
                continue

            # 确认离盘：从这里开始攒盘外握持采样。
            visit.inside_tray = False
            visit.exited_at_ts = ts
            visit.restart_holding()
            if not inside_raw:
                # 手此刻确实在盘外：这个读数就是离盘后的第一个证据。
                visit.last_outside_holding = holding_now
                if holding_now:
                    visit.holding_seen = True
            if due:
                visit.push_holding(holding_now, keep)

        # 手消失：先用已经攒到的采样定论，再结束进盘过程。
        trackers = {track.track_id: track for track in self._hand_tracks}
        for hand_id in list(self._visits):
            if hand_id in live_hands:
                continue
            visit = self._visits[hand_id]
            if ts - visit.last_seen_ts < self.cfg.holding_lost_timeout_s:
                continue
            if visit.exited_at_ts is None and self._vanished_outside(visit):
                # 手在盘外直接消失，离盘还没来得及按采样窗口确认（窗口
                # 按 k 帧一个点，走得太快就采不到）。最后观测确实在盘外，
                # 就按那一刻离盘结算，别把这段整段丢掉。
                visit.exited_at_ts = visit.last_seen_ts
                if visit.last_outside_holding:
                    visit.push_holding(True, keep)
                    visit.holding_seen = True
            if visit.exited_at_ts is not None:
                # 手已经确认离盘：用已攒到的采样定论，哪怕不足 n 个。
                # 之前要求「窗口里至少有一个采样」才结算，低帧率/很快就
                # 出画面时窗口是空的，这段进盘过程被直接丢掉，取料事件
                # 永远发不出来。
                #
                # 手框优先用当前轨迹，轨迹已被跟踪器删除（手出画面超过
                # 保留期）时用快照：手一消失就删轨迹的话，这里会拿到
                # None 而静默跳过，取料同样发不出来。
                hand = trackers.get(hand_id)
                hand_box = (
                    hand.observation_box()
                    if hand is not None
                    else visit.last_hand_box
                )
                if hand_box is not None:
                    self._settle_pick(
                        visit=visit,
                        hand_track_id=hand_id,
                        hand_box=hand_box,
                        ts=visit.exited_at_ts,
                        holding=None,
                        keep=keep,
                    )
            self._visits.pop(hand_id, None)
        for hand_id in list(self._outside_holding):
            if hand_id not in live_hands and hand_id not in self._visits:
                self._outside_holding.pop(hand_id, None)

    @staticmethod
    def _vanished_outside(visit: _HandVisit) -> bool:
        """手消失时最后观测在盘外，且已经确认进过盘。"""

        return visit.entered_confirmed and visit.last_outside_holding is not None

    def _settle_pick(
        self,
        *,
        visit: _HandVisit,
        hand_track_id: int,
        hand_box: BBox,
        ts: float,
        holding: Detection | None,
        keep: int,
    ) -> None:
        """攒够采样（或不得不定论）时给这一件下定论。"""

        dwell = ts - visit.entered_tray_ts
        if visit.holding_on_entry:
            self._logger.debug(
                "手 %s 带着丝杆进盘又离开，不判取料", hand_track_id
            )
            return
        if dwell < self.cfg.pick_confirm_s:
            # 进盘到离盘太快，多半是手从盘边划过。
            self._logger.debug(
                "手 %s 在盘内停留过短（%.2fs），不判取料",
                hand_track_id,
                dwell,
            )
            return
        if not visit.holding_verdict(keep):
            self._logger.debug(
                "手 %s 离盘后 %d 个采样点里没有见到手里的杆，不判取料",
                hand_track_id,
                len(visit.holding_samples),
            )
            return
        screw = holding
        if screw is None:
            # 定论这一帧恰好没检出杆：用占位证据，置信度记 0 便于追溯。
            screw = _holding_placeholder(hand_box)
        self._confirm_pick(
            ts=ts,
            hand_track_id=hand_track_id,
            screw=screw,
            dwell_s=dwell,
        )

    def _record_outside_holding(
        self, hand_id: int, holding: Detection | None, ts: float
    ) -> None:
        """记录一次「手在盘外、手里有没有杆」的观测。"""

        window = max(0.0, self._entry_grace_s())
        history = self._outside_holding.setdefault(hand_id, deque())
        history.append((holding is None, ts))
        while history and ts - history[0][1] > window:
            history.popleft()

    def _mostly_empty_outside(
        self, hand_id: int, ts: float, *, default_empty: bool = False
    ) -> bool:
        """进盘前**最近几次**盘外观测里，「空手」是否占多数。

        只看最近 ``pick_confirm_frames + 1`` 次观测，不用时间窗：时间窗在
        低帧率下会被放大到几秒，把上一件涂油时「手持杆」的观测也卷进来，
        于是每次进盘都被当成「带料进盘」。

        观测不足 2 次时直接采用那一次的结果——低帧率下盘外可能只采到一帧，
        此时没有更好的依据，退回单帧判断总好过一律拒绝取料。
        """

        history = self._outside_holding.get(hand_id)
        window = max(0.0, self._entry_grace_s())
        recent = (
            [(empty, t) for empty, t in history if ts - t <= window]
            if history
            else []
        )
        if not recent:
            # 这只手从未在盘外被观测到（例如第一次建轨迹就在盘内，或者
            # 太久没见已滚出时间窗）：没有依据，按调用方的保守值处理。
            # 之前这里直接返回 False（=“不是空手”），会让“空手进盘”一律
            # 被当成“带料进盘”，取料整件丢掉。
            return default_empty
        limit = max(2, int(self.cfg.pick_history_samples))
        recent = recent[-limit:]
        if len(recent) < 2:
            # 观测不足（低帧率下盘外只采到一帧）：按调用方给的保守值处理。
            # 进盘判定宁可放过（``default_empty=True``），因为漏判一件的
            # 代价高于多判一次——多出来的那次取料会因涂油未完成而进复核。
            return default_empty
        empty_count = sum(1 for empty, _ in recent if empty)
        # 平局按「空手」处理：进盘判定宁可放过一件（后续涂油判定会拦），
        # 也不要因为一两帧的杆交叠把整件取料吞掉。
        return empty_count * 2 >= len(recent)

    def _max_plausible_dt(self, current_gap: float | None = None) -> float:
        """单帧可信的最大间隔：按**已建立的节拍**的两倍计。

        掉帧时 ``dt`` 越大，累计时长越容易被凭空放大：一帧 0.95s 的间隔
        能让刷头只看过 3 帧就判完成（实测）。上限取「最近帧间隔中位数」的
        两倍，而不是配置里的 ``max_fps``——现场可能本来就只有 2~5fps，
        用配置值会把正常帧全判成不可信。还没建立节拍时，用本帧间隔自身
        作为节拍（第一帧无法区分「低帧率」和「掉帧」，按前者处理）。
        """

        base_dt = 1.0 / max(1, self.cfg.max_fps)
        if self._gap_samples:
            typical = median(self._gap_samples)
        elif current_gap is not None and current_gap > 0:
            typical = float(current_gap)
        else:
            typical = base_dt
        return 2.0 * max(typical, base_dt)

    def _entry_grace_s(self) -> float:
        """进盘前多久之内的「盘外空手」观测仍算数。

        窗口取 ``holding_lost_timeout_s``，但设 2s 的绝对上限：低帧率下
        乘上帧率倍率会长到 8s，那时「八秒前见过这只手空手」已经不足以
        说明本次是空手进盘。

        也不能取得太窄：工人从盘外走到盘内要 1~2s，窗口只有两三帧时，
        进盘那一刻永远查不到「最近空手」，真空手取料会被判成带料进盘
        而整件漏掉（示例视频第一件的取料即如此）。
        """

        return min(
            self.cfg.holding_lost_timeout_s * self._frame_scale,
            2.0,
        )

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
        hand_track_id: int,
        screw: Detection,
        dwell_s: float,
    ) -> None:
        task = self._ensure_cycle(ts, hand_track_id=hand_track_id)
        task.stage = TaskStage.PICKED
        task.stage_since = ts
        task.picked_at = ts
        task.hand_track_id = hand_track_id
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
                "hand_track_id": hand_track_id,
                "dwell_in_tray_s": round(dwell_s, 3),
                "screw_confidence": round(float(screw.confidence), 3),
            },
            start_ts=ts - dwell_s,
            end_ts=ts,
        )
        self._logger.info(
            "取料完成 %s（手部轨迹=%s，盘内停留 %.2fs）",
            task.task_id,
            hand_track_id,
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
        # 桥接空档与 dt 用同一个可信上限：原先 dt 夹到 1.0s、桥接却按
        # brush_hold_max_gap_s×scale 给到 2.0s，一次掉帧就能把 0.2s 的
        # 真实靠近凑成 1.0s 以上并直接判完成（实测）。时间不可信的帧
        # 不桥接，只累计可信的 dt。
        add = dt
        if task.last_brush_ts is not None and self._timing_reliable:
            gap = ts - task.last_brush_ts
            max_gap = min(
                self.cfg.brush_hold_max_gap_s * self._frame_scale,
                self._max_plausible_dt(),
            )
            # 加一点容差：max_gap 由乘法算出（如 0.2×2），实际 gap 恰等
            # 于它时会被浮点误差挡在门外（0.4 > 0.3999999999999999）。
            if 0 < gap <= max_gap + 1e-9:
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

        1. 优先取「与**本周期那只手**的交叠框交叠」的杆——握着的时候丝杆一定
           被手框遮住，这条约束把候选从满盘丝杆缩到手上那一根；只看本周期
           这只手，别的工人/另一只手握的杆不能把锚点抢走（实测会直接导致
           刷别处的杆也判完成）；
        2. 交叠候选里取离锚点最近的（``screw_anchor_max_step`` 按实际帧率
           缩放，但不超过 ``screw_anchor_max_step_abs``，防跳杆）；
        3. 手上没有交叠杆（手框偏了、杆被遮，或杆其实在另一只手里）时退化
           为按位置连续性取离锚点最近的杆，但**仅在锚点仍贴近最后一次确认
           握持的位置时**（``hand_screw_max_distance``）——否则锚点会在密集
           的静置丝杆之间逐帧漂走；
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
        # 绝对上限（不随帧率放大）：盘里丝杆间距只有 0.02~0.03，
        # frame_scale 顶到 4 时单帧步长会到 0.48，一跳就是相邻那根杆。
        #
        # 上限只允许**收紧**帧率放大的结果、不允许低于配置值本身：取料
        # 瞬间手带着杆快速划出，单帧位移实测 0.09~0.11，上限比这还小时
        # 锚点一步跨不过去就永久卡在盘内那根杆上，整件涂油累计为 0
        # （示例视频第一件即如此）。
        step = min(
            self.cfg.screw_anchor_max_step * self._frame_scale,
            max(
                self.cfg.screw_anchor_max_step_abs,
                self.cfg.screw_anchor_max_step,
            ),
        )
        anchor = task.screw_anchor
        hand_center = self._hand_center(task.hand_track_id)

        # 1) 握在手里的杆：与**本周期那只手**的框交叠的那些。
        cycle_hand = self._live_hand(task.hand_track_id)
        gripped: list[tuple[float, BBox]] = []
        for det in self._last_detections:
            if det.label != label:
                continue
            if cycle_hand is None or not _overlaps(
                det.box, cycle_hand.observation_box()
            ):
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

        grabbed = bool(gripped)
        if gripped and anchor is None:
            # 初始化：取交叠杆中离手中心最近的那根（握着的一定贴着这只手）。
            best = min(
                gripped, key=lambda item: _distance_to(item[1], hand_center, diag)
            )[1] if hand_center is not None else gripped[0][1]
        else:
            best = min(gripped, key=lambda item: item[0])[1] if gripped else None

        # 位置连续性只在「离最后一次确认握持的位置不远」时成立。锚点每帧
        # 都会被刷新，不加这道约束它会在密集的静置丝杆之间逐帧漂走
        # （实测漂进零件盒里的整排静置杆，手里那根反而累计不上）。
        continuity_ok = (
            task.screw_gripped_at is not None
            and anchor is not None
            and math.hypot(
                anchor[0] - task.screw_gripped_at[0],
                anchor[1] - task.screw_gripped_at[1],
            )
            / diag
            <= self.cfg.hand_screw_max_distance * self._frame_scale
        )

        # 2) 手上没有可见的杆（手框偏了/杆被遮/刷的是另一只手拿的杆）：
        #    按位置连续性找离锚点最近的杆。
        #
        # 安全性由两道约束共同保证：
        # - ``continuity_ok``：锚点仍贴「最后一次确认握持」的位置，不会在
        #   密集的静置丝杆之间逐帧漂走；
        # - ``_CONTINUITY_STEP_ABS``：单帧跨度比「握持证据」时小得多。没有
        #   握持证据却允许跨 0.15，会一步跳到另一只手握的杆上。
        #
        # 不能要求「本周期那只手不可见」才走这条分支：现场常常一只手拿杆、
        # 另一只手拿刷子，被跟踪的那只（拿刷子的那只）一直在画面里，此时
        # 杆本身才是唯一的定位依据（实测会让整件涂油累计为 0）。
        continuity_step = min(step, _CONTINUITY_STEP_ABS)
        if best is None and anchor is not None and continuity_ok:
            nearest: tuple[float, BBox] | None = None
            for det in self._last_detections:
                if det.label != label:
                    continue
                cx, cy = det.box.center
                distance = math.hypot(cx - anchor[0], cy - anchor[1]) / diag
                if distance > continuity_step:
                    continue
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, det.box)
            best = None if nearest is None else nearest[1]

        if best is not None:
            task.screw_anchor = best.center
            task.screw_anchor_ts = ts
            task.screw_observed = True
            task.last_screw_ts = ts
            if grabbed:
                task.screw_gripped_ts = ts
                task.screw_gripped_at = best.center
            return best

        # 3) hold 窗口内：按锚点位置继续（跨过一两帧漏检，不中断累计）。
        #
        # 窗口从「最后一次锚点更新」算起。加位置约束（``continuity_ok``）：
        # 锚点一旦漂离最后确认握持的位置就不再往下接，避免它在密集的静置
        # 丝杆之间逐帧漂走（实测漂进零件盒里的整排静置杆）。
        #
        # 注：杆被手完全遮住、超过窗口时按既定规则暂停累计、周期保持
        # PICKED（见 docs/05 已知限制）——现场确实会这样，属于已知取舍。
        if (
            anchor is not None
            and continuity_ok
            and task.screw_anchor_ts is not None
            and ts - task.screw_anchor_ts
            <= self.cfg.screw_anchor_hold_s * self._frame_scale
        ):
            task.screw_observed = False
            cx, cy = anchor
            return BBox(cx - 8.0, cy - 8.0, cx + 8.0, cy + 8.0)

        # 4) 锚点已失效：尝试**重新捕获**。
        #
        # 现场会这样：取完料拿着杆走出画面（去别处、被遮挡、或杆暂时检测
        # 不到），过一会儿再回来涂油。原实现在 hold 窗口过期后每帧直接返回
        # None，周期永久停在 PICKED，直到下一次取料把它判成 NG——中间回来
        # 涂了也不算。这里只要还在重新捕获时限内、且本帧又出现了「被手握着」
        # 的丝杆，就把它接成新锚点，继续累计涂油。
        reacquire = self._reacquire_anchor(task, ts)
        if reacquire is not None:
            reacquired, hand = reacquire
            gap = 0.0 if task.last_screw_ts is None else ts - task.last_screw_ts
            if hand.track_id != task.hand_track_id:
                # 换绑：本周期那只手的视觉 ID 已重建，跟着新 ID 继续判。
                self._logger.info(
                    "周期 %s 手部轨迹 %s → %s（视觉 ID 重建）",
                    task.task_id,
                    task.hand_track_id,
                    hand.track_id,
                )
                task.hand_track_id = hand.track_id
            task.screw_anchor = reacquired.center
            task.screw_anchor_ts = ts
            task.last_screw_ts = ts
            task.screw_observed = True
            if "SCREW_REACQUIRED" not in task.evidence:
                task.evidence.append("SCREW_REACQUIRED")
                self._logger.info(
                    "周期 %s 重新捕获手持丝杆（距上次观测 %.1fs）",
                    task.task_id,
                    gap,
                )
            return reacquired

        task.screw_observed = False
        return None

    def _reacquire_anchor(
        self, task: _Cycle, ts: float
    ) -> tuple[BBox, Track] | None:
        """锚点失效后重新捕获手持丝杆。

        条件是「本帧有丝杆检测与本周期那只手的框交叠」——握着的时候丝杆
        必然被手框遮住，这条约束足以把候选从满盘丝杆缩到手上那一根，不需要
        位置先验。手部轨迹已被回收时改为接最近的那只有效手。

        返回 ``(框, 手)``，调用方在换手时更新周期的 ``hand_track_id``。

        ``screw_reacquire_timeout_s=0`` 表示不限时：只要周期还活着就一直等，
        因为「多久算这一件不回来了」是工艺问题，交给下一次取料去判 NG 更准。
        """

        timeout = self.cfg.screw_reacquire_timeout_s
        if timeout > 0 and task.last_screw_ts is not None:
            if ts - task.last_screw_ts > timeout:
                return None

        # 只认**本周期那只手**握着的杆：别的工人/另一只手握着的是另一件料，
        # 接成新锚点会让刷别处的杆也算作这一件涂完了（已复现误判）。
        #
        # 例外是「本周期那只手的轨迹已经被跟踪器彻底回收」：手部视觉 ID 在
        # 遮挡/快速移动后必然重建，这时回来涂油的是同一个人，必须允许换绑。
        # 判据是「轨迹还在不在跟踪器里」而不是「本帧匹不匹配」——后者分不开
        # 「同一只手暂时漏检」和「别的工人伸手进来」。
        if self._hand_tracked(task.hand_track_id):
            hand = self._live_hand(task.hand_track_id)
            if hand is not None and task.screw_anchor is not None:
                hx, hy = hand.observation_box().center
                width, height = self._frame_size
                diag = max(1.0, math.hypot(width, height))
                if (
                    math.hypot(hx - task.screw_anchor[0], hy - task.screw_anchor[1])
                    / diag
                    > self.cfg.hand_screw_max_distance * self._frame_scale
                ):
                    # 本周期那只手还在、却已经跑到别处：这是另一只手/另一件料，
                    # 不能接成本周期的锚点（实测会直接导致刷别处的杆也判完成）。
                    return None
        else:
            # 换绑：本周期那只手的轨迹已经不在跟踪器里。这一件被拿走了很久
            # （手和杆都出过画面），回来时在哪都合理，不限制位置；但杆仍要
            # 被这只手握着（下面按「杆心落在手框内」筛）。
            hand = self._nearest_live_hand_to(task.screw_anchor)
        if hand is None:
            return None
        label = self.station.detector.screw_label
        hand_box = hand.observation_box()
        best: tuple[float, BBox] | None = None
        for det in self._last_detections:
            if det.label != label:
                continue
            # 与「持有判定」同一口径：杆心落在手框内，或两者明显交叠。
            # 只要求「有一点交叠」不够——手臂从零件盒上方经过时会蹭到盒里
            # 整排静置杆，锚点会被接过去，之后刷手里那根永远累计不上。
            gripped = (
                _box_center_in(det.box, hand_box)
                or hand_box.iou(det.box) >= 0.15
            )
            if not gripped:
                continue
            # 多根都满足时取离旧锚点最近的（离得远说明是别人手里的）。
            cx, cy = det.box.center
            score = (
                0.0
                if task.screw_anchor is None
                else math.hypot(cx - task.screw_anchor[0], cy - task.screw_anchor[1])
            )
            if best is None or score < best[0]:
                best = (score, det.box)
        if best is None:
            return None
        return best[1], hand

    def _hand_tracked(self, hand_track_id: int | None) -> bool:
        """该 id 的轨迹是否仍在跟踪器里（无论本帧是否匹配）。"""

        return (
            hand_track_id is not None
            and self._hand_tracker.get(hand_track_id) is not None
        )

    def _nearest_live_hand_to(
        self,
        anchor: tuple[float, float] | None,
        *,
        limit: float | None = None,
    ) -> Track | None:
        """离给定位置最近的本帧有效手；给了 ``limit`` 就只接受够近的。"""

        live = [
            hand
            for hand in self._hand_tracks
            if hand.matched and hand.hits >= max(1, self._hand_tracker.min_hits)
        ]
        if not live:
            return None
        if anchor is None:
            return live[0]
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))

        def distance(hand: Track) -> float:
            cx, cy = hand.observation_box().center
            return math.hypot(cx - anchor[0], cy - anchor[1]) / diag

        best = min(live, key=distance)
        if limit is not None and distance(best) > limit:
            return None
        return best

    def _live_hand(self, hand_track_id: int | None) -> Track | None:
        """本帧真正观测到、且达到 ``min_hits`` 的那只手（否则 None）。

        ``min_hits`` 之前对手部完全不生效：单帧误检就能建轨迹并写进盘状态。
        """

        if hand_track_id is None:
            return None
        hand = self._hand_tracker.get(hand_track_id)
        if hand is None or not hand.matched:
            return None
        if hand.hits < max(1, self._hand_tracker.min_hits):
            return None
        return hand

    def _hand_center(self, hand_track_id: int | None) -> tuple[float, float] | None:
        hand = self._live_hand(hand_track_id)
        return None if hand is None else hand.box.center

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
def _holding_placeholder(hand_box: BBox) -> Detection:
    """离盘帧没有杆检测时，用握手位置附近的一个占位「握持证据」。

    只在「离盘前不久已确认过握持」时才用到；占位框贴着手框下缘（手握杆
    时杆就在那个方向），置信度记 0 便于在事件里区分「实检」与「沿用」。
    """

    cx = hand_box.center[0]
    cy = hand_box.center[1] + hand_box.height * 0.5
    return Detection(
        label="",
        confidence=0.0,
        box=BBox(cx - 1.0, cy - 1.0, cx + 1.0, cy + 1.0),
    )


def _demote_for_continuation(det: Detection, cfg: PipelineConfig) -> Detection:
    """把「区外延续」用的检测压到低置信度带。

    ByteTrack 只让高置信度检测新建轨迹；把延续检测压到 ``new_thresh`` 以下
    就变成「只能接回已有轨迹，不能凭空新建」——否则区外同一片区域里的
    误检会不断长出新轨迹，而它们本来就不该被跟踪。
    """

    ceiling = cfg.bytetrack_new_thresh - 1e-3
    floor = cfg.bytetrack_low_thresh + 1e-3
    if ceiling <= floor:
        return det
    score = min(float(det.confidence), ceiling)
    if score == det.confidence:
        return det
    return Detection(
        label=det.label,
        confidence=score,
        box=det.box,
        track_id=det.track_id,
        angle=det.angle,
        point=det.point,
    )


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
