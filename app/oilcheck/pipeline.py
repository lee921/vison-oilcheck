"""取料与人工二次涂油事件判定（简化版）。

只做两条判定：

1. **取料完成**：跟踪到一只手在**未持有丝杆**的状态下进入涂油盘（手工标定的
   ``tray_roi`` 区域），随后同一只手**带着丝杆离开**涂油盘；
2. **人工二次涂油完成**：取料之后手持丝杆，刷头工作端持续在丝杆接触邻域内累计
   达到 ``brush_hold_s``。

两条判定各自发一条事件（``PICK`` / ``MANUAL_OILING``），一个周期结束后立即复位，
等待下一次取料。不做 OK/NG 分级、不做槽位跟踪、不做放料判定。

区域：

- **涂油盘区**：只用手工标定的 ``tray_roi``（必须覆盖盘体连同下方的手臂活动
  范围）。模型检出的 ``oil_station`` 框不参与判定，只供显示与排查；
- **工作追踪区**：``work_roi``，跟踪与判定的有效范围；
- **涂油区**：``oiling_roi``，标定与显示用，预留给后续涂油范围约束。

本模块不依赖 Qt、相机 SDK 或具体推理框架，测试直接回放 :class:`FrameDetections`。
"""

from __future__ import annotations

import logging
import math
from collections import deque
from statistics import median
from typing import Any

from app.review import SUMMARY_KEY, ReviewSummary

from .bytetrack_adapter import ByteTrackObjectTracker
from .config import PipelineConfig, StationConfig
from .constants import (
    _ANCHOR_SAME_SCREW_DISTANCE as _ANCHOR_SAME_SCREW_DISTANCE,
)
from .constants import (
    _BRUSH_ID_BASE as _BRUSH_ID_BASE,
)
from .constants import (
    _CARRY_FOLLOW_COS as _CARRY_FOLLOW_COS,
)
from .constants import (
    _CARRY_INHAND_FOLLOW_RATIO as _CARRY_INHAND_FOLLOW_RATIO,
)
from .constants import (
    _CARRY_INHAND_FRAMES as _CARRY_INHAND_FRAMES,
)
from .constants import (
    _CARRY_INHAND_MIN as _CARRY_INHAND_MIN,
)
from .constants import (
    _CARRY_INHAND_ROD_MOVE_MIN as _CARRY_INHAND_ROD_MOVE_MIN,
)
from .constants import (
    _CARRY_INHAND_S as _CARRY_INHAND_S,
)
from .constants import (
    _CARRY_LOCK_STEP as _CARRY_LOCK_STEP,
)
from .constants import (
    _CARRY_OUT_OF_TRAY_MAX_S as _CARRY_OUT_OF_TRAY_MAX_S,
)
from .constants import (
    _CARRY_ROD_FOLLOW_RATIO as _CARRY_ROD_FOLLOW_RATIO,
)
from .constants import (
    _CARRY_ROD_MOVE_MIN as _CARRY_ROD_MOVE_MIN,
)
from .constants import (
    _CONTINUITY_STEP_ABS as _CONTINUITY_STEP_ABS,
)
from .constants import (
    _FRAME_GAP_WINDOW as _FRAME_GAP_WINDOW,
)
from .constants import (
    _GRIP_PARKED_MIN_FRAC as _GRIP_PARKED_MIN_FRAC,
)
from .constants import (
    _HAND_ID_BASE as _HAND_ID_BASE,
)
from .constants import (
    _INVENTORY_SAME_IOU as _INVENTORY_SAME_IOU,
)
from .constants import (
    _OILING_ROD_OCCLUDED_RADIUS as _OILING_ROD_OCCLUDED_RADIUS,
)
from .constants import (
    _OILING_SAME_ROD_STEP as _OILING_SAME_ROD_STEP,
)
from .constants import (
    _OILING_STITCH_GAP_S as _OILING_STITCH_GAP_S,
)
from .constants import (
    _PICK_SAMPLE_CALIBRATION_FPS as _PICK_SAMPLE_CALIBRATION_FPS,
)
from .constants import (
    _PICK_SOURCE_HAND as _PICK_SOURCE_HAND,
)
from .constants import (
    _PICK_SOURCE_SCREW_EXIT as _PICK_SOURCE_SCREW_EXIT,
)
from .constants import (
    _ROD_MIN_ASPECT as _ROD_MIN_ASPECT,
)
from .constants import (
    _ROI_TRACK_LIMIT_ABS as _ROI_TRACK_LIMIT_ABS,
)
from .constants import (
    _RULE_TRUE_VETO_P as _RULE_TRUE_VETO_P,
)
from .constants import (
    _SCREW_EXIT_IDLE_RELEASE_S as _SCREW_EXIT_IDLE_RELEASE_S,
)
from .constants import (
    _SCREW_EXIT_MIN_FRAMES as _SCREW_EXIT_MIN_FRAMES,
)
from .constants import (
    _SCREW_EXIT_MIN_OUTSIDE_S as _SCREW_EXIT_MIN_OUTSIDE_S,
)
from .constants import (
    _SCREW_HISTORY_MAX as _SCREW_HISTORY_MAX,
)
from .constants import (
    _SCREW_HISTORY_S as _SCREW_HISTORY_S,
)
from .constants import (
    _SCREW_STATIC_BACK as _SCREW_STATIC_BACK,
)
from .constants import (
    _SCREW_STATIC_IOU as _SCREW_STATIC_IOU,
)
from .constants import (
    _SCREW_STATIC_LOOKBACK_S as _SCREW_STATIC_LOOKBACK_S,
)
from .constants import (
    _SCREW_TRACK_MISS as _SCREW_TRACK_MISS,
)
from .geometry import (
    _box_center_in as _box_center_in,
)
from .geometry import (
    _exit_depth as _exit_depth,
)
from .geometry import (
    _label_counts as _label_counts,
)
from .geometry import (
    _overlaps as _overlaps,
)
from .judge import Episode, EpisodeSegmenter, LogisticJudge
from .raw_features import (
    RawFrame,
)
from .review import (
    build_oiling_missing_summary,
)
from .rules.oiling import OilingRules
from .rules.pick_hand import HandPickRules
from .rules.pick_screw import ScrewExitRules
from .state import (
    _Cycle,
    _HandVisit,
    _OilingSession,
    _ScrewExit,
)
from .tracker import ObjectTracker, Track
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
)

EventCallback = Any  # Callable[[EventRecord], None]，避免循环导入


class StationPipeline(HandPickRules, ScrewExitRules, OilingRules):
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

        # 每只手的进盘过程；一只手下同一时间只会有一段。
        self._visits: dict[int, _HandVisit] = {}
        # 手在盘外时的持握**历史**，进盘判定用：值为 (是否空手, 时刻) 的
        # 队列，只保留时间窗口内（``_entry_grace_s``）的观测。进盘时看空手
        # 帧占比，而不是「最近一帧」——单帧既可能是手刚伸到盘边与静置杆
        # 交叠，也可能是手正把杆递进来，都不能定性。
        self._outside_holding: dict[int, deque[tuple[bool, float]]] = {}
        # 每只手最近一次记录盘外观测的时间戳，用于同帧去重。
        self._outside_holding_last_ts: dict[int, float] = {}
        self._cycles: dict[str, _Cycle] = {}
        self._task_seq = 0
        # 当前正在累计的涂油接触段（见 ``_OilingSession``）。与取料并行：
        # 工人常常一边取料一边就刷上了，不能等取料周期建立再开始判涂油。
        self._active_oiling: _OilingSession | None = None
        # 没有对应取料周期的涂油，事件键按段递增。
        self._free_oiling_seq = 0
        # 正在上报那一次涂油的接触起点（``_complete_oiling`` 组事件用）。
        self._oiling_start_ts: float | None = None
        # 已经判完、但当时还没有取料周期可挂的涂油（「一边取料一边刷」）。
        # 取料确认时如果就是这根杆，补挂到那条周期上，而不是当成无主涂油。
        self._pending_oilings: list[tuple[_OilingSession, float, Episode | None]] = []

        self._last_frame_ts: float | None = None
        self._frame_size: tuple[int, int] = (0, 0)
        self._last_detections: tuple[Detection, ...] = ()
        # 最近一小段时间的丝杆检测框，用于区分「盘里的静置杆」与「被手带走的
        # 杆」（见 ``_screw_parked_in_tray``）。每帧一份，带时间戳，按时间窗口
        # 裁剪——按帧数保留会让「0.53s 前」这个对照窗口随帧率变化。
        self._screw_history: deque[tuple[float, tuple[BBox, ...]]] = deque(
            maxlen=_SCREW_HISTORY_MAX
        )
        # 第二条取料判据：每根杆一条「在盘里待过 → 出盘」的记录
        # （见 ``_ScrewExit``）。键是按位置连续性维护的短时杆 ID。
        self._screw_exits: dict[int, _ScrewExit] = {}
        self._screw_exit_seq = 0
        # 最近一次通过本判据发出的取料时刻与位置，用于与手部判据去重。
        # 最近一次确认的取料（时刻与杆心），两条判据共用：同一件料被两条
        # 判据几乎同时判出时只报一次（见 ``_duplicate_pick``）。
        self._last_pick_ts: float | None = None
        self._last_pick_center: tuple[float, float] | None = None
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

        # 二级判定（逻辑回归）。没有权重文件时两者都是 None，判定退化成
        # 纯规则，行为与以前完全一致。见 app/oilcheck/judge.py。
        self._pick_judge: LogisticJudge | None = None
        self._oiling_judge: LogisticJudge | None = None
        # 逐帧概率分段器：**一段动作一个**（每趟进盘 / 每次刷涂），键是
        # (``"pick"``, 手部轨迹 ID) 或 (``"oiling"``, 涂油段的 id())。
        # 共用一个大分段器会把相邻两件的动作并成一段。
        self._segmenters: dict[tuple[str, int], EpisodeSegmenter] = {}
        # 已收段的结果：(类型, 归属键, 段)。归属键用于把段配给正确的事件。
        self._judge_episodes: list[tuple[str, int, Episode]] = []
        # 完整动作区间（涂油事件发出后，等这一段刷完才得到的整段起止）。
        self._last_full_episodes: list[tuple[str, Episode]] = []
        # 训练钩子：tools/train_judge.py 用来把特征与标签落盘。
        # 生产路径不设，零开销。
        self._training_pick_hook = None
        self._training_oiling_hook = None

    def attach_judges(
        self, pick: LogisticJudge | None, oiling: LogisticJudge | None
    ) -> None:
        """挂上二级判定模型（``None`` = 不启用）。"""

        self._pick_judge = pick
        self._oiling_judge = oiling
        self._segmenters.clear()
        self._judge_episodes.clear()
        self._last_full_episodes.clear()

    def set_training_hooks(self, *, pick=None, oiling=None) -> None:
        """设置训练用的特征回调 ``callable(values: dict, ts: float)``。"""

        self._training_pick_hook = pick
        self._training_oiling_hook = oiling

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
        self._outside_holding_last_ts.clear()
        self._active_oiling = None
        self._pending_oilings.clear()
        self._cycles.clear()
        # 任务序号**不归零**：重置后下一件若又叫 TASK-000001，PLC 侧的
        # hold_done_task_id 会把它当成「这一件结论已报完」，整件在 PLC 上
        # 没有记录；事件按 (cycle_id, task_id) 追溯也会撞车。序号只在
        # 进程启动时从 1 开始。
        self._hand_tracker.reset()
        self._brush_tracker.reset()
        self._last_frame_ts = None
        self._frame_size = (0, 0)
        self._last_detections = ()
        self._screw_history.clear()
        self._screw_exits.clear()
        self._screw_exit_seq = 0
        self._last_pick_ts = None
        self._last_pick_center = None
        self._hand_tracks = ()
        self._brush_tracks = ()
        self._frame_gap_events = 0
        self._last_frame_gap_s = 0.0
        self._gap_samples.clear()
        self._frame_scale = 1.0
        self._timing_reliable = True
        # 分段器随判定状态一起复位：上一段视频的进行中动作不能带进来。
        self._segmenters.clear()
        self._judge_episodes.clear()
        self._last_full_episodes.clear()

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

        # 涂油盘区完全来自手工 tray_roi，与模型检出的盘体框无关：盘体检测不再
        # 参与任何判定，判定结果因此可复现（同一段视频重跑结论一致）。
        tray_box = self._tray_region_box(frame_size)
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
        # 刷头出生准入：刷子干活时必然在手边或丝杆边，远离两者的「刷头」
        # 基本都是白色毛巾/导轨误检。只压制**新建轨迹**，已有轨迹的延续不受
        # 影响（分数压到 new_thresh 以下，仍可走第二阶段接回）。
        brushes = self._demote_isolated_brushes(
            brushes=brushes,
            hands=hands,
            screws=detections.by_label(cfg.screw_label),
            frame_size=frame_size,
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

        self._update_pick(ts=ts, dt=dt, tray_box=tray_box)
        # 第二条取料判据：丝杆自己从盘内移到盘外也算取料。放在手部判据之后，
        # 这样同一帧里两条都成立时，去重能看见手部判据刚发出的那一次。
        self._update_screw_exit_pick(ts=ts, dt=dt, tray_box=tray_box)
        # 逐帧给逻辑回归打分并分段，得到「取料/涂油从什么时候到什么时候」。
        # 放在规则判定之后：分段器用的是本帧已经算好的证据。
        self._update_judge_episodes(ts=ts)
        # 本帧的丝杆框在**判定之后**入库：``_screw_history`` 表示「此前」的
        # 位置，静置杆判据拿它当对照，多入一帧会整体错位。
        self._screw_history.append(
            (
                ts,
                tuple(
                    det.box
                    for det in detections.detections
                    if det.label == cfg.screw_label
                ),
            )
        )
        # 只保留最近 ``_SCREW_HISTORY_S`` 秒，但至少留两帧：低帧率下每帧
        # 间隔可能比这个时长还长，按时间硬裁会只剩一帧，「0.53 秒前的位置」
        # 这个对照就没法算了（静置判据随之失效，晃动的杆会被当成正在带走）。
        while (
            len(self._screw_history) > 2
            and ts - self._screw_history[0][0] > _SCREW_HISTORY_S
        ):
            self._screw_history.popleft()
        # 复核统计：处于 PICKED 阶段的周期，每帧都算一次「取料后经过的帧数」。
        # 放在 _update_oiling 之前，杆不可见而提前返回的帧也能计入。
        for task in self._cycles.values():
            if task.active and task.stage == TaskStage.PICKED:
                task.oiling_frames += 1
        self._update_oiling(ts=ts, dt=dt)
        # 超时检查放在涂油判定之后：本帧刚好凑够 brush_hold_s 的，先算完成。
        self._check_oiling_timeout(ts=ts)
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
            tracks=self._track_snapshots(),
            detections=self._display_detections(),
            frame_size=self._frame_size,
            updated_at=self._last_frame_ts or 0.0,
            stats={
                "frames_processed": self._frames_processed,
                "events_emitted": self._events_emitted,
                "tasks": len(self._cycles),
                "active_task_id": active.task_id if active else "",
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
        tray_box = self._tray_region_box(self._frame_size)
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

    def _demote_isolated_brushes(
        self,
        *,
        brushes: tuple[Detection, ...],
        hands: tuple[Detection, ...],
        screws: tuple[Detection, ...],
        frame_size: tuple[int, int],
    ) -> tuple[Detection, ...]:
        """把离手与丝杆都太远、注定是误检的刷头检测压到低置信度带。

        ``brush_spawn_max_distance=0`` 时原样返回。锚点用**当帧检测**而不是
        轨迹，避免漏检帧把真刷子也判成孤立：刷子抬手时手一定同时在画面里。
        """

        limit = float(self.cfg.brush_spawn_max_distance)
        if limit <= 0 or not brushes:
            return brushes
        anchors = [
            det.box.center for det in hands
        ] + [det.box.center for det in screws]
        if not anchors:
            return brushes
        width, height = frame_size
        diag = max(1.0, math.hypot(width, height))
        kept: list[Detection] = []
        for det in brushes:
            cx, cy = det.box.center
            near = any(
                math.hypot(cx - ax, cy - ay) / diag <= limit for ax, ay in anchors
            )
            kept.append(det if near else _demote_for_continuation(det, self.cfg))
        return tuple(kept)

    def _raw_frame(self, ts: float) -> RawFrame:
        """把当前这一帧的**原始检测框**打包给特征提取用。

        只交检测框、标定区域与丝杆历史。规则的中间结论（``_HandVisit`` /
        ``_CarryEvidence`` / ``_OilingSession``）一律不进来——模型的输入
        必须能从原始素材复算，训练脚本才采得到同样的特征。
        """

        width, height = self._frame_size
        station = self.station
        labels = station.detector
        return RawFrame(
            ts=ts,
            width=width,
            height=height,
            screws=[
                det for det in self._last_detections
                if det.label == labels.screw_label
            ],
            hands=[
                (track.track_id, track.observation_box())
                for track in self._hand_tracks
                if track.matched
            ],
            brushes=[
                (track.track_id, track.observation_box())
                for track in self._brush_tracks
                if track.matched
            ],
            tray=self._tray_region_box((width, height)),
            work=station.work_roi.pixel_bbox(width, height),
            oiling=station.oiling_roi.pixel_bbox(width, height),
            screw_history=[(entry_ts, list(boxes)) for entry_ts, boxes in self._screw_history],
            detector=labels,
        )

    def _tray_region_box(self, frame_size: tuple[int, int]) -> BBox | None:
        """涂油盘区：**只用**手工标定的 ``tray_roi``。

        盘区必须覆盖盘体**连同下方的手臂活动范围**。模型检出的 ``oil_station``
        框只框得住盘体本身（示例①里框底 y≈372），而工人取料时手在盘体下方
        100~280px（y≈422~654）——按检测框判「进盘」会把取料全部漏掉。

        以前这里还与 ``oil_station`` 框取并集、跟随盘体上下移动；那样做的问题
        是盘区随模型输出浮动：盘体漏检时预测框外推、框抖动都直接改变「在不在
        盘内」的判定，同一段动作在盘体移动的帧上会被判成「已离盘」。现在盘区
        完全由标定决定，判定结果只取决于手的位置，可复现、可回放对比；盘体
        上下移动靠 ``tray_roi`` 本身框够余量来吸收。

        """

        width, height = frame_size
        if width <= 0 or height <= 0:
            return None
        return self.station.tray_roi.pixel_bbox(width, height)

    # ------------------------------------------------------------------
    # 周期管理
    # ------------------------------------------------------------------
    def _check_oiling_timeout(self, *, ts: float) -> None:
        """取料后超时仍未完成二次涂油 ⇒ 判漏涂 NG。

        ``oiling_timeout_s = 0`` 时不启用，行为与原来一致：只由下一次取料
        触发 ``OILING_MISSING``。

        计时用媒体时间（``task.picked_at`` 到本帧 ts），与取料/涂油判定同一
        口径：慢放快放不改变结果，视频结束也不会凭空判一件、下一件全部 NG。

        超时判 NG 之后周期结束、``_cycles`` 里没有活动周期，之后真来了一次
        涂油不会被计入这一件（它已经判过 NG 了）。这是取舍：超时那一刻画面
        里已经没有正在刷的证据，再等待就失去了「及时发现漏涂」的意义。
        """

        if self.cfg.oiling_timeout_s <= 0:
            return
        task = next(
            (
                item
                for item in self._cycles.values()
                if item.active and item.stage == TaskStage.PICKED
            ),
            None,
        )
        if task is None or task.picked_at is None:
            return
        if ts - task.picked_at <= self.cfg.oiling_timeout_s:
            return
        self._supersede_unfinished(task, ts, trigger="oiling_timeout")

    def _supersede_unfinished(
        self,
        task: _Cycle,
        ts: float,
        *,
        trigger: str = "superseded_by_pick",
    ) -> None:
        """只取料、没完成二次涂油 ⇒ ``NG``。

        两个触发点，判的是同一件事（这一件没刷完），只是发现方式不同：

        - ``superseded_by_pick``：来了下一次取料。现场工艺里一件取走后没
          刷完就取下一件，就是漏涂；
        - ``oiling_timeout``：取料后超过 ``oiling_timeout_s`` 仍未完成，
          不再等下一次取料（``_check_oiling_timeout``）。

        周期立即结束，事件类型 ``OILING_MISSING``，结果 ``NEEDS_REVIEW``
        交给人工复核。
        """

        if task.stage != TaskStage.PICKED or task.picked_at is None:
            # 还没取料的周期不会漏涂，直接丢弃（当前流程不会出现，留作防御）。
            self._cycles.pop(task.task_id, None)
            return
        timed_out = trigger == "oiling_timeout"
        # 复核统计：收尾当前连续接触段，摘要里的「最长连续接触」才是完整的。
        self._close_brush_run(task)
        task.stage_since = ts
        task.completed_at = ts
        task.final_result = TaskResult.NG
        task.reason_code = (
            "OILING_MISSING_TIMEOUT" if timed_out else "OILING_MISSING_SUPERSEDED"
        )
        task.evidence.append(
            "MANUAL_OILING_TIMEOUT" if timed_out else "NEXT_PICK_BEFORE_MANUAL_OILING"
        )
        task.timeline.append(("NG", ts))
        extra: dict[str, Any] = {
            "hand_track_id": task.hand_track_id,
            "brush_hold_s": round(task.brush_hold_s, 3),
            "required_s": self.cfg.brush_hold_s,
            "picked_at": task.picked_at,
        }
        if timed_out:
            extra["timeout_s"] = self.cfg.oiling_timeout_s
            extra["elapsed_s"] = round(ts - (task.picked_at or ts), 3)
        else:
            extra["superseded_at"] = ts
        self._emit_event(
            task,
            event_type=EventType.OILING_MISSING,
            reason=(
                "oiling_missing_timeout"
                if timed_out
                else "oiling_missing_superseded_by_pick"
            ),
            result=EventResult.NEEDS_REVIEW,
            extra=extra,
            start_ts=task.picked_at,
            end_ts=ts,
            review=build_oiling_missing_summary(
                task,
                trigger="oiling_timeout" if timed_out else "superseded_by_pick",
                ts=ts,
                cfg=self.cfg,
            ),
        )
        if timed_out:
            self._logger.info(
                "周期 %s 取料后 %.1fs 仍未完成二次涂油（累计 %.2fs/%.2fs，"
                "超时 %.1fs），判 NG",
                task.task_id,
                extra["elapsed_s"],
                task.brush_hold_s,
                self.cfg.brush_hold_s,
                self.cfg.oiling_timeout_s,
            )
        else:
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
        """回收已结束的周期，只保留最近几条。

        保留策略是**按条数**而不是按时间：监控页的任务卡片要一直显示最近
        几件的结果，按时间删（旧的 ``completed_retain_s=10``）会让卡片栏在
        每件做完 10 秒后就空掉，现场看不到刚才那几件的结论。

        - ``completed_keep_count``：已结束周期保留最近 N 条（默认 4，与卡片
          栏格数一致）；
        - ``completed_retain_s``：兼容项，>0 时额外限制「结束后超过该时长
          也回收」，默认 0 表示不按时间清。

        活动周期永远不在这里回收：它要么由判定流程结束，要么由 ``reset``
        清空。``task_max_active`` 只用于给活动周期兜底，不参与结束周期。
        """

        finished = [
            task
            for task in self._cycles.values()
            if not task.active
        ]
        # 1) 兼容项：按时间回收（默认关闭）。
        if self.cfg.completed_retain_s > 0:
            for task in list(finished):
                end = task.completed_at or task.stage_since
                if ts - end >= self.cfg.completed_retain_s:
                    self._cycles.pop(task.task_id, None)
            finished = [t for t in finished if t.task_id in self._cycles]

        # 2) 按条数保留最近 N 条已结束周期。
        keep = max(1, int(self.cfg.completed_keep_count))
        if len(finished) > keep:
            finished.sort(
                key=lambda item: (
                    item.completed_at or item.stage_since,
                    item.created_at,
                )
            )
            for task in finished[: len(finished) - keep]:
                self._cycles.pop(task.task_id, None)

        # 3) 活动周期兜底：正常情况下同时只有一个，留点余量防异常。
        active = [task for task in self._cycles.values() if task.active]
        limit = max(1, int(self.cfg.task_max_active))
        if len(active) > limit:
            active.sort(key=lambda item: item.created_at)
            for task in active[: len(active) - limit]:
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
        review: ReviewSummary | None = None,
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
        # 通用复核摘要：复核页按这个结构渲染，不认识任何 SOP 专有键名。
        # 放在 extra 之后，避免被同名的 extra 键覆盖。
        if review is not None:
            evidence[SUMMARY_KEY] = review.to_dict()
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


