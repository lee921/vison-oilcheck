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
from dataclasses import dataclass, field
from statistics import median
from typing import Any

from app.review import SUMMARY_KEY, ReviewSummary

from .association import brush_work_point
from .bytetrack_adapter import ByteTrackObjectTracker
from .config import PipelineConfig, StationConfig
from .review import (
    build_oiling_missing_summary,
    build_oiling_summary,
    build_pick_summary,
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

# 手/刷头的 track id 命名空间，方便日志与 UI 区分。
_HAND_ID_BASE = 100_000
_BRUSH_ID_BASE = 200_000

# 帧间隔鲁棒估计的样本数。单帧长间隔（视频跳帧、推理卡顿）不应该改变
# 距离门限，所以用中位数而不是「上一帧的间隔」。
_FRAME_GAP_WINDOW = 15

# 区外延续跟踪的硬上限（归一化），不随 frame_scale 放大。
_ROI_TRACK_LIMIT_ABS = 0.10

# 「盘里的静置杆」判定窗口与门限。
#
# 盘内那排丝杆是**位置长期不动**的；被手带走的杆即使在中心还没离开盘区时，
# 框也已经在动。只按「杆心在不在盘区」排除静置杆会漏判：手工 tray_roi 覆盖
# 了盘体下方的手臂活动范围，取料瞬间手里那根杆的中心往往还落在盘区内
# （示例④ 第 2、3 次取料的整段离盘初期都是这样），于是被当成静置杆丢掉，
# 整件取料判不出来。
#
# 实测四段示例视频：静置杆与 _SCREW_STATIC_BACK 帧前的位置 IoU 在 0.5 以上
# （p50 0.9+），被带走的杆 0.3 以下；分界清晰，取 0.5。
_SCREW_STATIC_BACK = 8
_SCREW_STATIC_IOU = 0.5
_SCREW_HISTORY_MAX = _SCREW_STATIC_BACK + 2

# 手已出盘、手里那根杆还没出盘时，最多等这么久（按实际帧率缩放）。
#
# 取料的定义是「带杆**出**盘」，杆心在盘区内的读数是无效的，既不能进采样
# 窗口，也不能用来结算。这段时间要按「杆出盘」算：现场实测杆心比手晚出门
# 一秒上下是常态，示例⑤ 第 2 次取料更是在盘区下沿一边刷一边把杆往下带，
# 手出盘 6.53s、杆出盘 8.73s，隔了 2.2s。而按离盘后的固定结算时长
# （0.4s）计时，示例⑤ 会在杆出盘之前就按空手定论、整件取料丢掉。
#
# 上限只作兜底：正常情况下「还握着并带着杆」这个条件本身会先消失（手离开
# 那根杆、或杆出了盘），不会真等满。示例① 第 88~103 帧那种盘边点按的
# 假动作，手一撤这条就断，几帧内就按空手结算。
_CARRY_OUT_OF_TRAY_MAX_S = 5.0

# 判断「这根杆是不是进盘时那批存货」的 IoU 门限。
_INVENTORY_SAME_IOU = 0.5

# 离盘后「手边有盘内杆」要连续多少帧，才认作真的握着这根杆往盘外带。
#
# 示例⑥ 第 2 件连续 8 帧以上都握着同一根杆往外走；示例① 工人按夹具、
# 示例⑧ 工人把杆装上夹具，手只是从杆旁边划过，只有 1~2 帧命中。
# 取 4 帧（15fps 下约 0.27s），既能排除「划过」，又不至于要求太久。
_CARRY_CONFIRM_FRAMES = 4


# 对**位置没动过**的杆，握持要求杆有足够面积落在手框右上区（实测盘边蹭到的
# 静置杆 0.25~0.42，真正攥住的 ≥0.44）；对已经在动的杆，只要与右上区相交
# 即算——它不可能是静置杆。
_GRIP_PARKED_MIN_FRAC = 0.50

# 「纯位置连续性」跟进手持丝杆的单帧位移上限（不随帧率放大）。
#
# 与「杆与本周期那只手交叠」的证据强度不同：握持是强证据，取料瞬间手带杆
# 快速划出，单帧位移可达 0.11；纯位置连续性只是「上一帧在这、这一帧还在
# 附近」，允许的跨度必须小得多，否则锚点会一步跳到旁边另一根杆上（实测
# 在 0.15 的跨度下会被另一只手的杆抢走）。
_CONTINUITY_STEP_ABS = 0.08

# 「同一根杆」的判定距离（归一化）：位置没动过的杆只有在离锚点这么近时才
# 仍被当成锚点自己那根。盘里相邻丝杆间距 0.02~0.03，取 0.02 以内的重叠
# 不会把旁边的静置杆误认成锚点（见 ``_static_anchor_ok``）。
_ANCHOR_SAME_SCREW_DISTANCE = 0.02

# 「涂油还是同一根杆」的单帧位移容差（归一化，不随帧率放大）。
#
# 刷的时候手一直在动，杆框会随之晃动：实测同一根杆相邻帧的位置跳变在
# 0.03 以内，换一根杆则是 0.1 以上（盘里相邻杆间距 0.02~0.03，但工人
# 刷完一件换下一件要抬起来重新放，跨度远大于这个）。取 0.06 落在两者之间。
_OILING_SAME_ROD_STEP = 0.06

# 刷手抬开多久仍算同一次涂油（秒，按帧率放大）。
#
# 现场同一件料中途抬手的空档实测 0.5~1.3 s，不同料之间最短 1.07 s；
# 单靠时间分不开，所以这个上限只用来兜底「长时间不碰就该收段」，
# 真正的切分由杆的位置决定（见 ``_same_oiling_rod``）。
_OILING_STITCH_GAP_S = 1.6


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
    # 第一次见到这只手的时刻。只用于日志与排查，**不是**「盘内停留」的起点：
    # 手可能在盘外被连续跟踪很久才蹭过盘边。
    entered_tray_ts: float
    # 逐帧累计的「盘内时长」：只在 ``inside_raw`` 为真的帧上加上该帧的 dt。
    # 之前用 ``ts - entered_tray_ts`` 当停留时长，而起点是首次见到手的时刻，
    # 盘外跟踪得越久 dwell 越大，「盘边划过不判取料」的守卫被撑得越松
    # （实测盘外跟踪 3s 后只蹭过 1 帧也判成取料）。
    # 进盘时手里是否已经拿着丝杆。取料只认“空手进、带料出”。
    #
    # 不能在进盘当帧判：手一伸进盘，手框就与盘内静置的丝杆交叠，当帧判定
    # 永远是“持有”，取料会被全部吞掉（现场视频 140911 即如此）。因此按
    # **盘外观测历史**投票判定。
    holding_on_entry: bool
    # 本趟**第一次**观测到「手在盘区内」的时刻，即「盘内停留」的起点。
    # 用「首次进盘 → 结算」的区间，而不是逐帧累加 inside 帧的 dt：
    # 手在盘内时可能整段漏检（示例①第 3 次取料只有 4 帧中心落在盘区，
    # 逐帧累加只有 0.27s < pick_confirm_s，真实取料被误杀）。区间度量
    # 同时满足「盘外跟踪再久也不算进盘停留」——起点是进盘那一帧。
    first_inside_ts: float | None = None
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
    # 离盘后「手边有盘内杆」的连续帧数。攒够 ``_CARRY_CONFIRM_FRAMES`` 才
    # 认作真正握着往盘外带走（见 ``_update_pick`` 的 carry 分支）。
    carry_frames: int = 0
    # 是否已确认「进过盘」。一旦置真，这一段进盘过程就成立，直到离盘判定。
    entered_confirmed: bool = False
    # 逐帧累计的盘内停留时长（秒）。保留用于排查；判定用的是
    # ``first_inside_ts`` 到结算时刻的区间（见字段注释）。
    dwell_in_tray_s: float = 0.0
    # 这一段进盘过程中途接过几次手部轨迹（视觉 ID 重建）。只用于排查与日志。
    rebinds: int = 0
    # 进盘前一帧盘区里那批丝杆的框（存货快照）。
    # 用来区分「手里正带着走的那根」与「盘里的存货」：存货在这趟进盘前后
    # 一直在盘里；手上那根是**新进入盘区**的。只看位置稳不稳不够——工人
    # 把杆停在盘区下沿歇一下（示例⑤ 第 2 次取料停了 2s 多），位置就不再
    # 变化，会被 ``_screw_static`` 误当成存货，整件取料丢掉。
    entry_inventory: tuple[BBox, ...] = ()

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


# 握持判据里「手框右上角区域」的取法：右边 50%、上边 50% 的那一块。
# 实测拳头攥住杆身、杆从虎口（右上角）伸出去，杆框与这块区域明显相交；
# 而盘边蹭到的静置杆只在手框边缘有一点点交叠。
_GRIP_CORNER_RX = 0.5
_GRIP_CORNER_RY = 0.5
# 杆框至少要有多大比例落在右上角区域里才算「握着」。丝杆又细又长，
# 斜着握时与右上区只有部分重叠；示例③握持段 0.44~0.68，空手在盘边蹭到的
# 静置杆 0.25~0.42。这个值现在是 ``_gripped_by_corner`` 的默认门限，
# 调用方按杆的状态另选（见 ``_GRIP_PARKED_MIN_FRAC`` 与 ``min_frac=0``）。
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
    # ---- 复核用的统计量（只读埋点，不参与判定）----
    # 刷头工作点落进接触邻域内的帧数：直观区分「偶尔蹭到」与「一直在刷」。
    brush_near_frames: int = 0
    # 本周期内刷头工作点到杆心的最小归一化距离；从未靠近过为 None。
    brush_min_distance: float | None = None
    # 最长**连续**接触时长。断续靠近时它远小于 brush_hold_s 累计值。
    brush_longest_run_s: float = 0.0
    # 当前连续接触段的起点，以及段内最后一次「刷头靠近」的时刻（内部用）。
    # 用后者而不是「收尾那帧的 ts」算段长：杆不可见时会提前 return，
    # 拿收尾帧的 ts 会把看不见杆的时间也算进连续段里。
    brush_run_start_ts: float | None = None
    brush_last_near_ts: float | None = None
    # 手持丝杆被定位到的帧数（分子）与取料后经过的帧数（分母），
    # 两者之比即「手持丝杆可见比例」。分母用**全部**帧而不是「涂油判定
    # 阶段的帧」——杆不可见时 ``_update_oiling`` 会提前返回，那些帧根本
    # 不会进入涂油阶段，用后者算出来的比例永远是 100%。
    tap_frames: int = 0
    oiling_frames: int = 0
    timeline: list[tuple[str, float]] = field(default_factory=list)
    events_sent: set[str] = field(default_factory=set)

    @property
    def active(self) -> bool:
        return self.final_result == TaskResult.PROCESSING


@dataclass
@dataclass
class _OilingSession:
    """一段连续接触：刷头工作点持续贴着**同一根**手持丝杆。

    涂油的边界由杆决定，不由时间决定。现场同一件料上刷手会抬开 0.5~1.3 s
    再放回来，不同料之间最短只有 1.07 s——时间上分不开，位置能分开。
    """

    anchor: tuple[float, float] | None = None
    start_ts: float | None = None
    # 上一帧的处理时刻（用来算空档）。
    last_ts: float | None = None
    # 最近一次「确实贴着刷」的时刻：判断空档该不该桥接用这个。
    last_seen_ts: float | None = None
    cum_s: float = 0.0
    frames: int = 0
    min_distance: float = 1.0
    brush_track_id: int | None = None
    # 由几个接触片段拼成（中途抬手又放回来会加一）。
    parts: int = 1
    reported_ts: float | None = None


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
        self._pending_oilings: list[tuple[_OilingSession, float]] = []

        self._last_frame_ts: float | None = None
        self._frame_size: tuple[int, int] = (0, 0)
        self._last_detections: tuple[Detection, ...] = ()
        # 最近若干帧的丝杆检测框，用于区分「盘里的静置杆」与「被手带走的杆」
        # （见 ``_screw_parked_in_tray``）。每帧一份，按帧序追加。
        self._screw_history: deque[tuple[BBox, ...]] = deque(
            maxlen=_SCREW_HISTORY_MAX
        )
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
        # 本帧的丝杆框在**判定之后**入库：``_screw_history`` 表示「此前若干
        # 帧」的位置，静置杆判据拿 [-BACK] 与之比较，多入一帧会整体错位。
        self._screw_history.append(
            tuple(
                det.box
                for det in detections.detections
                if det.label == cfg.screw_label
            )
        )
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
    # 取料判定
    # ------------------------------------------------------------------
    def _update_pick(self, *, ts: float, dt: float, tray_box: BBox | None) -> None:
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
        # 先把「手部 ID 在进盘途中被重建」的过程转交给真正接手的那只手。
        self._hand_over_split_visits(tray_box, ts)
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
            # 记录「本趟第一次进盘」。起点不取首次见到手的时刻（那会把盘外
            # 跟踪时长算进去），也不逐帧累加盘内帧（手在盘内整段漏检时
            # 会严重低估，示例①第 3 次取料只有 4 帧命中盘区）。
            if inside_raw:
                if visit.first_inside_ts is None:
                    visit.first_inside_ts = ts
                visit.dwell_in_tray_s += max(0.0, dt)
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
                # 手已出盘、手里的杆还没出盘的那几帧**既不采样也不定论**。
                #
                # 杆心比手晚出盘是常态：示例④ 第 3 次取料手在 f348 出盘、
                # 杆在 f352 才出盘，中间 4 帧按「杆必须出盘」都是「手里没杆」。
                # 把这几帧记成 False，窗口 [F,F,T] 多数票就是「空手」，
                # 整件取料丢掉；掉一帧窗口错位，结论更随机。跳过这几帧后，
                # 采样窗口只包含杆真正出盘之后的读数。
                #
                # 也不能让这几帧照常走「攒够就定论 / 超时就定论」：示例⑤ 第 2、
                # 3 次取料手在 6.53s / 13.13s 出盘，手里那根杆一直在盘区下沿
                # 慢慢往下带，杆心出盘要 1s 以上（第 2 次 2.2s）。按离盘后的
                # 固定时长计算，0.4s 的结算期一到，采样窗口还是空的、「见过
                # 没见过手握杆」也还是假的，于是当场按空手定论，整件取料丢掉。
                # 等待的时限按 ``_carry_wait_s()``，不能用离盘后的固定时长。
                #
                # 「离盘」本身由采样窗口确认（可能比手真正出盘晚一两帧，
                # 所以这一段在 exited 之后才判得到）。判据是「手边有随手动、
                # 但杆心仍在盘区内的杆」——那是这一件正在出门，不是空手离开；
                # 进盘时就在盘里的存货不算（``entry_inventory``）。
                if self._carrying_out_of_tray(hand_box, visit.entry_inventory):
                    if ts - visit.exited_at_ts <= self._carry_wait_s():
                        # 「手边有随手动、杆心还在盘区内」是**带杆出门**的
                        # 暂定证据：工人把杆停在盘区下沿刷完再带走时，杆心一
                        # 直没出盘，holding_now 全程为假，不记这一条整件取料
                        # 会丢（示例⑥ 第 2 件 10.07s、示例⑦ 第 2/3 件）。
                        #
                        # 但「手在盘外 + 手边有盘内杆」本身不够：手框很大，
                        # 盘里/夹具上的杆都会落进来。示例① 工人按夹具、示例⑧
                        # 工人把杆装上夹具，都会被凭空判成一次取料。
                        #
                        # 差别在**持续握住同一根**：示例⑥ 第 2 件连续 8 帧
                        # 手边都是同一根杆、并一路把它带出盘；示例① 只在
                        # 手划过的 2 帧里碰到杆。所以这里按连续帧数认定。
                        visit.carry_frames += 1
                        if visit.carry_frames >= _CARRY_CONFIRM_FRAMES:
                            visit.holding_seen = True
                        # 等杆出盘的这段里，手确实在盘外：照常记「盘外空手」
                        # 历史，供**下一次**进盘定性。旧实现这几帧已经结算、
                        # 重建了进盘过程，每帧都在记；现在过程还挂着，不补
                        # 记录就断了窗口（示例③在 7.5fps 下下一件被误判成
                        # 「带料进盘」而漏掉）。
                        if due:
                            self._record_outside_holding(
                                hand.track_id, holding, ts
                            )
                        continue
                    # 等太久了还没把杆带出去（贴着盘沿刷、或检测框一直在盘区
                    # 内抖动）：不再等，用已经攒到的盘外读数定论。
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
                    # 记下这一趟进盘**之前**盘区里那批丝杆（存货快照）。
                    # 离盘后区分「手里带着走的那根」与「盘里的存货」用
                    # （见 ``_carrying_out_of_tray``）。
                    if self._screw_history:
                        visit.entry_inventory = self._screw_history[-1]
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
                self._outside_holding_last_ts.pop(hand_id, None)

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

        # 盘内停留 = 首次进盘 → 结算时刻的区间。
        # 原实现的起点是「首次见到这只手」：手在盘外被跟踪 3 秒再蹭一下
        # 盘区，dwell 就有 3 秒多，「盘边划过不判取料」的守卫形同虚设。
        # 起点改锚到真正进盘的那一帧，终点仍用结算时刻——离盘后攒采样
        # 的这段时间本来就计入阈值（现场标定的 pick_confirm_s 是在这套
        # 口径下调出来的），只换起点才不改变既有的取料行为。
        if visit.first_inside_ts is not None:
            dwell = max(0.0, ts - visit.first_inside_ts)
        else:
            dwell = visit.dwell_in_tray_s
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
        """记录一次「手在盘外、手里有没有杆」的观测。

        这条历史只用于**进盘定性**（这一趟是空手进盘还是带料进盘），
        记录前要剔除盘内的杆：

        手要经过盘边时，手框（实测 348×399 px，接近盘宽）必然把盘里那排
        静置杆框进去，这些杆的**中心**就落在手框内，命中「杆心在手框内」
        那条基础判据。一条这样的观测进了历史，本次进盘就被判成「带料进盘」，
        整件取料丢掉——示例② 第 95 帧附近的取料即如此（f94 手在盘外，
        盘内 (1544,319) 那根静置杆被判成握持）。

        只在这里剔除、不动 ``_holding_screw`` 本身：离盘后的握持采样仍要
        认盘内的杆——取料瞬间手里那根杆往往还压在盘区上方，那正是这一件
        料被带走的证据（视频② f463 那次取料的前两个采样点就落在盘区内，
        提前剔除会把整件取料判丢）。
        """

        if holding is not None and self._screw_parked_in_tray(holding):
            holding = None
        # 同一帧只记一次。三处调用点（新建轨迹、采样点、未确认进盘）
        # 看似互斥，实测同一 ``ts`` 会被调用 2~3 次：``pick_history_samples``
        # 默认 3，一帧就能把投票窗口灌满，多数票退化成「单帧随机数」。
        last = self._outside_holding_last_ts.get(hand_id)
        if last is not None and last == ts:
            return
        self._outside_holding_last_ts[hand_id] = ts
        window = max(0.0, self._entry_grace_s())
        history = self._outside_holding.setdefault(hand_id, deque())
        # 「手里没杆」是一次**状态复位**：从这一刻起的观测才是这一趟进盘前
        # 的有效依据。不清掉之前的「手持杆」观测，上一次取料后拿着杆走动的
        # 那串记录会一直留在窗口里，下一次空手进盘被投成「带料进盘」，整件
        # 取料判丢（示例③在 7.5fps 下第 2 件即如此）。
        if holding is None:
            history.clear()
        history.append((holding is None, ts))
        while history and ts - history[0][1] > window:
            history.popleft()

    def _screw_static(self, box: BBox) -> bool:
        """这根杆的位置是否**没动过**（与 ``_SCREW_STATIC_BACK`` 帧前同位）。

        只按位置稳定性判断，与杆在不在盘区里无关：盘里那排杆、零件盒里的
        整排杆都不动；被手带走的杆即使中心还没离开盘区，框也已经在动。
        实测四段示例视频里不动的杆对该窗口内位置的 IoU ≥0.5（中位 0.9+），
        被带走的杆 ≤0.3。

        历史不足时不作判断（返回 False）：开局/复位后的前几帧没有对照，
        此时不能凭它否定握持，否则真取料会被吞掉。
        """

        if len(self._screw_history) < _SCREW_STATIC_BACK:
            return False
        # 参考帧取窗口内**最早的连续两帧**（8 帧前与 7 帧前），而不是只比
        # ``[-BACK]`` 这一帧。
        #
        # 单帧比对时，只要那一帧恰好漏检了这根杆，位置从未变动的杆就会被
        # 判成「在动」；一旦锚点借此捕获了静置杆，`_static_anchor_ok` 的
        # 「锚点 ±0.02 同根」例外会让它一直合法，之后刷真杆的接触时长不再
        # 累计 → 漏涂 NG。
        #
        # 也不能拿窗口里**所有**帧来比：移动中的杆总能与最近一两帧的同位
        # 匹配上（实测这样静置判定直接失效，涂油全部判不出来）。只有保持
        # 时间差、又多一个参考帧，才能既跳过漏检帧、又区分静与动。
        window = list(self._screw_history)[-_SCREW_STATIC_BACK:]
        reference = window[:2]
        return any(
            box.iou(past_box) >= _SCREW_STATIC_IOU
            for frame in reference
            for past_box in frame
        )

    def _hand_over_split_visits(self, tray_box: BBox | None, ts: float) -> None:
        """手部 ID 在进盘途中被重建时，把没走完的进盘过程交给真正在盘里的那只手。

        跟踪器在掉帧/遮挡后会保留旧轨迹（``matched=False``，位置停在最后一次
        观测），同时给回来的手**新建**一条轨迹。于是出现这一幕：旧 ID 停在
        盘外不动、新 ID 在盘里干活，随后出盘又回落到旧 ID——进盘记在新 ID
        名下、离盘发生在旧 ID 名下，两边永远凑不齐，整件取料丢掉。示例④
        第 3 次取料即如此（f339 盘内是 100004，f347 出盘时已是 100003）。

        做法：找出「已确认进盘、但主人轨迹这一帧已失配」的过程，把它转给
        这一帧**确实在盘内**、且离它最后一次位置最近的那只手。宁可漏接也
        不要张冠李戴，所以三个条件都要满足：

        - 过程是进行中的（``entered_confirmed`` 且未离盘）；
        - 主人这一帧没有匹配上（``matched=False``）；
        - 接手的这只手这一帧在盘内，且距离在 ``hand_track_max_distance`` 内。
        """

        if not self._visits:
            return
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        matched = {
            track.track_id: track
            for track in self._hand_tracks
            if track.matched and track.hits >= max(1, self._hand_tracker.min_hits)
        }
        limit = self.cfg.hand_track_max_distance * self._frame_scale
        for owner_id, visit in list(self._visits.items()):
            if not visit.entered_confirmed or visit.exited_at_ts is not None:
                continue
            if owner_id in matched or visit.last_hand_box is None:
                continue
            ox, oy = visit.last_hand_box.center
            best: tuple[float, int] | None = None
            for track_id, track in matched.items():
                if track_id == owner_id:
                    continue
                # 候选轨迹自己也在走一段**已确认进盘**的过程时不动它，否则
                # 会把两只同时干活的手搅在一起。只是挂着个还没进盘的空壳
                # visit 的（掉帧期间新建、还没确认）可以直接顶掉：那几帧盘外
                # 握持历史本来就属于同一只手。
                other = self._visits.get(track_id)
                if other is not None and other.entered_confirmed:
                    continue
                hb = track.observation_box()
                hx, hy = hb.center
                if tray_box is not None and not (
                    tray_box.x1 <= hx <= tray_box.x2
                    and tray_box.y1 <= hy <= tray_box.y2
                ):
                    continue
                distance = math.hypot(ox - hx, oy - hy) / diag
                if distance > limit:
                    continue
                if best is None or distance < best[0]:
                    best = (distance, track_id)
            if best is None:
                continue
            _, new_id = best
            self._visits.pop(owner_id, None)
            self._visits.pop(new_id, None)
            visit.hand_track_id = new_id
            visit.last_seen_ts = ts
            visit.rebinds += 1
            visit.restart_holding()
            visit.holding_seen = False
            self._visits[new_id] = visit
            self._logger.debug(
                "手部轨迹 %s → %s 接管进盘过程（盘内已停留 %.2fs，接管 %d 次）",
                owner_id,
                new_id,
                visit.dwell_in_tray_s,
                visit.rebinds,
            )

    def _carrying_out_of_tray(
        self, hand_box: BBox, inventory: tuple[BBox, ...] = ()
    ) -> bool:
        """手已在盘外、但手边那根正随着动的杆还没出盘。

        这种帧不能当成「手里没杆」的采样点：杆心晚几帧出盘是常态（示例④
        第 3 次取料手 f348 出盘、杆 f352 才出盘），把它记成 False 会让
        采样窗口多数票投出「空手」。它是这一件正在出门的证据，只是还没
        跨过盘区边线。等待上限见 ``_CARRY_OUT_OF_TRAY_MAX_S``。

        **盘里的存货不算**。判据是「这根杆不在进盘时的存货快照里」——
        存货在这趟进盘前后一直在盘里；手上那根是这趟新进入盘区的。只看
        位置稳不稳不够：工人把杆停在盘区下沿歇一下（示例⑤ 第 2 次取料
        停了 2s 多），``_screw_static`` 就不再认为它在动，整件取料被当成
        「空手离盘」丢掉。

        ``inventory`` 为空（历史不足）时退回位置稳定性判据。
        """

        tray_box = self._tray_region_box(self._frame_size)
        if tray_box is None:
            return False
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        hx, hy = hand_box.center
        for det in self._last_detections:
            if det.label != self.station.detector.screw_label:
                continue
            cx, cy = det.box.center
            if math.hypot(cx - hx, cy - hy) / diag > self.cfg.hand_screw_max_distance:
                continue
            if not _box_center_in(det.box, tray_box):
                continue
            if inventory:
                # 与存货快照里任一根重合 ⇒ 它是盘里的料，与本件无关。
                if any(det.box.iou(box) >= _INVENTORY_SAME_IOU for box in inventory):
                    continue
                return True
            if self._screw_static(det.box):
                continue
            return True
        return False

    def _screw_parked_in_tray(self, screw: Detection) -> bool:
        """这根杆是不是盘里那根**没被动过**的静置杆。

        只看「中心在不在盘区」不够：手工 ``tray_roi`` 把盘体下方的手臂活动
        范围也框进去了，取料时手里那根杆的中心常常还落在盘区内（示例④第 2、
        3 次取料的离盘初期整段如此），按中心判定就会把正被带走的杆当成静置
        杆排除掉，整件取料判不出来。

        真正的区别在**位置稳定性**（见 :meth:`_screw_static`）。

        中心不在盘区的杆一律不是静置杆；历史不足时按静置处理（保守：
        盘区外的手在这一点上不做宽松推断）。
        """

        tray_box = self._tray_region_box(self._frame_size)
        if tray_box is None or not _box_center_in(screw.box, tray_box):
            return False
        if len(self._screw_history) < _SCREW_STATIC_BACK:
            return True
        return self._screw_static(screw.box)

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

    def _carry_wait_s(self) -> float:
        """手出盘后等「杆也出盘」的最长时长（按实际帧率缩放）。

        见 ``_CARRY_OUT_OF_TRAY_MAX_S``：这段时间不能用离盘后的固定结算
        时长，杆心比手晚出门一秒上下是常态。
        """

        return _CARRY_OUT_OF_TRAY_MAX_S * self._frame_scale

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
        """手里拿着的那根丝杆。严格按「空手进盘、带丝杆出盘」的口径：

        1. **杆必须已经出盘**。手在盘外时，杆心仍落在手工 ``tray_roi`` 里
           的一律不算手里拿着——那是盘里的料。取料的定义就是「带丝杆**出**
           盘」，杆心还在盘区里就还没出盘。这条不加，「离盘」这一帧手上
           蹭到的盘内杆会被当成取料证据：示例① 第 88~103 帧工人只是把手
           伸进盘里点按夹具，离盘瞬间杆心还在盘区内的那根杆和手框交叠，
           于是凭空多判一件取料；
        2. **杆心落在手框内**，或两者 IoU ≥ 0.15 —— 杆被手完全握住/遮住；
        3. **手框右上角区域与杆框相交** —— 实际握姿是拳头攥住杆身、杆从虎口
           （手框右上角）伸出去，此时杆心已经在手框外、IoU 也上不去。丝杆
           又细又长（面积只有手框的 17~26%），IoU 的结构上限就是面积比，
           0.15 这条门限在握持状态下几乎不可达（示例③实测握持段 IoU 只有
           0.03~0.13，而「杆心在手框内」会随手的朝向时真时假）。

        第 3 条只对**已经在动**的杆放开到「相交即可」；位置没动过的杆
        要求杆有 ``_GRIP_PARKED_MIN_FRAC`` 的面积落进右上区（盘边蹭到的
        实测 0.25~0.42，真攥住的 ≥0.44）。杆被带走时手常常是「握着杆身、
        杆从虎口伸出去」，与右上区的重叠刚好落在两档之间，用同一个 0.30
        会把整段离盘初期判成空手（示例④ 第 3 次取料 f353~f358 即如此）。

        盘内不启用第 2、3 条：手伸在盘上方必然蹭到盘里的杆，那条读数与
        实际有没有握着无关（盘内握持读数本来就不参与投票，见 ``_HandVisit``）。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        label = self.station.detector.screw_label
        hx, hy = hand_box.center
        tray_box = self._tray_region_box((width, height))
        hand_outside = tray_box is None or not _box_center_in(hand_box, tray_box)
        best: tuple[float, Detection] | None = None
        for det in self._last_detections:
            if det.label != label:
                continue
            cx, cy = det.box.center
            distance = math.hypot(cx - hx, cy - hy) / diag
            if distance > self.cfg.hand_screw_max_distance:
                continue
            if (
                hand_outside
                and tray_box is not None
                and _box_center_in(det.box, tray_box)
            ):
                # 杆心还在手工盘区内：手在盘外，说明这根还没被带出盘。
                continue
            gripped = _box_center_in(det.box, hand_box) or hand_box.iou(det.box) >= 0.15
            if not gripped and hand_outside:
                if self._screw_static(det.box):
                    # 位置没动过的杆不可能在手里：盘边/夹具边蹭到的不算。
                    gripped = (
                        _grip_corner_fraction(hand_box, det.box)
                        >= _GRIP_PARKED_MIN_FRAC
                    )
                else:
                    gripped = _gripped_by_corner(hand_box, det.box, min_frac=0.0)
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
        # 取料与涂油并行判定：工人一边取料一边刷时，涂油段起于取料结算
        # 之前，**不能**因为它还没挂上周期就清掉（示例⑤ 6.47~10.20 就是
        # 一整段连续涂抹，切成两段会多报一次）。
        #
        # 但「已经在某一件上报过的段」要收：那是上一件的涂油，新一件的
        # 刷动不该并进去（两件的杆可能落在同一位置，位置判才分不开）。
        self._close_oiling_if_settled()
        task.screw_observed = True
        # 记下取料那一刻手里那根杆的位置：之后这次涂油要么就发生在它上面，
        # 要么在它旁边（工人会走动）。没有它就无从判断「这次涂油属于哪一件」，
        # 另一只手握着别的杆刷够了会被错挂到本周期上（实测）。
        task.screw_anchor = screw.box.center
        task.screw_anchor_ts = ts
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
            review=build_pick_summary(task, dwell_s=dwell_s),
        )
        self._logger.info(
            "取料完成 %s（手部轨迹=%s，盘内停留 %.2fs）",
            task.task_id,
            hand_track_id,
            dwell_s,
        )
        # 「一边取料一边刷」：这次涂油在取料结算之前就判完了，补挂上来。
        self._drain_pending_oilings(task, ts)

    # ------------------------------------------------------------------
    # 人工二次涂油判定
    # ------------------------------------------------------------------
    def _update_oiling(self, *, ts: float, dt: float) -> None:
        """人工二次涂油：刷头贴着手持丝杆累计接触 ⇒ 完成。

        **与取料并行判定，不依赖作业周期**。现场就是这样：工人常常一边
        取料一边就刷上了（示例7 第 2 件、示例6 第 3 件），要等「手离开盘区」
        才建立取料周期的话，刷的那两秒正好落在周期建立之前，整段丢掉、
        随后还被下一次取料判成漏涂。

        「一次涂油」的边界由**杆**决定，和 ``tools/detect_operations.py``
        同一口径：换一根杆就另起一次；同一根杆上刷手抬一下再放回来
        （现场每件都有 0.5~1.3 s 遮挡）仍算同一次。
        """

        self._flush_stale_pending_oilings(ts)
        session = self._active_oiling
        contacts = self._oiling_contacts()
        if session is not None:
            # 只要还在跟同一根杆，不论中间空了几帧都优先续上。
            matched = next(
                (
                    item
                    for item in contacts
                    if self._same_oiling_rod(item[1], session)
                ),
                None,
            )
            if matched is None and contacts:
                # 场上还有别的杆在被刷：这一根换了，本段到此为止。
                self._close_oiling()
                session = None
        else:
            matched = None
        if matched is None and session is None and contacts:
            matched = contacts[0]

        if matched is not None:
            distance, box, brush = matched
            if session is None:
                # 首帧也把这一帧的 dt 记进去：刷头此刻确实贴着，和后续帧
                # 一样是有效接触。脚本没有 dt 概念、只能从第二帧起算，
                # 这里按判定的实际口径算，低帧率下才不会少算一帧。
                self._active_oiling = _OilingSession(
                    anchor=box.center,
                    start_ts=ts,
                    last_ts=ts,
                    last_seen_ts=ts,
                    cum_s=max(0.0, dt),
                    frames=1,
                    min_distance=distance,
                    brush_track_id=brush.track_id,
                )
                self._publish_oiling_progress(self._active_oiling)
                return
            gap = max(0.0, ts - session.last_ts)
            # 漏检/抬手的空档按帧率放大的桥接窗口补回来，超出就不再计入
            # ——把整段空档都算成接触会把「刷两下就走」凑成完成。
            bridge = min(
                self.cfg.brush_hold_max_gap_s * self._frame_scale,
                self._max_plausible_dt(),
            )
            if gap > bridge:
                session.parts += 1
            session.cum_s += min(gap, bridge)
            session.last_ts = ts
            session.last_seen_ts = ts
            session.frames += 1
            session.anchor = box.center
            session.min_distance = min(session.min_distance, distance)
            session.brush_track_id = brush.track_id
            self._publish_oiling_progress(session)
            if (
                session.reported_ts is None
                and session.cum_s >= self.cfg.brush_hold_s - 1e-9
            ):
                self._finish_oiling(ts)
            return

        if session is None:
            return
        # 本帧没贴着刷。三种情形要分开：
        # - 场上还有手持杆、只是刷头没贴上去 → 真的停手了，靠拼接窗口收段；
        # - 一根手持杆都没看见（被手完全挡住 / 整帧漏检）→ 接着往下判，
        #   把这段时间按桥接窗口补上：现场的杆本来就常被拳头压住，
        #   不补会让「刷了 2 秒、中间遮了 0.4 秒」凑不够 brush_hold_s；
        # - 超时 → 收段。
        if self._oiling_held_rods() or self._screw_visible_anywhere():
            # 场上有丝杆被检出：手确实松开了 / 刷头确实离开了，不补时间。
            # 只有「连一根杆都检不到」才算整帧遮挡，才往下桥接——
            # 用「有没有手持杆」区分不了这两种，杆画得出来但不在手里
            # 和杆被手完全压住是两回事。
            pass
        elif ts - session.last_seen_ts <= self._oiling_occluded_hold_s():
            bridge = min(
                self.cfg.brush_hold_max_gap_s * self._frame_scale,
                self._max_plausible_dt(),
            )
            gap = max(0.0, ts - session.last_ts)
            if gap > bridge:
                session.parts += 1
            session.cum_s += min(gap, bridge)
            session.last_ts = ts
            session.frames += 1
            self._publish_oiling_progress(session)
            if (
                session.reported_ts is None
                and session.cum_s >= self.cfg.brush_hold_s - 1e-9
            ):
                self._finish_oiling(ts)
            return
        # 本帧没贴着刷：短暂抬手/漏检先挂着，超过拼接窗口再收段。
        if ts - session.last_seen_ts > self._oiling_stitch_gap_s():
            self._close_oiling()

    def _publish_oiling_progress(self, session: _OilingSession) -> None:
        """把当前接触段的进度同步到正在等涂油的那个周期上。

        ``brush_hold_s`` 等字段供监控卡片、复核页与 PLC 灯效实时显示，
        不能只在判完的那一刻才写一次。
        """

        task = self._oiling_target_cycle(session)
        if task is None:
            return
        task.brush_hold_s = session.cum_s
        task.brush_track_id = session.brush_track_id
        task.brush_min_distance = session.min_distance
        task.brush_near_frames = session.frames
        task.last_seen_ts = session.last_seen_ts

    def _oiling_held_rods(self) -> tuple[BBox, ...]:
        """本帧「被手握着」的丝杆框。

        不要求杆心出盘：``tray_roi`` 是为取料划的，下沿常盖住工人刷油的
        位置，刷的时候杆压在盘区里是常态（示例6 第 1、3 件）。握着的事实
        用「杆框与手框交叠 + 杆心在手附近」判断。
        """

        label = self.station.detector.screw_label
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        limit = self.cfg.hand_screw_max_distance * self._frame_scale
        anchor = None if self._active_oiling is None else self._active_oiling.anchor
        boxes: list[BBox] = []
        for det in self._last_detections:
            if det.label != label:
                continue
            cx, cy = det.box.center
            for hand in self._hand_tracks:
                if not hand.matched:
                    continue
                hand_box = hand.observation_box()
                if not _overlaps(det.box, hand_box):
                    continue
                hx, hy = hand_box.center
                if math.hypot(cx - hx, cy - hy) / diag > limit:
                    continue
                # 盘里那根没被动过的存货不算「被握着」：手框很大，伸进盘里
                # 抓料时会与整排存货交叠，不排除就会把盘里的料当成正在刷的
                # 那根，让没刷的那一件也累计够了（实测会吞掉漏涂 NG）。
                #
                # 只排除**在盘区内**的静置杆：盘区外台面上的待刷杆同样不动，
                # 但它就是工人要刷的那根。
                # 当前这一段自己那根除外——拳头压着杆时它可能连续几帧不动。
                if (
                    len(self._screw_history) >= _SCREW_STATIC_BACK
                    and self._screw_parked_in_tray(det)
                    and not self._same_rod_pos(det.box, anchor, diag)
                ):
                    continue
                boxes.append(det.box)
                break
        return tuple(boxes)

    @staticmethod
    def _same_rod_pos(
        box: BBox, anchor: tuple[float, float] | None, diag: float
    ) -> bool:
        if anchor is None:
            return False
        cx, cy = box.center
        return (
            math.hypot(cx - anchor[0], cy - anchor[1]) / diag
            <= _OILING_SAME_ROD_STEP
        )

    def _oiling_contacts(self) -> list[tuple[float, BBox, Track]]:
        """本帧「刷头工作点贴着某只手里的杆」的全部候选，按距离升序。

        与 ``tools/detect_operations.py`` 的 ``OilingDetector._contacts``
        同一口径：遍历本帧全部刷头轨迹与全部手持杆，取刷头工作点到杆身
        中线的垂距 ≤ ``brush_contact_distance`` 的组合。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        held = self._oiling_held_rods()
        if not held:
            return []
        contacts: list[tuple[float, BBox, Track]] = []
        for brush in self._brush_tracks:
            if not brush.matched or brush.hits < self._brush_tracker.min_hits:
                continue
            point = brush_work_point(brush, self.station.detector)
            for box in held:
                distance = _distance_to_long_axis(box, point) / diag
                if distance <= self.cfg.brush_contact_distance:
                    contacts.append((distance, box, brush))
        contacts.sort(key=lambda item: item[0])
        return contacts

    def _same_oiling_rod(self, box: BBox, session: _OilingSession) -> bool:
        if session.anchor is None:
            return True
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        cx, cy = box.center
        return (
            math.hypot(cx - session.anchor[0], cy - session.anchor[1]) / diag
            <= _OILING_SAME_ROD_STEP
        )

    def _screw_visible_anywhere(self) -> bool:
        """本帧是否检出了任意一根丝杆（不要求在手边）。"""

        label = self.station.detector.screw_label
        return any(det.label == label for det in self._last_detections)

    def _oiling_occluded_hold_s(self) -> float:
        """杆整帧看不见时，还能按「还在刷」往下补多久。

        与「手持丝杆被预测保持多久」同一口径（``held_predict_hold_s``）：
        拳头压住杆是常态，补一两帧能救回真实接触；补得更久就等于凭空
        造时间，会把「刷两下就走」凑成完成。
        """

        # 不乘 frame_scale：那是给「判定门槛」用的放大，用在这里会让低帧率
        # 下把「手早就松开了」也补进来（实测 3 倍放大后能凭空补 1.5s）。
        # 只补得了「一两帧漏检」，最多一个可信帧间隔。
        return min(
            self.cfg.held_predict_hold_s,
            self._max_plausible_dt(),
        )

    def _oiling_stitch_gap_s(self) -> float:
        return max(
            _OILING_STITCH_GAP_S,
            self.cfg.brush_hold_max_gap_s,
        ) * self._frame_scale

    def _close_oiling(self) -> None:
        """收掉当前接触段；还没报过且累计够长的才算一次涂油。"""

        session = self._active_oiling
        if session is None:
            return
        self._active_oiling = None
        if (
            session.reported_ts is None
            and session.cum_s >= self.cfg.brush_hold_s - 1e-9
        ):
            self._finish_oiling(session.last_ts, session=session)

    def _finish_oiling(
        self, ts: float, *, session: _OilingSession | None = None
    ) -> None:
        """一次涂油达成：并入最近的取料周期，或单独报一条。"""

        session = session or self._active_oiling
        if session is None:
            return
        # 不把这一段从 ``_active_oiling`` 摘下来：刷手判完通常还在这根杆上
        # 继续刷，摘掉就会在下一帧另开一段、再报一次。等杆换了或离开，
        # ``_close_oiling`` 才会收段。
        session.reported_ts = ts
        task = self._oiling_target_cycle(session)
        if task is not None:
            self._attach_oiling(task, session, ts)
            return
        # 还没有取料周期可挂。分两种情形：
        #
        # - 有手正在盘区里（取料进行中）：工人常常一边取料一边就刷上了，
        #   取料要等手离开盘区才结算，那会儿这次涂油已经判完。先存着，
        #   取料确认时按杆补挂——立刻报成无主涂油会让这一件没有归属。
        # - 没有手在盘区里：这是纯涂油（录像开头就在刷、取完又补刷），
        #   现场按规则不算异常，直接报出去。
        if session.anchor is not None and self._pick_in_progress():
            self._pending_oilings.append((session, ts))
            return
        self._emit_free_oiling(session, ts)

    def _pick_in_progress(self) -> bool:
        """取料是不是正在进行中（进盘过程还挂着、尚未结算）。

        判据用「有没挂着未结算的进盘过程」而不是「盘区里有没有手」：
        工人常常站在盘边刷，手刚好在盘区下沿外一点点；按盘区判会在
        涂油判完的那一刻返回 False，这次涂油就被报成无主涂油
        （示例⑤ 6.47s、示例⑦ 11.93s 实测）。进盘过程挂着，说明这一趟
        还没定论，涂油应该等它。
        """

        # 「已确认进盘、还没结算」才算取料进行中。手在旁边走过（还没进盘）
        # 或只是被跟踪着，不算。
        #
        # 注意离盘之后、结算之前这段也算：那时正在攒盘外握持采样，
        # 取料马上就要确认了，这一件的涂油该等它挂上去（示例⑤ 6.47s、
        # 示例⑦ 11.93s 就落在这一段里）。
        if any(visit.entered_confirmed for visit in self._visits.values()):
            return True
        # 手已在盘区里但离盘还没按采样窗口确认：这也是取料进行中。
        tray_box = self._tray_region_box(self._frame_size)
        if tray_box is None:
            return False
        return any(
            visit.inside_tray
            and visit.last_hand_box is not None
            and _box_center_in(visit.last_hand_box, tray_box)
            for visit in self._visits.values()
        )

    def _flush_stale_pending_oilings(self, ts: float) -> None:
        """暂存太久、始终没等到对应取料的涂油，按无主涂油报出去。

        现场确实有「录前就在刷、整段都没有取料」的情形（示例③④⑦ 开头）。
        等 `oiling_timeout_s` 这么久还没有取料，就不再等；设 0 时用一个
        保守的固定值，保证视频结束前也能报出来。
        """

        if not self._pending_oilings:
            return
        wait = self.cfg.oiling_timeout_s if self.cfg.oiling_timeout_s > 0 else 10.0
        for session, oil_ts in list(self._pending_oilings):
            if ts - oil_ts < wait:
                continue
            self._pending_oilings.remove((session, oil_ts))
            self._emit_free_oiling(session, oil_ts)

    def _attach_oiling(
        self, task: _Cycle, session: _OilingSession, ts: float
    ) -> None:
        """把一次涂油记到某个周期上并判完成。"""

        if not task.active or task.stage != TaskStage.PICKED:
            # 这一件已经判过（多次涂油的情形），不重复上报。
            return
        self._oiling_start_ts = session.start_ts
        task.brush_hold_s = session.cum_s
        task.brush_track_id = session.brush_track_id
        task.brush_min_distance = session.min_distance
        task.brush_near_frames = session.frames
        if task.screw_anchor is None and session.anchor is not None:
            task.screw_anchor = session.anchor
        self._complete_oiling(task, ts)

    def _close_oiling_if_settled(self) -> None:
        """当前涂油段已经完整报过（且不是暂存待挂的），就收段。"""

        session = self._active_oiling
        if session is None or session.reported_ts is None:
            return
        # 还在 pending 里等这一件结算的，不能收——它马上就要挂到新周期上。
        if any(item[0] is session for item in self._pending_oilings):
            return
        self._close_oiling()

    def _drain_pending_oilings(self, task: _Cycle, ts: float) -> bool:
        """把暂存的涂油补挂到刚建立的周期上。

        只认「同一根杆」：暂存的是工人的涂油，取料确认的是从盘里拿出来的
        那一根，两者位置对得上才是同一件。
        """

        if not self._pending_oilings or task.screw_anchor is None:
            return False
        for session, oil_ts in list(self._pending_oilings):
            if session.anchor is None:
                continue
            if not self._same_anchor(task.screw_anchor, session.anchor):
                continue
            self._pending_oilings.remove((session, oil_ts))
            self._attach_oiling(task, session, oil_ts)
            return True
        return False

    def _oiling_target_cycle(self, session: _OilingSession) -> _Cycle | None:
        # 第一步就排除「这根杆在被别的手握着」：取料时记下了那是哪只手，
        # 画面里另一只手握着另一根杆刷，不该算到这一件上。

        """这次涂油该记到哪个取料周期上。

        只有「同一根杆」才认：周期在取料那一刻记下了手里那根杆的位置，
        这次刷的杆必须就是它。不校验的话，画面里另一只手握着别的杆刷够了，
        会被错挂到本周期上、把这一件提前判成已涂油（实测会吞掉真实漏涂）。
        """

        candidates = [
            task
            for task in self._cycles.values()
            if task.active and task.stage == TaskStage.PICKED
        ]
        if not candidates:
            return None
        session_anchor = session.anchor
        if session_anchor is not None:
            by_hand = [
                task
                for task in candidates
                if self._hand_holds_point(task.hand_track_id, session_anchor)
            ]
            if by_hand:
                candidates = by_hand
            elif any(task.hand_track_id is not None for task in candidates):
                # 取料那几只手里没有一只握着这根杆。要区分两种情形：
                #
                # - 取料那只手**还在画面里**（只是空着手）：这根杆是别的
                #   手里的，不归这一件。现场实测就有一只空手站在原处、
                #   另一只手刷另一根的情形，认了会把这一件提前判成已涂油。
                # - 取料那只手**已经不在画面里**（拿杆走开、或视觉 ID 重建）：
                #   无歧义，就是这一件，认。
                if self._any_cycle_hand_still_visible(candidates):
                    return None
            matched = [
                task
                for task in candidates
                if task.screw_anchor is not None
                and self._same_anchor(task.screw_anchor, session_anchor)
            ]
            if not matched:
                # 位置对不上。只有**确实存在另一个候选周期**时才据此拒绝：
                # 工人取完料走动一段再刷是常有的事，只有一个周期在等涂油
                # 时无歧义，认它；有两个以上在等，才按位置挑。
                if len(candidates) > 1:
                    return None
            else:
                candidates = matched
        # 多次涂油时挂到最近一次取料的那一件上。
        return max(candidates, key=lambda item: item.picked_at or item.created_at)

    def _any_cycle_hand_still_visible(self, tasks: list[_Cycle]) -> bool:
        """这些周期记下的取料那只手，还有没有一只在画面里。"""

        return any(
            task.hand_track_id is not None
            and self._live_hand(task.hand_track_id) is not None
            for task in tasks
        )

    def _other_hand_holds_other_rod(
        self, anchor: tuple[float, float]
    ) -> bool:
        """画面的其他手里是不是也握着杆（即「这根杆归谁」存在歧义）。"""

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        label = self.station.detector.screw_label
        for det in self._last_detections:
            if det.label != label:
                continue
            if (
                math.hypot(det.box.center[0] - anchor[0], det.box.center[1] - anchor[1])
                / diag
                <= _OILING_SAME_ROD_STEP
            ):
                continue
            # 这里按「任意手与任意杆交叠」判，不套 ``_oiling_held_rods`` 的
            # 盘内存货过滤——那只负责选当前要跟的杆，这里要回答的是
            # 「场上有几只手里有杆」。
            for hand in self._hand_tracks:
                if not hand.matched:
                    continue
                if _overlaps(det.box, hand.observation_box()):
                    return True
        return False

    def _hand_holds_point(
        self, hand_track_id: int | None, point: tuple[float, float]
    ) -> bool:
        """那只手此刻是不是正握着这个位置上的杆。"""

        if hand_track_id is None:
            return False
        hand = self._live_hand(hand_track_id)
        if hand is None:
            return False
        hand_box = hand.observation_box()
        if not _box_center_in(BBox(point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), hand_box):
            # 手框不一定「包含」杆心（握在手里时杆心常在手框边缘外），
            # 退一步按中心距离判。
            hx, hy = hand_box.center
            width, height = self._frame_size
            diag = max(1.0, math.hypot(width, height))
            if math.hypot(point[0] - hx, point[1] - hy) / diag > (
                self.cfg.hand_screw_max_distance
            ):
                return False
        return True

    def _same_anchor(
        self, a: tuple[float, float], b: tuple[float, float]
    ) -> bool:
        """两个位置是不是同一根杆（按涂油判定的同杆容差，放宽两倍）。

        工人取完料会走动，同一根杆后来的位置和取料那一刻可以差一截；
        这里只用来**否定**「明显是另一根杆」，宁可放宽也不要错挂。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        return (
            math.hypot(a[0] - b[0], a[1] - b[1]) / diag
            <= _OILING_SAME_ROD_STEP * 2
        )

    def _emit_free_oiling(self, session: _OilingSession, ts: float) -> None:
        """没有对应取料周期的涂油（录像开头就已在刷、或取完又补刷）。"""

        self._free_oiling_seq += 1
        event = EventRecord.create(
            timestamp_start=float(session.start_ts or ts),
            timestamp_end=float(ts),
            station_id=self.station.station_id,
            camera_id=self.station.camera_id,
            cycle_id=self.cycle_id,
            task_id="",
            event_type=EventType.MANUAL_OILING,
            result=EventResult.CONFIRMED,
            evidence={
                "task_id": "",
                "reason": "manual_oiling_completed",
                "cycle_attached": False,
                "brush_track_id": session.brush_track_id,
                "brush_hold_s": round(session.cum_s, 3),
                "required_s": self.cfg.brush_hold_s,
                "brush_min_distance": round(session.min_distance, 4),
                "brush_near_frames": session.frames,
                "contact_parts": session.parts,
                "detector": self.detector_name,
                "free_oiling_seq": self._free_oiling_seq,
            },
        )
        self._events_emitted += 1
        self._logger.info(
            "人工二次涂油完成（未挂作业周期，刷头靠近 %.2fs）", session.cum_s
        )
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:  # noqa: BLE001
                self._logger.exception("事件回调失败")

    def _static_anchor_ok(self, box: BBox, anchor: tuple[float, float] | None) -> bool:
        """这根杆能不能当「手里那根」的候选。

        位置没动过的杆（盘里的料、零件盒里的整排、夹具上的）一律排除——
        手框很大，随手一放就与整排杆交叠，那些杆不可能是正被带走/正被刷的
        这一根。示例① 第一次涂油时，锚点从工件漂到零件盒里的静置杆上，
        刷子在真杆上刷了 5 秒、接触累计却只有 0.13s，整件被判漏涂。

        例外：**就是锚点当前那根**。拳头压着杆、杆贴着夹具刷油时这一刻
        确实可能不动；允许它继续，否则锚点每抖一下就要丢一次。
        """

        if not self._screw_static(box):
            return True
        if anchor is None:
            return False
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        return (
            math.hypot(box.center[0] - anchor[0], box.center[1] - anchor[1]) / diag
            <= _ANCHOR_SAME_SCREW_DISTANCE
        )

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
        #
        # 加一条「位置没动过的不算」：手框很大（实测 348×399 px），手在盘中
        # 或零件盒边随手一放就与整排杆交叠；那些杆纹丝不动，不可能是手里
        # 正在被带走的这一根。锚点已经锁定的那根除外——拳头压着杆时它可能
        # 连续几帧不动。
        cycle_hand = self._live_hand(task.hand_track_id)
        gripped: list[tuple[float, BBox]] = []
        for det in self._last_detections:
            if det.label != label:
                continue
            if cycle_hand is None or not _overlaps(
                det.box, cycle_hand.observation_box()
            ):
                continue
            if not self._static_anchor_ok(det.box, anchor):
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
                if not self._static_anchor_ok(det.box, anchor):
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
            # 位置没动过的杆不算：手臂从零件盒/盘上方经过时整排杆都落在
            # 手框内，接成锚点之后刷手里那根永远累计不上（同
            # ``_static_anchor_ok``）。
            if not self._static_anchor_ok(det.box, task.screw_anchor):
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
        """本周期那只手的中心，用于丝杆锚点初始化。

        与其余判定（握持、进盘/离盘、锚点交叠）统一用 ``observation_box()``
        ——原始检测框。``box`` 是卡尔曼滤波后的状态，快速移动时明显滞后
        （实测 200px 的跳变只走到 155px）；锚点初始化时用滤波框会把杆锁到
        手「上一帧所在」的位置，取料瞬间尤其容易锁错。
        """

        hand = self._live_hand(hand_track_id)
        return None if hand is None else hand.observation_box().center

    def _has_live_brush(self) -> bool:
        """本帧是否有任一已匹配的刷头轨迹（用于区分漏检和远离）。"""

        return any(
            brush.matched and brush.hits >= self._brush_tracker.min_hits
            for brush in self._brush_tracks
        )

    def _brush_work_distance(self, screw_box: BBox, brush: Track) -> float:
        """刷头工作点到杆身中线的归一化距离（复核统计用）。

        与 :meth:`_nearest_brush` 同一算法（都用 ``brush_work_point``），
        不另立口径——否则复核时看到的距离与判定用的距离会对不上。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        point = brush_work_point(brush, self.station.detector)
        return _distance_to_long_axis(screw_box, point) / diag

    def _close_brush_run(self, task: _Cycle) -> None:
        """结束当前连续接触段，把它的时长并入「最长连续接触」。

        段长取「段内最后一次刷头靠近的时刻 - 段起点」，不用收尾那帧的 ts：
        杆不可见时 ``_update_oiling`` 会提前 return，那段空档并没有累计进
        ``brush_hold_s``，也不该算进连续接触时长。
        """

        if task.brush_run_start_ts is None:
            return
        end = task.brush_last_near_ts or task.brush_run_start_ts
        task.brush_longest_run_s = max(
            task.brush_longest_run_s, end - task.brush_run_start_ts
        )
        task.brush_run_start_ts = None
        task.brush_last_near_ts = None

    def _nearest_brush(self, screw_box: BBox) -> Track | None:
        """离杆身接触邻域最近、且已被本帧检测到的刷头。

        距离量的是刷头工作点到**杆身中线**的垂距，不是到杆心的距离：杆框
        细长，贴着杆身中下部刷时工作点离杆心天然就在门限附近，按杆心算
        会有一半的帧被判成「没靠近」（见 ``_distance_to_long_axis``）。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        best: tuple[float, Track] | None = None
        for brush in self._brush_tracks:
            if not brush.matched or brush.hits < self._brush_tracker.min_hits:
                continue
            point = brush_work_point(brush, self.station.detector)
            distance = _distance_to_long_axis(screw_box, point) / diag
            if distance > self.cfg.brush_contact_distance:
                continue
            if best is None or distance < best[0]:
                best = (distance, brush)
        return None if best is None else best[1]

    def _complete_oiling(self, task: _Cycle, ts: float) -> None:
        self._close_brush_run(task)
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
            # 起点取「取料时刻」和「这次接触起点」中较早的那个，终点取较晚
            # 的那个：补挂的涂油是在取料结算之前就刷上的，按取料时刻起点会
            # 得到倒挂区间（实测 8.73→7.47）。
            start_ts=min(
                task.picked_at or task.created_at,
                self._oiling_start_ts or ts,
            ),
            end_ts=max(ts, task.picked_at or task.created_at),
            review=build_oiling_summary(task, required_s=self.cfg.brush_hold_s),
        )
        self._logger.info(
            "人工二次涂油完成 %s（手持丝杆，刷头靠近 %.2fs）",
            task.task_id,
            task.brush_hold_s,
        )

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
