"""判定过程里的数据结构（作业周期、进盘过程、涂油段、出盘记录、carry 证据）。

这里只有**状态容器**与它们自己的小方法（``carried`` / ``size_compatible``
/ ``holding_verdict`` 等），不含任何跨帧判定流程——流程在
``pipeline.py`` 的 :class:`StationPipeline` 里。

拆出来是为了让 pipeline.py 聚焦在判定逻辑上；这些类本身是纯数据，便于
单独阅读与测试。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .constants import (
    _CARRY_FOLLOW_COS,
    _CARRY_INHAND_FOLLOW_RATIO,
    _CARRY_INHAND_ROD_MOVE_MIN,
    _CARRY_INHAND_S,
    _CARRY_ROD_FOLLOW_RATIO,
    _CARRY_ROD_MOVE_MIN,
)
from .geometry import _majority, _push
from .raw_features import OilingAggregator, PickAggregator
from .types import BBox, TaskResult, TaskStage


@dataclass
class _CarryEvidence:
    """离盘后「等杆出盘」那段（carry）的累积证据。

    carry 分支要回答的是一个比「手边有没有盘内的杆」更难的问题：**这根杆
    是不是真的被手带着往外走**。手在盘外、手边有盘内杆，这只是必要条件
    ——手框很大（实测 350×400 px，接近盘宽），盘里的存货、夹具上的杆都会
    落进来。

    - 示例⑧ 工人把杆装上白色夹具：手按着杆不动，杆自己的位置也一动不动，
      但「手边有盘内杆」帧帧成立，于是被判成一次取料；
    - 示例⑨ 手从盘边划过、紧挨着的那根杆压在夹具上，同样一动不动。

    真正取料的形态是反过来的：杆**跟着手一起动**（示例⑥/⑦ 一边刷一边把杆
    往下带，杆自身位移与手的位移同量级），或者杆**持续**落在手框里（攥在
    手里刷）。夹具上的杆两者都不成立：手划过时落手面积只有零星一两帧勉强
    够格，杆的位移也跟不上手（示例⑧⑨ 全程 ≤0.001）。
    """

    # 锁定的那根杆的初位置、上一帧位置，与当时的手位置（归一化）。
    start_rod: tuple[float, float] | None = None
    last_rod: tuple[float, float] | None = None
    start_hand: tuple[float, float] | None = None
    frames: int = 0
    # 上一帧的时刻，用来把「落手面积达标」累计成时长。
    last_ts: float | None = None
    # 锁定后丢掉的帧数（杆漏检、或候选偏离锁定位置）。
    gaps: int = 0
    # 杆框落在手框内的面积比例、手框与杆框的 IoU、虎口重叠比例的最大值。
    max_inhand: float = 0.0
    max_iou: float = 0.0
    max_grip: float = 0.0
    # 落手面积 ≥ ``_CARRY_INHAND_MIN`` 的帧数（持续性，不是单帧峰值）。
    inhand_frames: int = 0
    # 落手面积达标的累计**时长**（秒）。判定用它，``inhand_frames`` 只留作
    # 复核/诊断字段——按帧数数会让门限随帧率漂移。
    inhand_s: float = 0.0
    # 「杆自身位移」与「手自身位移」的最大值，以及取到最大值那一刻的位移向量。
    rod_move: float = 0.0
    hand_move: float = 0.0
    rod_vec: tuple[float, float] | None = None
    hand_vec: tuple[float, float] | None = None

    @property
    def carried(self) -> bool:
        """这段证据够不够认作「这根杆确实被手带着走」。

        两条任一成立：

        - **杆持续落在手里、并且确实跟着手在动**：落手面积达标的帧数
          ≥ ``_CARRY_INHAND_FRAMES``，**且**杆自身位移
          ≥ ``_CARRY_INHAND_ROD_MOVE_MIN``、达到手位移的
          ``_CARRY_INHAND_FOLLOW_RATIO`` 以上。面积这一条单独不够：工人
          手伸进盘里放杆时，手框会大面积盖住躺在盘里的那根杆（落手
          0.90、连续 7 帧），但杆自己几乎没动——那是装料，不是取料。
        - **杆跟着手动**：杆自身位移 ≥ ``_CARRY_ROD_MOVE_MIN``，且达到手
          位移的 ``_CARRY_ROD_FOLLOW_RATIO`` 以上、方向同向。只看杆有没有
          动不够——手一划过去、杆被碰歪一点也会动。

        实测（10 段示例视频，严格锁杆后）：

        - 真取料（走面积分支）：落手达标 1~32 帧，杆/手位移比 0.49~1.21；
        - 真取料（走位移分支）：杆/手位移比 0.6~1.2；
        - 示例⑧：落手达标 0 帧，杆位移 0.000 / 手 0.028；
        - 示例⑨：落手达标 0 帧，杆位移 0.001 / 手 0.061；
        - 示例①「手伸进盘点按夹具」：落手 0 帧，杆/手位移比 0.21；
        - **装料放杆（示例 091916 0.47s）**：落手达标 7 帧、面积 0.90，
          但杆位移 0.002 / 手 0.027（比值 0.07）→ 必须判否。
        """

        if self.inhand_s >= _CARRY_INHAND_S - 1e-9:
            # 面积够了，还要杆确实跟着手动了。放杆时杆本身几乎不动。
            if self.rod_move >= _CARRY_INHAND_ROD_MOVE_MIN and (
                self.hand_move <= 0.0
                or self.rod_move / self.hand_move >= _CARRY_INHAND_FOLLOW_RATIO
            ):
                # 手一帧都没动（``hand_move <= 0``）时比值没有意义，不能
                # 当成「跟着手走」的正面证据：现场「手在盘外、手边那根杆
                # 在盘里小幅晃动」也只有杆自己在动（实测 rod_move 0.0089、
                # hand_move 0.0）。这种情形要求杆的位移明显大过晃动门限，
                # 否则只凭「落手面积够 + 杆轻轻晃」就判成取料（示例①按夹具）。
                if self.hand_move > 0.0 or self.rod_move >= _CARRY_ROD_MOVE_MIN:
                    return True
        if self.rod_move < _CARRY_ROD_MOVE_MIN or self.hand_move < _CARRY_ROD_MOVE_MIN:
            return False
        # 杆跟着手动：两者朝同一个方向走。只看位移大小不够——夹具上的杆会
        # 随检测框抖动慢慢漂移（示例⑨ 0.0097 已接近门限），而手是往盘下沿
        # 去的，方向完全不同。
        return self._aligned(self.rod_vec, self.hand_vec, _CARRY_FOLLOW_COS)

    @staticmethod
    def _aligned(
        rod_vec: tuple[float, float] | None,
        hand_vec: tuple[float, float] | None,
        min_cos: float,
    ) -> bool:
        """两个位移向量是否同向（夹角的余弦 ≥ ``min_cos``）且长度相当。"""

        if rod_vec is None or hand_vec is None:
            return False
        rn = math.hypot(*rod_vec)
        hn = math.hypot(*hand_vec)
        if rn <= 0.0 or hn <= 0.0:
            return False
        if rn < hn * _CARRY_ROD_FOLLOW_RATIO:
            return False
        cosine = (rod_vec[0] * hand_vec[0] + rod_vec[1] * hand_vec[1]) / (rn * hn)
        return cosine >= min_cos

    def reset(self) -> None:
        self.__init__()


@dataclass
class _ScrewExit:
    """一根丝杆「确实在盘里 → 被带到盘外」的观测过程（第二条取料判据）。

    这条判据**完全不看手**，只看杆自己在哪，所以「确实在油盘中」要靠两条
    独立证据来保证：

    - ``static_s``：杆心在盘区内、且杆**没被拿动**的累计时长
      （``_screw_static``）。盘里那排杆长期不动；被手举着、从盘区上空经过
      的杆一直在动，累计不上。满 ``screw_exit_pick_inside_s`` 才算「这根杆
      确实躺在油盘里」，这是判取料的前提；
    - ``outside_since_ts``：杆心越过盘区边线 ``screw_exit_pick_margin`` 之后的
      起点。要连续在外一段时间才确认（见 ``_screw_exit_hold_s``），贴着
      边线抖一下不算。

    一根杆一条记录，键是专用丝杆跟踪器（``_screw_tracker``）的轨迹 ID。
    """

    track_id: int
    # 本帧杆心是否在盘区内。
    inside: bool = False
    # 「在盘里且**没被拿动**」的累计时长（秒）。只有杆静止时累加；杆一动就
    # 清零——这是「它原本躺在油盘里」，不是「它正从盘区上空经过」。
    static_s: float = 0.0
    # **已经确认这根杆确实躺在油盘里**：``static_s`` 够过
    # ``screw_exit_pick_inside_s`` 之后锁存。取料一开始杆就动、``static_s``
    # 随之清零，不留这个锁就再也成立不了——而它恰恰是取料的前提。
    confirmed: bool = False
    # 这一趟第一次被看到在盘区内的时刻。
    entered_ts: float | None = None
    # 杆在盘区里开始移动的时刻——取料动作的真正起点，作为事件区间起点。
    moving_since_ts: float | None = None
    # 杆心已越过盘区边线、待确认出盘的起点与已累计帧数。
    outside_since_ts: float | None = None
    outside_frames: int = 0
    # 连续未检出帧数。
    misses: int = 0
    # 这一趟已经报过取料：防止同一次出盘被重复上报。
    reported: bool = False
    # 上一帧的框尺寸（像素），用来要求「配对的是同一根杆」：环境里同为
    # screw 类的框尺寸差很远（盘里的杆 275×30、盘区下方的方块 150×150），
    # 只看位置最近会配错。
    last_size: tuple[float, float] | None = None
    # 记录建立以来见过的最大短边（杆的粗细）。盘里那根 270×30；盘区下方
    # 被误检成 screw 的方块短边 130 以上。出盘时用它把关，见
    # ``_advance_screw_exit``。
    max_short: float = 0.0
    # 上一帧的杆心（归一化）与像素速度，用于预测下一帧位置做关联。
    #
    # 不能用通用跟踪器维护杆的身份：盘里那排杆彼此只隔 0.02~0.03 对角线，
    # 跟踪器的关联代价会把相邻两根并成一条轨迹（实测 tid 从 (1529,222)
    # 一帧跳到 (1444,141)），「这根杆一直在盘里」的证据就永远攒不起来。
    last_center: tuple[float, float] | None = None
    last_ts: float | None = None
    velocity: tuple[float, float] = (0.0, 0.0)
    # 上一帧这根杆是不是**没动过**（``_screw_static``）。宽关联门限只给
    # 在动的记录用——静止的杆之间只隔 0.02~0.03，放宽就会互相串。
    static: bool = True
    # 记录已经存活多久（秒），以及在这期间它有没有**动过**。
    #
    # 盘里那排杆从视频开头就在、一直不动，如果给每根都留着记录，记录会
    # 越积越多（实测 122925 有 14 条同时活着），后来真的被取走的那根反而
    # 抢不到新记录——它被当成「已有记录的一帧」，那条记录的 ``inside=True``
    # 却从没攒出新证据。所以：静止记录活过 ``_SCREW_EXIT_IDLE_RELEASE_S``
    # 就释放，等它真动起来再重建；动过的记录（``active=True``）才长期留。
    age_s: float = 0.0
    active: bool = False

    def size_compatible(self, box: BBox) -> bool:
        """这个框和记录里那根杆的尺寸是不是同一量级。

        丝杆框的**长宽比**很稳定（细长条），但长度会随遮挡变化（实测同一
        根杆 151~293 px）。所以只看长短边各自的比例，宽放得比长严：宽度
        是杆的粗细，被挡住也不会变；长度是杆露出来多少，会变。
        """

        if self.last_size is None:
            return True
        last_w, last_h = self.last_size
        # 统一成「长边、短边」，不受杆的朝向影响。
        prev_long, prev_short = max(last_w, last_h), min(last_w, last_h)
        long_side, short_side = max(box.width, box.height), min(box.width, box.height)
        if prev_short <= 0 or prev_long <= 0 or short_side <= 0:
            return True
        # 盘里两根杆的框尺寸几乎一样（实测 270×30 与 272×30），所以尺寸
        # **不能**用来在它们之间分身份——那一层交给位置。这里只拦「根本不是
        # 同一类东西」：盘区下方那些 136×130 的方块、零件盒里的小件。
        # 短边（粗细）放宽到 0.4~2.5，长边（露出长度，会被遮挡）0.3~3.0。
        if not (0.4 <= short_side / prev_short <= 2.5):
            return False
        return 0.3 <= long_side / prev_long <= 3.0

    def restart_excursion(self) -> None:
        """重新开始一趟「在盘里 → 出盘」的观测（进盘或报过之后回盘）。"""

        self.inside = True
        self.static_s = 0.0
        self.confirmed = False
        self.moving_since_ts = None
        self.outside_since_ts = None
        self.outside_frames = 0
        self.reported = False



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
    # 带料进盘时，手里那根杆在进盘那一刻的位置。离盘时用来回答一个
    # ``holding_on_entry`` 回答不了的问题：带走的是**带进来的那根**
    # （放回去 / 换个位置再拿走，不是取料），还是**盘里的另一根**
    # （带进来一根、又取走一根，那仍然是取料）。
    entry_held: tuple[float, float] | None = None
    # 本趟**第一次**观测到「手在盘区内」的时刻，即「盘内停留」的起点。
    # 用「首次进盘 → 结算」的区间，而不是逐帧累加 inside 帧的 dt：
    # 手在盘内时可能整段漏检（示例①第 3 次取料只有 4 帧中心落在盘区，
    # 逐帧累加只有 0.27s < pick_confirm_s，真实取料被误杀）。区间度量
    # 同时满足「盘外跟踪再久也不算进盘停留」——起点是进盘那一帧。
    first_inside_ts: float | None = None
    # 已确认「在盘内」。
    inside_tray: bool = True
    last_seen_ts: float = 0.0
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
    # 离盘后「手边有盘内杆」的帧数。只作诊断字段；真正决定「握着杆往盘外
    # 带走」的是 ``carry.carried``（按落手面积时长与杆/手位移判）。
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
    # 离盘后「等杆出盘」那段的证据（见 ``_CarryEvidence``）。判定真正
    # 「握着杆往盘外走」用的是它，而不是亮着的 frames 计数：手在盘外、
    # 手边有盘内杆，既不等于握住了，也不等于杆会跟着动。
    carry: _CarryEvidence = field(default_factory=lambda: _CarryEvidence())
    # 二级判定模型的输入：**原始检测框**的逐帧测量聚合（见 raw_features）。
    # 只喂模型用，不参与任何规则判定——规则判定仍走上面那些字段。
    raw: PickAggregator = field(default_factory=PickAggregator)
    # 这趟进盘开始时盘区里的杆数。聚合特征要用「进盘到结算」的差值。
    raw_entry_rods: int | None = None
    # 上一个采样点的时刻。采样节拍按**时间**算，不按帧数：
    # ``pick_sample_every_frames`` 是「基准帧率下每 k 帧」，实际间隔取
    # ``k / max_fps × frame_scale``（见 :meth:`StationPipeline._sample_period_s`）。
    # 按帧数计时高帧率会让同一段动作攒到更多采样点、窗口更早攒满，
    # 判定结果随帧率漂移（实测 20fps 比 15fps 少判 4 件）。
    last_sample_ts: float | None = None

    def due(self, ts: float, period_s: float) -> bool:
        """本帧是不是采样点（按**时间**间隔，与帧率无关）。

        返回 0/1 语义：同一帧的读数只能证明「这一刻手是什么状态」，复制成
        多份等于伪造证据（实测把离盘后唯一一帧「手里有杆」放大成多数票，
        会把空手离开判成取料）。采样密度上限是「每帧一个」，低帧率时自然
        比高帧率稀，这是采样能力的边界，不是判定口径随帧率漂移。
        """

        if self.last_sample_ts is None:
            self.last_sample_ts = ts
            return True
        if ts - self.last_sample_ts < period_s - 1e-9:
            return False
        self.last_sample_ts = ts
        return True

    def push_inside(self, inside: bool, keep: int) -> None:
        _push(self.inside_samples, bool(inside), keep)

    def push_holding(self, holding: bool, keep: int) -> None:
        """记录盘外握持采样；离盘后重新开始累积（离盘前的属于取料前）。"""

        _push(self.holding_samples, bool(holding), keep)

    def restart_holding(self) -> None:
        """离盘后重置握持窗口：从此刻起重新累积。"""

        self.holding_samples.clear()

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
    # 本段里「杆被手压住整帧没检出、但刷头仍贴着锚点」累计补进来的时长，
    # 以及最后一次真接触之后连续补了多少帧。见
    # ``StationPipeline._oiling_rod_occluded_contact``。
    occluded_s: float = 0.0
    # ---- 复核 / 二级判定用的接触统计（每帧更新，不参与规则判定）----
    # 接触候选里那只手确实握着这根杆的帧数。与 ``frames`` 的差别在
    # 「杆在手边」还是「杆在手里」——正刷着时两者一致，刷到盘子边上的
    # 存货时前者成立、后者不成立。
    held_frames: int = 0
    # 这段接触里杆自身移动的最大归一化距离（手换位置刷时杆也应跟着动）。
    rod_move: float = 0.0
    # 杆第一次进入接触时的位置，用来算 ``rod_move``。
    rod_start: tuple[float, float] | None = None
    # 二级判定模型的输入：刷涂期间**原始检测框**的逐帧测量聚合。
    # 只喂模型用，不参与规则判定。
    raw: OilingAggregator = field(default_factory=OilingAggregator)


