"""PLC 警示灯：4 工步状态映射与后台写入线程。

协议与现场实测见 ``docs/12-PLC警示灯通信与控制.md``。

字节布局（首地址从 0 开始）::

    HR0 = VB0 | (VB1 << 8)      VB0 运行中标记 / VB1 良品记录
    HR1 = VB2 | (VB3 << 8)      VB2 不良记录   / VB3 恒 0（本方案不读）

本系统只有两个判定点（取料、人工二次涂油），现场流程是 4 个工步::

    等待取料 → 取料完成 → 涂油中 → 涂油完成/异常

工步位（bit(n-1) = 第 n 步）::

    状态             VB0        VB1        VB2
    等待取料       0b0001     0b0000        0
    取料完成       0b0010     0b0001        0
    涂油中         0b0100     0b0011        0
    涂油完成(良)   0b0000     0b0111        0
    涂油异常(NG)   0b0000     0b0011     0b0100
    未检测         0x00       保持       保持

三条语义约定：

- ``VB0`` 是「**正在检测**的工步」位。判定结束后不再是任何一步，所以
  涂油完成/异常时 ``VB0 = 0``——这是「这一件做完了」，不是「上位机掉线」
  （掉线也写 0，靠 VB1/VB2 是否已有结论区分）。
- ``VB1`` 是「**已经通过**的工步」累计位：第 1 步通过就 0b0001，前两步
  通过就 0b0011，三步全过就 0b0111。取料完成即置第 1 位。
- ``VB2`` 记「**失败**的那一步」位：只有漏涂时置第 3 位（``0b0100``），
  此时 VB1 停在 ``0b0011``（前两步已过）。

**PLC 只负责显示信号，不做任何处理**，所以运行期只写不读：VB3 恒写 0，
不做读-改-写，也没有人工改判回写。若将来 PLC 侧启用 VB3 输入，必须先改成
读-改-写（VB3 与 VB2 共用 HR1）。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .config import PlcConfig
from .plc_link import ModbusError, ModbusLinkError, PlcLink
from .types import TaskResult, TaskStage

logger = logging.getLogger("oilcheck.plc")


class ProcessStep(str, Enum):
    """现场流程的 4 个工步，外加「未检测」。"""

    IDLE = "IDLE"  # 工步 1 等待取料
    PICKED = "PICKED"  # 工步 2 取料完成
    OILING = "OILING"  # 工步 3 涂油中
    FINISHED = "FINISHED"  # 工步 4 涂油完成 / 异常
    OFFLINE = "OFFLINE"  # 未检测（相机离线、无启用工位、判定停摆）


STEP_LABELS: dict[ProcessStep, str] = {
    ProcessStep.IDLE: "等待取料",
    ProcessStep.PICKED: "取料完成",
    ProcessStep.OILING: "涂油中",
    ProcessStep.FINISHED: "涂油完成/异常",
    ProcessStep.OFFLINE: "未检测",
}

# 工步的显示序号（日志与 UI 用）。OFFLINE 没有工步号，记 0。
STEP_NUMBERS: dict[ProcessStep, int] = {
    ProcessStep.IDLE: 1,
    ProcessStep.PICKED: 2,
    ProcessStep.OILING: 3,
    ProcessStep.FINISHED: 4,
    ProcessStep.OFFLINE: 0,
}

# 工步的前进顺序。映射层用它保证同一件内工步只进不退。
_STEP_ORDER: dict[ProcessStep, int] = {
    ProcessStep.OFFLINE: -1,
    ProcessStep.IDLE: 0,
    ProcessStep.PICKED: 1,
    ProcessStep.OILING: 2,
    ProcessStep.FINISHED: 3,
}

# VB0 的运行中标记：**正在检测**哪一步（bit(n-1)）。判定结束的件不再是每一步，
# 所以涂油完成/异常时 VB0 写 0——PLC 看到 0 表示「本件已出结论或未检测」，
# 靠 VB1/VB2 里是否已有第 3 步的结论区分。
_VB0_BY_STEP: dict[ProcessStep, int] = {
    ProcessStep.IDLE: 0b0001,
    ProcessStep.PICKED: 0b0010,
    ProcessStep.OILING: 0b0100,
    ProcessStep.FINISHED: 0x00,
    ProcessStep.OFFLINE: 0x00,
}


@dataclass(frozen=True)
class PlcFrame:
    """要写到 PLC 的 4 个字节 + 可读的过程信息（日志/UI/审计用）。"""

    vb0: int
    vb1: int
    vb2: int
    step: ProcessStep
    task_id: str | None = None
    reason: str = ""

    @property
    def vb3(self) -> int:
        """VB3 恒 0：PLC 只显示，不提供人工改判输入。"""

        return 0

    @property
    def hr0(self) -> int:
        return (self.vb0 & 0xFF) | ((self.vb1 & 0xFF) << 8)

    @property
    def hr1(self) -> int:
        return (self.vb2 & 0xFF) | ((self.vb3 & 0xFF) << 8)

    @property
    def registers(self) -> tuple[int, int]:
        return (self.hr0, self.hr1)

    @property
    def step_number(self) -> int:
        return STEP_NUMBERS.get(self.step, 0)

    @property
    def step_label(self) -> str:
        return STEP_LABELS.get(self.step, self.step.value)

    def describe(self) -> str:
        return (
            f"VB0=0x{self.vb0:02X} VB1=0x{self.vb1:02X} VB2=0x{self.vb2:02X}"
            f" 工步{self.step_number} {self.step_label}"
            f"（HR0=0x{self.hr0:04X} HR1=0x{self.hr1:04X}）"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vb0": self.vb0,
            "vb1": self.vb1,
            "vb2": self.vb2,
            "vb3": self.vb3,
            "step": self.step.value,
            "step_number": self.step_number,
            "step_label": self.step_label,
            "task_id": self.task_id,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PlcEvidence:
    """一拍输入：来自 ``OilCheckService.station_snapshot()`` 的最小投影。"""

    online: bool
    task_id: str | None = None
    stage: TaskStage | None = None
    final_result: TaskResult | None = None
    brush_hold_s: float = 0.0
    # 是否存在**另一条**已判终态的周期（当前任务之外）。它表示「上一件刚
    # 判完」，而当前任务已经是下一件：中间那一帧终态很容易被 200ms 的轮询
    # 采样跨过去，漏涂（NG）就永远写不进 PLC。有它就先把上一件的结论报完。
    superseded_task_id: str | None = None
    superseded_result: TaskResult | None = None
    superseded_brush_hold_s: float = 0.0

    @classmethod
    def offline(cls) -> PlcEvidence:
        return cls(online=False)


@dataclass(frozen=True)
class _StepBits:
    """工步号配置：取料记在哪一步、涂油结论记在哪一步。

    默认 1/3：现场流程是「等待取料 → 取料完成 → 涂油中 → 涂油完成/异常」，
    取料占第 1 步、涂油占第 3 步（第 2 步是「取料已完成、等待涂油」的过程位）。
    """

    pick: int = 1
    oiling: int = 3

    @classmethod
    def from_steps(cls, steps: dict[str, int]) -> _StepBits:
        def _clamp(raw: Any, default: int) -> int:
            try:
                value = int(raw)
            except (TypeError, ValueError):
                return default
            return value if 1 <= value <= 8 else default

        return cls(
            pick=_clamp(steps.get("pick"), 1),
            oiling=_clamp(steps.get("oiling"), 3),
        )


def _bit(step: int) -> int:
    """第 n 步（1 起）对应的位值。"""

    return 1 << (step - 1)


def _cumulative_mask(last_step: int) -> int:
    """第 1 步到第 ``last_step`` 步全通过时的累计掩码（VB1 用）。

    现场语义是「已经通过的工步全部置 1」：前两步通过 = 0b0011，三步全过 =
    0b0111。所以掩码是低位连续的若干位，而不是只置最后一步。
    """

    last = max(1, min(8, int(last_step)))
    return (1 << last) - 1


@dataclass
class _MapperState:
    """``PlcStepMapper`` 的可变内部状态。

    单独拆出来是为了让「映射规则」与「状态」分开读：规则在
    :meth:`PlcStepMapper.update`，状态只有这 6 个字段。
    """

    last_task_id: str | None = None
    current_step: ProcessStep = ProcessStep.IDLE
    finished_at: float | None = None
    # 终态任务的结论，用于保持窗内维持 VB1/VB2；
    # 保持窗结束后回到工步 1 时清空（新件清零）。
    finished_result: TaskResult | None = None
    # 这一件的结论已经报过、保持窗也结束了：它继续留在快照里（保留策略是
    # 按条数，不是按时间），不能再报一次，否则工步会在 1 与 4 之间循环。
    hold_done_task_id: str | None = None
    # 快照里第一次出现「没有任务」的墙钟时刻。用于超时回落：用户点
    # 「重置状态」、示例回绕、切换检测器都会清空 _cycles，此前若停在
    # 「涂油中」，工步会一直冻在那里（警示灯整夜显示涂油中）。
    no_task_since: float | None = None
    last_frame: PlcFrame | None = None


# 「快照里没有任务」持续超过这个时长就回落 IDLE。取得比完成保持窗长，
# 避免清理把工步从 4 拉回 1；但又必须有限，否则重置后永久冻结。
_NO_TASK_IDLE_TIMEOUT_S = 5.0


def _resolve_step(ev: PlcEvidence) -> tuple[ProcessStep, str]:
    """按证据判定「本拍应该处于哪个工步」。不考虑历史与保持窗。"""

    if ev.final_result in (TaskResult.COMPLETED, TaskResult.NG):
        return ProcessStep.FINISHED, "周期已终态"
    if ev.final_result == TaskResult.PROCESSING:
        if ev.stage == TaskStage.PICKED:
            if ev.brush_hold_s > 0.0:
                return ProcessStep.OILING, "刷头已靠近"
            return ProcessStep.PICKED, "已取料未开始涂油"
        # WAITING_PICK：运行期不会停留（_ensure_cycle 建完立刻转 PICKED），
        # 见到它说明刚建周期，按工步 1 处理。
        return ProcessStep.IDLE, "周期刚建立"
    return ProcessStep.IDLE, "无活动周期"


class PlcStepMapper:
    """把作业周期状态映射成 PLC 的 4 个字节。纯逻辑，无 I/O，可单测。

    三条单调性约束，全部来自现场观察：

    1. 同一 ``task_id`` 内工步只能前进，不许回退。涂油累计时长可能因为
       刷子短暂离开而重新变 0，但「已经刷上了」这个事实不会撤销；
    2. 收到终态（``COMPLETED`` / ``NG``）后，该任务的任何后续中间态都忽略；
       保持窗结束后这一件的结论视为已报完，残留的终态证据不再触发工步 4；
    3. ``pipeline._cleanup`` 只保留最近 4 条结束周期，活动周期结束后到快照
       可见之间有观察空档。此时**不能**回工步 1，要等保持窗自然到期。
    """

    def __init__(self, *, hold_s: float = 1.0, steps: dict[str, int] | None = None):
        self._hold_s = max(0.0, float(hold_s))
        self._bits = _StepBits.from_steps(steps or {})
        self._state = _MapperState()

    @property
    def hold_s(self) -> float:
        return self._hold_s

    @property
    def last_frame(self) -> PlcFrame | None:
        return self._state.last_frame

    def reset(self) -> None:
        self._state = _MapperState()

    def update(self, ev: PlcEvidence, now: float) -> PlcFrame:
        """算出一帧。``now`` 是**墙钟**时间（不能用媒体时间）。"""

        st = self._state

        if not ev.online:
            # 未检测：只清 VB0，记录字节原样保留——PC 掉线不该把已经写出的
            # 良品/不良记录抹掉，那是这一件的历史。
            prev = st.last_frame
            frame = PlcFrame(
                vb0=0x00,
                vb1=prev.vb1 if prev else 0,
                vb2=prev.vb2 if prev else 0,
                step=ProcessStep.OFFLINE,
                task_id=ev.task_id or (prev.task_id if prev else None),
                reason="相机离线或未检测",
            )
            st.last_frame = frame
            return frame

        # ---- 1. 完成保持窗优先 ----
        # 窗内必须维持工步 4，哪怕下一件的取料已经发生：PLC 要靠这段窗口
        # 读走结论。窗结束后回工步 1，同一次写入把良品/不良记录清零。
        if st.current_step == ProcessStep.FINISHED and st.finished_at is not None:
            if now - st.finished_at < self._hold_s:
                frame = self._make_frame(
                    ProcessStep.FINISHED, st.last_task_id, "完成保持窗内"
                )
                st.last_frame = frame
                return frame
            finished_task = st.last_task_id
            st.hold_done_task_id = finished_task
            st.current_step = ProcessStep.IDLE
            st.finished_at = None
            st.finished_result = None
            # last_task_id 保留：这一件还在快照里，靠 hold_done_task_id 挡住
            # 重复上报；真有新件时 task_id 会变，自然走「新的一件」分支。
            frame = self._make_frame(
                ProcessStep.IDLE, finished_task, "保持窗结束，回到等待取料"
            )
            st.last_frame = frame
            return frame

        # ---- 2. 上一件的结论还没报，证据已经是下一件 ----
        # 漏涂的主要路径是被下一次取料顶掉（``_supersede_unfinished``）。
        # 200ms 的轮询很容易把那一帧终态跨过去，等到看见新周期时上一件的
        # NG 还没写进 PLC，工步会直接从 3 跳到 2。这里先补报上一件。
        superseded = ev.superseded_task_id
        if (
            superseded
            and superseded != ev.task_id
            and superseded != st.hold_done_task_id
        ):
            st.last_task_id = superseded
            st.current_step = ProcessStep.FINISHED
            st.finished_at = now
            st.finished_result = ev.superseded_result or TaskResult.NG
            frame = self._make_frame(
                ProcessStep.FINISHED, superseded, "上一件已终态，先报结论"
            )
            st.last_frame = frame
            return frame

        candidate, reason = _resolve_step(ev)

        if ev.task_id is None:
            # 快照里没有任务：可能是刚结束的件被 ``_cleanup`` 回收，也可能
            # 是启动初期还没有周期、用户点了「重置状态」、示例回绕或切换
            # 检测器。短暂保持当前工步让保持窗自己到期（否则清理会让工步
            # 从 4 跳回 1，PLC 读到假的「下一件」）；但不能无限保持——
            # 重置后没有下一件时工步会永久停在「涂油中」，警示灯整夜亮着。
            if st.no_task_since is None:
                st.no_task_since = now
            if now - st.no_task_since < _NO_TASK_IDLE_TIMEOUT_S:
                candidate = st.current_step
                reason = "快照暂无任务，保持当前工步"
            else:
                candidate = ProcessStep.IDLE
                st.current_step = ProcessStep.IDLE
                st.finished_at = None
                st.finished_result = None
                st.hold_done_task_id = None
                st.last_task_id = None
                reason = "快照暂无任务且已超时，回到等待取料"
        elif ev.task_id != st.last_task_id:
            # 新的一件：工步从头开始，不受上一件的单调性约束。
            st.last_task_id = ev.task_id
            st.no_task_since = None
            st.finished_at = None
            st.finished_result = None
            st.current_step = candidate
        elif ev.task_id == st.hold_done_task_id:
            # 这一件的结论已经报过、保持窗也结束了。它还会在快照里留一阵
            # （保留策略按条数），不能再报一次。
            candidate = ProcessStep.IDLE
            reason = "本件结论已报完"
        elif _STEP_ORDER[candidate] < _STEP_ORDER[st.current_step]:
            # 同一件：工步只进不退。涂油累计时长可能因刷子短暂离开变 0，
            # 但「已经刷上了」这个事实不会撤销。
            candidate = st.current_step
            reason = "工步不后退（保持上一拍）"
        else:
            st.no_task_since = None

        if candidate == ProcessStep.FINISHED:
            st.current_step = ProcessStep.FINISHED
            st.finished_at = now
            st.finished_result = (
                ev.final_result or st.finished_result or TaskResult.COMPLETED
            )
        else:
            st.current_step = candidate

        frame = self._make_frame(candidate, st.last_task_id, reason)
        st.last_frame = frame
        return frame

    # -- 内部 ---------------------------------------------------------
    def _state_step(self, step: ProcessStep) -> int:
        """这个状态对应「正在检测第几步」（1 起）。0 = 不在检测任何步。"""

        if step == ProcessStep.IDLE:
            return self._bits.pick
        if step == ProcessStep.PICKED:
            # 取料已过、等涂油：这一步夹在取料与涂油之间。
            return self._bits.pick + 1
        if step == ProcessStep.OILING:
            return self._bits.oiling
        return 0  # FINISHED / OFFLINE：本件已出结论，不在检测任何步

    def _passed_mask(self, through: int) -> int:
        """第 1 步到第 ``through`` 步已通过的累计掩码。"""

        if through < 1:
            return 0x00
        return _cumulative_mask(through)

    def _make_frame(
        self, step: ProcessStep, task_id: str | None, reason: str
    ) -> PlcFrame:
        current = self._state_step(step)
        vb0 = 0x00 if current <= 0 else _bit(current)
        if step in (ProcessStep.IDLE, ProcessStep.OFFLINE):
            vb1 = 0x00
            vb2 = 0x00
        elif step == ProcessStep.FINISHED:
            if self._state.finished_result == TaskResult.NG:
                # 漏涂：前序工步已通过，失败的是涂油那一步。
                vb1 = self._passed_mask(self._bits.oiling - 1)
                vb2 = _bit(self._bits.oiling)
            else:
                # 良品：从第 1 步累计到涂油步全通过。
                vb1 = self._passed_mask(self._bits.oiling)
                vb2 = 0x00
        else:
            # 取料完成 / 涂油中：只把「已经通过」的工步写进 VB1。
            vb1 = self._passed_mask(current - 1)
            vb2 = 0x00
        return PlcFrame(
            vb0=vb0,
            vb1=vb1,
            vb2=vb2,
            step=step,
            task_id=task_id,
            reason=reason,
        )


# ----------------------------------------------------------------------
# 后台写入线程
# ----------------------------------------------------------------------
EvidenceSource = Callable[[], PlcEvidence]


@dataclass
class PlcStatus:
    """给 UI 读的运行状态。"""

    enabled: bool = False
    dry_run: bool = True
    connected: bool = False
    step: ProcessStep = ProcessStep.OFFLINE
    frame: PlcFrame | None = None
    last_write_at: float | None = None
    last_error: str = ""
    write_failures: int = 0
    reconnects: int = 0
    # 启动时的连通性自检结果（None = 还没检测 / 未启用）
    startup_ok: bool | None = None
    startup_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "dry_run": self.dry_run,
            "connected": self.connected,
            "step": self.step.value,
            "step_number": STEP_NUMBERS.get(self.step, 0),
            "step_label": STEP_LABELS.get(self.step, ""),
            "frame": None if self.frame is None else self.frame.to_dict(),
            "last_write_at": self.last_write_at,
            "last_error": self.last_error,
            "write_failures": self.write_failures,
            "reconnects": self.reconnects,
            "startup_ok": self.startup_ok,
            "startup_message": self.startup_message,
        }


@dataclass
class _IoState:
    """写入线程的内部可变状态（与 :class:`PlcStatus` 分开，status 是投影）。"""

    next_attempt_mono: float = 0.0
    backoff_index: int = 0
    last_write_mono: float = 0.0
    last_written: tuple[int, int] | None = None


class PlcIoThread:
    """独立后台线程：每拍算一次映射，必要时写 PLC。

    两条硬性约束：

    1. **不能占用推理线程**。视觉每帧都在跑检测，网络 I/O 一阻塞就直接掉帧，
       破坏「盘内停留时长」这类时序证据；
    2. **只能有一条 socket**。实测第二条连接的 SYN 必然超时，多线程各自建连
       会把自己的会话饿死。
    """

    def __init__(
        self,
        *,
        cfg: PlcConfig,
        evidence_source: EvidenceSource,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._cfg = cfg
        self._evidence = evidence_source
        self._log = (logger_ or logger).getChild("io")
        self._link = PlcLink(
            cfg.host,
            cfg.port,
            unit=cfg.unit_id,
            connect_timeout_s=cfg.connect_timeout_s,
            request_timeout_s=cfg.request_timeout_s,
        )
        self._mapper = PlcStepMapper(hold_s=cfg.hold_s, steps=cfg.steps)
        self._backoff = tuple(
            max(0, int(x)) / 1000.0 for x in cfg.reconnect_backoff_ms
        ) or (0.2,)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status = PlcStatus(enabled=cfg.enabled, dry_run=cfg.dry_run)
        self._io = _IoState()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动 PLC 线程。

        **启动前先在后台测一次连通性**（不阻塞调用方）：现场最需要的是开机
        就明确知道「PLC 通不通」，而不是等到第一件判定才发现写不进去。
        结果写进 ``status()["startup_ok"]`` / ``["startup_message"]``，UI 与
        日志都能看到。

        检测失败**不阻止**启动——PLC 只是显示信号，拖住视觉检测代价更大；
        线程起来后会按退避自行重连。
        """

        if not self._cfg.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="oilcheck-plc", daemon=True
        )
        self._thread.start()
        self._log.info(
            "PLC 线程已启动（%s:%s unit=%d 只读=%s 保持窗=%.1fs）",
            self._cfg.host,
            self._cfg.port,
            self._cfg.unit_id,
            "是" if self._cfg.dry_run else "否",
            self._cfg.hold_s,
        )

    def check_connection(self) -> tuple[bool, str]:
        """测一次 PLC 连通性：建连 + 读 HR0..HR1。**不写任何寄存器**。

        返回 ``(是否成功, 说明)``。只读联调模式下也一样能测——读不需要
        什么权限，也不会改 PLC 状态。

        该 PLC 只允许一条 Modbus TCP 会话：生产模式下写入线程一直占着
        socket（写完不关、靠 1s 重写保活），第二条连接按现场实测会 SYN
        超时。所以这里先关掉写入线程自己的连接再探测，探测完让线程按
        退避自行重连，否则「测试连接」在生产模式下必然失败，提示还会把
        现场引向「检查博途在线监控」。
        """

        base = self._cfg.register_base
        owns_link = self._thread is not None and self._thread.is_alive()
        if owns_link:
            # 与写入线程共用同一个 PlcLink 对象：close() 之后线程下一拍
            # 会发现 not connected 并重新 open()。
            self._link.close()
        try:
            self._link.open()
            registers = self._link.read_holding(base, 2)
        except (ModbusLinkError, ModbusError) as exc:
            return False, str(exc)
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
        finally:
            self._link.close()
            if owns_link:
                # 让写入线程立刻重写完整帧：上次写入的值已不可信。
                self._io.last_written = None
                self._io.next_attempt_mono = 0.0
        return True, (
            f"{self._cfg.host}:{self._cfg.port} 可读，"
            f"HR{base}=0x{registers[0]:04X} HR{base + 1}=0x{registers[1]:04X}"
        )

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._link.close()
        with self._lock:
            self._status.connected = False
        self._log.info("PLC 线程已停止")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status.to_dict()

    def _startup_probe(self) -> None:
        """开机连通性检测：结果写进 status，失败只告警不退出。"""

        ok, message = self.check_connection()
        with self._lock:
            self._status.startup_ok = ok
            self._status.startup_message = message
            self._status.connected = ok
        # 一行讲清「连的谁、通不通、读到了什么」。INFO 级，控制台能直接看到，
        # 现场不用翻日志页就知道 Modbus 联上没有。
        head = (
            f"Modbus {self._cfg.host}:{self._cfg.port} "
            f"unit={self._cfg.unit_id} 基址={self._cfg.register_base}"
            f"（只读={'是' if self._cfg.dry_run else '否'}）"
        )
        if ok:
            self._log.info("PLC 启动检测通过：%s；%s", head, message)
            return
        self._log.warning(
            "PLC 启动检测失败：%s；%s。检测服务照常运行，PLC 线程会按退避重连；"
            "若持续失败，检查 %s:%s 是否被博途在线监控/组态软件占用"
            "（该 PLC 只允许一条 Modbus 会话）",
            head,
            message,
            self._cfg.host,
            self._cfg.port,
        )

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def _run(self) -> None:
        # 第一步就是连通性检测：放在线程里做，既满足「启动前测一次」，
        # 又不阻塞调用方（service.start() 跑在 UI 线程上）。
        self._startup_probe()
        interval = max(0.02, self._cfg.poll_interval_ms / 1000.0)
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._tick()
            except Exception:  # noqa: BLE001
                # PLC 侧的任何异常都不能让线程死掉：视觉判定还在跑，
                # 掉线告警比静默停止有用得多。
                self._log.exception("PLC 写入线程异常")
            elapsed = time.monotonic() - started
            self._stop.wait(max(0.0, interval - elapsed))

    def _tick(self) -> None:
        now_wall = time.time()
        now_mono = time.monotonic()
        try:
            evidence = self._evidence()
        except Exception as exc:  # noqa: BLE001
            self._log.warning("读取证据失败，按离线处理: %s", exc)
            evidence = PlcEvidence.offline()

        frame = self._mapper.update(evidence, now_wall)
        with self._lock:
            self._status.step = frame.step
            self._status.frame = frame

        if self._cfg.dry_run:
            self._log.debug("PLC 帧(只读) %s 原因=%s", frame.describe(), frame.reason)
            return

        due = (
            self._io.last_written != frame.registers
            or now_mono - self._io.last_write_mono >= self._cfg.refresh_interval_s
        )
        if not due:
            return
        if now_mono < self._io.next_attempt_mono:
            return
        self._flush(frame, now_mono)

    def _flush(self, frame: PlcFrame, now_mono: float) -> None:
        try:
            if not self._link.connected:
                self._link.open()
                with self._lock:
                    self._status.connected = True
                # 重连后不依赖「值变才写」：立刻写完整帧。PLC 可能刚重启，
                # 这段数据区回到了上电残留值（实测 HR1 上电为 0x00FF）。
                self._io.last_written = None
            else:
                with self._lock:
                    self._status.connected = True
            self._link.write_registers(self._cfg.register_base, list(frame.registers))
        except (ModbusLinkError, ModbusError, ValueError) as exc:
            self._on_write_error(exc, now_mono)
            return

        self._io.last_written = frame.registers
        self._io.last_write_mono = now_mono
        self._io.backoff_index = 0
        self._io.next_attempt_mono = 0.0
        with self._lock:
            self._status.connected = True
            self._status.last_write_at = time.time()
            self._status.last_error = ""
            self._status.frame = frame
        self._log.debug("PLC 已写入 %s 原因=%s", frame.describe(), frame.reason)

    def _on_write_error(self, exc: Exception, now_mono: float) -> None:
        delay = self._backoff[min(self._io.backoff_index, len(self._backoff) - 1)]
        self._io.backoff_index = min(self._io.backoff_index + 1, len(self._backoff) - 1)
        self._io.next_attempt_mono = now_mono + delay
        self._link.close()
        self._io.last_written = None
        is_link = isinstance(exc, ModbusLinkError)
        with self._lock:
            self._status.connected = False
            self._status.last_error = f"{type(exc).__name__}: {exc}"
            self._status.write_failures += 1
            if is_link:
                self._status.reconnects += 1
        if is_link:
            self._log.warning("PLC 通信失败，%.1fs 后重连：%s", delay, exc)
        else:
            # 协议层异常（地址越界/非法值）重试也没用，但保持退避节奏，
            # 避免每 200ms 刷一条同样的日志。
            self._log.error("PLC 拒绝了写入：%s", exc)
