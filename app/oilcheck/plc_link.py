"""PLC（警示灯）Modbus TCP 传输层。

只做一件事：把一次 Modbus 事务（读/写保持寄存器）做完，并把失败明确地
抛出来。连接管理、重连退避、状态映射都在 :mod:`app.oilcheck.plc_io`。

两条硬性约束来自现场实测（`docs/12-PLC警示灯通信与控制.md` §2）：

- **PLC 只允许一条 Modbus TCP 会话**：持有连接 1 时，第二条连接的 SYN
  会超时。所以这里始终复用一个 socket，并用锁把所有请求串行化。
- **空闲 10~18s 会断连，且不干净断开**（收不到 FIN，下次读才超时）。
  调用方必须保持亚秒级节奏，任何一次超时都当成断链处理。

协议细节：

- 首地址从 0 开始（不是 40001）；
- unit id 从站不校验（实测 0/1/2/255 都接受），默认 1；
- 只实现了 FC3（读保持寄存器）与 FC16（写多寄存器）——实测线圈、离散输入、
  输入寄存器、设备标识都返回异常码。
"""

from __future__ import annotations

import socket
import struct
import threading

# Modbus 异常码 → 中文说明。现场排障时日志里直接就是人话。
EXCEPTION_TEXT: dict[int, str] = {
    1: "非法功能码（从站未实现该功能）",
    2: "非法数据地址（寄存器地址越界）",
    3: "非法数据值（数量或取值超出范围）",
    4: "从站设备故障",
    5: "已确认（从站正在处理长任务）",
    6: "从站忙",
    8: "存储奇偶校验错误",
    10: "网关路径不可用",
    11: "网关目标设备无响应",
}


class ModbusError(RuntimeError):
    """从站返回的 Modbus 异常响应。

    这是**协议层**的成功往返：PLC 收到请求并明确拒绝了。与链路错误
    （超时、断连）要分开处理——链路错误才需要重连。
    """

    def __init__(self, code: int) -> None:
        self.code = code
        text = EXCEPTION_TEXT.get(code, "未知异常码")
        super().__init__(f"exception {code}（{text}）")


class ModbusLinkError(OSError):
    """链路层错误：超时、对端关闭、连接未建立。"""


class PlcLink:
    """单连接 Modbus TCP 客户端。

    线程安全：内部一把锁，所有请求串行。**不要**为不同线程各建一个实例
    ——PLC 不接受第二条会话。
    """

    def __init__(
        self,
        host: str,
        port: int = 502,
        *,
        unit: int = 1,
        connect_timeout_s: float = 1.0,
        request_timeout_s: float = 0.5,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.unit = int(unit) & 0xFF
        self.connect_timeout_s = float(connect_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self._sock: socket.socket | None = None
        self._tid = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._sock is not None

    def open(self) -> None:
        """建立连接。失败抛 :class:`ModbusLinkError`。"""

        with self._lock:
            self._open_locked()

    def _open_locked(self) -> None:
        self._close_locked()
        try:
            sock = socket.create_connection(
                (self.host, self.port), timeout=self.connect_timeout_s
            )
        except OSError as exc:
            raise ModbusLinkError(
                f"连接 {self.host}:{self.port} 失败: {type(exc).__name__}: {exc}"
            ) from exc
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # 请求超时用 SO_RCVTIMEO 之外的超时也行，但统一在 socket 上设更直接：
        # 读超时时 recv 抛 socket.timeout（OSError 子类），下面转成链路错误。
        sock.settimeout(self.request_timeout_s)
        self._sock = sock

    def close(self) -> None:
        with self._lock:
            self._close_locked()

    def _close_locked(self) -> None:
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def __enter__(self) -> PlcLink:
        self.open()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 事务
    # ------------------------------------------------------------------
    def _request_locked(self, pdu: bytes) -> bytes:
        sock = self._sock
        if sock is None:
            raise ModbusLinkError("连接未建立")
        self._tid = (self._tid + 1) & 0xFFFF or 1
        frame = struct.pack(">HHHB", self._tid, 0, len(pdu) + 1, self.unit) + pdu
        try:
            sock.sendall(frame)
            head = self._recv_exact(sock, 7)
            _tid, _proto, length, _unit = struct.unpack(">HHHB", head)
            if length < 2:
                raise ModbusLinkError(f"响应长度非法: {length}")
            body = self._recv_exact(sock, length - 1)
        except ModbusLinkError:
            self._close_locked()
            raise
        except OSError as exc:
            # 超时/对端关闭/网络不可达：一律当断链，关掉 socket 让调用方重连。
            # 留着半死连接会挡住新连接（PLC 单会话）。
            self._close_locked()
            raise ModbusLinkError(
                f"{self.host}:{self.port} 通信失败: {type(exc).__name__}: {exc}"
            ) from exc
        if body[0] & 0x80:
            raise ModbusError(body[1])
        return body

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        buf = b""
        while len(buf) < size:
            chunk = sock.recv(size - len(buf))
            if not chunk:
                raise ModbusLinkError("对端关闭连接")
            buf += chunk
        return buf

    # ------------------------------------------------------------------
    # 功能码
    # ------------------------------------------------------------------
    def read_holding(self, start: int, count: int) -> list[int]:
        """FC3 读保持寄存器。``count`` 1~125，本 PLC 实测上限 20。"""

        if not 1 <= count <= 125:
            raise ValueError(f"count 必须在 1~125，收到 {count}")
        with self._lock:
            body = self._request_locked(struct.pack(">BHH", 3, start, count))
        data = body[2:]
        expected = count * 2
        if len(data) != expected:
            raise ModbusLinkError(
                f"FC3 响应字节数不符：期望 {expected}，收到 {len(data)}"
            )
        return [struct.unpack(">H", data[i : i + 2])[0] for i in range(0, len(data), 2)]

    def write_registers(self, start: int, values: list[int]) -> None:
        """FC16 写多寄存器。

        一次写完连续的一段是**原子**的：PLC 不会读到「前半更新、后半还是旧值」
        的瞬时态。判断结论必须整块写，不能拆成多次 FC6。
        """

        if not values:
            raise ValueError("values 不能为空")
        if len(values) > 123:
            raise ValueError(f"FC16 单次最多 123 个寄存器，收到 {len(values)}")
        for value in values:
            if not 0 <= int(value) <= 0xFFFF:
                raise ValueError(f"寄存器值越界: {value}")
        payload = struct.pack(">HHB", start, len(values), len(values) * 2)
        payload += b"".join(struct.pack(">H", int(v)) for v in values)
        with self._lock:
            body = self._request_locked(bytes([16]) + payload)
        # FC16 正常响应的 PDU 是 5 字节：fc(1) + 起始地址(2) + 数量(2)。
        # 原先写 ``>= 6``，条件恒假，这段回显校验从未执行过（假从站发的
        # 也正是 5 字节）。核对一下，确认写的确实是这一段——写错地址在
        # PLC 上可能表现为「灯不对」而不是报错。
        if len(body) >= 5:
            echoed_start, echoed_count = struct.unpack(">HH", body[1:5])
            if echoed_start != start or echoed_count != len(values):
                raise ModbusLinkError(
                    f"FC16 回显不符：写 {start}/{len(values)}，"
                    f"回显 {echoed_start}/{echoed_count}"
                )

    def probe(self, pdu: bytes) -> str:
        """试探任意功能码：返回 ``OK`` / 异常说明 / 链路错误。

        给体检工具用——它要在不改变 PLC 状态的前提下探明支持哪些功能码。
        """

        try:
            with self._lock:
                self._request_locked(pdu)
        except ModbusError as exc:
            return str(exc)
        except ModbusLinkError as exc:
            return f"链路错误: {exc}"
        return "OK"
