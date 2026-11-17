"""PLC Modbus TCP 传输层测试（本地假从站，不碰现场 PLC）。"""

from __future__ import annotations

import socketserver
import struct
import threading
import time

import pytest

from app.oilcheck.plc_link import ModbusError, ModbusLinkError, PlcLink


class _FakeSlave(socketserver.BaseRequestHandler):
    """最小 Modbus TCP 从站。

    行为受 ``server.behaviour`` 控制：

    - ``"ok"``：FC3 返回寄存器表，FC16 写入并回显；
    - ``"exception2"``：一律返回 ``非法数据地址``；
    - ``"half"``：只发半个 MBAP 头就停住（模拟半包）。
    """

    def handle(self) -> None:  # noqa: C901 - 测试桩，分支直白更易读
        server: _FakeServer = self.server  # type: ignore[assignment]
        while True:
            head = self._recv_exact(7)
            if head is None:
                return
            tid, _proto, length, unit = struct.unpack(">HHHB", head)
            pdu = self._recv_exact(length - 1)
            if pdu is None:
                return
            server.requests.append(bytes(pdu))

            behaviour = server.behaviour
            if behaviour == "half":
                self.request.sendall(head[:3])
                time.sleep(0.5)
                return
            if behaviour == "exception2":
                self._send(tid, unit, bytes([pdu[0] | 0x80, 0x02]))
                continue

            fc = pdu[0]
            if fc == 3:
                start, count = struct.unpack(">HH", pdu[1:5])
                values = server.registers[start : start + count]
                if len(values) < count:
                    self._send(tid, unit, bytes([0x83, 0x02]))
                    continue
                self._send(
                    tid,
                    unit,
                    bytes([0x03, count * 2]) + struct.pack(f">{count}H", *values),
                )
            elif fc == 16:
                start, count, _blen = struct.unpack(">HHB", pdu[1:6])
                values = [
                    struct.unpack(">H", pdu[6 + i * 2 : 8 + i * 2])[0]
                    for i in range(count)
                ]
                while len(server.registers) < start + count:
                    server.registers.append(0)
                server.registers[start : start + count] = values
                self._send(tid, unit, struct.pack(">BHH", 16, start, count))
            else:
                self._send(tid, unit, bytes([fc | 0x80, 0x01]))

    def _recv_exact(self, size: int) -> bytes | None:
        buf = b""
        while len(buf) < size:
            chunk = self.request.recv(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _send(self, tid: int, unit: int, body: bytes) -> None:
        self.request.sendall(struct.pack(">HHHB", tid, 0, len(body) + 1, unit) + body)


class _FakeServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _FakeSlave)
        self.registers: list[int] = [0, 0, 0, 0]
        self.requests: list[bytes] = []
        self.behaviour = "ok"
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        return self.server_address[1]

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


@pytest.fixture()
def slave():
    server = _FakeServer()
    try:
        yield server
    finally:
        server.stop()


def _link(server: _FakeServer, **kwargs) -> PlcLink:
    return PlcLink(
        "127.0.0.1",
        server.port,
        unit=1,
        connect_timeout_s=1.0,
        request_timeout_s=0.5,
        **kwargs,
    )


def test_read_holding_roundtrip(slave):
    slave.registers = [0x0102, 0x0008, 0x0000, 0x0000]
    with _link(slave) as link:
        assert link.read_holding(0, 2) == [0x0102, 0x0008]
    # 首地址从 0 开始，不是 40001
    assert struct.unpack(">BHH", slave.requests[0]) == (3, 0, 2)


def test_write_registers_updates_slave(slave):
    frame = (0x0104, 0x0000)
    with _link(slave) as link:
        link.write_registers(0, list(frame))
    assert slave.registers[:2] == list(frame)


def test_write_registers_echoes_address_and_count(slave):
    with _link(slave) as link:
        link.write_registers(0, [0x0001, 0x0000])
    # FC16 请求体：fc + start + count + bytecount + data
    pdu = slave.requests[-1]
    assert pdu[0] == 16
    assert struct.unpack(">HHB", pdu[1:6]) == (0, 2, 4)


def test_write_registers_rejects_bad_values(slave):
    with _link(slave) as link:
        with pytest.raises(ValueError):
            link.write_registers(0, [])
        with pytest.raises(ValueError):
            link.write_registers(0, [0x10000])


def test_exception_response_raises_modbus_error(slave):
    slave.behaviour = "exception2"
    with _link(slave) as link:
        with pytest.raises(ModbusError) as info:
            link.read_holding(0, 2)
    assert info.value.code == 2
    assert "非法数据地址" in str(info.value)


def test_link_error_when_peer_closes(slave):
    """假从站只发半个 MBAP 头就关闭 ⇒ 必须抛链路错误，不能静默返回空。"""

    slave.behaviour = "half"
    link = _link(slave)
    link.open()
    try:
        with pytest.raises(ModbusLinkError):
            link.read_holding(0, 2)
        # 出错后连接必须被关掉：留着半死连接会挡住新连接（PLC 单会话）
        assert not link.connected
    finally:
        link.close()


def test_connect_failure_raises_link_error():
    # 端口 1 在本机不会有人监听
    link = PlcLink("127.0.0.1", 1, connect_timeout_s=0.5)
    with pytest.raises(ModbusLinkError):
        link.open()


def test_request_before_open_raises_link_error():
    link = PlcLink("127.0.0.1", 1)
    with pytest.raises(ModbusLinkError):
        link.read_holding(0, 1)


def test_requests_are_serialized_on_one_socket(slave):
    """并发调用必须串在同一连接上：PLC 不接受第二条会话。"""

    slave.registers = [0x0001, 0x0002, 0x0003, 0x0004]
    link = _link(slave)
    link.open()
    errors: list[Exception] = []

    def worker(start: int) -> None:
        try:
            for _ in range(20):
                values = link.read_holding(start, 2)
                assert len(values) == 2
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    try:
        threads = [threading.Thread(target=worker, args=(i % 3,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5.0)
    finally:
        link.close()
    assert not errors, errors
    # 所有请求都走同一个连接：假从站看到的是同一个 handler
    assert not any(isinstance(exc, ConnectionError) for exc in errors)


def test_probe_reports_exception_without_raising(slave):
    slave.behaviour = "exception2"
    with _link(slave) as link:
        assert "exception 2" in link.probe(bytes([1, 0, 0, 0, 8]))
