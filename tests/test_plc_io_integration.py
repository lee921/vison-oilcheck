"""端到端验证：真实 OilCheckService + PlcIoThread → 本地假 PLC 从站。

只跑在测试里，不碰现场 PLC。目的：确认「视觉判定 → 4 工步 → FC16 写字节」
这条链路整体成立，而不只是映射函数单测通过。
"""

from __future__ import annotations

import socketserver
import struct
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from app.oilcheck.config import OilCheckConfig, PlcConfig
from app.oilcheck.plc_io import PlcEvidence, PlcIoThread
from app.oilcheck.types import TaskResult, TaskStage


class _SlaveHandler(socketserver.BaseRequestHandler):
    """只实现 FC3/FC16 的最小从站，记录每一次写入。"""

    def handle(self) -> None:
        server: _SlaveServer = self.server  # type: ignore[assignment]
        while True:
            head = self._recv(7)
            if head is None:
                return
            tid, _proto, length, unit = struct.unpack(">HHHB", head)
            pdu = self._recv(length - 1)
            if pdu is None:
                return
            fc = pdu[0]
            if fc == 3:
                start, count = struct.unpack(">HH", pdu[1:5])
                values = server.registers[start : start + count]
                body = bytes([3, count * 2]) + struct.pack(f">{count}H", *values)
            elif fc == 16:
                start, count, blen = struct.unpack(">HHB", pdu[1:6])
                values = [
                    struct.unpack(">H", pdu[6 + i * 2 : 8 + i * 2])[0]
                    for i in range(count)
                ]
                with server.lock:
                    while len(server.registers) < start + count:
                        server.registers.append(0)
                    server.registers[start : start + count] = values
                    server.writes.append(tuple(values))
                body = struct.pack(">BHH", 16, start, count)
            else:
                body = bytes([fc | 0x80, 0x01])
            self.request.sendall(
                struct.pack(">HHHB", tid, 0, len(body) + 1, unit) + body
            )

    def _recv(self, size: int) -> bytes | None:
        buf = b""
        while len(buf) < size:
            chunk = self.request.recv(size - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf


class _SlaveServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _SlaveHandler)
        self.registers: list[int] = [0x0000, 0x00FF, 0, 0]
        self.writes: list[tuple[int, ...]] = []
        self.lock = threading.Lock()
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def port(self) -> int:
        return self.server_address[1]

    def stop(self) -> None:
        self.shutdown()
        self.server_close()


@pytest.fixture()
def slave():
    server = _SlaveServer()
    try:
        yield server
    finally:
        server.stop()


def _plc_cfg(port: int, *, dry_run: bool = False, hold_s: float = 0.15) -> PlcConfig:
    return replace(
        PlcConfig(),
        enabled=True,
        dry_run=dry_run,
        host="127.0.0.1",
        port=port,
        poll_interval_ms=20,
        refresh_interval_s=0.1,
        hold_s=hold_s,
        connect_timeout_s=1.0,
        request_timeout_s=0.5,
        reconnect_backoff_ms=(20, 50, 100),
    )


def _wait(predicate, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def test_writes_step_bytes_to_real_slave(slave):
    """取料 → 涂油中 → 完成，三个状态的灯色必须在从站寄存器上依次出现。"""

    state = {
        "evidence": PlcEvidence(
            online=True, task_id="T1", stage=None, final_result=None
        )
    }
    thread = PlcIoThread(
        cfg=_plc_cfg(slave.port), evidence_source=lambda: state["evidence"]
    )
    thread.start()
    try:
        # 等待取料 → 灯2 黄
        assert _wait(lambda: slave.registers[0] == 0x0200)
        assert slave.registers[1] == 0x0000

        # 取料完成 → 灯2、灯3 绿
        state["evidence"] = PlcEvidence(
            online=True,
            task_id="T1",
            stage=TaskStage.PICKED,
            final_result=TaskResult.PROCESSING,
            brush_hold_s=0.0,
        )
        assert _wait(lambda: slave.registers[0] == 0x0006)
        assert slave.registers[1] == 0x0000

        # 涂油中 → 灯2、灯3 绿，灯4 黄
        state["evidence"] = PlcEvidence(
            online=True,
            task_id="T1",
            stage=TaskStage.PICKED,
            final_result=TaskResult.PROCESSING,
            brush_hold_s=0.4,
        )
        assert _wait(lambda: slave.registers[0] == 0x0806)

        # 良品 → 灯2、灯3、灯4 全绿
        state["evidence"] = PlcEvidence(
            online=True,
            task_id="T1",
            stage=TaskStage.OILED,
            final_result=TaskResult.COMPLETED,
            brush_hold_s=1.1,
        )
        assert _wait(lambda: slave.registers[0] == 0x000E)
        assert slave.registers[1] == 0x0000
    finally:
        thread.stop()


def test_ng_writes_bad_bit_then_clears(slave):
    """漏涂写 VB2 工步 3 红灯，保持窗结束后回工步 1 并清零。"""

    state = {
        "evidence": PlcEvidence(
            online=True,
            task_id="T1",
            stage=TaskStage.PICKED,
            final_result=TaskResult.PROCESSING,
            brush_hold_s=0.3,
        )
    }
    thread = PlcIoThread(
        cfg=_plc_cfg(slave.port, hold_s=0.2),
        evidence_source=lambda: state["evidence"],
    )
    thread.start()
    try:
        assert _wait(lambda: slave.registers[0] == 0x0806)

        state["evidence"] = PlcEvidence(
            online=True,
            task_id="T1",
            stage=TaskStage.OILED,
            final_result=TaskResult.NG,
            brush_hold_s=0.3,
        )
        # 漏涂：灯4 同时绿+黄 → 红。VB0=0x0E，VB1=0x08；HR1 恒 0
        assert _wait(lambda: slave.registers[0] == 0x080E)
        assert slave.registers[1] == 0x0000

        # 保持窗结束后回等待取料（只剩灯2 黄）
        assert _wait(lambda: slave.registers[0] == 0x0200)
        assert slave.registers[1] == 0x0000
    finally:
        thread.stop()


def test_offline_clears_yellow_only(slave):
    """相机离线：黄灯全灭，绿灯与红灯保持（那是这一件的结论）。"""

    holder = {
        "evidence": PlcEvidence(
            online=True,
            task_id="T1",
            stage=TaskStage.OILED,
            final_result=TaskResult.COMPLETED,
            brush_hold_s=1.1,
        )
    }
    thread = PlcIoThread(
        cfg=_plc_cfg(slave.port, hold_s=5.0),
        evidence_source=lambda: holder["evidence"],
    )
    thread.start()
    try:
        # 良品完成：灯2、灯3、灯4 全绿
        assert _wait(lambda: slave.registers[0] == 0x000E)
        holder["evidence"] = PlcEvidence.offline()
        # 离线不动绿：最后一次写入仍是 VB0=0x0E / VB1=0x00
        assert _wait(
            lambda: bool(slave.writes)
            and slave.writes[-1][0] == 0x000E
            and slave.writes[-1][1] == 0x0000
        )
    finally:
        thread.stop()


def test_dry_run_never_writes(slave):
    """只读模式只算帧，一个字节都不许写。"""

    state = {"evidence": PlcEvidence(online=True)}
    thread = PlcIoThread(
        cfg=_plc_cfg(slave.port, dry_run=True),
        evidence_source=lambda: state["evidence"],
    )
    thread.start()
    try:
        time.sleep(0.3)
        state["evidence"] = PlcEvidence(
            online=True,
            task_id="T1",
            stage=TaskStage.PICKED,
            final_result=TaskResult.PROCESSING,
            brush_hold_s=0.0,
        )
        time.sleep(0.3)
    finally:
        thread.stop()
    assert slave.writes == []
    # 从站寄存器仍是上电残留值
    assert slave.registers[:2] == [0x0000, 0x00FF]


def test_periodic_rewrite_heals_after_slave_restart(slave):
    """PLC 重启把数据区刷回残留值，定时重写必须在 1 s 内自愈。"""

    state = {
        "evidence": PlcEvidence(
            online=True,
            task_id="T1",
            stage=TaskStage.PICKED,
            final_result=TaskResult.PROCESSING,
            brush_hold_s=0.3,
        )
    }
    thread = PlcIoThread(
        cfg=_plc_cfg(slave.port, hold_s=5.0),
        evidence_source=lambda: state["evidence"],
    )
    thread.start()
    try:
        assert _wait(lambda: slave.registers[0] == 0x0806)
        # 模拟 PLC 重启：数据区回到上电残留（VB2=0xFF，三盏全红）
        with slave.lock:
            slave.registers[:2] = [0x0000, 0x00FF]
        assert _wait(lambda: slave.registers[0] == 0x0806, timeout=2.0)
    finally:
        thread.stop()


def test_plc_config_enabled_builds_thread(tmp_path):
    """``plc.enabled=true`` 时 service 才构造 PLC 线程；关闭时返回未启用。"""

    from app.oilcheck.service import OilCheckService

    class _Manager:
        def cameras(self):
            return []

        def get_session(self, camera_id):  # noqa: ANN001, ARG002
            return None

    from app.oilcheck.config import OilCheckConfigService
    from app.oilcheck.events import EventStore

    cfg_service = OilCheckConfigService(tmp_path / "config")
    cfg_service.ensure_defaults()
    base = cfg_service.load()

    def _build(cfg: OilCheckConfig) -> OilCheckService:
        return OilCheckService(
            manager=_Manager(),
            config_service=cfg_service,
            config=cfg,
            event_store=EventStore(Path(tmp_path / "events")),
            data_root=Path(tmp_path),
        )

    service = _build(replace(base, plc=replace(base.plc, enabled=False)))
    try:
        assert service.plc_status()["enabled"] is False
    finally:
        service.shutdown()

    service = _build(
        replace(
            base,
            plc=replace(base.plc, enabled=True, host="127.0.0.1", port=1),
        )
    )
    try:
        assert service.plc_status()["enabled"] is True
        assert service.plc_status()["dry_run"] is True
    finally:
        service.shutdown()


# ----------------------------------------------------------------------
# 启动连通性检测
# ----------------------------------------------------------------------
def test_startup_probe_reports_ok(slave):
    """启动后 status 里要立刻有「检测通过」和读到的寄存器值。"""

    thread = PlcIoThread(
        cfg=_plc_cfg(slave.port),
        evidence_source=lambda: PlcEvidence(online=True),
    )
    thread.start()
    try:
        assert _wait(lambda: thread.status()["startup_ok"] is True)
        message = thread.status()["startup_message"]
        assert str(slave.port) in message
        # 探测是纯读，读到的是写入前的残留值
        assert "HR0=0x0000" in message
    finally:
        thread.stop()


def test_startup_primes_idle_frame_to_clear_residual_red(slave):
    """启动检测通过后要立刻用「等待取料」帧压掉上电残留的三盏红。

    实测该 PLC 上电 HR1=0x00FF（上电残留，与灯无关）。
    若不等主循环第一拍才覆盖，现场会先看到一段红。
    """

    assert slave.registers[:2] == [0x0000, 0x00FF]  # 上电残留值
    thread = PlcIoThread(
        cfg=_plc_cfg(slave.port),
        evidence_source=lambda: PlcEvidence(online=True),
    )
    thread.start()
    try:
        # 第一拍就是等待取料：灯1 黄、其余灭
        assert _wait(lambda: slave.registers[:2] == [0x0200, 0x0000])
        first = slave.writes[0] if slave.writes else None
        assert first == (0x0200, 0x0000), slave.writes[:3]
    finally:
        thread.stop()


def test_startup_prime_skipped_in_dry_run(slave):
    """只读联调不能写：启动压残留值也要跳过。"""

    thread = PlcIoThread(
        cfg=_plc_cfg(slave.port, dry_run=True),
        evidence_source=lambda: PlcEvidence(online=True),
    )
    thread.start()
    try:
        assert _wait(lambda: thread.status()["startup_ok"] is True)
        time.sleep(0.2)
    finally:
        thread.stop()
    assert slave.writes == []
    assert slave.registers[:2] == [0x0000, 0x00FF]


def test_startup_prime_skipped_when_probe_fails():
    """连不上时不硬写：启动检测失败就交给主循环按退避重连。"""

    thread = PlcIoThread(
        cfg=_plc_cfg(1),  # 端口 1 本机不会有人监听
        evidence_source=lambda: PlcEvidence(online=True),
    )
    thread.start()
    try:
        assert _wait(lambda: thread.status()["startup_ok"] is False)
        assert thread.status()["startup_message"]
    finally:
        thread.stop()


def test_startup_probe_failure_is_reported_not_fatal():
    """连不上时：startup_ok=False 且有原因，但线程照常起、服务不崩。"""

    # 端口 1 本机不会有人监听
    thread = PlcIoThread(
        cfg=_plc_cfg(1, hold_s=0.1),
        evidence_source=lambda: PlcEvidence(online=True),
    )
    try:
        thread.start()
        assert _wait(lambda: thread.status()["startup_ok"] is False)
        assert thread.status()["startup_message"]
        assert thread.status()["connected"] is False
    finally:
        thread.stop()


def test_startup_probe_runs_in_dry_run_too(slave):
    """只读模式也要报连通性——现场开机就想知道通不通。"""

    thread = PlcIoThread(
        cfg=_plc_cfg(slave.port, dry_run=True),
        evidence_source=lambda: PlcEvidence(online=True),
    )
    thread.start()
    try:
        assert _wait(lambda: thread.status()["startup_ok"] is True)
    finally:
        thread.stop()
    # 但 dry_run 仍然一个字节都不写
    assert slave.writes == []


def test_startup_probe_does_not_block_start(slave):
    """检测跑在线程里：start() 不能因为等网络而阻塞调用方。"""

    thread = PlcIoThread(
        cfg=_plc_cfg(slave.port),
        evidence_source=lambda: PlcEvidence(online=True),
    )
    try:
        started = time.monotonic()
        thread.start()
        elapsed = time.monotonic() - started
        # 建连实测 2~4 ms；给足余量，但必须远小于一次请求超时（500 ms）
        assert elapsed < 0.2, f"start() 阻塞了 {elapsed * 1000:.0f} ms"
    finally:
        thread.stop()


def test_check_connection_does_not_write(slave):
    """连通性检测是纯读，不许改 PLC 状态。"""

    thread = PlcIoThread(
        cfg=_plc_cfg(slave.port),
        evidence_source=lambda: PlcEvidence(online=True),
    )
    ok, message = thread.check_connection()
    assert ok is True
    assert "可读" in message
    assert slave.writes == []
    assert slave.registers[:2] == [0x0000, 0x00FF]
