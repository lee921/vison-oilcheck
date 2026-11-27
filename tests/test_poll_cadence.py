"""服务轮询线程的取帧节拍（app/oilcheck/service.py:_run_loop）。

现场症状：``pipeline.max_fps`` 配 15，实际只跑到 ~12.5fps；录制元数据里
76.6 秒收到 955 帧（12.5fps）也印证了这一点。

原先是「``wait(1/fps)`` → 干这一轮的活」，真实周期 = 等待 + 本轮耗时
（取帧、片段 JPEG 编码、PLC 快照……）。改成「等到下一个应取帧时刻再干」
之后，只要单轮耗时小于一拍，实际帧率就应贴近配置值。
"""

from __future__ import annotations

import threading
import time

import pytest

from app.oilcheck.config import OilCheckConfigService
from app.oilcheck.service import OilCheckService


class _FakeManager:
    def get_session(self, camera_id):  # noqa: ANN001, ARG002
        return None

    def cameras(self):
        return ()


class _FakeEventStore:
    pass


class _FakeRuntime:
    closed = False


def _make_service(tmp_path) -> OilCheckService:
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    import json

    (config_dir / "oilcheck.json").write_text(
        json.dumps({"pipeline": {"max_fps": 15}, "stations": []}),
        encoding="utf-8",
    )
    service = OilCheckConfigService(config_dir)
    cfg = service.load()
    return OilCheckService(
        manager=_FakeManager(),
        config_service=service,
        config=cfg,
        event_store=_FakeEventStore(),
        data_root=tmp_path,
    )


def test_loop_holds_configured_rate_when_each_iteration_takes_time(tmp_path):
    """单轮耗 45ms（小于 66.7ms 一拍）时，实际轮询率仍应接近 15Hz。

    旧实现下周期 = 66.7 + 45 ≈ 112ms（8.9Hz）；这正是现场「配 15 跑 12.5」
    的同一个机制，只是现场的单轮耗时小一些（约 13ms）。
    """

    service = _make_service(tmp_path)
    service._runtimes["ST01"] = _FakeRuntime()  # noqa: SLF001
    service._reload_if_file_changed = lambda: None  # noqa: SLF001

    ticks: list[float] = []

    def fake_poll(runtime):  # noqa: ANN001, ARG001
        ticks.append(time.monotonic())
        time.sleep(0.045)

    service._poll_station = fake_poll  # noqa: SLF001

    thread = threading.Thread(target=service._run_loop, daemon=True)  # noqa: SLF001
    thread.start()
    time.sleep(2.0)
    service._stop_event.set()  # noqa: SLF001
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(ticks) >= 5
    span = ticks[-1] - ticks[0]
    rate = (len(ticks) - 1) / span
    # 15Hz 配置、每轮 45ms：理想 15，旧实现约 8.9。给 10% 余量。
    assert rate >= 13.0, f"轮询率只有 {rate:.1f}Hz（配置 15Hz）"
    assert rate <= 16.5, f"轮询率 {rate:.1f}Hz 超过配置，等待逻辑可能失效"


def test_loop_backs_off_instead_of_spinning_when_a_tick_overruns(tmp_path):
    """某轮严重超时（例如写盘卡了 1 秒）后不能连续空转补拍。

    落后超过一整拍时从「现在」重新起拍：紧接着的一轮必须仍然等一拍，
    而不是立刻再跑一轮把 CPU 打满。
    """

    service = _make_service(tmp_path)
    service._runtimes["ST01"] = _FakeRuntime()  # noqa: SLF001
    service._reload_if_file_changed = lambda: None  # noqa: SLF001

    ticks: list[float] = []

    def slow_first(runtime):  # noqa: ANN001, ARG001
        ticks.append(time.monotonic())
        if len(ticks) == 1:
            time.sleep(0.5)  # 远超 66.7ms 一拍

    service._poll_station = slow_first  # noqa: SLF001

    thread = threading.Thread(target=service._run_loop, daemon=True)  # noqa: SLF001
    thread.start()
    time.sleep(1.5)
    service._stop_event.set()  # noqa: SLF001
    thread.join(timeout=2.0)

    # 1.5 秒里 15Hz 最多约 22 轮。若超时后空转补拍，会看到几十上百轮。
    assert 2 <= len(ticks) <= 20, f"职责循环跑了 {len(ticks)} 轮，疑似空转补拍"
    gaps = [b - a for a, b in zip(ticks, ticks[1:], strict=False)]
    assert all(gap >= 0.05 for gap in gaps), f"两轮间隔过短: {gaps}"


@pytest.mark.parametrize("fps", [5, 30])
def test_loop_respects_configured_fps(tmp_path, fps):
    service = _make_service(tmp_path)
    object.__setattr__(service._config.pipeline, "max_fps", fps)  # noqa: SLF001
    service._runtimes["ST01"] = _FakeRuntime()  # noqa: SLF001
    service._reload_if_file_changed = lambda: None  # noqa: SLF001
    ticks: list[float] = []
    service._poll_station = lambda runtime: ticks.append(time.monotonic())  # noqa: SLF001

    thread = threading.Thread(target=service._run_loop, daemon=True)  # noqa: SLF001
    thread.start()
    time.sleep(1.5)
    service._stop_event.set()  # noqa: SLF001
    thread.join(timeout=2.0)

    span = ticks[-1] - ticks[0]
    rate = (len(ticks) - 1) / span
    assert abs(rate - fps) / fps < 0.35, f"配置 {fps}Hz，实测 {rate:.1f}Hz"
