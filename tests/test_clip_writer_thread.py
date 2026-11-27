"""事件片段写盘必须在轮询线程之外（app/oilcheck/clips.py:_ClipWriter）。

现场症状：日志里 1.2~1.7 秒的「视频时间戳异常」，其中八成落在
「开始写事件片段 → 片段已保存」窗口内。

根因：``imdecode + VideoWriter.write`` 实测 14~41ms/帧，一段 60 帧就是
1~2.5 秒；原先这段开销跑在服务轮询线程里，而轮询线程同时是取帧线程，
写盘期间相机帧在 ``FrameBuffer`` 里被覆盖 —— 表现为长时间丢帧。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from app.oilcheck.clips import ClipConfig, EventClipService

WIDTH, HEIGHT = 640, 480


def _image(index: int) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), index % 255, dtype=np.uint8)


def _service(tmp_path: Path, **overrides) -> EventClipService:
    cfg = ClipConfig(
        enabled=True,
        root=tmp_path / "clips",
        pre_seconds=2.0,
        timeout_seconds=0.0,
        margin_seconds=0.0,
        post_seconds=1.0,
        fps=10,
        codec="mp4v",
        buffer_max_mb=64,
        max_pending=2,
        retain_days=0,
    )
    if overrides:
        cfg = ClipConfig(**{**cfg.__dict__, **overrides})
    return EventClipService(config=cfg)


def _push(service: EventClipService, index: int, ts: float) -> None:
    # 直接给原始帧：``push`` 内部做 JPEG 编码，这是它在现场的真实入口。
    service.push("cam", bgr=_image(index), ts=ts, pipeline_ts=ts)


def _register(service: EventClipService, start: float, end: float) -> bool:
    return service.on_event(
        event_id="evt-1",
        task_id="TASK-0001",
        event_type="OILING_MISSING",
        reason="oiling_missing_timeout",
        station_id="ST01",
        camera_id="cam",
        event_start=start,
        event_end=end,
        needs_review=True,
    )


def test_decode_happens_on_the_writer_thread(tmp_path, monkeypatch):
    """帧的解码（写盘前必做的一步）只能发生在写盘线程里。"""

    import app.oilcheck.clips as clips_module

    service = _service(tmp_path)
    for index in range(30):
        _push(service, index, 100.0 + index / 10.0)
    assert _register(service, 101.0, 101.5) is True

    threads: list[str] = []
    real_imdecode = clips_module.cv2.imdecode

    def slow_imdecode(*args, **kwargs):  # noqa: ANN002, ANN003
        threads.append(threading.current_thread().name)
        time.sleep(0.002)
        return real_imdecode(*args, **kwargs)

    monkeypatch.setattr(clips_module.cv2, "imdecode", slow_imdecode)

    for index in range(30, 40):
        _push(service, index, 100.0 + index / 10.0)

    assert service.wait_idle(timeout=10.0)
    assert threads, "写盘线程没被调用，片段没写出来"
    assert set(threads) == {"oilcheck-clip-writer"}, threads


def test_push_is_not_blocked_by_slow_writing(tmp_path, monkeypatch):
    """写盘线程每帧要 20ms 时，``push`` 自身耗时仍然只跟 JPEG 编码有关。"""

    import app.oilcheck.clips as clips_module

    service = _service(tmp_path)
    real_imdecode = clips_module.cv2.imdecode

    def slow_imdecode(*args, **kwargs):  # noqa: ANN002, ANN003
        time.sleep(0.02)
        return real_imdecode(*args, **kwargs)

    monkeypatch.setattr(clips_module.cv2, "imdecode", slow_imdecode)

    for index in range(30):
        _push(service, index, 200.0 + index / 10.0)
    _register(service, 201.0, 202.0)

    worst = 0.0
    for index in range(30, 80):
        started = time.perf_counter()
        _push(service, index, 200.0 + index / 10.0)
        worst = max(worst, time.perf_counter() - started)

    # 640×480 的 JPEG 编码约 1ms，入队几乎无成本；20ms 是写盘线程的单帧耗时。
    budget_ms = 8.0
    assert worst * 1000 < budget_ms, (
        f"push 最慢 {worst * 1000:.1f}ms，疑似在调用线程里写盘"
    )
    assert service.wait_idle(timeout=10.0)


def test_close_flushes_queued_clips(tmp_path):
    """退出时必须等写盘队列排空，否则留下的 mp4 没有 moov atom、打不开。"""

    service = _service(tmp_path)
    for index in range(30):
        _push(service, index, 300.0 + index / 10.0)
    _register(service, 301.0, 301.5)
    for index in range(30, 40):
        _push(service, index, 300.0 + index / 10.0)

    service.close()

    clips = list((tmp_path / "clips").rglob("*.mp4"))
    assert len(clips) == 1
    assert clips[0].stat().st_size > 0
    # 元数据也要写完（复核页靠它显示时长与帧数）。
    assert len(list((tmp_path / "clips").rglob("*.json"))) == 1


def test_clip_path_is_backfilled_after_writing(tmp_path):
    """``on_clip_ready`` 由写盘线程在文件落盘后回调，路径必须真实存在。"""

    ready: list[tuple[str, str]] = []
    service = _service(tmp_path)
    service._on_clip_ready = lambda eid, path: ready.append((eid, path))  # noqa: SLF001

    for index in range(30):
        _push(service, index, 400.0 + index / 10.0)
    _register(service, 401.0, 401.5)
    for index in range(30, 40):
        _push(service, index, 400.0 + index / 10.0)

    assert service.wait_idle(timeout=10.0)
    assert len(ready) == 1
    assert ready[0][0] == "evt-1"
    assert Path(ready[0][1]).exists()
