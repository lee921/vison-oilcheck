"""NG 事件片段存储（app/oilcheck/clips.py）的单元测试。

设计见 docs/14-NG事件片段存储设计.md。这里只测纯逻辑：环形缓冲、登记与
切片、时间轴换算、断流收尾、开关生效。不启动相机与 Qt。
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from app.oilcheck.clips import ClipConfig, EventClipService, _ClipBuffer, _Frame

WIDTH, HEIGHT = 64, 48


def _image(index: int) -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), index % 255, dtype=np.uint8)


def _encoded(index: int, *, ts: float, pipeline_ts: float | None = None) -> _Frame:
    """构造一帧已编码的数据（给缓冲/切片用例直接用）。"""

    ok, data = cv2.imencode(".jpg", _image(index))
    assert ok
    return _Frame(
        ts=float(ts),
        data=data,
        pipeline_ts=float(ts if pipeline_ts is None else pipeline_ts),
        width=WIDTH,
        height=HEIGHT,
    )


def _make(tmp_path: Path, **overrides) -> EventClipService:
    cfg = ClipConfig(
        enabled=True,
        root=tmp_path / "clips",
        pre_seconds=2.0,
        timeout_seconds=0.0,
        margin_seconds=0.0,
        post_seconds=1.0,
        fps=10,
        codec="mp4v",
        buffer_width=0,
        jpeg_quality=75,
        buffer_max_mb=64,
        max_pending=2,
        retain_days=0,
    )
    if overrides:
        cfg = ClipConfig(**{**cfg.__dict__, **overrides})
    return EventClipService(config=cfg)


def _settle(service: EventClipService) -> None:
    """等写盘线程把队列写完。

    写盘已经挪到独立线程（见 clips.py 模块文档）：``push`` 只收集帧，
    片段落盘是异步的。用例断言文件/元数据之前必须先等它排空，不能用
    ``sleep`` 猜时间——慢机器上会偶发失败。
    """

    assert service.wait_idle(timeout=10.0), "写盘队列没在 10s 内排空"


def _read_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = 0
    while cap.grab():
        count += 1
    cap.release()
    return count


def _push(service: EventClipService, camera: str, frame: _Frame) -> None:
    """走真实 ``push`` 入口：编码 + 入缓冲 + 推进已登记的片段。"""

    img = cv2.imdecode(frame.data, cv2.IMREAD_COLOR)
    assert img is not None
    service.push(camera, bgr=img, ts=frame.ts, pipeline_ts=frame.pipeline_ts)


def _register(
    service: EventClipService,
    camera: str,
    start: float,
    end: float,
) -> bool:
    return service.on_event(
        event_id="evt-1",
        task_id="TASK-0001",
        event_type="OILING_MISSING",
        reason="oiling_missing_timeout",
        station_id="ST01",
        camera_id=camera,
        event_start=start,
        event_end=end,
        needs_review=True,
    )


def test_ring_buffer_keeps_only_the_newest_frames():
    buffer = _ClipBuffer(capacity=5)
    for index in range(20):
        buffer.push(_encoded(index, ts=float(index)))
    kept = [frame.ts for frame in buffer.frames]
    assert kept == [15.0, 16.0, 17.0, 18.0, 19.0]


def test_clip_covers_lead_in_and_tail(tmp_path):
    service = _make(tmp_path)
    ready: list[tuple[str, str]] = []
    service._on_clip_ready = lambda eid, path: ready.append((eid, path))

    for index in range(40):
        _push(service, "cam", _encoded(index, ts=100.0 + index / 10.0))
    # 事件起点 2 秒前（缓冲能覆盖），终点此刻，再录 1 秒。
    now = 100.0 + 39 / 10.0
    assert _register(service, "cam", now - 2.0, now) is True
    for index in range(40, 50):
        _push(service, "cam", _encoded(index, ts=100.0 + index / 10.0))

    _settle(service)
    clips = list((tmp_path / "clips").rglob("*.mp4"))
    assert len(clips) == 1
    # 起点 2 秒前 → 终点后再 1 秒，约 3 秒 × 10fps = 30 帧（首尾各差一帧）。
    assert 28 <= _read_frame_count(clips[0]) <= 32
    assert ready and ready[0][0] == "evt-1"


def test_clip_meta_is_written(tmp_path):
    service = _make(tmp_path)
    for index in range(20):
        _push(service, "cam", _encoded(index, ts=50.0 + index / 10.0))
    now = 50.0 + 19 / 10.0
    _register(service, "cam", now - 0.5, now)
    for index in range(20, 30):
        _push(service, "cam", _encoded(index, ts=50.0 + index / 10.0))

    _settle(service)
    meta_paths = list((tmp_path / "clips").rglob("*.json"))
    assert len(meta_paths) == 1
    meta = json.loads(meta_paths[0].read_text(encoding="utf-8"))
    assert meta["type"] == "event_clip"
    assert meta["event_id"] == "evt-1"
    assert meta["reason"] == "oiling_missing_timeout"
    assert meta["frame_count"] > 0
    assert meta["truncated"] is False


def test_start_before_buffer_is_marked_truncated(tmp_path):
    service = _make(tmp_path)
    for index in range(20):
        _push(service, "cam", _encoded(index, ts=200.0 + index / 10.0))
    # 起点远早于缓冲能覆盖的范围（只留 2 秒）。
    assert _register(service, "cam", 150.0, 201.9) is True
    for index in range(20, 30):
        _push(service, "cam", _encoded(index, ts=200.0 + index / 10.0))

    _settle(service)
    meta = json.loads(next((tmp_path / "clips").rglob("*.json")).read_text(encoding="utf-8"))
    assert meta["truncated"] is True
    # 不伪造黑帧：从缓冲最早一帧开始写，仍然是一个能播放的文件。
    assert _read_frame_count(Path(meta["path"])) > 0


def test_media_time_offset_is_applied(tmp_path):
    service = _make(tmp_path)
    # 模拟回放 0.5×：墙钟走得比媒体时间快一倍。
    for index in range(30):
        media = index / 10.0
        wall = 1000.0 + index / 5.0
        _push(service, "cam", _encoded(index, ts=wall, pipeline_ts=media))

    media_now = 29 / 10.0
    # 事件时间用**媒体时间**给；服务应换算成墙钟再切片。
    _register(service, "cam", media_now - 1.0, media_now)
    # 换算后的终点在墙钟上还要走 2 秒（media 1 秒 × 2），补足这些帧。
    for index in range(30, 44):
        media = index / 10.0
        wall = 1000.0 + index / 5.0
        _push(service, "cam", _encoded(index, ts=wall, pipeline_ts=media))

    _settle(service)
    meta = json.loads(next((tmp_path / "clips").rglob("*.json")).read_text(encoding="utf-8"))
    span = meta["ended_at_epoch"] - meta["started_at_epoch"]
    # 媒体时间差 1 秒，换算成墙钟应对应 2 秒。首尾各差一帧（10fps），
    # 允许 ±0.2 秒。
    assert 1.7 <= span <= 2.3, f"span={span}"


def test_timeout_finalizes_without_more_frames(tmp_path):
    service = _make(tmp_path, finalize_timeout_s=0.0)
    for index in range(20):
        _push(service, "cam", _encoded(index, ts=10.0 + index / 10.0))
    _register(service, "cam", 11.0, 11.9)
    # 之后不再有新帧：靠 flush_idle 按时间收尾。
    service.flush_idle()
    _settle(service)
    assert len(list((tmp_path / "clips").rglob("*.mp4"))) == 1


def test_disabled_service_writes_nothing(tmp_path):
    service = _make(tmp_path, enabled=False)
    for index in range(20):
        _push(service, "cam", _encoded(index, ts=10.0 + index / 10.0))
    assert _register(service, "cam", 10.0, 11.0) is False
    service.flush_idle()
    _settle(service)
    assert not (tmp_path / "clips").exists()


def test_confirmed_event_is_not_registered(tmp_path):
    service = _make(tmp_path)
    for index in range(20):
        _push(service, "cam", _encoded(index, ts=10.0 + index / 10.0))
    ok = service.on_event(
        event_id="evt-2",
        task_id="TASK-0002",
        event_type="PICK",
        reason="",
        station_id="ST01",
        camera_id="cam",
        event_start=10.0,
        event_end=11.0,
        needs_review=False,
    )
    assert ok is False
    _settle(service)
    assert not list((tmp_path / "clips").rglob("*.mp4"))


def test_pending_limit_finalizes_oldest(tmp_path):
    service = _make(tmp_path, max_pending=1, post_seconds=100.0)
    for index in range(20):
        _push(service, "cam", _encoded(index, ts=10.0 + index / 10.0))
    _register(service, "cam", 10.0, 10.5)
    service.on_event(
        event_id="evt-3",
        task_id="TASK-0003",
        event_type="OILING_MISSING",
        reason="oiling_missing_superseded_by_pick",
        station_id="ST01",
        camera_id="cam",
        event_start=11.0,
        event_end=11.5,
        needs_review=True,
    )
    # 超限的那段只被标记收尾，由轮询线程落盘（判定线程不编码）。
    service.flush_idle()
    _settle(service)
    assert len(list((tmp_path / "clips").rglob("*.mp4"))) == 1
    assert service.pending_count("cam") == 1
    assert not list((tmp_path / "clips").rglob("TASK-0003*.json"))


def test_two_cameras_have_independent_buffers(tmp_path):
    service = _make(tmp_path)
    for index in range(20):
        _push(service, "cam-a", _encoded(index, ts=10.0 + index / 10.0))
    # cam-b 从未推帧：事件不该为它生成片段。
    ok = _register(service, "cam-b", 10.0, 11.0)
    assert ok is False


def test_margin_extends_both_ends(tmp_path):
    """默认前后各延长 1.5 秒，片段才包含完整动作。"""

    service = _make(tmp_path, margin_seconds=1.5, post_seconds=0.0, pre_seconds=4.0)
    for index in range(60):
        _push(service, "cam", _encoded(index, ts=300.0 + index / 10.0))
    # 事件口径的区间只有 0.5 秒（305.0~305.5）。
    _register(service, "cam", 305.0, 305.5)
    # 终点是 305.5 + 1.5 = 307.0，要推到它之后才能收尾。
    for index in range(60, 80):
        _push(service, "cam", _encoded(index, ts=300.0 + index / 10.0))

    _settle(service)
    meta = json.loads(next((tmp_path / "clips").rglob("*.json")).read_text(encoding="utf-8"))
    span = meta["ended_at_epoch"] - meta["started_at_epoch"]
    # 0.5 秒的事件区间 + 前后各 1.5 秒，共约 3.5 秒。
    assert 3.3 <= span <= 3.7


def test_full_process_from_pick_to_ng(tmp_path):
    """完整过程：取料前 1.5 秒 → 判出 NG 后 1.5 秒。

    漏涂的 ``event_end`` 就是「等满超时判出 NG」的时刻，这一段等待本来
    就属于完整过程，不能裁。真实时序是：事件在 403.0 判出的那一刻登记，
    此时缓冲里最新帧就是 403.0 附近，之后才收到 403.0~404.5 的帧。
    """

    service = _make(tmp_path, margin_seconds=1.5, post_seconds=0.0, pre_seconds=6.0)
    # 399.0 ~ 403.0：覆盖 401.0-1.5 = 399.5 这段历史。
    for index in range(41):
        _push(service, "cam", _encoded(index, ts=399.0 + index / 10.0))

    # 取料 401.0、判出 NG 403.0 —— 就在最新帧上登记（真实时序）。
    _register(service, "cam", 401.0, 403.0)
    # 之后继续录到 403.0+1.5 = 404.5 之后，片段才收尾。
    for index in range(41, 60):
        _push(service, "cam", _encoded(index, ts=399.0 + index / 10.0))

    _settle(service)
    meta = json.loads(next((tmp_path / "clips").rglob("*.json")).read_text(encoding="utf-8"))
    span = meta["ended_at_epoch"] - meta["started_at_epoch"]
    # 399.5 到 404.5，共 5 秒。
    assert 4.7 <= span <= 5.3, f"span={span}"
    assert meta["truncated"] is False


def test_buffer_capacity_covers_the_timeout_window(tmp_path):
    """缓冲容量必须盖住「超时窗 + 前后余量」，否则开头会被丢掉。"""

    service = _make(
        tmp_path,
        margin_seconds=1.5,
        post_seconds=0.0,
        pre_seconds=0.5,
        timeout_seconds=5.0,
        fps=10,
    )
    # 容量 = (0.5 + 5.0 + 1.5) × 10 + 4 = 74 帧。
    for index in range(80):
        _push(service, "cam", _encoded(index, ts=500.0 + index / 10.0))
    buffer = service._buffers["cam"]  # noqa: SLF001
    assert len(buffer.frames) >= 70
    # 取料在 501.0（约 8 秒前的帧已被 deque 挤掉也无妨，这里只验容量）。
    span = buffer.latest().ts - buffer.earliest().ts
    assert span >= 6.5


def test_buffer_respects_memory_cap(tmp_path):
    """内存上限生效：超过就丢最旧的帧，不会无限涨。"""

    service = _make(tmp_path, buffer_max_mb=1, pre_seconds=60.0, fps=10)
    buffer = service._buffers.setdefault("cam", None)  # noqa: SLF001
    # 直接往缓冲里塞「很大」的帧：约束在 _ClipBuffer 这一层，与编码无关。
    buffer = _ClipBuffer(capacity=1000, max_bytes=10_000)
    payload = np.zeros(4_000, dtype=np.uint8)
    for index in range(50):
        buffer.push(
            _Frame(
                ts=float(index),
                data=payload,
                pipeline_ts=float(index),
                width=WIDTH,
                height=HEIGHT,
            )
        )
    assert buffer.nbytes <= 10_000
    assert 0 < len(buffer.frames) < 50


def test_event_store_roundtrips_clip_reference(tmp_path):
    """片段路径要能跨重启读回，否则复核页重启后就打不开了。"""

    import time

    from app.oilcheck.events import EventStore
    from app.oilcheck.types import EventRecord, EventResult, EventType

    store = EventStore(tmp_path / "events")
    now = time.time()
    event = EventRecord.create(
        timestamp_start=now - 2.0,
        timestamp_end=now,
        station_id="ST01",
        camera_id="cam",
        event_type=EventType.OILING_MISSING,
        result=EventResult.NEEDS_REVIEW,
        task_id="TASK-0001",
    )
    store.add(event)
    assert store.get(event.event_id).clip_reference is None

    clip = tmp_path / "clips" / "x.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"fake")
    store.update_clip(event.event_id, str(clip))
    assert store.get(event.event_id).clip_reference == str(clip)

    # 模拟重启：新的 store 从磁盘读回。
    reopened = EventStore(tmp_path / "events")
    reopened.load_recent(days=1)
    back = reopened.get(event.event_id)
    assert back is not None
    assert back.clip_reference == str(clip)
