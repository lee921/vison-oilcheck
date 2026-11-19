from __future__ import annotations

import json
import time

from app.oilcheck.events import EventStore
from app.oilcheck.types import (
    EventRecord,
    EventResult,
    EventType,
    ReviewResult,
)


def _event(ts: float | None = None) -> EventRecord:
    ts = time.time() if ts is None else ts
    return EventRecord.create(
        timestamp_start=ts - 1.0,
        timestamp_end=ts,
        station_id="ST01",
        camera_id="cam_test",
        event_type=EventType.PICK,
        result=EventResult.NEEDS_REVIEW,
        task_id="TASK-000001",
        cycle_id="TEST",
        evidence={"reason": "test"},
    )


def test_event_store_persists_and_reloads(tmp_path):
    store = EventStore(tmp_path)
    event = _event()
    store.add(event)

    files = list(tmp_path.glob("events-*.jsonl"))
    assert len(files) == 1
    data = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert data["event_id"] == event.event_id
    assert data["event_type"] == "PICK"
    assert data["result"] == "NEEDS_REVIEW"
    assert data["task_id"] == "TASK-000001"
    assert data["timestamp_start"]

    loaded = EventStore(tmp_path)
    assert loaded.load_recent(days=1) == 1
    restored = loaded.get(event.event_id)
    assert restored is not None
    assert restored.task_id == event.task_id
    assert restored.review_result is None


def test_review_result_survives_reload(tmp_path):
    store = EventStore(tmp_path)
    event = _event()
    store.add(event)

    updated = store.submit_review(
        event.event_id,
        ReviewResult.FALSE_POSITIVE,
        note="操作调整，不是取料",
    )
    assert updated is not None
    assert updated.review_result == ReviewResult.FALSE_POSITIVE
    assert store.pending_review() == []

    reloaded = EventStore(tmp_path)
    assert reloaded.load_recent(days=1) == 1
    restored = reloaded.get(event.event_id)
    assert restored is not None
    assert restored.review_result == ReviewResult.FALSE_POSITIVE
    assert restored.review_note == "操作调整，不是取料"

    review_files = list(tmp_path.glob("reviews-*.jsonl"))
    assert len(review_files) == 1


def test_event_filters_by_station_and_limit(tmp_path):
    store = EventStore(tmp_path)
    for _ in range(5):
        event = _event()
        store.add(event)

    assert len(store.recent(3)) == 3
    assert len(store.recent(10, station_id="ST01")) == 5
    assert store.recent(10, station_id="ST02") == []
    counts = store.counts()
    assert counts["total"] == 5
    assert counts["pending_review"] == 5


def test_clear_removes_events_and_reviews(tmp_path):
    store = EventStore(tmp_path)
    first = _event()
    second = _event()
    store.add(first)
    store.add(second)
    store.submit_review(first.event_id, ReviewResult.FALSE_POSITIVE, note="误报")

    assert store.recent(10)
    assert list(tmp_path.glob("events-*.jsonl"))
    assert list(tmp_path.glob("reviews-*.jsonl"))

    removed = store.clear()

    assert removed == 2
    assert store.recent(10) == []
    assert store.counts() == {
        "total": 0,
        "pending_review": 0,
        "unconfirmed": 0,
    }
    assert not list(tmp_path.glob("events-*.jsonl"))
    assert not list(tmp_path.glob("reviews-*.jsonl"))

    reloaded = EventStore(tmp_path)
    assert reloaded.load_recent(days=1) == 0


def test_legacy_event_types_keep_readable_labels():
    """旧事件文件里的 PUT/BRUSH_CONTACT/TASK_RESULT 不能被误标成「取料」。"""

    from app.oilcheck.types import EventRecord

    for legacy in ("PUT", "BRUSH_CONTACT", "TASK_RESULT"):
        record = EventRecord.from_dict(
            {
                "event_id": "x",
                "event_type": legacy,
                "result": "CONFIRMED",
                "task_id": "TASK-1",
                "timestamp_start": "2026-11-06T10:00:00+08:00",
                "timestamp_end": "2026-11-06T10:00:01+08:00",
            }
        )
        assert record.event_type == legacy
        expected = {
            "PUT": "放料（历史）",
            "BRUSH_CONTACT": "刷头接触（历史）",
            "TASK_RESULT": "任务结果（历史）",
        }[legacy]
        assert record.event_label == expected
        assert record.to_dict()["event_type"] == legacy


def test_known_event_types_roundtrip():
    from app.oilcheck.types import EventRecord

    for kind, label in (("PICK", "取料"), ("MANUAL_OILING", "人工二次涂油")):
        record = EventRecord.from_dict(
            {"event_id": "x", "event_type": kind, "result": "CONFIRMED", "task_id": "T"}
        )
        assert record.event_type.value == kind
        assert record.event_label == label


def test_clear_while_add_is_appending_leaves_no_disk_residue(tmp_path):
    """``clear()`` 与追加必须互斥，否则清空后磁盘仍残留一条。

    原先追加在锁外：add 已经把事件写进内存、正在写盘时 clear 删掉了文件，
    追加随后又把这一行写回去——内存 0 条、磁盘 1 行。这里在
    ``_append_jsonl`` 里插 0.15s 延迟放大窗口，确定性地复现。
    """

    import threading
    import time

    store = EventStore(tmp_path, capacity=10_000)
    original = store._append_jsonl

    def slow_append(path, payload) -> None:
        time.sleep(0.15)
        original(path, payload)

    store._append_jsonl = slow_append  # type: ignore[method-assign]

    writer = threading.Thread(target=store.add, args=(_event(),), daemon=True)
    writer.start()
    time.sleep(0.05)  # 让 add 进入临界区/开始追加
    store.clear()
    writer.join(timeout=5)

    assert len(store.recent(10_000)) == 0
    lines = [
        line
        for path in tmp_path.glob("events-*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines == []
