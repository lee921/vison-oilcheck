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
