"""事件记录与人工复核持久化。

存储格式为 JSONL，按天分文件，追加写入：

- ``<数据目录>/events/events-YYYY-MM-DD.jsonl``
- ``<数据目录>/events/reviews-YYYY-MM-DD.jsonl``

采用追加日志而不是就地修改，便于追溯和故障恢复。内存中维护最近事件，
UI 通过 ``version`` 判断是否需要刷新。
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .types import EventRecord, EventResult, ReviewResult

logger = logging.getLogger("oilcheck.events")


class EventStore:
    def __init__(
        self,
        root: Path,
        *,
        capacity: int = 2000,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._root = Path(root)
        self._capacity = max(50, int(capacity))
        self._logger = logger_ or logger
        self._lock = threading.Lock()
        self._events: OrderedDict[str, EventRecord] = OrderedDict()
        self._version = 0

    @property
    def root(self) -> Path:
        return self._root

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def load_recent(self, *, days: int = 7) -> int:
        """启动时加载最近若干天的事件与复核结果。"""
        self._root.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - max(1, int(days)) * 86400
        loaded: OrderedDict[str, EventRecord] = OrderedDict()

        for path in sorted(self._events_dir().glob("events-*.jsonl")):
            for data in _iter_jsonl(path):
                try:
                    event = EventRecord.from_dict(data)
                except Exception:
                    continue
                if not event.event_id or event.timestamp_end < cutoff:
                    continue
                loaded[event.event_id] = event

        reviews: dict[str, tuple[ReviewResult, str, float]] = {}
        for path in sorted(self._events_dir().glob("reviews-*.jsonl")):
            for data in _iter_jsonl(path):
                event_id = str(data.get("event_id", ""))
                if not event_id:
                    continue
                try:
                    result = ReviewResult(str(data.get("review_result", "")))
                except ValueError:
                    continue
                reviewed_at = float(data.get("reviewed_at", 0.0) or 0.0)
                note = str(data.get("review_note", ""))
                prev = reviews.get(event_id)
                if prev is None or reviewed_at >= prev[2]:
                    reviews[event_id] = (result, note, reviewed_at)

        for event_id, (result, note, _ts) in reviews.items():
            event = loaded.get(event_id)
            if event is not None:
                loaded[event_id] = event.with_review(result, note=note)

        with self._lock:
            self._events = loaded
            while len(self._events) > self._capacity:
                self._events.popitem(last=False)
            self._version += 1
        self._logger.info("已加载 %d 条历史事件（最近 %d 天）", len(loaded), days)
        return len(loaded)

    def add(self, event: EventRecord) -> None:
        with self._lock:
            self._events[event.event_id] = event
            while len(self._events) > self._capacity:
                self._events.popitem(last=False)
            self._version += 1

        path = self._events_dir() / f"events-{_date_of(event.timestamp_end)}.jsonl"
        try:
            self._append_jsonl(path, event.to_dict())
        except OSError:
            self._logger.exception("写入事件文件失败: %s", path)

    def submit_review(
        self,
        event_id: str,
        review_result: ReviewResult,
        *,
        note: str = "",
        reviewer: str = "operator",
    ) -> EventRecord | None:
        with self._lock:
            event = self._events.get(event_id)
            if event is None:
                return None
            reviewed = event.with_review(review_result, note=note)
            self._events[event_id] = reviewed
            self._version += 1

        path = self._events_dir() / f"reviews-{_date_of(time.time())}.jsonl"
        payload = {
            "event_id": event_id,
            "review_result": review_result.value,
            "review_note": str(note),
            "reviewer": str(reviewer),
            "reviewed_at": time.time(),
        }
        try:
            self._append_jsonl(path, payload)
        except OSError:
            self._logger.exception("写入复核文件失败: %s", path)
        return reviewed

    def clear(self) -> int:
        """清空内存事件并删除本机所有事件/复核文件。

        只删除 ``events-*.jsonl`` 与 ``reviews-*.jsonl``；训练视频、配置和日志不受影响。
        返回被清空的内存事件数量。
        """
        with self._lock:
            removed = len(self._events)
            self._events.clear()
            self._version += 1
            files = list(self._events_dir().glob("events-*.jsonl"))
            files.extend(self._events_dir().glob("reviews-*.jsonl"))
            for path in files:
                try:
                    path.unlink()
                except FileNotFoundError:
                    continue
                except OSError as exc:
                    self._logger.warning("删除事件文件失败 (%s): %s", path, exc)
        self._logger.info(
            "已清空事件记录（内存 %d 条，文件 %d 个）", removed, len(files)
        )
        return removed

    def get(self, event_id: str) -> EventRecord | None:
        with self._lock:
            return self._events.get(event_id)

    def recent(
        self,
        limit: int = 100,
        *,
        station_id: str | None = None,
        pending_only: bool = False,
    ) -> list[EventRecord]:
        with self._lock:
            items: Iterable[EventRecord] = reversed(self._events.values())
            result: list[EventRecord] = []
            for event in items:
                if station_id and event.station_id != station_id:
                    continue
                if pending_only and not (
                    event.review_result is None
                    and event.result != EventResult.CONFIRMED
                ):
                    continue
                result.append(event)
                if len(result) >= max(1, int(limit)):
                    break
            return result

    def pending_review(self, limit: int = 200) -> list[EventRecord]:
        return self.recent(limit, pending_only=True)

    def counts(self) -> dict[str, int]:
        with self._lock:
            total = len(self._events)
            pending = sum(
                1
                for e in self._events.values()
                if e.review_result is None and e.result != EventResult.CONFIRMED
            )
            unconfirmed = sum(
                1
                for e in self._events.values()
                if e.result in (EventResult.NEEDS_REVIEW, EventResult.UNCONFIRMED)
            )
        return {"total": total, "pending_review": pending, "unconfirmed": unconfirmed}

    def _events_dir(self) -> Path:
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("跳过损坏的事件行: %s", path)
                    continue
                if isinstance(data, dict):
                    yield data
    except OSError as exc:
        logger.warning("读取事件文件失败 (%s): %s", path, exc)


def _date_of(ts: float) -> str:
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d")


def recent_dates(days: int) -> list[str]:
    now = datetime.now()
    return [(now - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(max(1, days))]
