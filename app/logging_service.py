from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterable
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


class LogStore:
    def __init__(self, capacity: int = 2000) -> None:
        self._capacity = capacity
        self._lines: deque[str] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def add(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)

    def extend(self, lines: Iterable[str]) -> None:
        with self._lock:
            self._lines.extend(lines)

    def snapshot(self) -> list[str]:
        with self._lock:
            return list(self._lines)


class InMemoryLogHandler(logging.Handler):
    def __init__(self, store: LogStore) -> None:
        super().__init__()
        self._store = store

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self._store.add(msg)
        except Exception:
            pass


def setup_logging(log_dir: Path, store: LogStore) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("oilcheck")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---------- 文件日志（按日期一个文件） ----------
    if not any(isinstance(h, TimedRotatingFileHandler) for h in logger.handlers):
        today = time.strftime("%Y-%m-%d")

        file_handler = TimedRotatingFileHandler(
            filename=str(log_dir / f"app-{today}.log"),
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
            utc=False,
        )

        file_handler.suffix = "%Y-%m-%d"
        file_handler.namer = lambda name: (
            str(log_dir / f"app-{name.split('.')[-1]}.log")
        )

        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # ---------- 内存日志 ----------
    if not any(isinstance(h, InMemoryLogHandler) for h in logger.handlers):
        memory_handler = InMemoryLogHandler(store)
        memory_handler.setLevel(logging.INFO)
        memory_handler.setFormatter(formatter)
        logger.addHandler(memory_handler)

    return logger
