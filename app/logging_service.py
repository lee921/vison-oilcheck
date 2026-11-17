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


# 打日志时带上 ``extra={FILE_ONLY: True}``，这条记录只进日志文件，不上控制台。
# 用于「现场知道就行、不该占启动输出」的提示（例如配置目录迁移提醒）。
FILE_ONLY = "oilcheck_file_only"


class _ConsoleFilter:
    """控制台过滤器：放过一切，唯独跳过标了 :data:`FILE_ONLY` 的记录。

    不继承 :class:`logging.Filter`：父类的 ``name`` 前缀匹配对这里没有意义，
    而且它默认 ``name=""`` 时会把所有记录都放行，容易掩盖配置错误。
    只要 ``filter(record)`` 这个方法即可（logging 只按鸭子类型调用）。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not getattr(record, FILE_ONLY, False)


def setup_logging(
    log_dir: Path, store: LogStore, *, console_level: str = "INFO"
) -> logging.Logger:
    """装配日志：文件（DEBUG）+ 控制台（``console_level`` 及以上）+ 内存。

    控制台只放 INFO 及以上：现场是命令行/双击启动，控制台是唯一不打开日志页
    就能看到启动结果的地方——PLC 连接、模型加载这些必须出现；DEBUG 太吵。
    """

    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("oilcheck")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---------- 控制台 ----------
    if not any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, TimedRotatingFileHandler)
        for h in logger.handlers
    ):
        console = logging.StreamHandler()
        console.setLevel(getattr(logging, str(console_level).upper(), logging.INFO))
        console.setFormatter(
            logging.Formatter(fmt="%(levelname)s %(name)s: %(message)s")
        )
        console.addFilter(_ConsoleFilter())
        logger.addHandler(console)

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
