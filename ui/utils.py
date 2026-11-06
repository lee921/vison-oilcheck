from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal


class UiInvoker(QObject):
    # 通过 Qt 信号将回调切回 UI 线程执行
    invoked = Signal(object)

    def __init__(
        self, parent: QObject | None = None, *, logger: logging.Logger | None = None
    ) -> None:
        super().__init__(parent)
        self._logger = logger or logging.getLogger("oilcheck.ui")
        self.invoked.connect(self._run)

    def invoke(self, fn: Callable[[], None]) -> None:
        # 异步通知 UI 线程执行
        self.invoked.emit(fn)

    def _run(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            self._logger.exception("UI 回调执行失败")
