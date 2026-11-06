from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import PlainTextEdit, PushButton

from app.logging_service import LogStore


class LogsPage(QWidget):
    def __init__(
        self,
        *,
        store: LogStore,
        log_dir: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("logsPage")

        self._store = store
        self._log_dir = log_dir

        self._text = PlainTextEdit(self)
        self._text.setReadOnly(True)

        self._last_lines: list[str] = []
        self._btn_open = PushButton("打开日志目录")
        self._btn_refresh = PushButton("刷新")

        top = QHBoxLayout()
        top.addWidget(self._btn_open)
        top.addWidget(self._btn_refresh)
        top.addStretch(1)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self._text, 1)

        self._btn_open.clicked.connect(self._on_open)
        self._btn_refresh.clicked.connect(self.refresh)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()

        self.refresh()

    def set_log_dir(self, log_dir: Path) -> None:
        self._log_dir = log_dir

    def refresh(self) -> None:
        lines = self._store.snapshot()
        if lines == self._last_lines:
            return

        scrollbar = self._text.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 2
        old_value = scrollbar.value()

        can_append = hasattr(self._text, "appendPlainText")
        # 支持增量追加，减少大文本刷新带来的闪烁
        if (
            can_append
            and self._last_lines
            and len(lines) >= len(self._last_lines)
            and lines[: len(self._last_lines)] == self._last_lines
        ):
            new_lines = lines[len(self._last_lines) :]
            if new_lines:
                self._text.appendPlainText("\n".join(new_lines))
        else:
            self._text.setPlainText("\n".join(lines))

        self._last_lines = lines

        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(old_value, scrollbar.maximum()))

    def _on_open(self) -> None:
        try:
            os.startfile(str(self._log_dir))  # type: ignore[attr-defined]
        except Exception:
            pass
