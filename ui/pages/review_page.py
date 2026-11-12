"""事件复核页：查看事件、打开片段、人工标注复核结果。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    ComboBox,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    PlainTextEdit,
    PushButton,
    TableWidget,
)

from app.oilcheck.events import EventStore
from app.oilcheck.types import EventResult, ReviewResult, format_timestamp
from app.review import fallback_lines, summary_from_event


class ReviewPage(QWidget):
    def __init__(
        self,
        *,
        event_store: EventStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("reviewPage")
        self._store = event_store
        self._event_ids: list[str] = []
        self._last_version = -1

        self._filter = ComboBox()
        self._filter.addItems(["待复核", "全部", "已复核"])
        self._filter.currentIndexChanged.connect(self._refresh)

        self._note = LineEdit()
        self._note.setPlaceholderText("复核备注（可选）")

        self._btn_refresh = PushButton("刷新")
        self._btn_open = PushButton("打开视频片段")
        self._btn_abnormal = PushButton("确认异常")
        self._btn_false = PushButton("误报")
        self._btn_unknown = PushButton("无法判断")

        self._stats = QLabel("-")
        self._stats.setStyleSheet("color:#9ca3af;")

        top = QHBoxLayout()
        top.addWidget(BodyLabel("筛选"))
        top.addWidget(self._filter)
        top.addSpacing(12)
        top.addWidget(self._stats)
        top.addStretch(1)
        top.addWidget(self._btn_refresh)

        action = QHBoxLayout()
        action.addWidget(self._note, 1)
        action.addWidget(self._btn_open)
        action.addWidget(self._btn_abnormal)
        action.addWidget(self._btn_false)
        action.addWidget(self._btn_unknown)

        self._table = TableWidget()
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            ["时间", "工位", "任务", "事件", "结果", "复核", "片段"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 150)
        self._table.setColumnWidth(1, 90)
        self._table.setColumnWidth(2, 160)
        self._table.setColumnWidth(3, 110)
        self._table.setColumnWidth(4, 90)
        self._table.setColumnWidth(5, 90)

        # 详情区：按通用复核摘要结构渲染，不认识任何 SOP 专有键名。
        # 旧事件（没有 review_summary）走 fallback_lines 平铺原始字段。
        self._detail = PlainTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setMinimumHeight(180)
        self._detail.setPlaceholderText("选中一条事件查看判定依据与实测指标")

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self._table)
        splitter.addWidget(self._detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.addLayout(top)
        root.addWidget(splitter, 1)
        root.addLayout(action)

        self._table.itemSelectionChanged.connect(self._refresh_detail)
        self._btn_refresh.clicked.connect(self._refresh)
        self._btn_open.clicked.connect(self._open_clip)
        self._btn_abnormal.clicked.connect(
            lambda: self._submit(ReviewResult.CONFIRMED_ABNORMAL)
        )
        self._btn_false.clicked.connect(
            lambda: self._submit(ReviewResult.FALSE_POSITIVE)
        )
        self._btn_unknown.clicked.connect(
            lambda: self._submit(ReviewResult.UNDETERMINABLE)
        )

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._refresh_if_changed)
        self._timer.start()

        self._refresh()

    def _refresh_if_changed(self) -> None:
        if self._store.version != self._last_version:
            self._refresh()

    def _refresh(self) -> None:
        current = self._selected_event()
        self._refresh_rows()
        # 重建表格会丢选中行：尽量恢复，否则详情区会在提交后立刻变空。
        if current is not None and current.event_id in self._event_ids:
            row = self._event_ids.index(current.event_id)
            self._table.selectRow(row)
        elif self._table.rowCount() > 0:
            self._table.selectRow(0)
        self._refresh_detail()

    def _refresh_rows(self) -> None:
        self._last_version = self._store.version
        mode = self._filter.currentText()
        events = self._store.recent(500)
        if mode == "待复核":
            events = [
                e
                for e in events
                if e.review_result is None and e.result != EventResult.CONFIRMED
            ]
        elif mode == "已复核":
            events = [e for e in events if e.review_result is not None]

        self._event_ids = [event.event_id for event in events]
        self._table.setRowCount(len(events))
        for row, event in enumerate(events):
            clip = "有" if event.clip_reference else "-"
            values = (
                format_timestamp(event.timestamp_end),
                event.station_id,
                event.task_id,
                event.event_label,
                event.result_label,
                event.review_label,
                clip,
            )
            for col, text in enumerate(values):
                self._table.setItem(row, col, QTableWidgetItem(text))

        counts = self._store.counts()
        self._stats.setText(
            f"事件 {counts['total']} · 待复核 {counts['pending_review']} · "
            f"未确认 {counts['unconfirmed']}"
        )
        has_rows = bool(events)
        for button in (
            self._btn_open,
            self._btn_abnormal,
            self._btn_false,
            self._btn_unknown,
        ):
            button.setEnabled(has_rows)

    def _refresh_detail(self) -> None:
        """把选中事件的证据渲染到详情区。

        有 ``review_summary`` 时按通用结构渲染（任何 SOP 都适用）；没有时
        回退到原始键值平铺——历史事件和尚未接入摘要的检测项也能看到全部
        字段，不会是一片空白。
        """

        event = self._selected_event()
        self._load_note(event)
        if event is None:
            self._detail.setPlainText("")
            return

        final = str(event.evidence.get("final_result") or "").strip()
        result_text = (
            f"{event.result_label}（周期结论 {final}）" if final else event.result_label
        )
        summary = summary_from_event(event.evidence)
        # 事件类型是枚举限定的（PICK / MANUAL_OILING / OILING_MISSING），
        # 别的 SOP 暂时只能复用其中之一。摘要里有更准确的检查项名时以它为准，
        # 免得「螺丝扭矩 NG」在表头显示成「人工二次涂油未完成」。
        check = summary.check_name if summary is not None else ""
        if not check or check in event.event_label:
            header = event.event_label
        else:
            header = f"{event.event_label} / {check}"
        lines: list[str] = [
            f"任务 {event.task_id} · {header} · {result_text}",
            f"开始 {format_timestamp(event.timestamp_start)}"
            f"    结束 {format_timestamp(event.timestamp_end)}",
        ]
        if event.timestamp_end > event.timestamp_start:
            lines.append(
                f"持续 {event.timestamp_end - event.timestamp_start:.2f} s"
            )
        lines.append(f"复核：{event.review_label}")
        if event.review_note:
            lines.append(f"复核备注：{event.review_note}")
        lines.append("")

        if summary is not None:
            lines.extend(summary.lines())
        else:
            lines.extend(fallback_lines(event.evidence))
        self._detail.setPlainText("\n".join(lines))

    def select_event(self, event_id: str) -> None:
        self._refresh()
        if not event_id:
            return
        try:
            row = self._event_ids.index(event_id)
        except ValueError:
            return
        self._table.selectRow(row)
        self._table.scrollToItem(self._table.item(row, 0))

    def _load_note(self, event) -> None:
        """把已保存的复核备注回填到输入框，便于改判时在此基础上修改。"""

        self._note.setText("" if event is None else (event.review_note or ""))

    def _selected_event(self):
        row = self._table.currentRow()
        if row < 0 or row >= len(self._event_ids):
            return None
        return self._store.get(self._event_ids[row])

    def _submit(self, result: ReviewResult) -> None:
        event = self._selected_event()
        if event is None:
            return
        note = self._note.text().strip()
        updated = self._store.submit_review(
            event.event_id, result, note=note, reviewer="operator"
        )
        if updated is None:
            InfoBar.error(
                "复核失败",
                "事件不存在或已从内存中移除",
                position=InfoBarPosition.TOP_RIGHT,
                duration=2500,
                parent=self,
            )
            return
        InfoBar.success(
            "已记录复核",
            f"{updated.task_id} → {updated.review_label}",
            position=InfoBarPosition.TOP_RIGHT,
            duration=1500,
            parent=self,
        )
        # 刷新后详情区会从事件里把刚保存的备注回填回来，不必清空输入框。
        self._refresh()

    def _open_clip(self) -> None:
        event = self._selected_event()
        if event is None:
            return
        if not event.clip_reference:
            InfoBar.warning(
                "无视频片段",
                "第一版未启用事件片段保存；请使用训练录制视频或现场录像复核。",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3500,
                parent=self,
            )
            return
        path = Path(event.clip_reference)
        if not path.exists():
            InfoBar.error(
                "片段不存在",
                str(path),
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
