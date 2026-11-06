"""实时监控页。

页面结构对应方案第 7 节：

- 顶部：系统在线、工位、检测模式、录制状态；
- 左侧：视频 + 取料 ROI + 二次涂油 ROI + 检测框；
- 右侧：单根物料状态、最近事件、复核入口。
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    ComboBox,
    InfoBar,
    InfoBarPosition,
    MessageBox,
    PrimaryPushButton,
    PushButton,
    TableWidget,
)

from app.camera_manager import CameraManager
from app.config_service import ConfigService
from app.models import AppConfig
from app.oilcheck.config import default_slot_layouts
from app.oilcheck.roi import Roi
from app.oilcheck.service import OilCheckService
from app.oilcheck.types import (
    Detection,
    OperationSnapshot,
    OperationState,
    SlotOccupancy,
    SlotSnapshot,
    StationSnapshot,
    format_timestamp,
)
from ui.utils import UiInvoker
from ui.widgets.oilcheck_video import OilCheckVideoView, RoiOverlay

STATE_COLORS: dict[OperationState, str] = {
    OperationState.NOT_STARTED: "#6b7280",
    OperationState.PICKED: "#2563eb",
    OperationState.OILING_CANDIDATE: "#d97706",
    OperationState.PENDING_CONFIRMATION: "#b91c1c",
    OperationState.IDENTITY_UNCERTAIN: "#7c3aed",
    OperationState.OILING_CONFIRMED: "#15803d",
}

SLOT_COLORS: dict[SlotOccupancy, str] = {
    SlotOccupancy.OCCUPIED: "#15803d",
    SlotOccupancy.EMPTY: "#6b7280",
    SlotOccupancy.UNKNOWN: "#b45309",
}

SLOT_REASON_LABELS = {
    "INITIAL": "初始化",
    "RESET": "已重置",
    "SCREW_STABLE": "稳定有料",
    "SCREW_STABLE_UNDER_HAND": "遮挡下有料",
    "CLEAR_EMPTY_EVIDENCE": "稳定空槽",
    "OCCLUSION_TIMEOUT": "遮挡超时",
    "TRAY_UNRELIABLE": "盘体不可靠",
    "NO_TRAY": "未检测到盘体",
}

PICKUP_STATUS_LABELS = {
    "CONFIRMED": "已确认",
    "PENDING": "待确认",
    "UNCONFIRMED": "未确认",
}

IDENTITY_STATUS_LABELS = {
    "ASSOCIATED": "已关联",
    "UNCERTAIN": "不确定",
}

CAMERA_STATE_LABELS = {
    "disconnected": "未连接",
    "connecting": "连接中",
    "connected": "已连接",
    "previewing": "预览中",
    "stopping": "停止中",
    "error": "错误",
    "missing": "相机缺失",
    "demo": "演示（无相机）",
}

MODE_LABELS = {
    "null": "无模型（待接入）",
    "simulated": "演示模式",
    "yolo": "模型检测",
}


class SlotCard(QFrame):
    """机器涂油盘单个槽位的状态卡片。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("slotCard")
        self.setStyleSheet(
            "QFrame#slotCard{background:#1b1c1f;border:1px solid #2a2c31;"
            "border-radius:8px;}"
        )
        self.setMinimumHeight(64)

        self._title = QLabel("-")
        self._title.setStyleSheet("color:#e5e7eb;font-weight:600;")
        self._chip = QLabel("未知")
        self._chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chip.setStyleSheet(
            "background:#b45309;color:#ffffff;border-radius:9px;"
            "padding:2px 8px;font-size:11px;"
        )
        self._detail = QLabel("-")
        self._detail.setStyleSheet("color:#6b7280;font-size:11px;")

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._chip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        layout.addLayout(top)
        layout.addWidget(self._detail)

    def set_slot(self, slot: SlotSnapshot | None) -> None:
        if slot is None:
            self._title.setText("-")
            self._chip.setText("未知")
            self._detail.setText("-")
            return
        self._title.setText(slot.slot_name)
        color = SLOT_COLORS.get(slot.occupancy, "#b45309")
        self._chip.setText(slot.occupancy_label)
        self._chip.setStyleSheet(
            f"background:{color};color:#ffffff;border-radius:9px;"
            "padding:2px 8px;font-size:11px;"
        )
        self._detail.setText(
            f"{slot.visibility_label} · "
            f"{SLOT_REASON_LABELS.get(slot.reason, slot.reason)}"
        )
        self.setToolTip(
            f"{slot.slot_name}\n"
            f"占用：{slot.occupancy.value}\n"
            f"可见性：{slot.visibility.value}\n"
            f"原因：{slot.reason}\n"
            f"最近变化：{format_timestamp(slot.last_change_ts)}\n"
            f"置信度：{slot.confidence if slot.confidence is not None else '-'}"
        )


class OperationCard(QFrame):
    """一条人工取料/二次涂油操作记录。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("operationCard")
        self.setStyleSheet(
            "QFrame#operationCard{background:#1b1c1f;border:1px solid #2a2c31;"
            "border-radius:8px;}"
        )
        self.setMinimumHeight(88)

        self._title = QLabel("-")
        self._title.setStyleSheet("color:#e5e7eb;font-weight:600;")
        self._part = QLabel("-")
        self._part.setStyleSheet("color:#9ca3af;font-size:11px;")
        self._chip = QLabel("未开始")
        self._chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chip.setStyleSheet(
            "background:#374151;color:#ffffff;border-radius:9px;"
            "padding:2px 10px;font-size:12px;"
        )
        self._detail = QLabel("-")
        self._detail.setStyleSheet("color:#6b7280;font-size:11px;")

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._part)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addWidget(self._chip)
        bottom.addSpacing(8)
        bottom.addWidget(self._detail, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addLayout(bottom)

    def set_operation(
        self, operation: OperationSnapshot | None, *, fallback_name: str = ""
    ) -> None:
        if operation is None:
            self._title.setText(fallback_name or "-")
            self._part.setText("-")
            self._chip.setText("空闲")
            self._chip.setStyleSheet(
                "background:#374151;color:#ffffff;border-radius:9px;"
                "padding:2px 10px;font-size:12px;"
            )
            self._detail.setText("-")
            return

        self._title.setText(f"{operation.logical_part_id} · {operation.slot_name}")
        self._part.setText(operation.operation_id)
        color = STATE_COLORS.get(operation.state, "#374151")
        self._chip.setText(operation.state_label)
        self._chip.setStyleSheet(
            f"background:{color};color:#ffffff;border-radius:9px;"
            "padding:2px 10px;font-size:12px;"
        )
        reason = operation.reason_code or "-"
        pickup = PICKUP_STATUS_LABELS.get(
            operation.pickup_status, operation.pickup_status
        )
        identity = IDENTITY_STATUS_LABELS.get(
            operation.identity_status, operation.identity_status
        )
        self._detail.setText(f"取料 {pickup} · 关联 {identity}")
        self.setToolTip(
            f"{operation.operation_id}\n"
            f"来源槽位：{operation.source_slot}\n"
            f"状态：{operation.state.value}\n"
            f"取料：{operation.pickup_status}\n"
            f"关联：{operation.identity_status}\n"
            f"原因：{reason}\n"
            f"证据：{', '.join(operation.evidence) or '-'}"
        )


class MonitorPage(QWidget):
    reviewRequested = Signal(str)

    def __init__(
        self,
        *,
        service: OilCheckService,
        manager: CameraManager,
        cfg_service: ConfigService,
        app_cfg: AppConfig,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("monitorPage")

        self._service = service
        self._manager = manager
        self._cfg_service = cfg_service
        self._app_cfg = app_cfg
        self._ui = UiInvoker(self)

        self._stations = list(service.stations())
        self._current_index = 0
        self._slot_cards: dict[int, SlotCard] = {}
        self._operation_cards: dict[int, OperationCard] = {}
        self._event_ids: list[str] = []
        self._last_frame_ts = 0.0
        self._last_events_version = -1

        self._build_ui()
        self._apply_station(0)

        self._video_timer = QTimer(self)
        self._video_timer.setInterval(int(1000 / max(1, self._app_cfg.ui_fps)))
        self._video_timer.timeout.connect(self._refresh_video)
        self._video_timer.start()

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(200)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self._station_combo = ComboBox()
        for station in self._stations:
            self._station_combo.addItem(f"{station.name}（{station.station_id}）")
        self._station_combo.currentIndexChanged.connect(self._apply_station)

        self._status_chip = self._make_chip("服务未启动", "#4b5563")
        self._mode_chip = self._make_chip("模式：-", "#374151")
        self._rate_chip = self._make_chip("检测：-", "#374151")
        self._record_chip = self._make_chip("录制：关闭", "#374151")

        self._btn_camera = PrimaryPushButton("开始")
        self._btn_demo = PushButton("演示模式")
        self._btn_demo.setCheckable(True)
        self._btn_reset = PushButton("重置状态")
        self._btn_snapshot = PushButton("截图")

        top = QHBoxLayout()
        top.addWidget(BodyLabel("工位"))
        top.addWidget(self._station_combo)
        top.addSpacing(8)
        top.addWidget(self._status_chip)
        top.addWidget(self._mode_chip)
        top.addWidget(self._rate_chip)
        top.addWidget(self._record_chip)
        top.addStretch(1)
        top.addWidget(self._btn_camera)
        top.addWidget(self._btn_demo)
        top.addWidget(self._btn_reset)
        top.addWidget(self._btn_snapshot)

        self._video = OilCheckVideoView()
        self._video_info = QLabel("等待相机画面")
        self._video_info.setStyleSheet("color:#6b7280;font-size:11px;")

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.addWidget(self._video, 1)
        left.addWidget(self._video_info)

        self._slot_grid = QGridLayout()
        self._slot_grid.setSpacing(8)

        self._card_grid = QGridLayout()
        self._card_grid.setSpacing(8)

        self._event_table = TableWidget()
        self._event_table.setColumnCount(4)
        self._event_table.setHorizontalHeaderLabels(
            ["时间", "事件", "物料", "结果/复核"]
        )
        self._event_table.verticalHeader().setVisible(False)
        self._event_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._event_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._event_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self._event_table.horizontalHeader().setStretchLastSection(True)
        self._event_table.setColumnWidth(0, 100)
        self._event_table.setColumnWidth(1, 92)
        self._event_table.setColumnWidth(2, 140)
        self._event_table.setMinimumHeight(220)
        self._event_table.cellDoubleClicked.connect(self._on_event_double_clicked)

        self._btn_review = PushButton("打开复核页")
        self._btn_review.clicked.connect(lambda: self.reviewRequested.emit(""))
        self._btn_clear = PushButton("清空事件")
        self._btn_clear.clicked.connect(self._on_clear_events)

        events_title = QLabel("最近事件")
        events_title.setStyleSheet("color:#e5e7eb;font-weight:600;")

        right = QVBoxLayout()
        right.setContentsMargins(12, 0, 0, 0)
        right.setSpacing(8)
        slots_title = QLabel("槽位状态（机器涂油盘）")
        slots_title.setStyleSheet("color:#e5e7eb;font-weight:600;")
        right.addWidget(slots_title)
        right.addLayout(self._slot_grid)
        right.addSpacing(6)
        materials_title = QLabel("人工操作（最近 4 条）")
        materials_title.setStyleSheet("color:#e5e7eb;font-weight:600;")
        right.addWidget(materials_title)
        right.addLayout(self._card_grid)
        right.addSpacing(8)
        right.addWidget(events_title)
        right.addWidget(self._event_table, 1)
        bottom_buttons = QHBoxLayout()
        bottom_buttons.addWidget(self._btn_review, 1)
        bottom_buttons.addWidget(self._btn_clear)
        right.addLayout(bottom_buttons)

        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setMinimumWidth(400)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        body.addLayout(left, 3)
        body.addWidget(right_widget, 2)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.addLayout(top)
        root.addLayout(body, 1)

        self._btn_camera.clicked.connect(self._on_camera_toggle)
        self._btn_demo.toggled.connect(self._on_demo_toggled)
        self._btn_reset.clicked.connect(self._on_reset)
        self._btn_snapshot.clicked.connect(self._on_snapshot)

    @staticmethod
    def _make_chip(text: str, color: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            f"background:{color};color:#ffffff;border-radius:9px;"
            "padding:2px 10px;font-size:12px;"
        )
        return label

    @staticmethod
    def _set_chip(label: QLabel, text: str, color: str) -> None:
        if label.text() == text:
            return
        label.setText(text)
        label.setStyleSheet(
            f"background:{color};color:#ffffff;border-radius:9px;"
            "padding:2px 10px;font-size:12px;"
        )

    # ------------------------------------------------------------------
    # 工位切换
    # ------------------------------------------------------------------
    def _apply_station(self, index: int) -> None:
        if not self._stations:
            self._video.clear("未配置工位，请编辑 config/oilcheck.json")
            self._video_info.setText("未配置工位")
            return
        index = max(0, min(index, len(self._stations) - 1))
        self._current_index = index
        station = self._stations[index]

        while self._slot_grid.count():
            item = self._slot_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        while self._card_grid.count():
            item = self._card_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._slot_cards.clear()
        self._operation_cards.clear()

        layouts = station.slot_layout or default_slot_layouts(4)
        for pos, layout in enumerate(layouts[:8]):
            card = SlotCard()
            self._slot_cards[layout.index] = card
            self._slot_grid.addWidget(card, pos // 2, pos % 2)

        operation_count = 4
        for pos in range(operation_count):
            card = OperationCard()
            self._operation_cards[pos + 1] = card
            self._card_grid.addWidget(card, pos // 2, pos % 2)

        self._event_ids.clear()
        self._last_events_version = -1
        self._event_table.setRowCount(0)
        self._last_frame_ts = 0.0
        self._btn_demo.blockSignals(True)
        self._btn_demo.setChecked(False)
        self._btn_demo.blockSignals(False)

        self._refresh_status()
        self._refresh_video()

    # ------------------------------------------------------------------
    # 定时刷新
    # ------------------------------------------------------------------
    def _refresh_video(self) -> None:
        station = self._current_station()
        if station is None:
            return

        snapshot = self._service.station_snapshot(station.station_id)
        camera_state = (
            snapshot.stats.get("camera_state") if snapshot else None
        ) or self._service.camera_state(station.camera_id)

        frame = self._service.latest_frame(station.camera_id)
        has_live_picture = camera_state in {"previewing", "demo"}
        if not has_live_picture:
            # 相机停止后清掉最后一帧，避免监控页看起来仍在实时预览。
            if self._video.has_frame or self._last_frame_ts:
                self._last_frame_ts = 0.0
                label = CAMERA_STATE_LABELS.get(camera_state, camera_state)
                self._video.clear(f"相机{label}，点击“开始”启动预览")
                self._video_info.setText("画面已停止")
        elif frame is not None and frame.ts > self._last_frame_ts:
            self._last_frame_ts = frame.ts
            self._video.set_frame(frame.bgr, frame.ts)
            if camera_state == "demo":
                self._video_info.setText(
                    f"画面 {frame.width}×{frame.height} · 演示合成画面（不写入训练录制）"
                )
            else:
                self._video_info.setText(f"画面 {frame.width}×{frame.height}")

        frame_size = snapshot.frame_size if snapshot is not None else (0, 0)
        rois: tuple[RoiOverlay, ...] = ()
        if snapshot is not None:
            tray = snapshot.tray
            if tray is not None and tray.box is not None:
                tray_roi = _box_to_roi(tray.box, frame_size)
                if tray_roi is not None:
                    rois += (
                        RoiOverlay(
                            "机器涂油盘",
                            tray_roi,
                            "#ffd60a",
                            dashed=True,
                        ),
                    )
            for slot in snapshot.slots:
                if slot.box is None:
                    continue
                slot_roi = _box_to_roi(slot.box, frame_size)
                if slot_roi is None:
                    continue
                color = SLOT_COLORS.get(slot.occupancy, "#b45309")
                rois += (
                    RoiOverlay(
                        f"{slot.slot_name} · {slot.occupancy_label}",
                        slot_roi,
                        color,
                        dashed=slot.occupancy != SlotOccupancy.OCCUPIED,
                    ),
                )

        detections: tuple[Detection, ...] = ()
        if snapshot is not None:
            excluded = {
                station.detector.hand_label,
                station.detector.brush_label,
                station.detector.tray_label,
            }
            detections = tuple(
                det for det in snapshot.detections if det.label not in excluded
            ) + tuple(
                Detection(
                    label=f"{track.label}#{track.track_id}",
                    confidence=track.confidence,
                    box=track.box,
                )
                for track in snapshot.tracks
            )

        mode = "" if snapshot is None else snapshot.mode
        tray_state = ""
        if snapshot is not None and snapshot.tray is not None:
            tray_state = " · 盘体可靠" if snapshot.tray.reliable else " · 盘体不可靠"
        banner = (
            f"{station.name} · {CAMERA_STATE_LABELS.get(camera_state, camera_state)}"
            f" · {MODE_LABELS.get(mode, mode)}{tray_state}"
        )
        self._video.set_overlays(
            rois=rois,
            detections=detections,
            banner=banner,
            hand_label=station.detector.hand_label,
            rod_label=station.detector.rod_label,
            brush_label=station.detector.brush_label,
            tray_label=station.detector.tray_label,
        )

    def _refresh_status(self) -> None:
        station = self._current_station()
        if station is None:
            return
        snapshot = self._service.station_snapshot(station.station_id)
        self._update_slot_cards(snapshot)
        self._update_operation_cards(snapshot)
        self._update_status_chips(station, snapshot)
        self._update_events(station.station_id)

        session = self._manager.get_session(station.camera_id)
        previewing = session is not None and session.state.value == "previewing"
        camera_state = (
            snapshot.stats.get("camera_state") if snapshot else None
        ) or self._service.camera_state(station.camera_id)
        if camera_state == "demo":
            self._btn_camera.setText("演示画面")
            self._btn_camera.setEnabled(False)
        else:
            self._btn_camera.setText("停止" if previewing else "开始")
            self._btn_camera.setEnabled(True)
        self._btn_snapshot.setEnabled(previewing)

    def _update_slot_cards(self, snapshot: StationSnapshot | None) -> None:
        slots = {} if snapshot is None else {s.slot_index: s for s in snapshot.slots}
        for index, card in self._slot_cards.items():
            card.set_slot(slots.get(index))

    def _update_operation_cards(self, snapshot: StationSnapshot | None) -> None:
        operations = list(snapshot.operations if snapshot else ())
        cards = list(self._operation_cards.values())
        for index, card in enumerate(cards):
            operation = operations[index] if index < len(operations) else None
            card.set_operation(operation)

    def _update_status_chips(self, station, snapshot: StationSnapshot | None) -> None:
        camera_state = (
            snapshot.stats.get("camera_state") if snapshot else None
        ) or self._service.camera_state(station.camera_id)
        online = bool(snapshot and snapshot.online)
        self._set_chip(
            self._status_chip,
            f"相机：{CAMERA_STATE_LABELS.get(camera_state, camera_state)}",
            "#15803d" if online else "#b91c1c",
        )

        mode = "" if snapshot is None else snapshot.mode
        mode_color = (
            "#b45309"
            if mode == "simulated"
            else "#374151" if mode == "null" else "#15803d"
        )
        self._btn_demo.blockSignals(True)
        self._btn_demo.setChecked(mode == "simulated")
        self._btn_demo.blockSignals(False)
        self._set_chip(
            self._mode_chip, f"模式：{MODE_LABELS.get(mode, mode)}", mode_color
        )

        stats = {} if snapshot is None else snapshot.stats
        processing = stats.get("processing_ms")
        if processing is None:
            rate_text = "检测：-"
        else:
            rate_text = f"检测：{float(processing):.1f}ms · {int(stats.get('detections', 0))}目标"
        if snapshot is not None and snapshot.slots:
            occupied = sum(
                1 for slot in snapshot.slots if slot.occupancy == SlotOccupancy.OCCUPIED
            )
            unknown = sum(
                1 for slot in snapshot.slots if slot.occupancy == SlotOccupancy.UNKNOWN
            )
            rate_text += f" · 槽位 {occupied}/{len(snapshot.slots)} 有料"
            if unknown:
                rate_text += f" · {unknown} 未知"
        self._set_chip(self._rate_chip, rate_text, "#374151")

        status = self._service.recording_status(station.camera_id)
        if status.active:
            elapsed = 0.0
            if status.started_at:
                elapsed = max(0.0, datetime.now().timestamp() - status.started_at)
            self._set_chip(
                self._record_chip,
                f"录制中 {int(elapsed // 60):02d}:{int(elapsed % 60):02d} · "
                f"{status.frames_written}帧",
                "#b91c1c",
            )
        else:
            self._set_chip(self._record_chip, "录制：关闭", "#374151")
        if status.last_error:
            self._set_chip(
                self._record_chip, f"录制错误：{status.last_error}", "#b91c1c"
            )

    def _update_events(self, station_id: str) -> None:
        version = self._service.event_store.version
        if version == self._last_events_version:
            return
        self._last_events_version = version
        events = self._service.recent_events(20, station_id=station_id)
        self._event_ids = [event.event_id for event in events]
        self._event_table.setRowCount(len(events))
        for row, event in enumerate(events):
            result_text = event.result_label
            if event.review_result is not None:
                result_text = f"{event.result_label}/{event.review_label}"
            values = (
                datetime.fromtimestamp(event.timestamp_end).strftime("%H:%M:%S"),
                event.event_label,
                event.logical_part_id,
                result_text,
            )
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if event.result.value != "CONFIRMED":
                    item.setForeground(QColor("#fbbf24"))
                self._event_table.setItem(row, col, item)

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------
    def _on_camera_toggle(self) -> None:
        station = self._current_station()
        if station is None:
            return
        session = self._manager.get_session(station.camera_id)
        if session is None:
            InfoBar.error(
                "相机不存在",
                f"请先在设备页添加相机 {station.camera_id}",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return

        if session.state.value == "previewing":
            future = self._manager.stop_async(station.camera_id)
            action = "停止"
        else:
            future = self._manager.start_async(station.camera_id)
            action = "启动"

        self._btn_camera.setEnabled(False)

        def _done() -> None:
            self._btn_camera.setEnabled(True)
            try:
                future.result()
                InfoBar.success(
                    action,
                    f"{action}成功：{station.name}",
                    position=InfoBarPosition.TOP_RIGHT,
                    duration=1500,
                    parent=self,
                )
            except Exception as exc:  # noqa: BLE001
                InfoBar.error(
                    f"{action}失败",
                    f"{type(exc).__name__}: {exc}（详见日志页）",
                    position=InfoBarPosition.TOP_RIGHT,
                    duration=3000,
                    parent=self,
                )

        future.add_done_callback(lambda _f: self._ui.invoke(_done))

    def _on_demo_toggled(self, checked: bool) -> None:
        station = self._current_station()
        if station is None:
            return
        kind = "simulated" if checked else "default"
        if self._service.set_station_detector_kind(station.station_id, kind) is None:
            InfoBar.error(
                "切换失败",
                "检测器初始化失败，请查看日志",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
        else:
            InfoBar.info(
                "检测模式",
                (
                    "已切换到演示模式，事件仅用于流程联调"
                    if checked
                    else "已恢复配置的检测模式"
                ),
                position=InfoBarPosition.TOP_RIGHT,
                duration=2000,
                parent=self,
            )
        self._last_events_version = -1
        self._refresh_status()

    def _on_reset(self) -> None:
        station = self._current_station()
        if station is None:
            return
        self._service.reset_station(station.station_id)
        InfoBar.success(
            "已重置",
            f"{station.name} 的物料状态已复位",
            position=InfoBarPosition.TOP_RIGHT,
            duration=1500,
            parent=self,
        )

    def _on_snapshot(self) -> None:
        station = self._current_station()
        if station is None:
            return
        session = self._manager.get_session(station.camera_id)
        if session is None:
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = (self._cfg_service.repo_root / self._app_cfg.snapshot_dir).resolve()
        out_path = out_dir / f"{station.camera_id}_{stamp}.bmp"
        try:
            session.snapshot(out_path)
            InfoBar.success(
                "截图",
                str(out_path),
                position=InfoBarPosition.TOP_RIGHT,
                duration=1800,
                parent=self,
            )
        except Exception as exc:  # noqa: BLE001
            InfoBar.error(
                "截图失败",
                str(exc),
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )

    def _on_clear_events(self) -> None:
        dialog = MessageBox(
            "清空事件",
            "将删除本机所有事件与复核记录（训练视频、配置和日志不受影响）。"
            "此操作不可撤销，是否继续？",
            self,
        )
        if not dialog.exec():
            return
        removed = self._service.clear_events()
        self._last_events_version = -1
        self._refresh_status()
        InfoBar.success(
            "已清空",
            f"已删除 {removed} 条事件及对应复核记录",
            position=InfoBarPosition.TOP_RIGHT,
            duration=1800,
            parent=self,
        )

    def _on_event_double_clicked(self, row: int, _column: int) -> None:
        if 0 <= row < len(self._event_ids):
            self.reviewRequested.emit(self._event_ids[row])

    def _current_station(self):
        if not self._stations:
            return None
        return self._stations[self._current_index]

    def set_app_config(self, app_cfg: AppConfig) -> None:
        self._app_cfg = app_cfg
        self._video_timer.setInterval(int(1000 / max(1, app_cfg.ui_fps)))

    def refresh_stations(self, selected_station_id: str | None = None) -> None:
        """配置变化后刷新工位下拉框。"""
        current = selected_station_id
        if current is None and self._current_station() is not None:
            current = self._current_station().station_id

        self._stations = list(self._service.stations())
        self._station_combo.blockSignals(True)
        self._station_combo.clear()
        index = 0
        for i, station in enumerate(self._stations):
            self._station_combo.addItem(f"{station.name}（{station.station_id}）")
            if station.station_id == current:
                index = i
        self._station_combo.setCurrentIndex(index)
        self._station_combo.blockSignals(False)
        self._apply_station(index)


def _box_to_roi(box, frame_size: tuple[int, int]) -> Roi | None:
    width, height = frame_size
    if width <= 0 or height <= 0:
        return None
    return Roi.from_dict(
        {
            "kind": "rect",
            "points": [
                [box.x1 / width, box.y1 / height],
                [box.x2 / width, box.y2 / height],
            ],
        }
    )
