"""实时监控页。

页面结构尽量保持原样：

- 顶部：系统在线、工位、检测模式、录制状态；
- 左侧：视频 + ROI + 检测/跟踪结果；
- 右侧：检测区域状态、最近任务、最近事件、复核入口。

检测逻辑改为“单根丝杆任务”模型：取料状态、手持状态、刷涂状态与覆盖进度
直接来自 :class:`TaskSnapshot`。
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
    QProgressBar,
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
from app.oilcheck.roi import Roi
from app.oilcheck.service import OilCheckService
from app.oilcheck.types import (
    Detection,
    StationSnapshot,
    TaskResult,
    TaskSnapshot,
    TaskStage,
)
from ui.utils import UiInvoker
from ui.widgets.oilcheck_video import OilCheckVideoView, RoiOverlay

STAGE_COLORS: dict[TaskStage, str] = {
    TaskStage.WAITING_PICK: "#6b7280",
    TaskStage.PICK_CANDIDATE: "#d97706",
    TaskStage.LEAVING_TRAY: "#d97706",
    TaskStage.PICK_CONFIRMED: "#2563eb",
    TaskStage.HOLDING: "#2563eb",
    TaskStage.BRUSH_CONTACT: "#0891b2",
    TaskStage.BRUSHING: "#7c3aed",
    TaskStage.OILING_COMPLETED: "#15803d",
    TaskStage.PENDING_CONFIRMATION: "#b91c1c",
    TaskStage.NG: "#b91c1c",
}

RESULT_COLORS: dict[TaskResult, str] = {
    TaskResult.PROCESSING: "#374151",
    TaskResult.OK: "#15803d",
    TaskResult.NG: "#b91c1c",
    TaskResult.PENDING_CONFIRMATION: "#b45309",
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


class RegionCard(QFrame):
    """检测区域/跟踪状态卡片（保留原槽位卡片的位置与样式）。"""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("regionCard")
        self.setStyleSheet(
            "QFrame#regionCard{background:#1b1c1f;border:1px solid #2a2c31;"
            "border-radius:8px;}"
        )
        self.setMinimumHeight(64)

        self._title = QLabel(title)
        self._title.setStyleSheet("color:#e5e7eb;font-weight:600;")
        self._chip = QLabel("-")
        self._chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chip.setStyleSheet(
            "background:#374151;color:#ffffff;border-radius:9px;"
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

    def set_state(
        self, chip: str, color: str, detail: str, *, tooltip: str = ""
    ) -> None:
        self._chip.setText(chip)
        self._chip.setStyleSheet(
            f"background:{color};color:#ffffff;border-radius:9px;"
            "padding:2px 8px;font-size:11px;"
        )
        self._detail.setText(detail)
        self.setToolTip(tooltip or detail)


class TaskCard(QFrame):
    """一条单根丝杆任务记录。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("taskCard")
        self.setStyleSheet(
            "QFrame#taskCard{background:#1b1c1f;border:1px solid #2a2c31;"
            "border-radius:8px;}"
        )
        self.setMinimumHeight(104)

        self._title = QLabel("-")
        self._title.setStyleSheet("color:#e5e7eb;font-weight:600;")
        self._stage = QLabel("-")
        self._stage.setStyleSheet("color:#9ca3af;font-size:11px;")
        self._chip = QLabel("空闲")
        self._chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._chip.setStyleSheet(
            "background:#374151;color:#ffffff;border-radius:9px;"
            "padding:2px 10px;font-size:12px;"
        )
        self._detail = QLabel("-")
        self._detail.setStyleSheet("color:#6b7280;font-size:11px;")
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(True)
        self._progress.setFormat("覆盖 %p%")
        self._progress.setFixedHeight(14)
        self._progress.setStyleSheet(
            "QProgressBar{background:#25272c;color:#d1d5db;border:none;"
            "border-radius:7px;font-size:10px;text-align:center;}"
            "QProgressBar::chunk{background:#2ea6ff;border-radius:7px;}"
        )

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._title)
        top.addStretch(1)
        top.addWidget(self._stage)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.addWidget(self._chip)
        bottom.addSpacing(8)
        bottom.addWidget(self._detail, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(5)
        layout.addLayout(top)
        layout.addLayout(bottom)
        layout.addWidget(self._progress)

    def set_task(self, task: TaskSnapshot | None) -> None:
        if task is None:
            self._title.setText("-")
            self._stage.setText("-")
            self._chip.setText("空闲")
            self._chip.setStyleSheet(
                "background:#374151;color:#ffffff;border-radius:9px;"
                "padding:2px 10px;font-size:12px;"
            )
            self._detail.setText("-")
            self._progress.setValue(0)
            return

        self._title.setText(task.task_id)
        self._stage.setText(task.stage_label)
        color = RESULT_COLORS.get(task.final_result, "#374151")
        self._chip.setText(task.result_label)
        self._chip.setStyleSheet(
            f"background:{color};color:#ffffff;border-radius:9px;"
            "padding:2px 10px;font-size:12px;"
        )
        self._detail.setText(
            f"取料 {task.pick_label} · 手持 {task.holding_label} · "
            f"涂油 {task.oiling_label}"
        )
        self._progress.setValue(task.coverage_percent)
        timeline = " → ".join(
            f"{name}@{datetime.fromtimestamp(ts).strftime('%H:%M:%S')}"
            for name, ts in task.timeline
        )
        self.setToolTip(
            f"{task.task_id}\n"
            f"阶段：{task.stage.value}（{task.reason_code or '-'}）\n"
            f"丝杆轨迹：{task.screw_track_id}  手部轨迹：{task.hand_track_id}  "
            f"刷头轨迹：{task.brush_track_id}\n"
            f"有效刷涂：{task.effective_brush_duration_s:.2f}s  "
            f"相对运动：{task.brush_motion:.3f}\n"
            f"覆盖区段：{list(task.axis_segments)}\n"
            f"证据：{', '.join(task.evidence) or '-'}\n"
            f"时间线：{timeline or '-'}"
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
        self._region_cards: dict[str, RegionCard] = {}
        self._task_cards: dict[int, TaskCard] = {}
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

        self._region_grid = QGridLayout()
        self._region_grid.setSpacing(8)

        self._task_grid = QGridLayout()
        self._task_grid.setSpacing(8)

        self._event_table = TableWidget()
        self._event_table.setColumnCount(4)
        self._event_table.setHorizontalHeaderLabels(
            ["时间", "事件", "任务", "结果/复核"]
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
        regions_title = QLabel("检测区域与跟踪")
        regions_title.setStyleSheet("color:#e5e7eb;font-weight:600;")
        right.addWidget(regions_title)
        right.addLayout(self._region_grid)
        right.addSpacing(6)
        tasks_title = QLabel("任务（最近 4 条）")
        tasks_title.setStyleSheet("color:#e5e7eb;font-weight:600;")
        right.addWidget(tasks_title)
        right.addLayout(self._task_grid)
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

        while self._region_grid.count():
            item = self._region_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        while self._task_grid.count():
            item = self._task_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._region_cards.clear()
        self._task_cards.clear()

        for pos, (key, title) in enumerate(
            (
                ("tray", "涂油盘参考区"),
                ("hand", "手持操作区"),
                ("brush", "刷涂区"),
                ("exclude", "物料盒排除区"),
            )
        ):
            card = RegionCard(title)
            self._region_cards[key] = card
            self._region_grid.addWidget(card, pos // 2, pos % 2)

        for pos in range(4):
            card = TaskCard()
            self._task_cards[pos + 1] = card
            self._task_grid.addWidget(card, pos // 2, pos % 2)

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

        rois: list[RoiOverlay] = [
            RoiOverlay("涂油盘区", station.tray_roi, "#ffd60a"),
            RoiOverlay("手持操作区", station.hand_roi, "#64d2ff"),
            RoiOverlay("刷涂区", station.brush_roi, "#ff375f"),
        ]
        for roi in station.exclude_rois:
            rois.append(RoiOverlay("排除区", roi, "#8a8f98", dashed=True))

        frame_size = snapshot.frame_size if snapshot is not None else (0, 0)
        if snapshot is not None and snapshot.tray is not None:
            tray = snapshot.tray
            if tray.box is not None:
                tray_roi = _box_to_roi(tray.box, frame_size)
                if tray_roi is not None:
                    rois.append(
                        RoiOverlay(
                            f"检测盘体 · {'可靠' if tray.reliable else '预测'}",
                            tray_roi,
                            "#f59e0b",
                            dashed=True,
                        )
                    )

        detections: tuple[Detection, ...] = ()
        if snapshot is not None:
            excluded = {
                station.detector.hand_label,
                station.detector.brush_head_label,
                station.detector.brush_rod_label,
                station.detector.tray_label,
            }
            detections = tuple(
                det for det in snapshot.detections if det.label not in excluded
            ) + tuple(
                Detection(
                    label=f"{track.label}#{track.track_id}",
                    confidence=track.confidence,
                    box=track.box,
                    angle=track.angle,
                    point=track.point,
                )
                for track in snapshot.tracks
            )

        mode = "" if snapshot is None else snapshot.mode
        tray_state = ""
        if snapshot is not None and snapshot.tray is not None:
            tray_state = " · 盘体可靠" if snapshot.tray.reliable else " · 盘体预测"
        active_task = ""
        if snapshot is not None:
            active_task = str(snapshot.stats.get("active_task_id", ""))
        task_state = f" · {active_task}" if active_task else ""
        banner = (
            f"{station.name} · {CAMERA_STATE_LABELS.get(camera_state, camera_state)}"
            f" · {MODE_LABELS.get(mode, mode)}{tray_state}{task_state}"
        )
        self._video.set_overlays(
            rois=tuple(rois),
            detections=detections,
            banner=banner,
            hand_label=station.detector.hand_label,
            screw_label=station.detector.screw_label,
            brush_label=station.detector.brush_head_label,
            brush_rod_label=station.detector.brush_rod_label,
            tray_label=station.detector.tray_label,
        )

    def _refresh_status(self) -> None:
        station = self._current_station()
        if station is None:
            return
        snapshot = self._service.station_snapshot(station.station_id)
        self._update_region_cards(station, snapshot)
        self._update_task_cards(snapshot)
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

    def _update_region_cards(self, station, snapshot: StationSnapshot | None) -> None:
        tray = snapshot.tray if snapshot is not None else None
        tracks = snapshot.tracks if snapshot is not None else ()
        hand_count = sum(
            1
            for t in tracks
            if t.label == station.detector.hand_label and t.observed
        )
        brush_count = sum(
            1
            for t in tracks
            if t.label == station.detector.brush_head_label and t.observed
        )
        screw_count = sum(
            1
            for t in tracks
            if t.label == station.detector.screw_label and t.observed
        )
        if tray is not None and tray.visible:
            motion = {"MOVING": "移动中", "STILL": "静止"}.get(tray.motion, "未知")
            self._region_cards["tray"].set_state(
                "可见" if tray.reliable else "预测",
                "#15803d" if tray.reliable else "#b45309",
                f"{motion} · 置信度 {tray.confidence or 0:.2f}",
            )
        else:
            self._region_cards["tray"].set_state(
                "配置 ROI" if snapshot is not None else "等待",
                "#374151",
                "使用固定涂油盘 ROI" if snapshot is not None else "-",
            )
        self._region_cards["hand"].set_state(
            f"{hand_count} 只手", "#15803d" if hand_count else "#374151",
            f"丝杆轨迹 {screw_count} 条",
        )
        self._region_cards["brush"].set_state(
            f"{brush_count} 个刷头",
            "#15803d" if brush_count else "#374151",
            "刷头工作端接触与相对运动",
        )
        self._region_cards["exclude"].set_state(
            f"{len(station.exclude_rois)} 个区域",
            "#b45309" if station.exclude_rois else "#374151",
            "物料盒区域不参与推理",
        )

    def _update_task_cards(self, snapshot: StationSnapshot | None) -> None:
        tasks = list(snapshot.tasks if snapshot else ())[-4:][::-1]
        cards = list(self._task_cards.values())
        for index, card in enumerate(cards):
            task = tasks[index] if index < len(tasks) else None
            card.set_task(task)

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
        if snapshot is not None:
            tasks = snapshot.tasks
            if tasks:
                active = sum(
                    1 for task in tasks if task.final_result == TaskResult.PROCESSING
                )
                ok = sum(1 for task in tasks if task.final_result == TaskResult.OK)
                rate_text += f" · 任务 {len(tasks)}（进行中 {active}，OK {ok}）"
            candidates = int(stats.get("pick_candidates", 0))
            if candidates:
                rate_text += f" · 候选 {candidates}"
        health: list[str] = []
        gap_events = int(stats.get("frame_gap_events", 0) or 0)
        dropped = int(stats.get("dropped_frames", 0) or 0)
        if gap_events:
            health.append(f"时间跳变 {gap_events}")
        if dropped:
            health.append(f"丢帧 {dropped}")
        if health:
            rate_text += " · " + "，".join(health)
        self._set_chip(
            self._rate_chip, rate_text, "#b45309" if health else "#374151"
        )

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
                event.task_id,
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
            f"{station.name} 的任务状态已复位",
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
