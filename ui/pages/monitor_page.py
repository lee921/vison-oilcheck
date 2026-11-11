"""实时监控页。

页面结构尽量保持原样：

- 顶部：系统在线、工位、检测模式、录制状态；
- 左侧：视频 + ROI + 检测/跟踪结果；
- 右侧：检测区域状态、最近任务、最近事件、复核入口。

检测逻辑只有两条判定：取料完成（空手进盘、带料离盘）与人工二次涂油完成
（手持丝杆时刷子靠近累计到阈值）。阶段与进度直接来自 :class:`TaskSnapshot`。
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
    CheckBox,
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
from app.oilcheck.service import OilCheckService
from app.oilcheck.types import (
    BBox,
    Detection,
    StationSnapshot,
    TaskResult,
    TaskSnapshot,
    TaskStage,
)
from ui.utils import UiInvoker
from ui.widgets.oilcheck_video import PALETTE, OilCheckVideoView, RoiOverlay

# 回放速率选项：0.5× 是现场逐帧核对用的慢放，1× 原速。
SPEED_OPTIONS: tuple[tuple[str, float], ...] = (
    ("0.5×", 0.5),
    ("1×", 1.0),
    ("2×", 2.0),
)

STAGE_COLORS: dict[TaskStage, str] = {
    TaskStage.WAITING_PICK: "#6b7280",
    TaskStage.PICKED: "#2563eb",
    TaskStage.OILED: "#15803d",
}

RESULT_COLORS: dict[TaskResult, str] = {
    TaskResult.PROCESSING: "#374151",
    TaskResult.COMPLETED: "#15803d",
    TaskResult.NG: "#b91c1c",
}

CAMERA_STATE_LABELS = {
    "video": "示例回放",
    "video_stopped": "回放已停止",
    "disconnected": "未连接",
    "connecting": "连接中",
    "connected": "已连接",
    "previewing": "预览中",
    "stopping": "停止中",
    "error": "错误",
    "missing": "相机缺失",
}

MODE_LABELS = {
    "null": "无模型（待接入）",
    "yolo": "模型检测",
}

LIVE_SOURCE = "__live__"


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
        self._progress.setFormat("刷子靠近 %p%")
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
            f"手部轨迹 {task.hand_track_id or '-'} · "
            f"刷子靠近 {task.brush_hold_s:.2f}s / "
            f"{task.brush_hold_target_s:.2f}s"
        )
        self._progress.setValue(task.brush_hold_percent)
        timeline = " → ".join(
            f"{name}@{datetime.fromtimestamp(ts).strftime('%H:%M:%S')}"
            for name, ts in task.timeline
        )
        self.setToolTip(
            f"{task.task_id}\n"
            f"阶段：{task.stage.value}（{task.reason_code or '-'}）\n"
            f"手部轨迹：{task.hand_track_id or '-'}  "
            f"刷头轨迹：{task.brush_track_id or '-'}\n"
            f"刷子靠近累计：{task.brush_hold_s:.2f}s / "
            f"{task.brush_hold_target_s:.2f}s\n"
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
        # 当前画面对应的示例视频帧号文本（实时相机为空）。
        self._frame_pos_text = ""
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

        self._source_combo = ComboBox()
        self._source_signature: tuple[tuple[str, str], ...] | None = None
        self._reload_sources()

        # 回放速率：只在示例视频下显示。默认取 demo.speed，切换视频后沿用
        # 用户当前选择。
        self._speed_combo = ComboBox()
        self._speed_combo.setFixedWidth(78)
        self._speed_combo.setToolTip("示例回放速率：只影响播放快慢，不影响判定")
        for text, value in SPEED_OPTIONS:
            self._speed_combo.addItem(text, userData=value)
        self._demo_speed = float(self._service.demo_speed())
        self._select_speed(self._demo_speed)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)

        self._btn_camera = PrimaryPushButton("开始")
        self._btn_reset = PushButton("重置状态")
        self._btn_snapshot = PushButton("截图")
        self._chk_track_ids = CheckBox("显示追踪 ID")
        self._chk_track_ids.setChecked(True)
        self._chk_track_ids.setToolTip(
            "在画面上标出工作追踪区内的跟踪目标 ID：\n"
            "手/刷头为视觉轨迹 ID（#100004），丝杆为丝杆视觉轨迹 ID（#300001）。\n"
            "丢失中的轨迹以虚线框和 ? 标出。"
        )
        self._chk_track_ids.toggled.connect(self._on_track_ids_toggled)

        top = QHBoxLayout()
        top.addWidget(BodyLabel("工位"))
        top.addWidget(self._station_combo)
        top.addSpacing(6)
        top.addWidget(BodyLabel("画面源"))
        top.addWidget(self._source_combo)
        top.addSpacing(6)
        self._speed_label = BodyLabel("速率")
        top.addWidget(self._speed_label)
        top.addWidget(self._speed_combo)
        # 默认隐藏，_refresh_status 会按当前画面源决定是否显示。
        self._sync_speed_visibility("", "")
        top.addSpacing(8)
        top.addWidget(self._status_chip)
        top.addWidget(self._mode_chip)
        top.addWidget(self._rate_chip)
        top.addWidget(self._record_chip)
        top.addStretch(1)
        top.addWidget(self._chk_track_ids)
        top.addWidget(self._btn_camera)
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

        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        self._btn_camera.clicked.connect(self._on_camera_toggle)
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

        # 涂油盘区不在监控页展示：它是判定用的内部参考区，不是操作对象。
        cards = [
            ("work", "工作追踪区"),
            ("oiling", "涂油区"),
        ]
        for pos, (key, title) in enumerate(cards):
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
        self._frame_pos_text = ""
        self._sync_source_combo()

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
        has_live_picture = camera_state in {"previewing", "video"}
        if not has_live_picture:
            if self._video.has_frame or self._last_frame_ts:
                self._last_frame_ts = 0.0
                label = CAMERA_STATE_LABELS.get(camera_state, camera_state)
                self._video.clear(f"相机{label}，点击“开始”启动预览")
                self._video_info.setText("画面已停止")
                self._frame_pos_text = ""
        elif frame is not None and frame.ts > self._last_frame_ts:
            self._last_frame_ts = frame.ts
            self._video.set_frame(frame.bgr, frame.ts)
            if camera_state == "video":
                info = self._service.video_status(station.station_id)
                progress = float(info.get("progress") or 0.0) * 100.0
                # 帧号与这一帧画面同时记录，横幅才不会比画面快一帧。
                self._frame_pos_text = _frame_position(info)
                speed = float(info.get("speed") or 1.0)
                speed_text = "" if abs(speed - 1.0) < 1e-6 else f" · {speed:g}×"
                self._video_info.setText(
                    f"画面 {frame.width}×{frame.height} · 示例回放"
                    f" {info.get('name', '')} · {self._frame_pos_text}"
                    f"{speed_text} · {progress:.0f}%（不写入训练录制）"
                )
            else:
                self._frame_pos_text = ""
                self._video_info.setText(f"画面 {frame.width}×{frame.height}")

        # 涂油盘区的实际位置由模型检出的 oil_station 框体现，画面上叠加
        # 工作追踪区（判定范围）与涂油区（标定/显示用，虚线）。
        rois: list[RoiOverlay] = [
            RoiOverlay("工作追踪区", station.work_roi, "#64d2ff"),
            RoiOverlay("涂油区", station.oiling_roi, "#32d74b", dashed=True),
        ]

        # 画面上的框全部来自配置里的类别名，未知类别也会被画出来。
        # 检测框与跟踪 ID 分开绘制：检测框表达“模型看到了什么”，
        # 跟踪 ID 表达“管线正在跟踪谁”，两者可以单独关闭。
        detections: tuple[Detection, ...] = ()
        if snapshot is not None:
            detections = tuple(snapshot.detections)

        mode = "" if snapshot is None else snapshot.mode
        active_task = ""
        counts_text = ""
        if snapshot is not None:
            active_task = str(snapshot.stats.get("active_task_id", ""))
            counts = snapshot.stats.get("label_counts") or {}
            if counts:
                counts_text = " · " + " ".join(
                    f"{name}:{count}" for name, count in counts.items()
                )
        task_state = f" · {active_task}" if active_task else ""
        # 示例回放时把帧号放在横幅最前面，现场对照录像定位问题最直接。
        frame_state = f" · {self._frame_pos_text}" if self._frame_pos_text else ""
        banner = (
            f"{station.name} · {CAMERA_STATE_LABELS.get(camera_state, camera_state)}"
            f"{frame_state} · {MODE_LABELS.get(mode, mode)}"
            f"{task_state}{counts_text}"
        )
        # 跟踪结果只包含工作追踪区/涂油盘区内的目标（管线已做过区域过滤），
        # 因此画面上的 ID 就是“被跟踪的目标”。
        tracks = () if snapshot is None else tuple(snapshot.tracks)
        self._video.set_overlays(
            rois=tuple(rois),
            detections=detections,
            tracks=tracks,
            banner=banner,
            label_colors=_label_colors(station.detector),
        )

    def _on_track_ids_toggled(self, enabled: bool) -> None:
        self._video.set_show_track_ids(bool(enabled))

    def _refresh_status(self) -> None:
        station = self._current_station()
        if station is None:
            return
        snapshot = self._service.station_snapshot(station.station_id)
        # 画面源清单可能随配置重载变化，先重建再同步选中项。
        self._sync_sources()
        # 画面源可能被服务层直接切换（配置重载、外部调用），下拉保持同步。
        self._sync_source_combo()
        self._update_region_cards(station, snapshot)
        self._update_task_cards(snapshot)
        self._update_status_chips(station, snapshot)
        self._update_events(station.station_id)
        camera_state = (
            snapshot.stats.get("camera_state") if snapshot else None
        ) or self._service.camera_state(station.camera_id)
        self._sync_speed_visibility(camera_state, station.station_id)

        session = self._manager.get_session(station.camera_id)
        previewing = session is not None and session.state.value == "previewing"
        camera_state = (
            snapshot.stats.get("camera_state") if snapshot else None
        ) or self._service.camera_state(station.camera_id)
        on_video = camera_state.startswith("video")
        if on_video:
            # 回放由“画面源”下拉控制，相机启停按钮此时无意义。
            self._btn_camera.setText("示例回放中")
            self._btn_camera.setEnabled(False)
        else:
            self._btn_camera.setText("停止" if previewing else "开始")
            self._btn_camera.setEnabled(True)
        self._btn_snapshot.setEnabled(previewing)

    def _update_region_cards(self, station, snapshot: StationSnapshot | None) -> None:
        counts, suffix = work_area_counts(station, snapshot)
        screw_count = counts[station.detector.screw_label]
        hand_count = counts[station.detector.hand_label]
        brush_count = counts[station.detector.brush_label]
        self._region_cards["work"].set_state(
            f"丝杆 {screw_count} 个",
            "#15803d" if screw_count else "#374151",
            f"手 {hand_count} 只 · 刷具 {brush_count} 个"
            + (suffix if suffix else "（区内跟踪）"),
        )
        self._region_cards["oiling"].set_state(
            "已标定" if station.oiling_roi is not None else "未标定",
            "#2563eb",
            "预留涂油范围约束（本轮仅显示）",
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
        mode_color = "#374151" if mode in {"", "null"} else "#15803d"
        self._set_chip(
            self._mode_chip, f"模式：{MODE_LABELS.get(mode, mode)}", mode_color
        )

        stats = {} if snapshot is None else snapshot.stats
        processing = stats.get("processing_ms")
        imgsz = stats.get("imgsz")
        fps = stats.get("process_fps")
        if processing is None:
            rate_text = "检测：-"
        else:
            parts = [f"imgsz {int(imgsz)}"] if imgsz else []
            parts.append(f"{float(processing):.1f}ms")
            if fps is not None:
                parts.append(f"{int(fps)}fps")
            rate_text = "检测：" + " · ".join(parts)
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
        if self._service.active_video(station.station_id) is not None:
            InfoBar.info(
                "示例回放中",
                "请先在上方“画面源”切回实时相机",
                position=InfoBarPosition.TOP_RIGHT,
                duration=2000,
                parent=self,
            )
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

    # ------------------------------------------------------------------
    # 画面源（实时相机 / 示例视频）
    # ------------------------------------------------------------------
    def _reload_sources(self) -> None:
        """按配置里的 demo.videos 重建画面源下拉。"""

        combo = getattr(self, "_source_combo", None)
        if combo is None:
            return
        videos = self._service.demo_videos()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("实时相机", userData=LIVE_SOURCE)
        for video in videos:
            combo.addItem(video.name, userData=video.name)
        combo.blockSignals(False)
        self._source_signature = tuple((v.name, v.path) for v in videos)

    def _sync_sources(self) -> None:
        """配置里的视频清单变了就重建下拉。

        服务层会在配置文件被外部改动后自动重载（例如手工补上 ``demo.videos``），
        但界面只在页面构建时读一次清单，不跟着重建的话下拉里永远是旧的几条，
        现场会以为「配置改了没用」。清单很短，直接比较签名。
        """

        signature = tuple(
            (v.name, v.path) for v in self._service.demo_videos()
        )
        if signature == getattr(self, "_source_signature", None):
            return
        self._reload_sources()

    def _select_speed(self, speed: float) -> None:
        """把速率下拉同步到给定倍数（找不到就选最接近的一项）。"""

        combo = getattr(self, "_speed_combo", None)
        if combo is None:
            return
        index = 0
        best = float("inf")
        for pos in range(combo.count()):
            value = float(combo.itemData(pos))
            delta = abs(value - float(speed))
            if delta < best:
                best = delta
                index = pos
        combo.blockSignals(True)
        combo.setCurrentIndex(index)
        combo.blockSignals(False)

    def _current_speed(self) -> float:
        return float(self._speed_combo.currentData() or 1.0)

    def _sync_speed_visibility(self, camera_state: str, station_id: str) -> None:
        """速率只对示例回放有意义，实时相机隐藏；显示时同步到实际速率。

        同步到实际速率而不是只看下拉：配置重载后视频源会按 ``demo.speed``
        重建，下拉需要跟着反映真实状态。
        """

        if getattr(self, "_speed_combo", None) is None:
            return
        show = camera_state == "video"
        self._speed_label.setVisible(show)
        self._speed_combo.setVisible(show)
        if not show:
            return
        info = self._service.video_status(station_id)
        speed = info.get("speed")
        if speed:
            self._select_speed(float(speed))

    def _on_speed_changed(self, _index: int) -> None:
        station = self._current_station()
        if station is None:
            return
        speed = self._current_speed()
        self._service.set_station_video_speed(station.station_id, speed)

    def _sync_source_combo(self) -> None:
        """把下拉框同步到该工位当前实际使用的画面源。"""

        combo = getattr(self, "_source_combo", None)
        station = self._current_station()
        if combo is None or station is None:
            return
        video = self._service.active_video(station.station_id)
        target = LIVE_SOURCE if video is None else video.name

        current = combo.currentData()
        if current == target:
            return
        index = combo.findData(target)
        combo.blockSignals(True)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def _on_source_changed(self, _index: int) -> None:
        station = self._current_station()
        if station is None:
            return
        data = self._source_combo.currentData()
        name = None if data in (None, "", LIVE_SOURCE) else str(data)
        ok = self._service.set_station_video(station.station_id, name)
        if not ok:
            InfoBar.error(
                "切换画面源失败",
                "示例视频未在配置中登记、文件不存在或启动失败，详见日志",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            self._sync_source_combo()
            return

        self._last_frame_ts = 0.0
        self._frame_pos_text = ""
        self._last_events_version = -1
        self._video.clear("示例回放启动中…" if name else "等待相机画面")
        # 新视频源按配置的默认速率启动，这里把下拉当前选择应用过去，
        # 免得切一次视频速率就被重置。
        if name:
            self._service.set_station_video_speed(
                station.station_id, self._current_speed()
            )
        InfoBar.info(
            "画面源",
            f"已切换到：{name}" if name else "已切回实时相机",
            position=InfoBarPosition.TOP_RIGHT,
            duration=2000,
            parent=self,
        )
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


def work_area_counts(
    station, snapshot: StationSnapshot | None
) -> tuple[dict[str, int], str]:
    """统计工作追踪区内的目标数，返回 ``({label: 数量}, 附加说明)``。

    - **手/刷头**：按跟踪轨迹统计，只算本帧实际观测到、且中心在区内的；
      ``snapshot.tracks`` 比「区内」宽——它还包括涂油盘区内的目标，以及移出区外
      但仍在跟踪的目标（``roi_track_margin``）。
    - **丝杆**：不建轨迹，只做逐帧检测，因此按本帧检测框在区内的个数统计。
      不要用「跟踪数 or 全画面检测数」兜底——跟踪数为 0 时会静默回退成全画面
      检测总数，卡片显示的就完全不是区内数了。
    - **raw_mode**：没有轨迹且本来就不过滤区域，只能按全画面检测计数，
      返回值里带上说明，避免误读。
    """

    cfg = station.detector
    labels = (cfg.hand_label, cfg.brush_label, cfg.screw_label)
    counts = (snapshot.stats.get("label_counts") if snapshot else None) or {}
    if snapshot is None:
        return ({label: int(counts.get(label, 0)) for label in labels}, "")
    if bool(snapshot.stats.get("raw_mode")):
        return (
            {label: int(counts.get(label, 0)) for label in labels},
            "（raw_mode 全画面计数）",
        )

    width, height = snapshot.frame_size
    work_roi = station.work_roi

    def in_work_area(box: BBox) -> bool:
        return work_roi.contains_box_center(box, width, height)

    inside_tracks = [
        track
        for track in snapshot.tracks
        if track.observed and in_work_area(track.box)
    ]
    result = dict.fromkeys(labels, 0)
    for track in inside_tracks:
        if track.label in result:
            result[track.label] += 1
    result[cfg.screw_label] = sum(
        1
        for det in snapshot.detections
        if det.label == cfg.screw_label and in_work_area(det.box)
    )
    return (result, "")


def _frame_position(info: dict) -> str:
    """把帧序号格式化成「第 123/456 帧」（1 基，与 ROI 标定窗口一致）。"""

    index = int(info.get("frame_index") or 0)
    total = int(info.get("frame_count") or 0)
    if total > 0:
        return f"第 {index + 1}/{total} 帧"
    return f"第 {index + 1} 帧"


def _label_colors(detector) -> dict[str, str]:
    """按配置里的类别名分配固定配色。

    类别名全部取自 ``detector`` 配置，UI 不再硬编码 ``hand`` / ``brush``
    等字符串；模型后续新增的类别由视频控件按调色板顺延取色。
    """

    names = (
        detector.hand_label,
        detector.brush_label,
        detector.tray_label,
        detector.screw_label,
    )
    colors: dict[str, str] = {}
    for index, name in enumerate(names):
        if name and name not in colors:
            colors[name] = PALETTE[index % len(PALETTE)]
    return colors

