"""设置页：应用级参数 + 取料/涂油判定参数 + 录制参数。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    ComboBox,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    PrimaryPushButton,
)

from app.camera_manager import CameraManager
from app.config_service import ConfigService
from app.models import AppConfig
from app.oilcheck.config import OilCheckConfig, OilCheckConfigService
from app.oilcheck.roi import Roi
from ui.widgets.roi_calibration import RoiCalibrationDialog


class SettingsPage(QWidget):
    def __init__(
        self,
        *,
        cfg_service: ConfigService,
        app_cfg: AppConfig,
        manager: CameraManager,
        oilcheck_cfg_service: OilCheckConfigService,
        oilcheck_cfg: OilCheckConfig,
        on_saved: Callable[[AppConfig], None],
        on_oilcheck_saved: Callable[[OilCheckConfig], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settingsPage")
        self.setStyleSheet(
            "QSpinBox,QDoubleSpinBox{background:#1b1c1f;color:#e5e7eb;"
            "border:1px solid #3a3c42;border-radius:6px;padding:4px 8px;"
            "min-height:24px;}"
            "QSpinBox:focus,QDoubleSpinBox:focus{border-color:#2ea6ff;}"
            "QLabel{color:#e5e7eb;}"
            "QCheckBox{color:#e5e7eb;}"
        )

        self._cfg_service = cfg_service
        self._app_cfg = app_cfg
        self._manager = manager
        self._oil_cfg_service = oilcheck_cfg_service
        self._oil_cfg = oilcheck_cfg
        self._on_saved = on_saved
        self._on_oilcheck_saved = on_oilcheck_saved

        # 应用参数
        self._layout = ComboBox()
        self._layout.addItems(["1", "4", "9", "16"])
        self._ui_fps = QSpinBox()
        self._ui_fps.setRange(1, 60)
        self._decode_fps = QSpinBox()
        self._decode_fps.setRange(1, 60)
        self._offline_timeout = QSpinBox()
        self._offline_timeout.setRange(1, 600)
        self._snapshot_dir = LineEdit()
        self._log_dir = LineEdit()

        # 判定参数
        self._max_fps = QSpinBox()
        self._max_fps.setRange(1, 60)
        self._pick_hand_distance = _double_spin(0.005, 0.5, 0.005, 3)
        self._pick_confirm = _double_spin(0.0, 10.0, 0.05, 2)
        self._brush_distance = _double_spin(0.005, 0.5, 0.005, 3)
        self._brush_hold = _double_spin(0.0, 60.0, 0.1, 2)
        self._holding_lost_timeout = _double_spin(0.0, 30.0, 0.1, 2)

        # 盘体跟踪参数
        self._tray_distance = _double_spin(0.01, 0.5, 0.01, 3)
        self._tray_missing = _double_spin(0.0, 10.0, 0.1, 2)
        self._tray_confidence = _double_spin(0.0, 1.0, 0.05, 2)

        # 手与丝杆关联
        self._hand_screw_distance = _double_spin(0.005, 1.0, 0.005, 3)
        self._held_infer_distance = _double_spin(0.005, 1.0, 0.005, 3)
        # 高级
        self._held_predict_hold = _double_spin(0.0, 5.0, 0.05, 2)
        self._hand_backend = ComboBox()
        self._hand_backend.addItems(["auto", "bytetrack", "object"])
        self._brush_backend = ComboBox()
        self._brush_backend.addItems(["auto", "bytetrack", "object"])

        # 手部/刷头跟踪参数
        self._hand_track_distance = _double_spin(0.005, 0.5, 0.005, 3)
        self._hand_track_missing = _double_spin(0.0, 10.0, 0.1, 2)
        self._hand_track_min_hits = QSpinBox()
        self._hand_track_min_hits.setRange(1, 20)
        self._brush_track_distance = _double_spin(0.005, 0.5, 0.005, 3)
        self._brush_track_missing = _double_spin(0.0, 10.0, 0.1, 2)
        self._brush_track_min_hits = QSpinBox()
        self._brush_track_min_hits.setRange(1, 20)
        self._task_max_active = QSpinBox()
        self._task_max_active.setRange(1, 64)
        self._completed_retain = _double_spin(0.0, 600.0, 1.0, 1)

        # 录制参数
        self._training_dir = LineEdit()
        self._record_fps = QSpinBox()
        self._record_fps.setRange(1, 60)
        self._segment_seconds = _double_spin(10.0, 86400.0, 30.0, 1)
        self._codec = LineEdit()
        self._auto_start = QCheckBox("程序启动后自动开始录制")

        self._btn_save = PrimaryPushButton("保存全部")
        self._btn_roi = PrimaryPushButton("打开 ROI 标定")
        self._btn_roi.setFixedWidth(200)
        self._btn_roi.clicked.connect(self._open_roi_calibration)
        self._roi_summary = QLabel("尚未配置")
        self._roi_summary.setWordWrap(True)
        self._roi_summary.setStyleSheet("color:#9ca3af;font-size:11px;")

        app_form = QFormLayout()
        app_form.addRow("预览布局", self._layout)
        app_form.addRow("UI 刷新帧率", self._ui_fps)
        app_form.addRow("解码最大帧率", self._decode_fps)
        app_form.addRow("离线超时（秒）", self._offline_timeout)
        app_form.addRow("截图目录", self._snapshot_dir)
        app_form.addRow("日志目录", self._log_dir)

        roi_form = QFormLayout()
        roi_form.addRow("涂油盘区 / 工作追踪区 / 涂油区", self._btn_roi)
        roi_form.addRow("", self._roi_summary)

        pipeline_form = QFormLayout()
        pipeline_form.addRow("检测最大帧率", self._max_fps)
        pipeline_form.addRow("手-丝杆接近距离（归一化）", self._pick_hand_distance)
        pipeline_form.addRow("盘内停留确认时长（秒）", self._pick_confirm)
        pipeline_form.addRow("刷头-丝杆接触距离（归一化）", self._brush_distance)
        pipeline_form.addRow("刷子靠近判完成时长（秒）", self._brush_hold)
        pipeline_form.addRow("手持丢失判定超时（秒）", self._holding_lost_timeout)

        assoc_form = QFormLayout()
        assoc_form.addRow("手-丝杆最大关联距离（归一化）", self._hand_screw_distance)
        assoc_form.addRow("手持丝杆推断距离（归一化）", self._held_infer_distance)

        advanced_form = QFormLayout()
        advanced_form.addRow("手持位置预测保持（秒）", self._held_predict_hold)
        advanced_form.addRow("手部跟踪后端", self._hand_backend)
        advanced_form.addRow("刷头跟踪后端", self._brush_backend)

        tray_form = QFormLayout()
        tray_form.addRow("盘体最大关联距离（归一化）", self._tray_distance)
        tray_form.addRow("盘体短时漏检保留（秒）", self._tray_missing)
        tray_form.addRow("盘体最低置信度", self._tray_confidence)

        tracking_form = QFormLayout()
        tracking_form.addRow("手部最大关联距离（归一化）", self._hand_track_distance)
        tracking_form.addRow("手部短时漏检保留（秒）", self._hand_track_missing)
        tracking_form.addRow("手部最少命中帧数", self._hand_track_min_hits)
        tracking_form.addRow("刷头最大关联距离（归一化）", self._brush_track_distance)
        tracking_form.addRow("刷头短时漏检保留（秒）", self._brush_track_missing)
        tracking_form.addRow("刷头最少命中帧数", self._brush_track_min_hits)
        tracking_form.addRow("最大同时任务数", self._task_max_active)
        tracking_form.addRow("完成后保留（秒）", self._completed_retain)

        record_form = QFormLayout()
        record_form.addRow("训练视频目录", self._training_dir)
        record_form.addRow("录制帧率", self._record_fps)
        record_form.addRow("分段时长（秒）", self._segment_seconds)
        record_form.addRow("编码 FourCC", self._codec)
        record_form.addRow("", self._auto_start)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 12, 0)
        content_layout.addWidget(_section("应用设置"))
        content_layout.addLayout(app_form)
        content_layout.addSpacing(12)
        content_layout.addWidget(_section("ROI 标定"))
        content_layout.addLayout(roi_form)
        content_layout.addSpacing(12)
        content_layout.addWidget(_section("取料事件与二次涂油判定"))
        content_layout.addLayout(pipeline_form)
        content_layout.addSpacing(12)
        content_layout.addWidget(_section("涂油盘定位"))
        content_layout.addLayout(tray_form)
        content_layout.addSpacing(12)
        content_layout.addWidget(_section("手与丝杆关联"))
        content_layout.addLayout(assoc_form)
        content_layout.addSpacing(12)
        content_layout.addWidget(_section("高级：手持预测 / 跟踪后端"))
        content_layout.addLayout(advanced_form)
        content_layout.addSpacing(12)
        content_layout.addWidget(_section("手部 / 刷头跟踪"))
        content_layout.addLayout(tracking_form)
        content_layout.addSpacing(12)
        content_layout.addWidget(_section("训练视频录制"))
        content_layout.addLayout(record_form)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet(
            "QScrollArea{background:transparent;border:none;}"
            "QScrollArea > QWidget > QWidget{background:transparent;}"
        )

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._btn_save)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.addWidget(scroll, 1)
        root.addLayout(buttons)

        self._btn_save.clicked.connect(self.save)
        self._load_app(app_cfg)
        self._load_oilcheck(oilcheck_cfg)

    def _load_app(self, app_cfg: AppConfig) -> None:
        self._app_cfg = app_cfg
        self._layout.setCurrentText(str(app_cfg.layout))
        self._ui_fps.setValue(int(app_cfg.ui_fps))
        self._decode_fps.setValue(int(app_cfg.decode_max_fps))
        self._offline_timeout.setValue(int(app_cfg.offline_timeout_s))
        self._snapshot_dir.setText(app_cfg.snapshot_dir)
        self._log_dir.setText(app_cfg.log_dir)

    def _load_oilcheck(self, cfg: OilCheckConfig) -> None:
        self._oil_cfg = cfg
        pipeline = cfg.pipeline
        self._max_fps.setValue(int(pipeline.max_fps))
        self._pick_hand_distance.setValue(float(pipeline.pick_hand_distance))
        self._pick_confirm.setValue(float(pipeline.pick_confirm_s))
        self._brush_distance.setValue(float(pipeline.brush_contact_distance))
        self._brush_hold.setValue(float(pipeline.brush_hold_s))
        self._holding_lost_timeout.setValue(float(pipeline.holding_lost_timeout_s))

        self._tray_distance.setValue(float(pipeline.tray_max_distance))
        self._tray_missing.setValue(float(pipeline.tray_max_missing_s))
        self._tray_confidence.setValue(float(pipeline.tray_min_confidence))
        self._hand_screw_distance.setValue(
            float(pipeline.hand_screw_max_distance)
        )
        self._held_infer_distance.setValue(float(pipeline.held_infer_distance))
        self._held_predict_hold.setValue(float(pipeline.held_predict_hold_s))
        self._hand_backend.setCurrentText(str(pipeline.hand_tracker_backend))
        self._brush_backend.setCurrentText(str(pipeline.brush_tracker_backend))

        self._hand_track_distance.setValue(float(pipeline.hand_track_max_distance))
        self._hand_track_missing.setValue(float(pipeline.hand_track_max_missing_s))
        self._hand_track_min_hits.setValue(int(pipeline.hand_track_min_hits))
        self._brush_track_distance.setValue(float(pipeline.brush_track_max_distance))
        self._brush_track_missing.setValue(float(pipeline.brush_track_max_missing_s))
        self._brush_track_min_hits.setValue(int(pipeline.brush_track_min_hits))
        self._task_max_active.setValue(int(pipeline.task_max_active))
        self._completed_retain.setValue(float(pipeline.completed_retain_s))

        recording = cfg.recording
        self._training_dir.setText(recording.training_dir)
        self._record_fps.setValue(int(recording.fps))
        self._segment_seconds.setValue(float(recording.segment_seconds))
        self._codec.setText(recording.codec)
        self._auto_start.setChecked(bool(recording.auto_start))
        self._update_roi_summary(cfg)

    def _update_roi_summary(self, cfg: OilCheckConfig) -> None:
        lines = []
        for station in cfg.stations:
            suffix = "" if station.enabled else "（未启用）"
            lines.append(
                f"{station.name}{suffix}：涂油盘区 "
                f"{_format_roi(station.tray_roi)}；工作追踪区 "
                f"{_format_roi(station.work_roi)}；涂油区 "
                f"{_format_roi(station.oiling_roi)}"
            )
        self._roi_summary.setText(
            "\n".join(lines) or "未配置工位，请先在 oilcheck.json 中添加"
        )
        self._update_config_source()

    def _update_config_source(self) -> None:
        """把实际生效的配置文件路径写进按钮提示。

        配置只在首次运行时从仓库模板复制，之后一直读数据目录那份；改仓库
        里的模板不会生效。这条信息不放版面上（不需要常看），放在「打开 ROI
        标定」的悬停提示里，排查时移上去就能看到。
        """

        info = self._oil_cfg_service.source_info()
        stamp = "未找到"
        if info.get("mtime"):
            stamp = datetime.fromtimestamp(float(info["mtime"])).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        self._btn_roi.setToolTip(
            f"生效配置：{info['path']}\n"
            f"修改时间：{stamp}\n"
            f"数据目录：{info['data_root']}"
        )

    def _open_roi_calibration(self) -> None:
        dialog = RoiCalibrationDialog(
            manager=self._manager,
            cfg_service=self._oil_cfg_service,
            cfg=self._oil_cfg,
            on_saved=self._on_oilcheck_saved,
            parent=self,
        )
        dialog.exec()
        # 无论保存还是取消，都从磁盘重新读取一次，避免界面与服务配置不一致。
        self.set_oilcheck_config(self._oil_cfg_service.load())

    def set_app_config(self, app_cfg: AppConfig) -> None:
        self._load_app(app_cfg)

    def set_oilcheck_config(self, cfg: OilCheckConfig) -> None:
        self._load_oilcheck(cfg)

    def save(self) -> None:
        app_cfg = AppConfig.from_dict(
            {
                **self._app_cfg.to_dict(),
                "layout": int(self._layout.currentText()),
                "ui_fps": int(self._ui_fps.value()),
                "decode_max_fps": int(self._decode_fps.value()),
                "offline_timeout_s": int(self._offline_timeout.value()),
                "snapshot_dir": self._snapshot_dir.text().strip() or "snapshots",
                "log_dir": self._log_dir.text().strip() or "logs",
            }
        )
        pipeline = replace(
            self._oil_cfg.pipeline,
            max_fps=int(self._max_fps.value()),
            pick_hand_distance=float(self._pick_hand_distance.value()),
            pick_confirm_s=float(self._pick_confirm.value()),
            brush_contact_distance=float(self._brush_distance.value()),
            brush_hold_s=float(self._brush_hold.value()),
            holding_lost_timeout_s=float(self._holding_lost_timeout.value()),
            tray_max_distance=float(self._tray_distance.value()),
            tray_max_missing_s=float(self._tray_missing.value()),
            tray_min_confidence=float(self._tray_confidence.value()),
            hand_screw_max_distance=float(self._hand_screw_distance.value()),
            held_infer_distance=float(self._held_infer_distance.value()),
            held_predict_hold_s=float(self._held_predict_hold.value()),
            hand_tracker_backend=self._hand_backend.currentText(),
            brush_tracker_backend=self._brush_backend.currentText(),
            hand_track_max_distance=float(self._hand_track_distance.value()),
            hand_track_max_missing_s=float(self._hand_track_missing.value()),
            hand_track_min_hits=int(self._hand_track_min_hits.value()),
            brush_track_max_distance=float(self._brush_track_distance.value()),
            brush_track_max_missing_s=float(self._brush_track_missing.value()),
            brush_track_min_hits=int(self._brush_track_min_hits.value()),
            task_max_active=int(self._task_max_active.value()),
            completed_retain_s=float(self._completed_retain.value()),
        )
        recording = replace(
            self._oil_cfg.recording,
            training_dir=self._training_dir.text().strip()
            or self._oil_cfg.recording.training_dir,
            fps=int(self._record_fps.value()),
            segment_seconds=float(self._segment_seconds.value()),
            codec=self._codec.text().strip() or "mp4v",
            auto_start=self._auto_start.isChecked(),
        )
        oil_cfg = replace(self._oil_cfg, pipeline=pipeline, recording=recording)

        try:
            self._cfg_service.save_app_config(app_cfg)
            self._oil_cfg_service.save(oil_cfg)
        except Exception as exc:  # noqa: BLE001
            InfoBar.error(
                "保存失败",
                str(exc),
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return

        self._app_cfg = app_cfg
        self._oil_cfg = oil_cfg
        self._on_saved(app_cfg)
        self._on_oilcheck_saved(oil_cfg)

        InfoBar.success(
            "已保存",
            "app.json 与 oilcheck.json 已更新，运行配置已同步",
            position=InfoBarPosition.TOP_RIGHT,
            duration=1800,
            parent=self,
        )


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("color:#f3f4f6;font-weight:700;font-size:13px;")
    return label


def _format_roi(roi: Roi) -> str:
    if roi.kind == "rect":
        (x1, y1), (x2, y2) = roi.points
        return f"[{x1:.3f}, {y1:.3f}] – [{x2:.3f}, {y2:.3f}]"
    return f"多边形（{len(roi.points)} 点）"


def _double_spin(
    minimum: float, maximum: float, step: float, decimals: int
) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    return spin
