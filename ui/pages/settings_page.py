"""设置页：应用级参数 + 取料/涂油判定参数 + 录制参数。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

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
        self._slot_occupied = _double_spin(0.0, 10.0, 0.05, 2)
        self._slot_empty = _double_spin(0.0, 10.0, 0.05, 2)
        self._slot_unknown = _double_spin(0.0, 30.0, 0.1, 2)
        self._pick_abort = _double_spin(0.0, 30.0, 0.1, 2)
        self._occlusion_timeout = _double_spin(0.0, 60.0, 0.1, 2)
        self._pick_window = _double_spin(0.0, 30.0, 0.1, 2)
        self._brush_min = _double_spin(0.0, 60.0, 0.1, 2)
        self._brush_gap = _double_spin(0.0, 30.0, 0.1, 2)
        self._brush_motion = _double_spin(0.0, 10.0, 0.01, 3)
        self._brush_distance = _double_spin(0.005, 0.5, 0.005, 3)
        self._brush_require_hand = QCheckBox("要求同时检测到手部")
        self._identity_timeout = _double_spin(0.0, 30.0, 0.1, 2)
        self._event_cooldown = _double_spin(0.0, 60.0, 0.1, 2)

        # 盘体跟踪参数
        self._tray_distance = _double_spin(0.01, 0.5, 0.01, 3)
        self._tray_missing = _double_spin(0.0, 10.0, 0.1, 2)
        self._tray_confidence = _double_spin(0.0, 1.0, 0.05, 2)

        # 手部/刷子跟踪参数
        self._hand_track_distance = _double_spin(0.005, 0.5, 0.005, 3)
        self._hand_track_missing = _double_spin(0.0, 10.0, 0.1, 2)
        self._hand_track_min_hits = QSpinBox()
        self._hand_track_min_hits.setRange(1, 20)
        self._brush_track_distance = _double_spin(0.005, 0.5, 0.005, 3)
        self._brush_track_missing = _double_spin(0.0, 10.0, 0.1, 2)
        self._brush_track_min_hits = QSpinBox()
        self._brush_track_min_hits.setRange(1, 20)
        self._operation_max_active = QSpinBox()
        self._operation_max_active.setRange(1, 64)
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
        roi_form.addRow("盘体区 / 二次涂油区", self._btn_roi)
        roi_form.addRow("", self._roi_summary)

        pipeline_form = QFormLayout()
        pipeline_form.addRow("检测最大帧率", self._max_fps)
        pipeline_form.addRow("槽位有料确认时长（秒）", self._slot_occupied)
        pipeline_form.addRow("槽位空置确认时长（秒）", self._slot_empty)
        pipeline_form.addRow("遮挡/定位失效转未知（秒）", self._slot_unknown)
        pipeline_form.addRow("手离开后取料取消（秒）", self._pick_abort)
        pipeline_form.addRow("遮挡转待确认超时（秒）", self._occlusion_timeout)
        pipeline_form.addRow("取料证据关联窗口（秒）", self._pick_window)
        pipeline_form.addRow("二次涂油最短刷动时长（秒）", self._brush_min)
        pipeline_form.addRow("刷动动作最大间隙（秒）", self._brush_gap)
        pipeline_form.addRow("刷子最小相对位移", self._brush_motion)
        pipeline_form.addRow("刷子-物料最大距离（归一化）", self._brush_distance)
        pipeline_form.addRow("", self._brush_require_hand)
        pipeline_form.addRow("身份关联丢失超时（秒）", self._identity_timeout)
        pipeline_form.addRow("事件冷却（秒）", self._event_cooldown)

        tray_form = QFormLayout()
        tray_form.addRow("盘体最大关联距离（归一化）", self._tray_distance)
        tray_form.addRow("盘体短时漏检保留（秒）", self._tray_missing)
        tray_form.addRow("盘体最低置信度", self._tray_confidence)

        tracking_form = QFormLayout()
        tracking_form.addRow("手部最大关联距离（归一化）", self._hand_track_distance)
        tracking_form.addRow("手部短时漏检保留（秒）", self._hand_track_missing)
        tracking_form.addRow("手部最少命中帧数", self._hand_track_min_hits)
        tracking_form.addRow("刷子最大关联距离（归一化）", self._brush_track_distance)
        tracking_form.addRow("刷子短时漏检保留（秒）", self._brush_track_missing)
        tracking_form.addRow("刷子最少命中帧数", self._brush_track_min_hits)
        tracking_form.addRow("最大同时操作物料数", self._operation_max_active)
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
        content_layout.addWidget(_section("取料与二次涂油判定"))
        content_layout.addLayout(pipeline_form)
        content_layout.addSpacing(12)
        content_layout.addWidget(_section("机器涂油盘跟踪"))
        content_layout.addLayout(tray_form)
        content_layout.addSpacing(12)
        content_layout.addWidget(_section("手部 / 刷子跟踪"))
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
        self._slot_occupied.setValue(float(pipeline.slot_occupied_confirm_s))
        self._slot_empty.setValue(float(pipeline.slot_empty_confirm_s))
        self._slot_unknown.setValue(float(pipeline.slot_occlusion_unknown_s))
        self._pick_abort.setValue(float(pipeline.pick_abort_s))
        self._occlusion_timeout.setValue(float(pipeline.occlusion_timeout_s))
        self._pick_window.setValue(float(pipeline.pick_window_s))
        self._brush_min.setValue(float(pipeline.brush_min_duration_s))
        self._brush_gap.setValue(float(pipeline.brush_max_gap_s))
        self._brush_motion.setValue(float(pipeline.brush_min_motion))
        self._brush_distance.setValue(float(pipeline.brush_part_distance))
        self._brush_require_hand.setChecked(bool(pipeline.brush_require_hand))
        self._identity_timeout.setValue(float(pipeline.identity_timeout_s))
        self._event_cooldown.setValue(float(pipeline.event_cooldown_s))

        self._tray_distance.setValue(float(pipeline.tray_max_distance))
        self._tray_missing.setValue(float(pipeline.tray_max_missing_s))
        self._tray_confidence.setValue(float(pipeline.tray_min_confidence))

        self._hand_track_distance.setValue(float(pipeline.hand_track_max_distance))
        self._hand_track_missing.setValue(float(pipeline.hand_track_max_missing_s))
        self._hand_track_min_hits.setValue(int(pipeline.hand_track_min_hits))
        self._brush_track_distance.setValue(float(pipeline.brush_track_max_distance))
        self._brush_track_missing.setValue(float(pipeline.brush_track_max_missing_s))
        self._brush_track_min_hits.setValue(int(pipeline.brush_track_min_hits))
        self._operation_max_active.setValue(int(pipeline.operation_max_active))
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
                f"{station.name}{suffix}：盘体/取料区 "
                f"{_format_roi(station.tray_roi or station.pick_roi)}；"
                f"二次涂油区 {_format_roi(station.oiling_roi)}"
            )
        self._roi_summary.setText(
            "\n".join(lines) or "未配置工位，请先在 oilcheck.json 中添加"
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
        brush_min_duration = float(self._brush_min.value())
        brush_max_gap = float(self._brush_gap.value())
        brush_min_motion = float(self._brush_motion.value())
        pipeline = replace(
            self._oil_cfg.pipeline,
            max_fps=int(self._max_fps.value()),
            slot_occupied_confirm_s=float(self._slot_occupied.value()),
            slot_empty_confirm_s=float(self._slot_empty.value()),
            slot_occlusion_unknown_s=float(self._slot_unknown.value()),
            pick_abort_s=float(self._pick_abort.value()),
            occlusion_timeout_s=float(self._occlusion_timeout.value()),
            pick_window_s=float(self._pick_window.value()),
            brush_min_duration_s=brush_min_duration,
            brush_max_gap_s=brush_max_gap,
            brush_min_motion=brush_min_motion,
            brush_part_distance=float(self._brush_distance.value()),
            brush_require_hand=self._brush_require_hand.isChecked(),
            identity_timeout_s=float(self._identity_timeout.value()),
            event_cooldown_s=float(self._event_cooldown.value()),
            tray_max_distance=float(self._tray_distance.value()),
            tray_max_missing_s=float(self._tray_missing.value()),
            tray_min_confidence=float(self._tray_confidence.value()),
            hand_track_max_distance=float(self._hand_track_distance.value()),
            hand_track_max_missing_s=float(self._hand_track_missing.value()),
            hand_track_min_hits=int(self._hand_track_min_hits.value()),
            brush_track_max_distance=float(self._brush_track_distance.value()),
            brush_track_max_missing_s=float(self._brush_track_missing.value()),
            brush_track_min_hits=int(self._brush_track_min_hits.value()),
            operation_max_active=int(self._operation_max_active.value()),
            completed_retain_s=float(self._completed_retain.value()),
            # 保留旧字段，便于旧配置/外部脚本读取。
            oiling_min_duration_s=brush_min_duration,
            oiling_max_gap_s=brush_max_gap,
            oiling_min_motion=brush_min_motion,
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
