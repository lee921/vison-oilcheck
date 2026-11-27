"""设置页：应用级参数 + 取料/涂油判定参数 + 录制参数。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
    PushButton,
)

from app.camera_manager import CameraManager
from app.config_service import ConfigService
from app.models import AppConfig
from app.oilcheck.config import OilCheckConfig, OilCheckConfigService
from app.oilcheck.roi import Roi
from app.oilcheck.service import OilCheckService
from ui.utils import UiInvoker
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
        oilcheck_service: OilCheckService | None = None,
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
        self._oilcheck_service = oilcheck_service
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
        # 第二条取料判据：丝杆自己出盘区。开关 + 两个关键门限。
        self._screw_exit_enabled = QCheckBox("启用「丝杆移出盘区」取料判据")
        self._screw_exit_enabled.setToolTip(
            "只看杆、不看手：杆在盘区内放稳过，随后被检到移到盘区外，"
            "就算一次取料。手被遮挡/漏检时由它兜底。与手部判据并行，"
            "同一件料只报一次。"
        )
        self._screw_exit_inside = _double_spin(0.0, 10.0, 0.05, 2)
        self._screw_exit_inside.setToolTip(
            "杆必须在盘区内**放稳**（位置不动）这么久，才算「确实在盘里」。"
            "这是判取料的前提，防止杆从盘区上空划过被误判。"
        )
        self._brush_distance = _double_spin(0.005, 0.5, 0.005, 3)
        self._brush_hold = _double_spin(0.0, 60.0, 0.1, 2)
        self._oiling_timeout = _double_spin(0.0, 3600.0, 1.0, 1)
        self._holding_lost_timeout = _double_spin(0.0, 30.0, 0.1, 2)

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
        # NG 事件片段：只给需要人工复核的事件（当前是漏涂）存前后画面。
        self._event_clips = QCheckBox("保存 NG 事件片段（复核页可直接播放）")
        self._event_clips.setToolTip(
            "漏涂等需要复核的事件，前后各留 1.5 秒余量录成 mp4。\n"
            "片段写盘与训练录制共用磁盘和编码资源，确认硬盘余量后再打开。"
        )
        self._clip_margin = _double_spin(0.0, 30.0, 1.5, 0.5)
        self._clip_post = _double_spin(0.0, 60.0, 2.0, 0.5)
        self._clip_buffer_width = QSpinBox()
        self._clip_buffer_width.setRange(0, 3840)
        self._clip_buffer_width.setSpecialValueText("原生分辨率")
        self._clip_buffer_width.setSuffix(" px")
        self._clip_buffer_width.setToolTip(
            "缓冲的缩放宽，0 = 不缩放。缓冲存的是 JPEG：\n"
            "原生 2560×1440 约 75MB，1280×720 约 45MB。复核看动作，720p 通常够。"
        )
        self._clip_jpeg_quality = QSpinBox()
        self._clip_jpeg_quality.setRange(1, 100)
        self._clip_buffer_max_mb = QSpinBox()
        self._clip_buffer_max_mb.setRange(32, 8192)
        self._clip_buffer_max_mb.setSuffix(" MB")
        self._clip_buffer_max_mb.setToolTip("环形缓冲的内存上限，超过就丢最旧的帧")
        self._clip_retain_days = QSpinBox()
        self._clip_retain_days.setRange(0, 3650)
        self._clip_retain_days.setSuffix(" 天")
        self._clip_retain_days.setToolTip("启动时清理超过该天数的片段；0 = 不清理")

        # PLC 警示灯（默认关闭 + 只读）
        self._plc_enabled = QCheckBox("启用 PLC 警示灯输出")
        self._plc_enabled.setToolTip(
            "默认关闭。现场按 docs/12 §8 确认字节序后再开。\n"
            "该 PLC 只允许一条 Modbus TCP 会话，不要同时开博途在线监控。"
        )
        self._plc_dry_run = QCheckBox("只读联调（算帧并打日志，不写 PLC）")
        self._plc_dry_run.setToolTip(
            "联调第一周建议保持勾选：日志里能看到每一拍应写的字节，"
            "可与 PLC 在线监控逐位核对。"
        )
        self._plc_host = LineEdit()
        self._plc_host.setPlaceholderText("192.168.1.88")
        self._plc_port = QSpinBox()
        self._plc_port.setRange(1, 65535)
        self._plc_hold_s = _double_spin(0.0, 60.0, 0.5, 1)
        self._plc_hold_s.setToolTip(
            "良品（三盏绿）/ 漏涂 NG（红灯）在 PLC 上停留多久再回等待取料。"
            "现场默认 2 秒"
        )
        self._btn_plc_test = PushButton("测试连接")
        self._btn_plc_test.clicked.connect(self._on_plc_test)
        # 测试连接会阻塞在 socket 上，放到单线程执行器里跑，避免卡住 UI。
        self._plc_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="oilcheck-plc-test"
        )
        self._ui = UiInvoker(self, logger=logging.getLogger("oilcheck.ui.settings"))

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
        pipeline_form.addRow("", self._screw_exit_enabled)
        pipeline_form.addRow("丝杆在盘内放稳时长（秒）", self._screw_exit_inside)
        pipeline_form.addRow("刷头-丝杆接触距离（归一化）", self._brush_distance)
        pipeline_form.addRow("刷子靠近判完成时长（秒）", self._brush_hold)
        pipeline_form.addRow("未涂油超时判 NG（秒，0=不限时）", self._oiling_timeout)
        pipeline_form.addRow("手持丢失判定超时（秒）", self._holding_lost_timeout)

        assoc_form = QFormLayout()
        assoc_form.addRow("手-丝杆最大关联距离（归一化）", self._hand_screw_distance)
        assoc_form.addRow("手持丝杆推断距离（归一化）", self._held_infer_distance)

        advanced_form = QFormLayout()
        advanced_form.addRow("手持位置预测保持（秒）", self._held_predict_hold)
        advanced_form.addRow("手部跟踪后端", self._hand_backend)
        advanced_form.addRow("刷头跟踪后端", self._brush_backend)

        tracking_form = QFormLayout()
        tracking_form.addRow("手部最大关联距离（归一化）", self._hand_track_distance)
        tracking_form.addRow("手部短时漏检保留（秒）", self._hand_track_missing)
        tracking_form.addRow("手部最少命中帧数", self._hand_track_min_hits)
        tracking_form.addRow("刷头最大关联距离（归一化）", self._brush_track_distance)
        tracking_form.addRow("刷头短时漏检保留（秒）", self._brush_track_missing)
        tracking_form.addRow("刷头最少命中帧数", self._brush_track_min_hits)
        tracking_form.addRow("最大同时任务数", self._task_max_active)
        tracking_form.addRow("完成记录最长保留（秒，0=不限）", self._completed_retain)

        record_form = QFormLayout()
        record_form.addRow("训练视频目录", self._training_dir)
        record_form.addRow("录制帧率", self._record_fps)
        record_form.addRow("分段时长（秒）", self._segment_seconds)
        record_form.addRow("编码 FourCC", self._codec)
        record_form.addRow("", self._auto_start)
        record_form.addRow("", self._event_clips)
        record_form.addRow("片段前后余量（秒）", self._clip_margin)
        record_form.addRow("片段事件后再录（秒）", self._clip_post)
        record_form.addRow("片段缓冲缩放宽", self._clip_buffer_width)
        record_form.addRow("片段 JPEG 质量", self._clip_jpeg_quality)
        record_form.addRow("片段缓冲内存上限", self._clip_buffer_max_mb)
        record_form.addRow("片段保留", self._clip_retain_days)

        plc_form = QFormLayout()
        plc_form.addRow("", self._plc_enabled)
        plc_form.addRow("", self._plc_dry_run)
        plc_form.addRow("PLC IP", self._plc_host)
        plc_form.addRow("端口", self._plc_port)
        plc_form.addRow("完成保持窗（秒）", self._plc_hold_s)
        plc_form.addRow("", self._btn_plc_test)

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
        content_layout.addSpacing(12)
        content_layout.addWidget(_section("PLC 警示灯（Modbus TCP）"))
        content_layout.addLayout(plc_form)
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
        self._screw_exit_enabled.setChecked(bool(pipeline.screw_exit_pick_enabled))
        self._screw_exit_inside.setValue(float(pipeline.screw_exit_pick_inside_s))
        self._brush_distance.setValue(float(pipeline.brush_contact_distance))
        self._brush_hold.setValue(float(pipeline.brush_hold_s))
        self._oiling_timeout.setValue(float(pipeline.oiling_timeout_s))
        self._holding_lost_timeout.setValue(float(pipeline.holding_lost_timeout_s))

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
        self._event_clips.setChecked(bool(recording.event_clips_enabled))
        self._clip_margin.setValue(float(recording.clip_margin_seconds))
        self._clip_post.setValue(float(recording.clip_post_seconds))
        self._clip_buffer_width.setValue(int(recording.clip_buffer_width))
        self._clip_jpeg_quality.setValue(int(recording.clip_jpeg_quality))
        self._clip_buffer_max_mb.setValue(int(recording.clip_buffer_max_mb))
        self._clip_retain_days.setValue(int(recording.clip_retain_days))

        plc = cfg.plc
        self._plc_enabled.setChecked(bool(plc.enabled))
        self._plc_dry_run.setChecked(bool(plc.dry_run))
        self._plc_host.setText(plc.host)
        self._plc_port.setValue(int(plc.port))
        self._plc_hold_s.setValue(float(plc.hold_s))
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

        配置固定在仓库的 ``config/`` 下，UI 保存的位置就是判定读取的位置。
        这条信息不放版面上（不需要常看），放在「打开 ROI 标定」的悬停提示里，
        排查时移上去就能看到。
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
            f"配置目录：{info.get('config_dir', '-')}"
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
        # 读不出来（文件被误保存成 GBK/坏 JSON）时保留当前界面配置，不要
        # 用 `load()` 的返回值把它覆盖掉。
        try:
            self.set_oilcheck_config(self._oil_cfg_service.load())
        except Exception as exc:  # noqa: BLE001
            InfoBar.error(
                "检测配置无法读取",
                f"{exc}\n界面保留当前配置，修复文件后重试。",
                position=InfoBarPosition.TOP_RIGHT,
                duration=6000,
                parent=self,
            )

    def set_app_config(self, app_cfg: AppConfig) -> None:
        self._load_app(app_cfg)

    def set_oilcheck_config(self, cfg: OilCheckConfig) -> None:
        self._load_oilcheck(cfg)

    def _on_plc_test(self) -> None:
        """测试连接：只做一次 FC3 读，不写任何寄存器。

        该 PLC 只允许一条 Modbus 会话，所以这里用完立刻断开——否则主程序
        的 PLC 线程会一直连不上（表现为 1 秒级超时）。
        """

        host = self._plc_host.text().strip() or self._oil_cfg.plc.host
        port = int(self._plc_port.value())
        self._btn_plc_test.setEnabled(False)
        self._btn_plc_test.setText("测试中...")

        def _probe() -> tuple[bool, str]:
            # 走 service：由 PlcIoThread 先关掉自己的连接再探测。自建
            # PlcLink 会与写入线程抢唯一会话，生产模式下必然超时。
            service = self._oilcheck_service
            if service is not None:
                try:
                    return service.check_plc_connection()
                except Exception as exc:  # noqa: BLE001
                    return False, f"{type(exc).__name__}: {exc}"
            from app.oilcheck.plc_link import ModbusError, ModbusLinkError, PlcLink

            link = PlcLink(host, port, connect_timeout_s=2.0, request_timeout_s=2.0)
            try:
                link.open()
                registers = link.read_holding(self._oil_cfg.plc.register_base, 2)
            except (ModbusLinkError, ModbusError) as exc:
                return False, str(exc)
            finally:
                link.close()
            return True, (
                f"HR{self._oil_cfg.plc.register_base}=0x{registers[0]:04X}  "
                f"HR{self._oil_cfg.plc.register_base + 1}=0x{registers[1]:04X}"
            )

        future = self._plc_executor.submit(_probe)
        future.add_done_callback(
            lambda f: self._ui.invoke(lambda: self._on_plc_tested(f))
        )

    def _on_plc_tested(self, future: object) -> None:
        self._btn_plc_test.setEnabled(True)
        self._btn_plc_test.setText("测试连接")
        try:
            ok, detail = future.result()  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        if ok:
            InfoBar.success(
                "PLC 连接正常",
                detail,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
        else:
            InfoBar.error(
                "PLC 连接失败",
                f"{detail}\n该 PLC 只允许一条 Modbus 会话，"
                f"请确认没有博途在线监控/组态软件占用。",
                position=InfoBarPosition.TOP_RIGHT,
                duration=5000,
                parent=self,
            )

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
            screw_exit_pick_enabled=self._screw_exit_enabled.isChecked(),
            screw_exit_pick_inside_s=float(self._screw_exit_inside.value()),
            brush_contact_distance=float(self._brush_distance.value()),
            brush_hold_s=float(self._brush_hold.value()),
            oiling_timeout_s=float(self._oiling_timeout.value()),
            holding_lost_timeout_s=float(self._holding_lost_timeout.value()),
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
            event_clips_enabled=self._event_clips.isChecked(),
            clip_margin_seconds=float(self._clip_margin.value()),
            clip_post_seconds=float(self._clip_post.value()),
            clip_buffer_width=int(self._clip_buffer_width.value()),
            clip_jpeg_quality=int(self._clip_jpeg_quality.value()),
            clip_buffer_max_mb=int(self._clip_buffer_max_mb.value()),
            clip_retain_days=int(self._clip_retain_days.value()),
        )
        plc = replace(
            self._oil_cfg.plc,
            enabled=self._plc_enabled.isChecked(),
            dry_run=self._plc_dry_run.isChecked(),
            host=self._plc_host.text().strip() or self._oil_cfg.plc.host,
            port=int(self._plc_port.value()),
            hold_s=float(self._plc_hold_s.value()),
        )
        oil_cfg = replace(
            self._oil_cfg, pipeline=pipeline, recording=recording, plc=plc
        )

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
