"""ROI 标定对话框：截取相机/示例视频画面，拖拽绘制涂油盘区、工作追踪区与涂油区。"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    BodyLabel,
    ComboBox,
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    PushButton,
    Slider,
)

from app.camera_manager import CameraManager
from app.oilcheck.config import (
    DemoVideo,
    OilCheckConfig,
    OilCheckConfigService,
    StationConfig,
)
from app.oilcheck.roi import Roi
from app.paths import resolve_runtime_file, runtime_root
from ui.utils import UiInvoker
from ui.widgets.roi_canvas import RoiCanvas

logger = logging.getLogger("oilcheck.ui.roi_calibration")

_ROI_NAMES = {
    "tray": "涂油盘区",
    "work": "工作追踪区",
    "oiling": "涂油区",
}

# 标定模式顺序：先判定用的大区域，再仅显示用的涂油区。
_MODE_ORDER = ("tray", "work", "oiling")


def _roi_name(key: str) -> str:
    return _ROI_NAMES.get(key, key)


def _format_seconds(value: float) -> str:
    total = max(0, int(round(value)))
    return f"{total // 60:02d}:{total % 60:02d}"


class RoiCalibrationDialog(QDialog):
    def __init__(
        self,
        *,
        manager: CameraManager,
        cfg_service: OilCheckConfigService,
        cfg: OilCheckConfig,
        on_saved,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("ROI 标定")
        self.resize(1120, 760)
        self.setStyleSheet(
            "QDialog{background:#1b1c1f;}"
            "QLabel{color:#e5e7eb;}"
            "QPushButton{color:#e5e7eb;}"
            "QPushButton:checked{background:#0f6cbd;color:#ffffff;border-radius:4px;}"
        )

        self._manager = manager
        self._cfg_service = cfg_service
        self._cfg = cfg
        self._on_saved = on_saved
        self._ui = UiInvoker(self)
        # 标定属于配置层操作，未启用的工位也允许预先标定。
        self._stations = list(cfg.stations)
        self._camera_id = ""
        self._mode = "tray"
        self._rois: dict[str, Roi | None] = {key: None for key in _MODE_ORDER}
        self._frozen = False
        self._last_frame_ts = 0.0
        self._camera_starting = False
        # 画面源：None 为实时相机，字符串为 demo.videos 中登记的示例视频名。
        self._video_name: str | None = None
        self._video: DemoVideo | None = None
        self._capture: cv2.VideoCapture | None = None
        self._video_frames = 0
        self._video_fps = 0.0

        self._station_combo = ComboBox()
        for station in self._stations:
            suffix = "" if station.enabled else " · 未启用"
            self._station_combo.addItem(
                f"{station.name}（{station.station_id}）{suffix}"
            )
        self._station_combo.currentIndexChanged.connect(self._on_station_changed)
        self._source_combo = ComboBox()
        self._source_combo.addItem("实时相机", userData=None)
        for video in cfg.demo.videos:
            self._source_combo.addItem(video.name, userData=video.name)
        self._source_combo.currentIndexChanged.connect(self._on_source_changed)
        self._camera_label = QLabel("-")
        self._camera_label.setStyleSheet("color:#9ca3af;")
        self._frame_info = QLabel("画面：-")
        self._frame_info.setStyleSheet("color:#6b7280;font-size:11px;")

        self._btn_camera = PushButton("打开相机")
        self._btn_camera.clicked.connect(self._ensure_camera)
        self._btn_capture = PushButton("截取画面")
        self._btn_capture.clicked.connect(self._toggle_capture)

        # 示例视频位置条：拖动选择要截取的那一帧，无需相机即可标定。
        self._video_slider = Slider()
        self._video_slider.setFixedHeight(22)
        self._video_slider.setOrientation(Qt.Orientation.Horizontal)
        self._video_slider.setRange(0, 1000)
        self._video_slider.setMinimumWidth(240)
        self._video_slider.setEnabled(False)
        self._video_slider.valueChanged.connect(self._on_video_seek)
        self._video_pos = QLabel("示例帧：-")
        self._video_pos.setStyleSheet("color:#9ca3af;font-size:11px;")
        self._video_row = QWidget()
        video_layout = QHBoxLayout(self._video_row)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.addWidget(BodyLabel("视频位置"))
        video_layout.addWidget(self._video_slider, 1)
        video_layout.addWidget(self._video_pos)
        self._video_row.setVisible(False)

        self._video_read_timer = QTimer(self)
        self._video_read_timer.setSingleShot(True)
        self._video_read_timer.setInterval(80)
        self._video_read_timer.timeout.connect(self._apply_video_seek)

        self._mode_buttons: dict[str, PushButton] = {}
        for key in _MODE_ORDER:
            button = PushButton(_ROI_NAMES[key])
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, mode=key: self._set_mode(mode))
            self._mode_buttons[key] = button
        self._mode_buttons["tray"].setChecked(True)
        self._btn_clear = PushButton("清除当前区域")
        self._btn_clear.clicked.connect(self._clear_current)

        self._status = QLabel("选择工位后打开相机")
        self._status.setStyleSheet("color:#9ca3af;")

        self._btn_save = PrimaryPushButton("保存并应用")
        self._btn_save.clicked.connect(self._save)
        self._btn_cancel = PushButton("取消")
        self._btn_cancel.clicked.connect(self.reject)

        self._canvas = RoiCanvas()
        self._canvas.roiChanged.connect(self._on_roi_changed)

        top = QHBoxLayout()
        top.addWidget(BodyLabel("工位"))
        top.addWidget(self._station_combo)
        top.addSpacing(12)
        # 没登记示例视频时整组隐藏，避免出现只有“实时相机”一项的下拉。
        self._source_widgets = (BodyLabel("画面源"), self._source_combo)
        for widget in self._source_widgets:
            widget.setVisible(bool(cfg.demo.videos))
            top.addWidget(widget)
        top.addSpacing(12)
        top.addWidget(self._camera_label)
        top.addSpacing(8)
        top.addWidget(self._frame_info)
        top.addStretch(1)
        top.addWidget(self._btn_camera)
        top.addWidget(self._btn_capture)

        tools = QHBoxLayout()
        for key in _MODE_ORDER:
            tools.addWidget(self._mode_buttons[key])
        tools.addWidget(self._btn_clear)
        tools.addSpacing(12)
        tools.addWidget(self._status, 1)
        tools.addWidget(self._btn_save)
        tools.addWidget(self._btn_cancel)

        hint = QLabel(
            "操作：打开相机 → 截取画面 → 选择区域类型 → 在画面上拖拽矩形 → 保存并应用。"
            "现场没有相机时，在“画面源”里选择已登记的示例视频，拖动“视频位置”"
            "挑一帧代表性画面即可直接绘制。"
            "涂油盘区用于判断「空手进盘、带料离盘」，请把盘体连同下方手臂"
            "活动范围一起框进去（模型只框得出盘体本身，运行时两者取并集"
            "跟随盘体移动）；工作追踪区是跟踪与判定的有效范围；"
            "涂油区本轮只做标定与显示，预留给后续涂油范围约束。"
            "拖到画面外会自动裁剪到边缘；坐标为归一化值，保存后检测服务会重新加载。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6b7280;font-size:11px;")

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.addLayout(top)
        root.addSpacing(6)
        root.addLayout(tools)
        root.addWidget(self._video_row)
        root.addWidget(self._canvas, 1)
        root.addWidget(hint)

        self._live_timer = QTimer(self)
        self._live_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._live_timer.setInterval(66)
        self._live_timer.timeout.connect(self._update_live)

        if self._stations:
            self._on_station_changed(0)
        else:
            self._status.setText("未配置工位：请先在 oilcheck.json 中添加工位")
            for button in (
                self._btn_camera,
                self._btn_capture,
                *self._mode_buttons.values(),
                self._btn_clear,
                self._btn_save,
            ):
                button.setEnabled(False)
            self._canvas.clear("未配置工位")

    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # 构造阶段可能已经发起启动请求，此时窗口尚不可见，预览轮询会跳过；
        # 显示后重新检查一次相机状态。示例视频模式只需重新取一次当前帧。
        if self._video_name is not None:
            self._reload_video_frame()
            return
        if self._camera_id:
            self._ensure_camera()

    def _current_station(self) -> StationConfig | None:
        index = self._station_combo.currentIndex()
        if 0 <= index < len(self._stations):
            return self._stations[index]
        return None

    def _on_station_changed(self, index: int) -> None:
        if not (0 <= index < len(self._stations)):
            return
        station = self._stations[index]
        self._camera_id = station.camera_id
        self._rois = {key: None for key in _MODE_ORDER}
        self._rois["tray"] = station.tray_roi
        self._rois["work"] = station.work_roi
        self._rois["oiling"] = station.oiling_roi
        self._camera_label.setText(f"相机：{station.camera_id}")
        self._frame_info.setText("画面：-")
        self._canvas.set_rois(self._rois)
        self._canvas.set_mode("work")
        self._mode = "tray"
        for key, button in self._mode_buttons.items():
            button.setChecked(key == "work")
        self._frozen = False
        self._last_frame_ts = 0.0
        self._camera_starting = False
        self._canvas.set_editable(False)
        self._btn_capture.setText("截取画面")
        if self._video_name is not None:
            self._reload_video_frame()
            return
        self._status.setText("正在打开相机...")
        self._live_timer.start()
        self._ensure_camera()

    def _ensure_camera(self) -> None:
        if self._video_name is not None:
            return
        cam_id = self._camera_id
        if not cam_id or self._camera_starting:
            return
        session = self._manager.get_session(cam_id)
        if session is None:
            message = f"相机不可用：{cam_id}（请在“设备”页启用该相机）"
            self._status.setText(message)
            self._btn_camera.setEnabled(True)
            self._btn_camera.setText("重新打开相机")
            InfoBar.warning(
                "相机不可用",
                message,
                position=InfoBarPosition.TOP_RIGHT,
                duration=3500,
                parent=self,
            )
            return
        state = session.state.value
        if state == "previewing":
            self._btn_camera.setEnabled(True)
            self._btn_camera.setText("相机已打开")
            self._live_timer.start()
            self._status.setText("实时画面：点击“截取画面”后开始绘制")
            return
        if state in {"connecting", "stopping"}:
            # 监控页可能正在启动/停止同一台相机，等待状态稳定后再判断。
            self._btn_camera.setEnabled(False)
            self._btn_camera.setText("正在打开...")
            self._status.setText("相机正在切换状态，请稍候...")
            self._wait_preview(0, cam_id)
            return

        self._btn_camera.setEnabled(False)
        self._btn_camera.setText("正在打开...")
        self._status.setText("正在打开相机...")
        self._camera_starting = True
        future = self._manager.start_async(cam_id)
        future.add_done_callback(
            lambda f: self._ui.invoke(lambda: self._finish_start(f, cam_id))
        )

    def _finish_start(self, future, cam_id: str) -> None:
        if cam_id != self._camera_id:
            return
        self._camera_starting = False
        try:
            future.result()
        except Exception as exc:  # noqa: BLE001
            self._btn_camera.setEnabled(True)
            self._btn_camera.setText("重新打开相机")
            self._status.setText(f"相机启动失败：{exc}")
            InfoBar.error(
                "相机启动失败",
                f"{type(exc).__name__}: {exc}",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3500,
                parent=self,
            )
            return
        self._wait_preview(0, cam_id)

    def _wait_preview(self, attempts: int, cam_id: str) -> None:
        if cam_id != self._camera_id:
            return
        if not self.isVisible():
            return
        session = self._manager.get_session(cam_id)
        state = "missing" if session is None else session.state.value
        if state == "previewing":
            self._btn_camera.setEnabled(True)
            self._btn_camera.setText("相机已打开")
            self._status.setText("实时画面：点击“截取画面”后开始绘制")
            self._live_timer.start()
            return
        if state == "error" or attempts >= 50:
            self._btn_camera.setEnabled(True)
            self._btn_camera.setText("重新打开相机")
            self._status.setText(f"相机未进入预览状态（{state}）")
            return
        QTimer.singleShot(100, lambda: self._wait_preview(attempts + 1, cam_id))

    def _update_live(self) -> None:
        if self._frozen or self._video_name is not None or not self._camera_id:
            return
        frame = self._manager.latest_frame(self._camera_id)
        if frame is None or frame.ts <= self._last_frame_ts:
            return
        self._last_frame_ts = frame.ts
        if self._canvas.set_frame(frame.bgr):
            self._frame_info.setText(
                f"画面：{frame.width}×{frame.height}（原始分辨率）"
            )
        self._canvas.set_rois(self._rois)
        self._canvas.set_editable(False)

    def _toggle_capture(self) -> None:
        if self._video_name is not None:
            return
        if self._frozen:
            self._frozen = False
            self._last_frame_ts = 0.0
            self._canvas.set_editable(False)
            self._btn_capture.setText("截取画面")
            self._status.setText("实时画面：点击“截取画面”后开始绘制")
            self._live_timer.start()
            return

        if not self._camera_id:
            return
        frame = self._manager.latest_frame(self._camera_id)
        if frame is None:
            InfoBar.warning(
                "没有画面",
                "相机尚未产生画面，请先打开相机",
                position=InfoBarPosition.TOP_RIGHT,
                duration=2500,
                parent=self,
            )
            return

        if self._canvas.set_frame(frame.bgr):
            self._frame_info.setText(
                f"画面：{frame.width}×{frame.height}（原始分辨率）"
            )
        self._canvas.set_rois(self._rois)
        self._canvas.set_editable(True)
        self._frozen = True
        self._live_timer.stop()
        self._btn_capture.setText("重新实时")
        self._status.setText(
            f"已截取画面 {frame.width}×{frame.height}：选择目标后，在画面上拖拽矩形"
        )

    def _set_mode(self, mode: str) -> None:
        self._mode = mode
        self._canvas.set_mode(mode)
        for key, button in self._mode_buttons.items():
            button.setChecked(key == mode)
        name = _roi_name(mode)
        if self._video_name is not None:
            self._status.setText(f"当前绘制：{name}（在示例视频帧上拖拽矩形）")
        elif self._frozen:
            self._status.setText(f"当前绘制：{name}（在画面上拖拽矩形）")
        else:
            self._status.setText(f"当前目标：{name}；请先点击“截取画面”")

    def _clear_current(self) -> None:
        if not self._canvas.clear_roi(self._mode):
            self._status.setText(f"{_roi_name(self._mode)}尚未设置")

    def _on_roi_changed(self, target: str, roi: Roi | None) -> None:
        self._rois[target] = roi
        self._canvas.set_rois(self._rois)
        name = _roi_name(target)
        if roi is None:
            self._status.setText(f"{name}已清除")
            return
        width, height = self._canvas.source_size
        pixel_text = ""
        if width > 0 and height > 0:
            box = roi.pixel_bbox(width, height)
            pixel_text = f"，{box.width:.0f}×{box.height:.0f}px"
        points = " ".join(f"({x:.3f}, {y:.3f})" for x, y in roi.points)
        self._status.setText(f"{name}已更新：{points}{pixel_text}")

    # ------------------------------------------------------------------
    # 画面源：实时相机 / 示例视频帧
    # ------------------------------------------------------------------
    def _on_source_changed(self, _index: int) -> None:
        name = self._source_combo.currentData()
        if not name:
            self._use_live_source()
        else:
            self._use_video_source(str(name))

    def _use_live_source(self) -> None:
        self._video_read_timer.stop()
        self._close_video()
        self._btn_camera.setEnabled(True)
        self._btn_capture.setEnabled(True)
        self._frozen = False
        self._last_frame_ts = 0.0
        self._canvas.set_editable(False)
        self._btn_capture.setText("截取画面")
        self._camera_label.setText(f"相机：{self._camera_id}")
        self._status.setText("实时画面：点击“截取画面”后开始绘制")
        self._live_timer.start()
        self._ensure_camera()

    def _use_video_source(self, name: str) -> None:
        """打开示例视频文件，用其中一帧代替相机画面标定 ROI。"""

        self._live_timer.stop()
        self._frozen = True
        self._stop_camera()

        video = self._cfg.demo.video(name)
        capture: cv2.VideoCapture | None = None
        if video is not None:
            path = Path(resolve_runtime_file(video.path, repo_root=runtime_root()))
            if path.exists():
                capture = cv2.VideoCapture(str(path))
                if not capture.isOpened():
                    capture.release()
                    capture = None
        if capture is None:
            self._source_combo.blockSignals(True)
            self._source_combo.setCurrentIndex(0)
            self._source_combo.blockSignals(False)
            self._use_live_source()
            InfoBar.warning(
                "示例视频不可用",
                f"{name}：未在 demo.videos 中登记或文件无法打开，详见日志",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3500,
                parent=self,
            )
            return

        self._close_video()
        self._video_name = name
        self._video = video
        self._capture = capture
        self._video_frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._video_fps = fps if 0.0 < fps < 240.0 else 0.0
        self._camera_label.setText(f"示例视频：{name}")
        self._btn_camera.setEnabled(False)
        self._btn_capture.setEnabled(False)
        self._video_row.setVisible(True)
        self._video_slider.setEnabled(True)
        self._show_video_frame(0)

    def _close_video(self) -> None:
        self._video_read_timer.stop()
        capture = self._capture
        self._capture = None
        if capture is not None:
            capture.release()
        self._video_name = None
        self._video = None
        self._video_frames = 0
        self._video_fps = 0.0
        self._video_slider.blockSignals(True)
        self._video_slider.setValue(0)
        self._video_slider.blockSignals(False)
        self._video_slider.setEnabled(False)
        self._video_row.setVisible(False)

    def _on_video_seek(self, _value: int) -> None:
        self._video_read_timer.start()

    def _apply_video_seek(self) -> None:
        self._show_video_frame(self._slider_frame())

    def _reload_video_frame(self) -> None:
        if self._capture is not None:
            self._show_video_frame(self._slider_frame())

    def _slider_frame(self) -> int:
        if self._video_frames <= 1:
            return 0
        return round(self._video_slider.value() / 1000.0 * (self._video_frames - 1))

    def _sync_slider(self, index: int) -> None:
        value = 0
        if self._video_frames > 1:
            value = round(index / (self._video_frames - 1) * 1000.0)
        self._video_slider.blockSignals(True)
        self._video_slider.setValue(value)
        self._video_slider.blockSignals(False)

    def _show_video_frame(self, index: int) -> None:
        capture = self._capture
        if capture is None:
            return
        index = max(0, min(int(index), max(0, self._video_frames - 1)))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok or bgr is None:
            self._status.setText(f"示例视频读取失败：第 {index + 1} 帧")
            return
        self._sync_slider(index)
        height, width = bgr.shape[:2]
        if self._canvas.set_frame(bgr):
            self._frame_info.setText(f"画面：{width}×{height}（示例视频帧）")
        total = self._video_frames or "?"
        position = f"第 {index + 1}/{total} 帧"
        if self._video_fps > 0:
            position += f" · {_format_seconds(index / self._video_fps)}"
        self._video_pos.setText(position)
        self._canvas.set_rois(self._rois)
        self._canvas.set_editable(True)
        self._frozen = True
        name = self._video.name if self._video is not None else self._video_name
        self._status.setText(f"{name}：拖动“视频位置”选择画面帧，选择目标后拖拽矩形")

    def _save(self) -> None:
        station = self._current_station()
        if station is None:
            return
        tray = self._rois.get("tray")
        work = self._rois.get("work")
        if tray is None or work is None:
            InfoBar.warning(
                "ROI 不完整",
                "涂油盘区和工作追踪区都需要设置",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return
        # 涂油区缺省沿用工位现值（旧配置迁移后就是工作追踪区），不强制绘制。
        oiling = self._rois.get("oiling") or station.oiling_roi

        updated = replace(
            station,
            tray_roi=tray,
            work_roi=work,
            oiling_roi=oiling,
        )
        stations = tuple(
            updated if s.station_id == station.station_id else s
            for s in self._cfg.stations
        )
        cfg = replace(self._cfg, stations=stations)
        try:
            self._cfg_service.save(cfg)
        except Exception as exc:  # noqa: BLE001
            InfoBar.error(
                "保存失败",
                str(exc),
                position=InfoBarPosition.TOP_RIGHT,
                duration=3500,
                parent=self,
            )
            return

        self._cfg = cfg
        self._on_saved(cfg)
        InfoBar.success(
            "ROI 已保存",
            f"{station.name}：涂油盘区、工作追踪区、涂油区",
            position=InfoBarPosition.TOP_RIGHT,
            duration=1800,
            parent=self,
        )
        self.accept()

    def done(self, result: int) -> None:
        self._live_timer.stop()
        self._video_read_timer.stop()
        self._close_video()
        self._stop_camera()
        super().done(result)

    def _stop_camera(self) -> None:
        """标定结束（保存/取消/关闭窗口）后停止本工位相机预览。"""
        cam_id = self._camera_id
        if not cam_id:
            return
        session = self._manager.get_session(cam_id)
        if session is None:
            return
        if session.state.value not in {
            "connecting",
            "connected",
            "previewing",
            "error",
        }:
            return
        try:
            future = self._manager.stop_async(cam_id)
        except Exception:  # noqa: BLE001
            logger.exception("提交停止相机预览请求失败")
            return
        future.add_done_callback(self._on_stop_camera_done)

    @staticmethod
    def _on_stop_camera_done(future) -> None:
        try:
            future.result()
        except Exception:  # noqa: BLE001
            logger.exception("关闭 ROI 标定后停止相机预览失败")
