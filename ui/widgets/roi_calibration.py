"""ROI 标定对话框：打开相机、截取画面、拖拽绘制取料区与涂油区。"""

from __future__ import annotations

import logging
from dataclasses import replace

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
)

from app.camera_manager import CameraManager
from app.oilcheck.config import (
    OilCheckConfig,
    OilCheckConfigService,
    StationConfig,
)
from app.oilcheck.roi import Roi
from ui.utils import UiInvoker
from ui.widgets.roi_canvas import RoiCanvas

logger = logging.getLogger("oilcheck.ui.roi_calibration")


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
        self._mode = "pick"
        self._rois: dict[str, Roi | None] = {"pick": None, "oiling": None}
        self._frozen = False
        self._last_frame_ts = 0.0
        self._camera_starting = False

        self._station_combo = ComboBox()
        for station in self._stations:
            suffix = "" if station.enabled else " · 未启用"
            self._station_combo.addItem(
                f"{station.name}（{station.station_id}）{suffix}"
            )
        self._station_combo.currentIndexChanged.connect(self._on_station_changed)
        self._camera_label = QLabel("-")
        self._camera_label.setStyleSheet("color:#9ca3af;")
        self._frame_info = QLabel("画面：-")
        self._frame_info.setStyleSheet("color:#6b7280;font-size:11px;")

        self._btn_camera = PushButton("打开相机")
        self._btn_camera.clicked.connect(self._ensure_camera)
        self._btn_capture = PushButton("截取画面")
        self._btn_capture.clicked.connect(self._toggle_capture)

        self._btn_pick = PushButton("设置盘体区")
        self._btn_pick.setCheckable(True)
        self._btn_pick.setChecked(True)
        self._btn_pick.clicked.connect(lambda: self._set_mode("pick"))
        self._btn_oil = PushButton("设置涂油区")
        self._btn_oil.setCheckable(True)
        self._btn_oil.clicked.connect(lambda: self._set_mode("oiling"))
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
        top.addWidget(self._camera_label)
        top.addSpacing(8)
        top.addWidget(self._frame_info)
        top.addStretch(1)
        top.addWidget(self._btn_camera)
        top.addWidget(self._btn_capture)

        tools = QHBoxLayout()
        tools.addWidget(self._btn_pick)
        tools.addWidget(self._btn_oil)
        tools.addWidget(self._btn_clear)
        tools.addSpacing(12)
        tools.addWidget(self._status, 1)
        tools.addWidget(self._btn_save)
        tools.addWidget(self._btn_cancel)

        hint = QLabel(
            "操作：打开相机 → 截取画面 → 选择“盘体区/涂油区”→ 在画面上拖拽矩形 → 保存并应用。"
            "盘体区用于跟踪上下移动的机器涂油盘；4 个槽位按盘体局部几何自动计算。"
            "拖到画面外会自动裁剪到边缘；坐标为归一化值，保存后检测服务会重新加载。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6b7280;font-size:11px;")

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.addLayout(top)
        root.addSpacing(6)
        root.addLayout(tools)
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
                self._btn_pick,
                self._btn_oil,
                self._btn_clear,
                self._btn_save,
            ):
                button.setEnabled(False)
            self._canvas.clear("未配置工位")

    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        # 构造阶段可能已经发起启动请求，此时窗口尚不可见，预览轮询会跳过；
        # 显示后重新检查一次相机状态。
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
        self._rois = {"pick": station.pick_roi, "oiling": station.oiling_roi}
        self._camera_label.setText(f"相机：{station.camera_id}")
        self._frame_info.setText("画面：-")
        self._canvas.set_rois(self._rois["pick"], self._rois["oiling"])
        self._canvas.set_mode("pick")
        self._mode = "pick"
        self._btn_pick.setChecked(True)
        self._btn_oil.setChecked(False)
        self._frozen = False
        self._last_frame_ts = 0.0
        self._camera_starting = False
        self._canvas.set_editable(False)
        self._btn_capture.setText("截取画面")
        self._status.setText("正在打开相机...")
        self._live_timer.start()
        self._ensure_camera()

    def _ensure_camera(self) -> None:
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
        if self._frozen or not self._camera_id:
            return
        frame = self._manager.latest_frame(self._camera_id)
        if frame is None or frame.ts <= self._last_frame_ts:
            return
        self._last_frame_ts = frame.ts
        if self._canvas.set_frame(frame.bgr):
            self._frame_info.setText(
                f"画面：{frame.width}×{frame.height}（原始分辨率）"
            )
        self._canvas.set_rois(self._rois["pick"], self._rois["oiling"])
        self._canvas.set_editable(False)

    def _toggle_capture(self) -> None:
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
        self._canvas.set_rois(self._rois["pick"], self._rois["oiling"])
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
        self._btn_pick.setChecked(mode == "pick")
        self._btn_oil.setChecked(mode == "oiling")
        name = "盘体/取料区" if mode == "pick" else "二次涂油区"
        if self._frozen:
            self._status.setText(f"当前绘制：{name}（在画面上拖拽矩形）")
        else:
            self._status.setText(f"当前目标：{name}；请先点击“截取画面”")

    def _clear_current(self) -> None:
        if not self._canvas.clear_roi(self._mode):
            name = "盘体/取料区" if self._mode == "pick" else "二次涂油区"
            self._status.setText(f"{name}尚未设置")

    def _on_roi_changed(self, target: str, roi: Roi | None) -> None:
        self._rois[target] = roi
        self._canvas.set_rois(self._rois["pick"], self._rois["oiling"])
        name = "盘体/取料区" if target == "pick" else "二次涂油区"
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

    def _save(self) -> None:
        station = self._current_station()
        if station is None:
            return
        pick = self._rois.get("pick")
        oiling = self._rois.get("oiling")
        if pick is None or oiling is None:
            InfoBar.warning(
                "ROI 不完整",
                "盘体区和二次涂油区都需要设置",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return

        tray_roi = station.tray_roi
        if tray_roi is None or tray_roi == station.pick_roi:
            tray_roi = pick
        updated = replace(
            station,
            pick_roi=pick,
            oiling_roi=oiling,
            tray_roi=tray_roi,
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
            f"{station.name}：盘体区与二次涂油区已更新",
            position=InfoBarPosition.TOP_RIGHT,
            duration=1800,
            parent=self,
        )
        self.accept()

    def done(self, result: int) -> None:
        self._live_timer.stop()
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
