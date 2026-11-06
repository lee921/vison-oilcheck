"""训练视频录制页：左侧实时/回放画面，右侧已录制分段。

- 左侧默认显示当前相机实时画面；开始录制时确保切回实时；
- 双击右侧分段，在左侧播放该 MP4；
- 播放期间录制可继续，点击“实时画面”回到实时预览。
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QEvent, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
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
from app.oilcheck.service import OilCheckService
from app.oilcheck.types import format_timestamp
from ui.utils import UiInvoker
from ui.widgets.oilcheck_video import OilCheckVideoView


class RecordingPage(QWidget):
    def __init__(
        self,
        *,
        service: OilCheckService,
        manager: CameraManager,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("recordingPage")
        self._service = service
        self._manager = manager
        self._ui = UiInvoker(self)
        self._camera_ids: list[str] = []
        self._segment_paths: list[str] = []
        self._checked_paths: set[str] = set()

        self._mode = "live"  # live | file
        self._cap: cv2.VideoCapture | None = None
        self._play_path: Path | None = None
        self._play_interval_ms = 66
        self._paused = False
        self._last_live_ts = 0.0

        self._camera_combo = ComboBox()
        self._camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        self._reload_cameras()

        self._status = QLabel("录制未开始")
        self._status.setStyleSheet("color:#e5e7eb;font-weight:600;")
        self._detail = QLabel("-")
        self._detail.setStyleSheet("color:#9ca3af;font-size:12px;")
        self._path = QLabel("-")
        self._path.setStyleSheet("color:#6b7280;font-size:11px;")
        self._path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self._btn_preview = PushButton("预览")
        self._btn_preview.clicked.connect(self._on_preview_clicked)
        self._btn_toggle = PrimaryPushButton("开始录制")
        self._btn_open_dir = PushButton("打开录制目录")
        self._btn_refresh = PushButton("刷新分段")

        self._preview = OilCheckVideoView()
        self._preview.installEventFilter(self)
        self._preview_status = QLabel("实时画面")
        self._preview_status.setStyleSheet("color:#e5e7eb;font-weight:600;")
        self._btn_live = PushButton("实时画面")
        self._btn_live.setEnabled(False)
        self._btn_live.clicked.connect(self._show_live)

        top = QHBoxLayout()
        top.addWidget(BodyLabel("相机"))
        top.addWidget(self._camera_combo)
        top.addSpacing(12)
        top.addWidget(self._status)
        top.addStretch(1)
        top.addWidget(self._btn_preview)
        top.addWidget(self._btn_toggle)
        top.addWidget(self._btn_open_dir)
        top.addWidget(self._btn_refresh)

        info = QVBoxLayout()
        info.setSpacing(2)
        info.addWidget(self._detail)
        info.addWidget(self._path)

        preview_header = QHBoxLayout()
        preview_header.addWidget(self._preview_status)
        preview_header.addStretch(1)
        preview_header.addWidget(self._btn_live)

        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(6)
        left.addLayout(preview_header)
        left.addWidget(self._preview, 1)

        segments_title = QLabel("已录制视频（双击播放，勾选后删除）")
        segments_title.setStyleSheet("color:#e5e7eb;font-weight:600;")
        self._btn_delete = PushButton("删除勾选")
        self._btn_delete.clicked.connect(self._delete_selected_segments)
        self._select_all = CheckBox("全选")
        self._select_all.stateChanged.connect(self._on_select_all_changed)

        segments_header = QHBoxLayout()
        segments_header.addWidget(segments_title)
        segments_header.addStretch(1)
        segments_header.addWidget(self._select_all)
        segments_header.addWidget(self._btn_delete)

        self._table = TableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(
            ["选择", "开始时间", "时长", "帧数", "文件"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setColumnWidth(0, 46)
        self._table.setColumnWidth(1, 145)
        self._table.setColumnWidth(2, 84)
        self._table.setColumnWidth(3, 60)
        self._table.itemChanged.connect(self._on_table_item_changed)
        self._table.cellDoubleClicked.connect(self._play_selected_segment)

        hint = QLabel(
            "录制的是相机原始画面（不带检测框），用于外部训练项目标注；"
            "每段视频旁会生成同名 JSON 元数据。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#6b7280;font-size:11px;")

        right = QVBoxLayout()
        right.setContentsMargins(12, 0, 0, 0)
        right.setSpacing(6)
        right.addLayout(segments_header)
        right.addWidget(self._table, 1)
        right.addWidget(hint)

        left_widget = QWidget()
        left_widget.setLayout(left)
        right_widget = QWidget()
        right_widget.setLayout(right)
        right_widget.setMinimumWidth(440)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        body.addWidget(left_widget, 3)
        body.addWidget(right_widget, 2)

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 12, 16, 12)
        root.addLayout(top)
        root.addLayout(info)
        root.addSpacing(6)
        root.addLayout(body, 1)

        self._btn_toggle.clicked.connect(self._toggle_recording)
        self._btn_open_dir.clicked.connect(self._open_dir)
        self._btn_refresh.clicked.connect(self._refresh_segments)

        fps = max(1, min(30, int(self._service.config.pipeline.max_fps)))
        self._live_timer = QTimer(self)
        self._live_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._live_timer.setInterval(max(33, int(1000 / fps)))
        self._live_timer.timeout.connect(self._update_live_frame)
        self._live_timer.start()

        self._play_timer = QTimer(self)
        self._play_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._play_timer.timeout.connect(self._next_play_frame)

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(1000)
        self._status_timer.timeout.connect(self._refresh_status)
        self._status_timer.start()

        self._segment_timer = QTimer(self)
        self._segment_timer.setInterval(5000)
        self._segment_timer.timeout.connect(self._refresh_segments)
        self._segment_timer.start()

        self._show_live()
        self._refresh_status()
        self._refresh_segments()

    # ------------------------------------------------------------------
    # 相机与录制
    # ------------------------------------------------------------------
    def _reload_cameras(self) -> None:
        self._camera_ids = [
            camera.id for camera in self._manager.cameras() if camera.enabled
        ]
        self._camera_combo.blockSignals(True)
        self._camera_combo.clear()
        for camera in self._manager.cameras():
            if not camera.enabled:
                continue
            self._camera_combo.addItem(camera.name or camera.id)
        self._camera_combo.blockSignals(False)
        if self._camera_ids:
            self._camera_combo.setCurrentIndex(0)

    def _current_camera_id(self) -> str | None:
        index = self._camera_combo.currentIndex()
        if 0 <= index < len(self._camera_ids):
            return self._camera_ids[index]
        return None

    def _camera_name(self, camera_id: str) -> str:
        for camera in self._manager.cameras():
            if camera.id == camera_id:
                return camera.name or camera.id
        return camera_id

    def _on_camera_changed(self, _index: int) -> None:
        if not hasattr(self, "_status"):
            return
        self._show_live()
        self._refresh_status()

    def _toggle_recording(self) -> None:
        camera_id = self._current_camera_id()
        if camera_id is None:
            InfoBar.warning(
                "没有相机",
                "请先在设备页添加并启用相机",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return
        session = self._manager.get_session(camera_id)
        state = "missing" if session is None else session.state.value
        self._btn_preview.setText("停止预览" if state == "previewing" else "预览")

        status = self._service.recording_status(camera_id)
        if status.active:
            self._service.stop_training_recording(camera_id)
            InfoBar.info(
                "录制",
                "训练视频录制已停止",
                position=InfoBarPosition.TOP_RIGHT,
                duration=1800,
                parent=self,
            )
            self._refresh_status()
            self._refresh_segments()
            return

        # 开始录制时切回实时画面，并确保相机已经启动预览；否则录制器没有帧可写。
        self._show_live()
        self._ensure_preview(camera_id, lambda: self._start_recording(camera_id))

    def _on_preview_clicked(self) -> None:
        camera_id = self._current_camera_id()
        if camera_id is None:
            InfoBar.warning(
                "没有相机",
                "请先在设备页添加并启用相机",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return
        session = self._manager.get_session(camera_id)
        if session is None:
            InfoBar.error(
                "相机不存在",
                f"请先在设备页添加相机 {camera_id}",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return

        if session.state.value == "previewing":
            if self._service.recording_status(camera_id).active:
                InfoBar.warning(
                    "正在录制",
                    "请先停止录制，再停止预览",
                    position=InfoBarPosition.TOP_RIGHT,
                    duration=2500,
                    parent=self,
                )
                return
            self._btn_preview.setEnabled(False)
            self._btn_toggle.setEnabled(False)
            future = self._manager.stop_async(camera_id)
            future.add_done_callback(
                lambda f: self._ui.invoke(lambda: self._finish_stop_preview(f))
            )
            return

        self._show_live()
        self._ensure_preview(camera_id, self._show_live)

    def _ensure_preview(self, camera_id: str, on_ready) -> None:
        session = self._manager.get_session(camera_id)
        if session is None:
            InfoBar.error(
                "相机不存在",
                f"请先在设备页添加相机 {camera_id}",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return
        if session.state.value == "previewing":
            on_ready()
            return

        self._btn_preview.setEnabled(False)
        self._btn_toggle.setEnabled(False)
        InfoBar.info(
            "正在启动相机",
            "相机进入预览状态后继续",
            position=InfoBarPosition.TOP_RIGHT,
            duration=1800,
            parent=self,
        )
        future = self._manager.start_async(camera_id)
        future.add_done_callback(
            lambda f: self._ui.invoke(
                lambda: self._finish_start_camera(f, camera_id, on_ready)
            )
        )

    def _finish_start_camera(self, future, camera_id: str, on_ready) -> None:
        try:
            future.result()
        except Exception as exc:  # noqa: BLE001
            self._btn_preview.setEnabled(True)
            self._btn_toggle.setEnabled(True)
            InfoBar.error(
                "相机启动失败",
                f"{type(exc).__name__}: {exc}（详见日志页）",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3500,
                parent=self,
            )
            return

        self._wait_for_preview(camera_id, 0, on_ready)

    def _wait_for_preview(self, camera_id: str, attempts: int, on_ready) -> None:
        if not self.isVisible() or self._current_camera_id() != camera_id:
            self._btn_preview.setEnabled(True)
            self._btn_toggle.setEnabled(True)
            return

        session = self._manager.get_session(camera_id)
        state = "missing" if session is None else session.state.value
        if state == "previewing":
            self._btn_preview.setEnabled(True)
            self._btn_toggle.setEnabled(True)
            self._refresh_status()
            on_ready()
            return
        if state == "error" or attempts >= 50:
            self._btn_preview.setEnabled(True)
            self._btn_toggle.setEnabled(True)
            InfoBar.error(
                "相机启动失败",
                f"相机未进入预览状态（{state}），请检查连接与日志",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3500,
                parent=self,
            )
            return
        QTimer.singleShot(
            100,
            lambda: self._wait_for_preview(camera_id, attempts + 1, on_ready),
        )

    def _finish_stop_preview(self, future) -> None:
        self._btn_preview.setEnabled(True)
        self._btn_toggle.setEnabled(True)
        try:
            future.result()
        except Exception as exc:  # noqa: BLE001
            InfoBar.error(
                "停止预览失败",
                f"{type(exc).__name__}: {exc}（详见日志页）",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return
        self._show_live()
        self._refresh_status()

    def _start_recording(self, camera_id: str) -> None:
        self._service.start_training_recording(camera_id)
        InfoBar.info(
            "录制",
            "已开始录制，等待相机画面",
            position=InfoBarPosition.TOP_RIGHT,
            duration=2000,
            parent=self,
        )
        self._refresh_status()
        self._refresh_segments()

    def _refresh_status(self) -> None:
        camera_id = self._current_camera_id()
        if camera_id is None:
            self._status.setText("没有可用相机")
            self._detail.setText("-")
            self._path.setText("-")
            self._btn_toggle.setText("开始录制")
            return

        session = self._manager.get_session(camera_id)
        state = "missing" if session is None else session.state.value
        self._btn_preview.setText("停止预览" if state == "previewing" else "预览")

        status = self._service.recording_status(camera_id)
        if status.active:
            elapsed = 0.0
            if status.started_at:
                elapsed = max(0.0, datetime.now().timestamp() - status.started_at)
            self._status.setText(
                f"录制中 · {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
            )
            self._status.setStyleSheet("color:#f87171;font-weight:600;")
            self._detail.setText(
                f"已写 {status.frames_written} 帧 · 丢帧 {status.frames_dropped} · "
                f"当前分段 {Path(status.segment_path).name if status.segment_path else '-'}"
            )
            self._btn_toggle.setText("停止录制")
        else:
            self._status.setText("录制未开始")
            self._status.setStyleSheet("color:#9ca3af;font-weight:600;")
            if status.last_error:
                self._detail.setText(f"上次错误：{status.last_error}")
            else:
                self._detail.setText(
                    f"已写 {status.frames_written} 帧 · 丢帧 {status.frames_dropped}"
                )
            self._btn_toggle.setText("开始录制")
        self._path.setText(f"存储目录：{self._service.recording_root}")

    # ------------------------------------------------------------------
    # 实时预览
    # ------------------------------------------------------------------
    def _update_live_frame(self) -> None:
        if self._mode != "live":
            return
        camera_id = self._current_camera_id()
        if camera_id is None:
            self._preview.clear("没有可用相机")
            self._last_live_ts = 0.0
            return
        session = self._manager.get_session(camera_id)
        state = "missing" if session is None else session.state.value
        if state != "previewing":
            # 相机停止后不要再显示最后一帧，否则看起来像仍在实时预览。
            if self._preview.has_frame or self._last_live_ts:
                self._preview.clear(
                    f"相机未预览（{state}），点击“预览”或“开始录制”启动"
                )
            self._last_live_ts = 0.0
            return
        frame = self._service.latest_frame(camera_id)
        if frame is None:
            if self._last_live_ts == 0.0:
                self._preview.clear("等待相机画面")
            return
        if frame.ts <= self._last_live_ts:
            return
        self._last_live_ts = frame.ts
        self._preview.set_frame(frame.bgr, frame.ts)

        status = self._service.recording_status(camera_id)
        banner = f"{self._camera_name(camera_id)} · 实时画面"
        if status.active:
            banner += " · 录制中"
        self._preview.set_overlays(rois=(), detections=(), banner=banner)

    def _show_live(self) -> None:
        self._stop_playback()
        self._mode = "live"
        self._last_live_ts = 0.0
        self._preview_status.setText("实时画面")
        self._btn_live.setEnabled(False)
        self._update_live_frame()

    # ------------------------------------------------------------------
    # 录像回放
    # ------------------------------------------------------------------
    def _start_playback(self, path: Path) -> None:
        self._stop_playback()
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            InfoBar.error(
                "无法播放",
                f"视频文件无法打开：{path.name}",
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return

        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 15.0
        interval = max(10, int(1000 / fps))

        self._cap = capture
        self._play_path = path
        self._mode = "file"
        self._preview_status.setText(f"录像回放：{path.name}")
        self._btn_live.setEnabled(True)
        self._preview.set_overlays(rois=(), detections=(), banner=f"录像：{path.name}")
        self._play_timer.start(interval)
        self._next_play_frame()

    def _next_play_frame(self) -> None:
        capture = self._cap
        if capture is None:
            return
        ok, frame = capture.read()
        if not ok:
            self._play_timer.stop()
            capture.release()
            self._cap = None
            name = self._play_path.name if self._play_path else "-"
            self._preview_status.setText("播放结束（双击重播）")
            self._preview.set_overlays(
                rois=(),
                detections=(),
                banner=f"播放结束：{name} · 双击重播",
            )
            return

        self._preview.set_frame(frame, time.time())
        name = self._play_path.name if self._play_path else "-"
        self._preview.set_overlays(
            rois=(),
            detections=(),
            banner=f"录像：{name} · 双击暂停",
        )

    def _toggle_play_pause(self) -> None:
        if self._mode != "file":
            return
        if self._cap is None:
            # 已播放结束：双击从头重播。
            if self._play_path is not None:
                self._start_playback(self._play_path)
            return

        name = self._play_path.name if self._play_path else "-"
        if self._paused:
            self._paused = False
            self._play_timer.start(self._play_interval_ms)
            self._preview_status.setText("录像回放（双击暂停）")
            self._preview.set_overlays(
                rois=(),
                detections=(),
                banner=f"录像：{name} · 双击暂停",
            )
        else:
            self._paused = True
            self._play_timer.stop()
            self._preview_status.setText("已暂停（双击继续）")
            self._preview.set_overlays(
                rois=(),
                detections=(),
                banner="已暂停 · 双击继续",
            )

    def _stop_playback(self) -> None:
        self._play_timer.stop()
        self._paused = False
        capture = self._cap
        self._cap = None
        self._play_path = None
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 分段列表
    # ------------------------------------------------------------------
    def _refresh_segments(self) -> None:
        segments = self._service.recording_segments(300)
        self._segment_paths = [str(item.get("path", "")) for item in segments]
        self._table.blockSignals(True)
        self._table.setRowCount(len(segments))
        for row, item in enumerate(segments):
            started = float(item.get("started_at_epoch", 0.0) or 0.0)
            ended = float(item.get("ended_at_epoch", 0.0) or 0.0)
            fps = max(1, int(item.get("fps", 15) or 15))
            frames = int(item.get("frame_count", 0) or 0)
            if item.get("status") == "recording" or not ended:
                duration = "录制中"
            else:
                duration_s = max(0.0, ended - started)
                duration = f"{int(duration_s // 60):02d}:{int(duration_s % 60):02d}"
                if not frames:
                    frames = int(duration_s * fps)
            path = Path(str(item.get("path", "")))
            path_key = str(path)

            tooltip = (
                f"{format_timestamp(started)}\n"
                f"相机：{item.get('camera_name') or item.get('camera_id', '')}\n"
                f"工位：{item.get('station_name') or item.get('station_id', '')}\n"
                f"丢帧：{item.get('dropped_frames', 0)}\n"
                f"{path}"
            )
            check = QTableWidgetItem("")
            check.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            check.setCheckState(
                Qt.CheckState.Checked
                if path_key in self._checked_paths
                else Qt.CheckState.Unchecked
            )
            check.setToolTip(tooltip)
            self._table.setItem(row, 0, check)

            values = (
                datetime.fromtimestamp(started).strftime("%m-%d %H:%M:%S"),
                duration,
                str(frames),
                path.name,
            )
            for offset, text in enumerate(values, start=1):
                cell = QTableWidgetItem(text)
                cell.setToolTip(tooltip)
                self._table.setItem(row, offset, cell)
        self._table.blockSignals(False)

        self._checked_paths &= {path for path in self._segment_paths if path}
        self._update_select_all_state()

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 0:
            return
        row = item.row()
        if not (0 <= row < len(self._segment_paths)):
            return
        path = self._segment_paths[row]
        if item.checkState() == Qt.CheckState.Checked:
            self._checked_paths.add(path)
        else:
            self._checked_paths.discard(path)
        self._update_select_all_state()

    def _on_select_all_changed(self, _state: int) -> None:
        checked = self._select_all.isChecked()
        self._table.blockSignals(True)
        for row in range(self._table.rowCount()):
            item = self._table.item(row, 0)
            if item is None:
                continue
            item.setCheckState(
                Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            )
        self._table.blockSignals(False)
        if checked:
            self._checked_paths = set(self._segment_paths)
        else:
            self._checked_paths.clear()
        self._update_select_all_state()

    def _update_select_all_state(self) -> None:
        self._select_all.blockSignals(True)
        all_checked = bool(self._segment_paths) and all(
            path in self._checked_paths for path in self._segment_paths
        )
        self._select_all.setChecked(all_checked)
        self._select_all.blockSignals(False)

    def _open_dir(self) -> None:
        root = self._service.recording_root
        root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(root)))

    def _delete_selected_segments(self) -> None:
        paths = [path for path in self._segment_paths if path in self._checked_paths]
        if not paths:
            InfoBar.warning(
                "未勾选视频",
                "请先勾选要删除的视频，可点击“全选”",
                position=InfoBarPosition.TOP_RIGHT,
                duration=2500,
                parent=self,
            )
            return

        dialog = MessageBox(
            "删除已录制视频",
            f"将删除勾选的 {len(paths)} 个视频及其元数据，并同步更新录制清单。"
            "此操作不可撤销，是否继续？",
            self,
        )
        if not dialog.exec():
            return

        deleting = {str(Path(path)) for path in paths}
        if self._play_path is not None and str(self._play_path) in deleting:
            # Windows 下必须先释放文件句柄，否则删除会失败。
            self._show_live()

        deleted, errors = self._service.delete_recording_segments(paths)
        self._refresh_segments()

        if deleted and not errors:
            InfoBar.success(
                "已删除",
                f"已删除 {deleted} 个录制视频",
                position=InfoBarPosition.TOP_RIGHT,
                duration=1800,
                parent=self,
            )
        elif deleted and errors:
            InfoBar.warning(
                "部分删除完成",
                f"已删除 {deleted} 个；{errors[0]}",
                position=InfoBarPosition.TOP_RIGHT,
                duration=4000,
                parent=self,
            )
        else:
            InfoBar.error(
                "删除失败",
                errors[0] if errors else "没有文件被删除",
                position=InfoBarPosition.TOP_RIGHT,
                duration=4000,
                parent=self,
            )

    def _play_selected_segment(self, row: int, column: int) -> None:
        if column == 0:
            return
        if not (0 <= row < len(self._segment_paths)):
            return
        path = Path(self._segment_paths[row])
        if not path.exists():
            InfoBar.error(
                "视频不存在",
                str(path),
                position=InfoBarPosition.TOP_RIGHT,
                duration=3000,
                parent=self,
            )
            return
        self._start_playback(path)

    def reload_cameras(self) -> None:
        self._reload_cameras()
        self._show_live()
        self._refresh_status()

    def eventFilter(self, obj, event) -> bool:  # noqa: ANN001
        if obj is self._preview and event.type() == QEvent.Type.MouseButtonDblClick:
            if event.button() == Qt.MouseButton.LeftButton:
                self._toggle_play_pause()
                return True
        return super().eventFilter(obj, event)

    def showEvent(self, event) -> None:  # noqa: N802
        self._last_live_ts = 0.0
        self._live_timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._live_timer.stop()
        self._stop_playback()
        self._mode = "live"
        self._last_live_ts = 0.0
        self._preview_status.setText("实时画面")
        self._btn_live.setEnabled(False)
        super().hideEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._stop_playback()
        self._live_timer.stop()
        super().closeEvent(event)
