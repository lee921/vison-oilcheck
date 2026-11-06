from __future__ import annotations

import json
import logging
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    InfoBar,
    InfoBarPosition,
    PrimaryPushButton,
    PushButton,
    TableWidget,
)

from app.camera_manager import CameraManager
from app.config_service import ConfigService
from app.models import CameraConfig

from ..utils import UiInvoker
from ..widgets.camera_edit_dialog import CameraEditDialog


class DevicePage(QWidget):
    def __init__(
        self,
        *,
        manager: CameraManager,
        cfg_service: ConfigService,
        on_saved: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("devicePage")

        self._manager = manager
        self._cfg_service = cfg_service
        self._on_saved = on_saved
        self._logger = logging.getLogger("oilcheck.ui.device_page")
        self._ui = UiInvoker(self, logger=self._logger)

        self._btn_add = PrimaryPushButton("新增")
        self._btn_edit = PushButton("编辑")
        self._btn_delete = PushButton("删除")
        self._btn_test = PushButton("测试连接")
        self._btn_reload = PushButton("刷新")
        self._btn_import = PushButton("导入")
        self._btn_export = PushButton("导出")
        self._btn_save = PrimaryPushButton("保存")

        top = QHBoxLayout()
        top.addWidget(self._btn_add)
        top.addWidget(self._btn_edit)
        top.addWidget(self._btn_delete)
        top.addWidget(self._btn_test)
        top.addSpacing(12)
        top.addWidget(self._btn_reload)
        top.addWidget(self._btn_import)
        top.addWidget(self._btn_export)
        top.addStretch(1)
        top.addWidget(self._btn_save)

        self._table = TableWidget(self)
        self._table.setColumnCount(8)
        self._table.setHorizontalHeaderLabels(
            ["启用", "ID", "名称", "IP", "端口", "通道", "码流", "模式"]
        )
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)

        root = QVBoxLayout(self)
        root.addLayout(top)
        root.addWidget(self._table, 1)

        self._btn_add.clicked.connect(self._on_add)
        self._btn_edit.clicked.connect(self._on_edit)
        self._btn_delete.clicked.connect(self._on_delete)
        self._btn_test.clicked.connect(self._on_test)
        self._btn_reload.clicked.connect(self.reload)
        self._btn_import.clicked.connect(self._on_import)
        self._btn_export.clicked.connect(self._on_export)
        self._btn_save.clicked.connect(self.save)

        self._cameras: list[CameraConfig] = []
        self.reload()

    def reload(self) -> None:
        self._cameras = self._cfg_service.load_cameras()
        self._render()

    def _render(self) -> None:
        self._table.setRowCount(len(self._cameras))
        for row, cam in enumerate(self._cameras):
            enabled = QTableWidgetItem("")
            enabled.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            enabled.setCheckState(
                Qt.CheckState.Checked if cam.enabled else Qt.CheckState.Unchecked
            )
            self._table.setItem(row, 0, enabled)

            self._table.setItem(row, 1, QTableWidgetItem(cam.id))
            self._table.setItem(row, 2, QTableWidgetItem(cam.name))
            self._table.setItem(row, 3, QTableWidgetItem(cam.ip))
            self._table.setItem(row, 4, QTableWidgetItem(str(cam.port)))
            self._table.setItem(row, 5, QTableWidgetItem(str(cam.channel)))
            stream_text = "主码流" if cam.stream_type == "main" else "子码流"
            mode_text = "直渲染" if cam.preferred_mode == "direct" else "解码取帧"
            self._table.setItem(row, 6, QTableWidgetItem(stream_text))
            self._table.setItem(row, 7, QTableWidgetItem(mode_text))

        self._table.resizeColumnsToContents()

    def _selected_index(self) -> int | None:
        items = self._table.selectedItems()
        if not items:
            return None
        return items[0].row()

    def _on_add(self) -> None:
        dlg = CameraEditDialog(parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        cam = dlg.camera()
        self._cameras.append(cam)
        self._render()
        self.save()

    def _on_edit(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        cam = self._cameras[idx]
        dlg = CameraEditDialog(camera=cam, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._cameras[idx] = dlg.camera()
        self._render()
        self.save()

    def _on_delete(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        self._cameras.pop(idx)
        self._render()
        self.save()

    def _on_test(self) -> None:
        idx = self._selected_index()
        if idx is None:
            return
        cam = self._cameras[idx]
        self._btn_test.setEnabled(False)
        self._btn_test.setText("测试中...")
        self._logger.info(
            "测试连接点击 (id=%s, name=%s, ip=%s, port=%s, user=%s)",
            cam.id,
            cam.name,
            cam.ip,
            cam.port,
            cam.username,
        )
        fut = self._manager.test_camera_async(cam)
        fut.add_done_callback(
            lambda f: self._notify_future(f, "测试连接", cam.name or cam.id)
        )

    def _on_import(self) -> None:
        start_dir = str(self._cfg_service.config_dir)
        path, _ = QFileDialog.getOpenFileName(
            self, "导入摄像头配置", start_dir, "JSON 文件 (*.json)"
        )
        if not path:
            return

        try:
            with Path(path).open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("格式不正确：需要一个 JSON 数组")
            cameras: list[CameraConfig] = []
            for item in data:
                if isinstance(item, dict):
                    cam = CameraConfig.from_dict(item)
                    if cam.id:
                        cameras.append(cam)
            self._cameras = cameras
            self._render()
            InfoBar.success(
                "已导入",
                f"已加载 {len(cameras)} 路摄像头（点击“保存”写入配置）",
                position=InfoBarPosition.TOP_RIGHT,
                duration=2500,
                parent=self,
            )
        except Exception as e:
            InfoBar.error(
                "导入失败",
                str(e),
                position=InfoBarPosition.TOP_RIGHT,
                duration=3500,
                parent=self,
            )

    def _on_export(self) -> None:
        self._sync_enabled_flags()
        default_name = str(self._cfg_service.config_dir / "cameras.export.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "导出摄像头配置", default_name, "JSON 文件 (*.json)"
        )
        if not path:
            return

        out_path = Path(path)
        if out_path.suffix.lower() != ".json":
            out_path = out_path.with_suffix(".json")

        try:
            payload = [c.to_dict() for c in self._cameras]
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            InfoBar.success(
                "已导出",
                str(out_path),
                position=InfoBarPosition.TOP_RIGHT,
                duration=2500,
                parent=self,
            )
        except Exception as e:
            InfoBar.error(
                "导出失败",
                str(e),
                position=InfoBarPosition.TOP_RIGHT,
                duration=3500,
                parent=self,
            )

    def _sync_enabled_flags(self) -> None:
        updated: list[CameraConfig] = []
        for row, cam in enumerate(self._cameras):
            item = self._table.item(row, 0)
            enabled = bool(item and item.checkState() == Qt.CheckState.Checked)
            updated.append(
                cam
                if cam.enabled == enabled
                else cam.__class__.from_dict({**cam.to_dict(), "enabled": enabled})
            )
        self._cameras = updated

    def save(self) -> None:
        # 将表格勾选状态同步回配置
        self._sync_enabled_flags()
        self._cfg_service.save_cameras(self._cameras)
        self._manager.set_cameras(self._cameras)
        self._on_saved()

        InfoBar.success(
            "已保存",
            "cameras.json 已更新",
            position=InfoBarPosition.TOP_RIGHT,
            duration=1500,
            parent=self.window(),
        )

    def _notify_future(self, future: Future, action: str, name: str) -> None:
        def _show() -> None:
            try:
                future.result()
                InfoBar.success(
                    action,
                    f"连接正常：{name}",
                    position=InfoBarPosition.TOP_RIGHT,
                    duration=2000,
                    parent=self.window(),
                )
            except Exception as e:
                InfoBar.error(
                    f"{action}失败",
                    f"{type(e).__name__}: {e}（详见“日志”页）",
                    position=InfoBarPosition.TOP_RIGHT,
                    duration=3500,
                    parent=self.window(),
                )
            finally:
                self._btn_test.setEnabled(True)
                self._btn_test.setText("测试连接")

        # 切回 UI 线程更新按钮与提示
        self._ui.invoke(_show)
