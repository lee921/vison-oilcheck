from __future__ import annotations

from dataclasses import replace
from typing import cast
from uuid import uuid4

from PySide6.QtWidgets import QDialog, QFormLayout, QHBoxLayout, QVBoxLayout, QWidget
from qfluentwidgets import (
    CheckBox,
    ComboBox,
    LineEdit,
    PasswordLineEdit,
    PrimaryPushButton,
    PushButton,
    SpinBox,
    SwitchButton,
)

from app.models import CameraConfig, ReconnectConfig


class CameraEditDialog(QDialog):
    def __init__(
        self, *, camera: CameraConfig | None = None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("摄像头")
        self.setModal(True)

        self._camera = camera

        self._enabled = CheckBox("启用")
        self._id = LineEdit()
        self._name = LineEdit()
        self._ip = LineEdit()
        self._port = SpinBox()
        self._port.setRange(1, 65535)
        self._username = LineEdit()
        self._password = PasswordLineEdit()
        self._channel = SpinBox()
        self._channel.setRange(1, 9999)
        self._stream = ComboBox()
        self._stream.addItem("主码流", userData="main")
        self._stream.addItem("子码流", userData="sub")
        self._mode = ComboBox()
        self._mode.addItem("解码取帧", userData="decode")
        self._mode.addItem("直渲染", userData="direct")

        self._reconnect_enable = SwitchButton()
        self._offline_timeout = SpinBox()
        self._offline_timeout.setRange(1, 600)
        self._max_backoff = SpinBox()
        self._max_backoff.setRange(1, 600)

        form = QFormLayout()
        form.addRow("", self._enabled)
        form.addRow("ID", self._id)
        form.addRow("名称", self._name)
        form.addRow("IP", self._ip)
        form.addRow("端口", self._port)
        form.addRow("用户名", self._username)
        form.addRow("密码", self._password)
        form.addRow("通道", self._channel)
        form.addRow("码流", self._stream)
        form.addRow("模式", self._mode)
        form.addRow("自动重连", self._reconnect_enable)
        form.addRow("离线超时（秒）", self._offline_timeout)
        form.addRow("最大退避（秒）", self._max_backoff)

        btn_ok = PrimaryPushButton("确定")
        btn_cancel = PushButton("取消")
        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(btn_cancel)
        btns.addWidget(btn_ok)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addLayout(btns)

        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        self._load(camera)

    def _load(self, camera: CameraConfig | None) -> None:
        if camera is None:
            self._enabled.setChecked(True)
            self._id.setText(f"cam_{uuid4().hex[:6]}")
            self._name.setText("")
            self._ip.setText("")
            self._port.setValue(8000)
            self._username.setText("admin")
            self._password.setText("")
            self._channel.setValue(1)
            self._stream.setCurrentIndex(1)
            self._mode.setCurrentIndex(0)
            self._reconnect_enable.setChecked(True)
            self._offline_timeout.setValue(10)
            self._max_backoff.setValue(30)
            return

        self._enabled.setChecked(bool(camera.enabled))
        self._id.setText(camera.id)
        self._name.setText(camera.name)
        self._ip.setText(camera.ip)
        self._port.setValue(int(camera.port))
        self._username.setText(camera.username)
        self._password.setText(camera.password)
        self._channel.setValue(int(camera.channel))
        self._stream.setCurrentIndex(0 if camera.stream_type == "main" else 1)
        self._mode.setCurrentIndex(1 if camera.preferred_mode == "direct" else 0)
        self._reconnect_enable.setChecked(bool(camera.reconnect.enable))
        self._offline_timeout.setValue(int(camera.reconnect.offline_timeout_s))
        self._max_backoff.setValue(int(camera.reconnect.max_backoff_s))

    def camera(self) -> CameraConfig:
        cam_id = self._id.text().strip()
        if not cam_id:
            cam_id = f"cam_{uuid4().hex[:6]}"

        reconnect = ReconnectConfig(
            enable=bool(self._reconnect_enable.isChecked()),
            max_backoff_s=int(self._max_backoff.value()),
            offline_timeout_s=int(self._offline_timeout.value()),
        )

        base = self._camera or CameraConfig.from_dict({"id": cam_id})
        stream_type = cast(str, self._stream.currentData() or "sub")
        preferred_mode = cast(str, self._mode.currentData() or "decode")
        return replace(
            base,
            id=cam_id,
            name=self._name.text().strip() or cam_id,
            enabled=bool(self._enabled.isChecked()),
            ip=self._ip.text().strip(),
            port=int(self._port.value()),
            username=self._username.text().strip(),
            password=self._password.text(),
            channel=int(self._channel.value()),
            stream_type=stream_type,
            preferred_mode=preferred_mode,
            reconnect=reconnect,
        )
