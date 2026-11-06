from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

StreamType = Literal["main", "sub"]
PreferredMode = Literal["direct", "decode"]


@dataclass(frozen=True)
class ReconnectConfig:
    enable: bool = True
    max_backoff_s: int = 30
    offline_timeout_s: int = 10

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            enable=bool(data.get("enable", True)),
            max_backoff_s=int(data.get("max_backoff_s", 30)),
            offline_timeout_s=int(data.get("offline_timeout_s", 10)),
        )

    def to_dict(self) -> dict:
        return {
            "enable": self.enable,
            "max_backoff_s": self.max_backoff_s,
            "offline_timeout_s": self.offline_timeout_s,
        }


@dataclass(frozen=True)
class CameraConfig:
    id: str
    name: str
    enabled: bool
    ip: str
    port: int
    username: str
    password: str
    total_detections: int = 0
    channel: int = 1
    stream_type: StreamType = "sub"
    preferred_mode: PreferredMode = "decode"
    reconnect: ReconnectConfig = ReconnectConfig()

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        try:
            total_detections = int(data.get("total_detections", 0))
        except Exception:
            total_detections = 0
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            enabled=bool(data.get("enabled", True)),
            ip=str(data.get("ip", "")),
            port=int(data.get("port", 8000)),
            username=str(data.get("username", "admin")),
            password=str(data.get("password", "")),
            total_detections=total_detections,
            channel=int(data.get("channel", 1)),
            stream_type="main" if data.get("stream_type") == "main" else "sub",
            preferred_mode=(
                "direct" if data.get("preferred_mode") == "direct" else "decode"
            ),
            reconnect=ReconnectConfig.from_dict(dict(data.get("reconnect", {}))),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "ip": self.ip,
            "port": self.port,
            "username": self.username,
            "password": self.password,
            "total_detections": self.total_detections,
            "channel": self.channel,
            "stream_type": self.stream_type,
            "preferred_mode": self.preferred_mode,
            "reconnect": self.reconnect.to_dict(),
        }


@dataclass(frozen=True)
class RelayConfig:
    enabled: bool = False
    port: int = 10000
    timeout_s: float = 1.0

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        # host 是历史字段；目标设备由现场固定接线映射决定。
        return cls(
            enabled=bool(data.get("enabled", False)),
            port=int(data.get("port", 10000)),
            timeout_s=float(data.get("timeout_s", 1.0)),
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "port": self.port,
            "timeout_s": self.timeout_s,
        }


@dataclass(frozen=True)
class AppConfig:
    layout: int = 4
    ui_fps: int = 15
    decode_max_fps: int = 15
    offline_timeout_s: int = 10
    snapshot_dir: str = "snapshots"
    log_dir: str = "logs"
    relay: RelayConfig = RelayConfig()

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls(
            layout=int(data.get("layout", 4)),
            ui_fps=int(data.get("ui_fps", 15)),
            decode_max_fps=int(data.get("decode_max_fps", 15)),
            offline_timeout_s=int(data.get("offline_timeout_s", 10)),
            snapshot_dir=str(data.get("snapshot_dir", "snapshots")),
            log_dir=str(data.get("log_dir", "logs")),
            relay=RelayConfig.from_dict(dict(data.get("relay", {}))),
        )

    def to_dict(self) -> dict:
        return {
            "layout": self.layout,
            "ui_fps": self.ui_fps,
            "decode_max_fps": self.decode_max_fps,
            "offline_timeout_s": self.offline_timeout_s,
            "snapshot_dir": self.snapshot_dir,
            "log_dir": self.log_dir,
            "relay": self.relay.to_dict(),
        }
