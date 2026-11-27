from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal, Self

StreamType = Literal["main", "sub"]
PreferredMode = Literal["direct", "decode"]

logger = logging.getLogger("oilcheck.config")

# 现场手写 JSON 的常见错误：`enabled: "false"`、`port: "abc"`、`ui_fps: 1e400`。
# 这些配置在 main.py 里早于 setup_logging 读取，一旦抛异常就是「窗口闪一下
# 就没了」且没有任何日志。解析器逐项退回默认值并记 warning——日志就绪后
# 能看到，程序也能带着可用配置起来。
_TRUE_STRINGS = {"1", "true", "yes", "on"}
_FALSE_STRINGS = {"0", "false", "no", "off", ""}


def _bad(section: str, key: str, value: Any, default: Any) -> None:
    logger.warning(
        "配置项 %s.%s 非法（%r），已回退默认值 %r", section, key, value, default
    )


def _as_bool(section: str, key: str, value: Any, default: bool) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE_STRINGS:
            return True
        if text in _FALSE_STRINGS:
            return False
        _bad(section, key, value, default)
        return default
    if isinstance(value, (bool, int)):
        return bool(value)
    if value is None:
        return default
    _bad(section, key, value, default)
    return default


def _as_int(
    section: str, key: str, value: Any, default: int,
    lo: int | None = None, hi: int | None = None,
) -> int:
    # int(inf) 抛 OverflowError，int(nan) 抛 ValueError，两者都要接住。
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        _bad(section, key, value, default)
        return default
    if lo is not None and number < lo:
        number = lo
    if hi is not None and number > hi:
        number = hi
    return number


def _as_float(
    section: str, key: str, value: Any, default: float,
    lo: float | None = None, hi: float | None = None,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        _bad(section, key, value, default)
        return default
    if not math.isfinite(number):
        _bad(section, key, value, default)
        return default
    if lo is not None and number < lo:
        number = lo
    if hi is not None and number > hi:
        number = hi
    return number


def _as_dict(section: str, key: str, value: Any) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    _bad(section, key, value, {})
    return {}


@dataclass(frozen=True)
class ReconnectConfig:
    enable: bool = True
    max_backoff_s: int = 30
    offline_timeout_s: int = 10

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        data = dict(data or {})
        d = cls()
        return cls(
            enable=_as_bool("reconnect", "enable", data.get("enable", d.enable), d.enable),
            max_backoff_s=_as_int(
                "reconnect", "max_backoff_s", data.get("max_backoff_s", d.max_backoff_s),
                1, 3600,
            ),
            offline_timeout_s=_as_int(
                "reconnect", "offline_timeout_s",
                data.get("offline_timeout_s", d.offline_timeout_s), 1, 3600,
            ),
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
        data = dict(data or {})
        return cls(
            id=str(data.get("id", "")),
            name=str(data.get("name", "")),
            enabled=_as_bool("cameras", "enabled", data.get("enabled", True), True),
            ip=str(data.get("ip", "")),
            port=_as_int("cameras", "port", data.get("port", 8000), 8000, 1, 65535),
            username=str(data.get("username", "admin")),
            password=str(data.get("password", "")),
            total_detections=_as_int(
                "cameras", "total_detections", data.get("total_detections", 0), 0, 0
            ),
            channel=_as_int("cameras", "channel", data.get("channel", 1), 1, 1, 64),
            stream_type="main" if data.get("stream_type") == "main" else "sub",
            preferred_mode=(
                "direct" if data.get("preferred_mode") == "direct" else "decode"
            ),
            reconnect=ReconnectConfig.from_dict(
                _as_dict("cameras", "reconnect", data.get("reconnect"))
            ),
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
        data = dict(data or {})
        d = cls()
        return cls(
            enabled=_as_bool("relay", "enabled", data.get("enabled", d.enabled), d.enabled),
            port=_as_int("relay", "port", data.get("port", d.port), d.port, 1, 65535),
            timeout_s=_as_float(
                "relay", "timeout_s", data.get("timeout_s", d.timeout_s),
                d.timeout_s, 0.05, 60.0,
            ),
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
    # 解码侧最大帧率。必须**大于**相机码流帧率：这一项是节流上限，取成和
    # 码流一样的值（例如 25fps 相机配 25）时，帧间隔正好压在 1/fps 边界上，
    # 抖动会让每第二帧被判「早到」丢掉，恒定丢一半。现场默认 50。
    decode_max_fps: int = 50
    offline_timeout_s: int = 10
    snapshot_dir: str = "snapshots"
    log_dir: str = "logs"
    relay: RelayConfig = RelayConfig()

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        data = dict(data or {})
        d = cls()
        return cls(
            layout=_as_int("app", "layout", data.get("layout", d.layout), d.layout, 1, 4),
            ui_fps=_as_int("app", "ui_fps", data.get("ui_fps", d.ui_fps), d.ui_fps, 1, 60),
            decode_max_fps=_as_int(
                "app", "decode_max_fps", data.get("decode_max_fps", d.decode_max_fps),
                d.decode_max_fps, 1, 60,
            ),
            offline_timeout_s=_as_int(
                "app", "offline_timeout_s",
                data.get("offline_timeout_s", d.offline_timeout_s),
                d.offline_timeout_s, 1, 3600,
            ),
            snapshot_dir=str(data.get("snapshot_dir") or d.snapshot_dir),
            log_dir=str(data.get("log_dir") or d.log_dir),
            relay=RelayConfig.from_dict(_as_dict("app", "relay", data.get("relay"))),
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
