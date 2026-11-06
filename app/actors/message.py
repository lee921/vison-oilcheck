"""Actor 并发模型的消息类型定义。"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


class Message:
    """Actor 消息基类。"""

    pass


@dataclass(frozen=True, kw_only=True)
class Command(Message):
    """命令消息：由外部调用发起的请求。"""

    reply: Future[Any] | None = None


@dataclass(frozen=True)
class Connect(Command):
    """连接摄像头的命令。"""

    pass


@dataclass(frozen=True)
class StartPreview(Command):
    """启动预览的命令。"""

    target_hwnd: int | None = None


@dataclass(frozen=True)
class Stop(Command):
    """停止预览的命令。"""

    pass


@dataclass(frozen=True)
class Reset(Command):
    """重置会话状态的命令。"""

    pass


@dataclass(frozen=True)
class Snapshot(Command):
    """截图命令。"""

    output_path: Path


@dataclass(frozen=True)
class Shutdown(Command):
    """关闭 Actor 的命令。"""

    pass


@dataclass(frozen=True)
class CallbackData(Message):
    """SDK 回调数据（从回调线程序列化进入 Actor）。"""

    kind: Literal["SYSHEAD", "STREAMDATA"]
    payload: bytes


@dataclass(frozen=True)
class DecodedData(Message):
    """PlayCtrl 回调产生的 YUV 解码数据。"""

    payload: bytes
    width: int
    height: int
    frame_type: int
