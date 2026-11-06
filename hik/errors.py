"""HikSDK 会话操作的统一错误模型。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class ErrorSource(str, Enum):
    """错误来源。"""

    SDK = "SDK"
    PLAYCTRL = "PlayCtrl"
    DECODE = "Decode"
    IO = "IO"
    INTERNAL = "Internal"


@dataclass(frozen=True)
class SessionError:
    """会话操作中的错误详情。"""

    source: ErrorSource
    code: int | None
    message: str
    fatal: bool = False
    ts: float = 0.0

    def __init__(
        self,
        source: ErrorSource | str,
        code: int | None,
        message: str,
        fatal: bool = False,
        ts: float = 0.0,
    ):
        if isinstance(source, str):
            source = ErrorSource(source)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "fatal", fatal)
        object.__setattr__(self, "ts", ts or time.time())

    @classmethod
    def sdk_error(cls, code: int, message: str, fatal: bool = False) -> SessionError:
        """创建 SDK 错误。"""
        return cls(ErrorSource.SDK, code, message, fatal)

    @classmethod
    def playctrl_error(
        cls, code: int, message: str, fatal: bool = False
    ) -> SessionError:
        """创建 PlayCtrl 错误。"""
        return cls(ErrorSource.PLAYCTRL, code, message, fatal)

    @classmethod
    def decode_error(cls, message: str) -> SessionError:
        """创建解码错误。"""
        return cls(ErrorSource.DECODE, None, message, fatal=False)

    @classmethod
    def io_error(cls, message: str) -> SessionError:
        """创建 IO 错误。"""
        return cls(ErrorSource.IO, None, message, fatal=False)

    @classmethod
    def internal_error(cls, message: str) -> SessionError:
        """创建内部错误。"""
        return cls(ErrorSource.INTERNAL, None, message, fatal=False)

    def __str__(self) -> str:
        code_str = f"[{self.code}]" if self.code is not None else ""
        fatal_str = " FATAL" if self.fatal else ""
        return f"{self.source.value}{code_str}: {self.message}{fatal_str}"
