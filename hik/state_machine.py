"""HikSDK 会话状态机。"""

from __future__ import annotations

import logging
from enum import Enum
from typing import TYPE_CHECKING

from .errors import SessionError

if TYPE_CHECKING:
    pass


class SessionState(str, Enum):
    """会话状态定义。"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    PREVIEWING = "previewing"
    STOPPING = "stopping"
    ERROR = "error"


class SessionEvent(str, Enum):
    """触发状态迁移的事件。"""

    # 用户/系统触发
    CONNECT_REQUEST = "connect_request"
    START_PREVIEW = "start_preview"
    STOP_REQUEST = "stop_request"
    RESET = "reset"

    # SDK/PlayCtrl 回调触发
    LOGIN_OK = "login_ok"
    LOGIN_FAILED = "login_failed"
    PREVIEW_OK = "preview_ok"
    PREVIEW_FAILED = "preview_failed"
    STOPPED = "stopped"
    ERROR = "error"


# 合法状态迁移：(current_state, event) -> next_state
_TRANSITIONS: dict[tuple[SessionState, SessionEvent], SessionState] = {
    # 从 DISCONNECTED 出发
    (SessionState.DISCONNECTED, SessionEvent.CONNECT_REQUEST): SessionState.CONNECTING,
    (
        SessionState.DISCONNECTED,
        SessionEvent.START_PREVIEW,
    ): SessionState.CONNECTING,  # 隐式连接
    (SessionState.DISCONNECTED, SessionEvent.RESET): SessionState.DISCONNECTED,
    # 从 CONNECTING 出发
    (SessionState.CONNECTING, SessionEvent.LOGIN_OK): SessionState.CONNECTED,
    (SessionState.CONNECTING, SessionEvent.LOGIN_FAILED): SessionState.ERROR,
    (SessionState.CONNECTING, SessionEvent.ERROR): SessionState.ERROR,
    (SessionState.CONNECTING, SessionEvent.STOP_REQUEST): SessionState.STOPPING,
    # 从 CONNECTED 出发
    (
        SessionState.CONNECTED,
        SessionEvent.START_PREVIEW,
    ): SessionState.CONNECTED,  # 已发起预览，等待 PREVIEW_OK
    (SessionState.CONNECTED, SessionEvent.PREVIEW_OK): SessionState.PREVIEWING,
    (SessionState.CONNECTED, SessionEvent.PREVIEW_FAILED): SessionState.ERROR,
    (SessionState.CONNECTED, SessionEvent.ERROR): SessionState.ERROR,
    (SessionState.CONNECTED, SessionEvent.STOP_REQUEST): SessionState.STOPPING,
    # 从 PREVIEWING 出发
    (SessionState.PREVIEWING, SessionEvent.STOP_REQUEST): SessionState.STOPPING,
    (SessionState.PREVIEWING, SessionEvent.ERROR): SessionState.ERROR,
    # 从 STOPPING 出发
    (SessionState.STOPPING, SessionEvent.STOPPED): SessionState.DISCONNECTED,
    (SessionState.STOPPING, SessionEvent.ERROR): SessionState.ERROR,
    # 从 ERROR 出发
    (SessionState.ERROR, SessionEvent.RESET): SessionState.DISCONNECTED,
    (SessionState.ERROR, SessionEvent.STOP_REQUEST): SessionState.STOPPING,
}


class InvalidTransitionError(RuntimeError):
    """非法状态迁移。"""

    def __init__(self, state: SessionState, event: SessionEvent) -> None:
        super().__init__(f"非法状态迁移: {state.value} + {event.value}")
        self.state = state
        self.event = event


class SessionStateMachine:
    """会话生命周期状态机。"""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._state = SessionState.DISCONNECTED
        self._error: SessionError | None = None
        self._logger = logger or logging.getLogger("oilcheck.state_machine")

    @property
    def state(self) -> SessionState:
        """当前状态。"""
        return self._state

    @property
    def last_error(self) -> SessionError | None:
        """处于 ERROR 状态时的最后一次错误。"""
        return self._error

    def transition(
        self, event: SessionEvent, error: SessionError | None = None
    ) -> SessionState:
        """
        根据事件迁移到新状态。

        Args:
            event: 触发迁移的事件
            error: 可选错误详情（用于 ERROR 状态）

        Returns:
            新状态

        Raises:
            InvalidTransitionError: 非法迁移时抛出
        """
        key = (self._state, event)
        if key not in _TRANSITIONS:
            raise InvalidTransitionError(self._state, event)

        old_state = self._state
        new_state = _TRANSITIONS[key]

        # 进入 ERROR 状态时记录错误信息
        if new_state == SessionState.ERROR:
            self._error = error or SessionError.internal_error(
                f"状态迁移发生未知错误（from {old_state.value}）"
            )
        elif new_state != SessionState.ERROR:
            self._error = None

        self._state = new_state
        self._logger.debug(
            "状态迁移: %s -> %s (事件=%s, 错误=%s)",
            old_state.value,
            new_state.value,
            event.value,
            self._error,
        )
        return new_state

    def can_start_preview(self) -> bool:
        """判断是否允许启动预览。"""
        return self._state in (SessionState.CONNECTED,)

    def can_stop(self) -> bool:
        """判断是否允许停止会话。"""
        return self._state in (
            SessionState.CONNECTING,
            SessionState.CONNECTED,
            SessionState.PREVIEWING,
            SessionState.ERROR,
        )

    def reset(self) -> SessionState:
        """重置到 DISCONNECTED 状态。"""
        return self.transition(SessionEvent.RESET)
