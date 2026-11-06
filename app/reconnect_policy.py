"""摄像头会话的重连策略。"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum

from hik.errors import SessionError
from hik.state_machine import SessionState

from .models import ReconnectConfig


class PolicyAction(str, Enum):
    """策略决策动作。"""

    NONE = "none"
    RECONNECT = "reconnect"
    STOP = "stop"


@dataclass(frozen=True)
class ReconnectDecision:
    """重连策略的决策结果。"""

    action: PolicyAction
    delay_s: int = 0
    reason: str = ""


class ReconnectPolicy:
    """
    摄像头会话重连策略。

    根据以下信息决定是否重连：
    - 会话状态
    - 最近活动时间
    - 错误信息
    - 配置（超时、退避等）
    """

    def __init__(
        self,
        *,
        offline_timeout_s: int = 10,
        max_backoff_s: int = 30,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        初始化重连策略。

        Args:
            offline_timeout_s: 超过多少秒无活动视为离线
            max_backoff_s: 退避上限（秒）
            logger: 可选日志记录器
        """
        self._offline_timeout_s = offline_timeout_s
        self._max_backoff_s = max_backoff_s
        self._logger = logger or logging.getLogger("oilcheck.reconnect")

    def should_reconnect(
        self,
        state: SessionState,
        last_activity_ts: float | None,
        error: SessionError | None,
        reconnect_enabled: bool,
    ) -> ReconnectDecision:
        """
        判断是否需要尝试重连。

        Args:
            state: 当前会话状态
            last_activity_ts: 最近活动时间戳（秒）
            error: 最近一次错误（如果有）
            reconnect_enabled: 是否开启重连

        Returns:
            ReconnectDecision，包含动作与延迟
        """
        if not reconnect_enabled:
            return ReconnectDecision(PolicyAction.NONE, reason="已禁用重连")

        # 致命错误直接停止
        if error and error.fatal:
            return ReconnectDecision(
                PolicyAction.STOP,
                reason=f"致命错误: {error.source.value}/{error.code}",
            )

        # 检查会话状态
        if state != SessionState.PREVIEWING:
            if state == SessionState.ERROR:
                # 错误状态自动重试（致命错误除外）
                return ReconnectDecision(
                    PolicyAction.RECONNECT,
                    delay_s=1,
                    reason=f"错误状态: {error.message if error else '未知'}",
                )
            return ReconnectDecision(PolicyAction.NONE, reason=f"状态: {state.value}")

        # 检查离线超时
        now = time.time()
        if last_activity_ts is None:
            return ReconnectDecision(PolicyAction.NONE, reason="暂无活动")
        if not self.offline(last_activity_ts, self._offline_timeout_s):
            return ReconnectDecision(PolicyAction.NONE, reason="活跃")

        offline_duration = now - last_activity_ts

        # 离线则尝试重连
        return ReconnectDecision(
            PolicyAction.RECONNECT,
            delay_s=0,
            reason=f"离线时长 {offline_duration:.1f}s",
        )

    def offline(self, last_activity_ts: float | None, timeout_s: int) -> bool:
        """根据最近活动时间判断是否离线。"""
        if last_activity_ts is None:
            return False
        return (time.time() - last_activity_ts) > timeout_s

    def next_backoff(self, attempt: int) -> int:
        """
        计算指数退避延迟。

        Args:
            attempt: 重试次数（从 0 开始）

        Returns:
            退避秒数（不超过 max_backoff_s）
        """
        if attempt < 0:
            return 1

        # 指数退避：2^attempt
        backoff = min(2**attempt, self._max_backoff_s)
        return max(1, backoff)


@dataclass
class ReconnectState:
    """记录重连尝试的状态。"""

    attempt: int = 0
    next_attempt_ts: float = 0.0
    last_action_ts: float = 0.0

    def should_attempt(self, now: float) -> bool:
        """判断是否到达下一次重试时间。"""
        return now >= self.next_attempt_ts

    def schedule_next(self, delay_s: int, now: float) -> None:
        """安排下一次重连时间。"""
        self.attempt += 1
        self.last_action_ts = now
        self.next_attempt_ts = now + delay_s

    def reset(self) -> None:
        """重置重连状态。"""
        self.attempt = 0
        self.next_attempt_ts = 0.0
        self.last_action_ts = 0.0


class ReconnectSupervisor:
    """协调重连决策与退避调度。"""

    def __init__(
        self,
        *,
        app_offline_timeout_s: int,
        logger: logging.Logger | None = None,
    ) -> None:
        self._app_offline_timeout_s = app_offline_timeout_s
        self._logger = logger or logging.getLogger("oilcheck.reconnect")
        self._lock = threading.Lock()
        self._state: dict[str, ReconnectState] = {}

    def update_app_config(self, *, app_offline_timeout_s: int) -> None:
        with self._lock:
            self._app_offline_timeout_s = max(1, int(app_offline_timeout_s))

    def reset(self, cam_id: str) -> None:
        with self._lock:
            state = self._state.setdefault(cam_id, ReconnectState())
            state.reset()

    def forget(self, cam_id: str) -> None:
        with self._lock:
            self._state.pop(cam_id, None)

    def evaluate(
        self,
        *,
        cam_id: str,
        reconnect_cfg: ReconnectConfig,
        state: SessionState,
        last_activity_ts: float | None,
        error: SessionError | None,
        now: float,
    ) -> ReconnectDecision | None:
        with self._lock:
            app_offline_timeout_s = self._app_offline_timeout_s
        offline_timeout = reconnect_cfg.offline_timeout_s or app_offline_timeout_s
        policy = ReconnectPolicy(
            offline_timeout_s=offline_timeout,
            max_backoff_s=int(reconnect_cfg.max_backoff_s),
            logger=self._logger,
        )

        decision = policy.should_reconnect(
            state,
            last_activity_ts,
            error,
            reconnect_cfg.enable,
        )

        with self._lock:
            st = self._state.setdefault(cam_id, ReconnectState())

            if decision.action == PolicyAction.NONE:
                if st.attempt != 0:
                    st.reset()
                return None

            if decision.action == PolicyAction.STOP:
                st.reset()
                return decision

            if not st.should_attempt(now):
                return None

            delay = max(decision.delay_s, policy.next_backoff(st.attempt))
            st.schedule_next(delay, now)
            return decision
