"""重连策略与会话兜底的纯逻辑测试。

这些路径原先完全没有覆盖，而 R12~R14、R17 都是「纯逻辑、可离线复现」
的问题：不依赖 SDK，一个假状态机就能挡住回归。
"""

from __future__ import annotations

import time
from concurrent.futures import Future

import pytest

from app.actors.dispatcher import ActorDispatcher
from app.actors.message import Command
from app.models import ReconnectConfig
from app.reconnect_policy import (
    PolicyAction,
    ReconnectPolicy,
    ReconnectSupervisor,
)
from hik.state_machine import SessionEvent, SessionState, SessionStateMachine


def _decision(state: SessionState, *, last_activity: float | None = None, error=None):
    policy = ReconnectPolicy(offline_timeout_s=10, max_backoff_s=30)
    return policy.should_reconnect(state, last_activity, error, True)


def test_non_previewing_non_error_states_do_not_reconnect():
    for state in (
        SessionState.DISCONNECTED,
        SessionState.CONNECTING,
        SessionState.STOPPING,
    ):
        assert _decision(state).action == PolicyAction.NONE


def test_error_state_requests_reconnect():
    assert _decision(SessionState.ERROR).action == PolicyAction.RECONNECT


def test_backoff_never_resets_while_session_is_transient():
    """经过 CONNECTING / DISCONNECTED 时退避不能清零。

    原实现只要 decision 是 NONE 就 ``st.reset()``：重连过程中会话必然
    经过这两个状态，attempt 永远回到 0，退避恒为 1s，max_backoff_s
    形同虚设——一台断网相机每几秒就被重新登录一次。
    """

    supervisor = ReconnectSupervisor(app_offline_timeout_s=10)
    cfg = ReconnectConfig(enable=True, max_backoff_s=30, offline_timeout_s=10)
    now = time.time()
    delays: list[float] = []

    def tick(state: SessionState) -> None:
        supervisor.evaluate(
            cam_id="cam",
            reconnect_cfg=cfg,
            state=state,
            last_activity_ts=None,
            error=None,
            now=now,
        )

    for _ in range(5):
        # 一次完整循环：断线 → 登录中（这两步都是 NONE）→ 登录失败（ERROR）
        tick(SessionState.DISCONNECTED)
        tick(SessionState.CONNECTING)
        # 到达重试时刻前，ERROR 只会走「还没到点」分支
        tick(SessionState.ERROR)
        with supervisor._lock:
            state = supervisor._state["cam"]
            delays.append(state.next_attempt_ts - now)
            now = state.next_attempt_ts + 0.001

    assert delays[0] < delays[-1], delays
    assert delays[-1] >= 2.0, delays


def test_healthy_session_clears_backoff():
    supervisor = ReconnectSupervisor(app_offline_timeout_s=10)
    cfg = ReconnectConfig(enable=True, max_backoff_s=30, offline_timeout_s=10)
    now = time.time()
    supervisor.evaluate(
        cam_id="cam",
        reconnect_cfg=cfg,
        state=SessionState.ERROR,
        last_activity_ts=None,
        error=None,
        now=now,
    )
    with supervisor._lock:
        assert supervisor._state["cam"].attempt == 1
    supervisor.evaluate(
        cam_id="cam",
        reconnect_cfg=cfg,
        state=SessionState.PREVIEWING,
        last_activity_ts=now,
        error=None,
        now=now,
    )
    with supervisor._lock:
        assert supervisor._state["cam"].attempt == 0


def test_error_state_can_be_reset_to_disconnected():
    """ERROR 必须有可走出的迁移，否则界面没有任何恢复入口。"""

    machine = SessionStateMachine()
    machine.transition(SessionEvent.CONNECT_REQUEST)  # → CONNECTING
    machine.transition(SessionEvent.LOGIN_FAILED)  # → ERROR
    assert machine.state == SessionState.ERROR
    assert machine.transition(SessionEvent.RESET) == SessionState.DISCONNECTED


def test_connecting_start_preview_is_an_invalid_transition():
    """记录现状：CONNECTING + START_PREVIEW 没有迁移。

    ``_do_start_preview`` 必须在此之前把会话复位（见 session_actor）。
    """

    machine = SessionStateMachine()
    machine.transition(SessionEvent.CONNECT_REQUEST)
    from hik.state_machine import InvalidTransitionError

    with pytest.raises(InvalidTransitionError):
        machine.transition(SessionEvent.START_PREVIEW)


def test_dispatcher_fails_pending_commands_on_stop():
    """线程停止后，队列里残留命令的 Future 必须被置异常。

    否则调用方阻塞在 ``reply.result()``（无超时）上，退出时
    ``executor.shutdown(wait=True)`` 会把进程钉死。
    """

    seen: list[Command] = []

    def handler(message) -> None:  # noqa: ANN001
        if isinstance(message, Command):
            seen.append(message)
            if message.reply is not None:
                message.reply.set_result(None)

    dispatcher = ActorDispatcher("test", handler)
    dispatcher.start()

    # 先让线程处理一条（确保它真的在跑）
    first = Command(reply=Future())
    assert dispatcher.send(first)
    first.reply.result(timeout=2)

    # 停掉线程，再把命令塞进「已死」的队列
    dispatcher.stop()
    assert dispatcher.join(timeout=2)

    pending = Command(reply=Future())
    assert dispatcher.send(pending)
    dispatcher._fail_pending()
    with pytest.raises(RuntimeError, match="Actor 已停止"):
        pending.reply.result(timeout=1)
