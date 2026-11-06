"""Actor 消息调度器，用于串行化操作。"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable

from .message import Message, Shutdown


class ActorDispatcher:
    """
    Actor 消息调度器。

    使用专用线程顺序处理消息，保证操作串行与线程安全。
    """

    def __init__(
        self,
        name: str,
        handler: Callable[[Message], None],
        *,
        logger: logging.Logger | None = None,
        queue_size: int = 1000,
        on_stop: Callable[[], None] | None = None,
    ) -> None:
        """
        初始化调度器。

        Args:
            name: Actor 名称（用于日志）
            handler: 消息处理函数
            logger: 可选日志记录器
            queue_size: 队列上限（0 表示不限制）
        """
        self._name = name
        self._handler = handler
        self._logger = logger or logging.getLogger(f"oilcheck.actor.{name}")
        self._queue: queue.Queue[Message] = queue.Queue(maxsize=queue_size)
        self._thread: threading.Thread | None = None
        self._running = False
        self._stopped = threading.Event()
        self._on_stop = on_stop

    def start(self) -> None:
        """启动 Actor 线程。"""
        if self._running:
            return

        self._running = True
        self._stopped.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name=f"Actor-{self._name}", daemon=False
        )
        self._thread.start()
        self._logger.info("Actor 已启动: %s", self._name)

    def stop(self) -> None:
        """请求停止 Actor。"""
        if not self._running:
            return

        self._running = False
        # 发送关闭消息，确保线程干净退出
        try:
            self._queue.put_nowait(Shutdown(reply=None))
        except queue.Full:
            self._logger.warning("发送关闭消息失败：队列已满")

    def join(self, timeout: float | None = None) -> bool:
        """
        等待 Actor 线程结束。

        Args:
            timeout: 等待的最大秒数

        Returns:
            True 表示线程结束，False 表示超时
        """
        if self._thread is None:
            return True

        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def send(
        self, message: Message, block: bool = True, timeout: float | None = None
    ) -> bool:
        """
        向 Actor 发送消息。

        Args:
            message: 要发送的消息
            block: 队列满时是否阻塞
            timeout: 阻塞最大时长（None 表示无限）

        Returns:
            发送成功返回 True，否则返回 False
        """
        try:
            if block:
                self._queue.put(message, block=True, timeout=timeout)
            else:
                self._queue.put_nowait(message)
            return True
        except queue.Full:
            self._logger.warning("Actor 队列已满: %s", self._name)
            return False

    def _run_loop(self) -> None:
        """Actor 主循环。"""
        self._logger.debug("Actor 循环已启动: %s", self._name)

        while self._running:
            try:
                # 带超时读取，便于及时检查 _running
                try:
                    message = self._queue.get(block=True, timeout=0.1)
                except queue.Empty:
                    continue

                # 处理关闭消息
                if isinstance(message, Shutdown):
                    self._logger.debug("Actor 收到关闭消息: %s", self._name)
                    break

                # 顺序处理消息，避免并发冲突
                try:
                    self._handler(message)
                except Exception:
                    self._logger.exception(
                        "Actor 处理消息异常: %s, message=%s",
                        self._name,
                        type(message).__name__,
                    )

            except Exception:
                self._logger.exception("Actor 循环异常: %s", self._name)
                break

        self._stopped.set()
        self._logger.info("Actor 已停止: %s", self._name)
        if self._on_stop is not None:
            try:
                self._on_stop()
            except Exception:
                self._logger.exception("执行 on_stop 回调失败: %s", self._name)

    @property
    def is_running(self) -> bool:
        """判断 Actor 是否运行中。"""
        return self._running and self._thread is not None and self._thread.is_alive()
