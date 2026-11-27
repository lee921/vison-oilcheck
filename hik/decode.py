from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Frame:
    """解码后的 BGR 帧。

    ``ts`` 是取到该帧的墙钟时间，用于预览刷新、录制分段时间和在线判定。
    ``media_ts`` 是该帧在**视频时间轴**上的时间：示例回放变速（如 0.5×）
    时它仍按视频自身帧周期前进，检测管线用它计时，判定结果与播放速度无关。
    实时相机没有这个概念，保持 ``None``，管线回退到 ``ts``。
    """

    bgr: Any
    width: int
    height: int
    ts: float
    media_ts: float | None = None


class FrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # 新帧到达通知：判定侧用 :meth:`wait_newer` 阻塞等下一帧，
        # 而不是按固定节拍轮询「最新帧」。
        self._updated = threading.Condition(self._lock)
        self._frame: Frame | None = None

    def update_bgr(self, bgr: Any, width: int, height: int) -> None:
        # 多线程环境下更新最新帧，确保线程安全
        frame = Frame(bgr=bgr, width=width, height=height, ts=time.time())
        with self._updated:
            self._frame = frame
            self._updated.notify_all()

    def get(self) -> Frame | None:
        # 读取最新帧（只读快照）
        with self._lock:
            return self._frame

    def wait_newer(self, since_ts: float, timeout: float) -> Frame | None:
        """等一帧 ``ts > since_ts`` 的新帧；超时返回 ``None``。

        判定侧原先按 ``1/max_fps`` 固定节拍取「最新帧」：相机帧间隔不均匀
        （实测 25fps 主码流中位 38ms、p90 75ms），有的节拍落在两帧之间，
        取到的还是上一帧，于是整拍白跑——配置 15fps 实测只到 13.6fps，
        日志里对应一条「视频时间戳异常」。这里改成「有更新的帧就返回」，
        节拍只用于限速，不再决定采样相位。
        """

        deadline = time.monotonic() + max(0.0, timeout)
        with self._updated:
            while True:
                frame = self._frame
                if frame is not None and frame.ts > since_ts:
                    return frame
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._updated.wait(remaining)

    def last_ts(self) -> float | None:
        with self._lock:
            return None if self._frame is None else self._frame.ts

    def clear(self) -> None:
        """清除缓存帧，停止预览后避免 UI 继续显示最后一帧。"""
        with self._updated:
            self._frame = None
            # 唤醒等着的判定线程，让它立刻拿到 None 而不是等到超时。
            self._updated.notify_all()
