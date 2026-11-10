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
        self._frame: Frame | None = None

    def update_bgr(self, bgr: Any, width: int, height: int) -> None:
        # 多线程环境下更新最新帧，确保线程安全
        frame = Frame(bgr=bgr, width=width, height=height, ts=time.time())
        with self._lock:
            self._frame = frame

    def get(self) -> Frame | None:
        # 读取最新帧（只读快照）
        with self._lock:
            return self._frame

    def last_ts(self) -> float | None:
        with self._lock:
            return None if self._frame is None else self._frame.ts

    def clear(self) -> None:
        """清除缓存帧，停止预览后避免 UI 继续显示最后一帧。"""
        with self._lock:
            self._frame = None
