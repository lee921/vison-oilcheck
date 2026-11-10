"""示例视频帧源。

演示模式不再脚本伪造检测，而是回放真实录像：画面来自 mp4 文件，检测结果
来自训练好的模型。本模块只负责“按视频节拍把帧放进缓冲区”，与会话层解耦，
因此不依赖海康 SDK，也不改变相机链路。

帧对象复用 :class:`hik.decode.Frame`，与实时相机保持同一份数据契约。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from hik.decode import Frame

logger = logging.getLogger("oilcheck.video_source")

# 允许的播放速率：0.5× 用于现场逐帧核对，2× 用于快速过完整段录像。
_SPEED_MIN = 0.1
_SPEED_MAX = 4.0


def _clamp_speed(value: float) -> float:
    try:
        speed = float(value)
    except (TypeError, ValueError):
        return 1.0
    if speed <= 0:
        return 1.0
    return max(_SPEED_MIN, min(_SPEED_MAX, speed))


class VideoFileSource:
    """把 mp4 文件伪装成一个持续产出最新帧的帧源。

    读取线程按 ``min(视频 fps, fps_limit) × speed`` 的节拍顺序解码；``loop``
    为真时播放结束后自动回到开头。

    每一帧带两个时间：

    - ``ts``：取到该帧的墙钟时间（预览刷新、录制分段、在线判定用）；
    - ``media_ts``：**视频时间轴**时间，每送出一帧前进 ``1/native_fps``，
      与播放速率无关。检测管线用 ``media_ts`` 计时，所以 0.5× 慢放时
      ``pick_confirm_s`` / ``brush_hold_s`` 的判定结果与 1× 完全一致，不会
      因为看得慢而把“刷一下就过”的动作判成完成。

    用媒体时间而不是系统时间做判定也避开了两个坑：暂停时不出帧、时间自然
    不走；循环回绕时继续单调递增，不会制造时间倒退或巨大空洞。
    """

    def __init__(
        self,
        path: Path | str,
        *,
        loop: bool = True,
        fps_limit: float = 15.0,
        speed: float = 1.0,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._path = Path(path)
        self._loop = bool(loop)
        self._fps_limit = max(1.0, float(fps_limit))
        self._speed = _clamp_speed(speed)
        self._logger = logger_ or logger

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame: Frame | None = None
        self._paused = False
        self._progress = 0.0
        self._error = ""
        self._native_fps = 0.0
        self._frame_count = 0
        # 当前正在播放的帧序号（0 基）与总帧数。播放/跳转/循环都会更新，
        # 供 UI 在画面左上角显示“第 N/M 帧”，方便对照现场录像定位问题。
        self._frame_index = 0
        # 已送出的帧数，用于推进媒体时间；不随 seek / 循环重置。
        self._frames_delivered = 0
        # 待执行的跳转目标帧号；None 表示顺序播放。用 -1 之外的哨兵是为了
        # 让 seek(0) 也能生效。
        self._seek_to: int | None = None
        self._width = 0
        self._height = 0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def path(self) -> Path:
        return self._path

    @property
    def name(self) -> str:
        return self._path.name

    @property
    def playing(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @property
    def progress(self) -> float:
        """播放进度 0~1。"""

        with self._lock:
            return self._progress

    @property
    def frame_index(self) -> int:
        """当前帧序号（0 基）。"""

        with self._lock:
            return self._frame_index

    @property
    def frame_count(self) -> int:
        """视频总帧数；读不到时为 0。"""

        with self._lock:
            return self._frame_count

    @property
    def speed(self) -> float:
        """播放速率倍数（1.0 为原速）。"""

        with self._lock:
            return self._speed

    def set_speed(self, speed: float) -> None:
        """调整播放速率，下一帧生效，不影响当前播放位置。"""

        value = _clamp_speed(speed)
        with self._lock:
            self._speed = value

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._error

    def start(self) -> None:
        if self.playing:
            return
        if not self._path.exists():
            raise FileNotFoundError(f"示例视频不存在: {self._path}")
        self._stop.clear()
        with self._lock:
            self._paused = False
            self._error = ""
            self._frames_delivered = 0
        self._thread = threading.Thread(
            target=self._run, name=f"oilcheck-video-{self._path.stem}", daemon=True
        )
        self._thread.start()
        self._logger.info("示例视频已启动: %s", self._path)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        with self._lock:
            self._frame = None
            self._progress = 0.0
            self._frame_index = 0
            self._seek_to = None

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False

    def seek(self, progress: float) -> None:
        """跳转到 0~1 的相对位置（下一帧生效）。"""

        ratio = max(0.0, min(1.0, float(progress)))
        with self._lock:
            count = self._frame_count
            self._progress = ratio
            self._seek_to = int(ratio * max(0, count - 1))

    def get(self) -> Frame | None:
        with self._lock:
            return self._frame

    def last_ts(self) -> float | None:
        frame = self.get()
        return None if frame is None else frame.ts

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "path": str(self._path),
                "name": self._path.name,
                "playing": self.playing,
                "paused": self._paused,
                "progress": round(self._progress, 4),
                "frame_index": self._frame_index,
                "frame_count": self._frame_count,
                "speed": self._speed,
                "width": self._width,
                "height": self._height,
                "native_fps": round(self._native_fps, 2),
                "error": self._error,
            }

    # ------------------------------------------------------------------
    # 解码线程
    # ------------------------------------------------------------------
    def _run(self) -> None:
        import cv2  # type: ignore[import-not-found]

        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            with self._lock:
                self._error = f"视频无法打开: {self._path}"
            self._logger.error("示例视频无法打开: %s", self._path)
            return

        try:
            native_fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
            if native_fps <= 0 or native_fps > 240:
                native_fps = 15.0
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = min(float(native_fps), self._fps_limit)
            media_base = time.time()
            with self._lock:
                self._native_fps = float(native_fps)
                self._frame_count = total
                self._width = width
                self._height = height

            while not self._stop.is_set():
                started = time.perf_counter()

                with self._lock:
                    target = self._seek_to
                    self._seek_to = None
                if target is not None:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(target)))

                ok, bgr = capture.read()
                if not ok:
                    if not self._loop:
                        self._logger.info("示例视频播放结束: %s", self._path)
                        break
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, bgr = capture.read()
                    if not ok:
                        with self._lock:
                            self._error = "视频读取失败"
                        break

                if self._stop.is_set():
                    break

                with self._lock:
                    paused = self._paused
                if paused:
                    self._stop.wait(0.05)
                    continue

                h, w = bgr.shape[:2]
                with self._lock:
                    speed = self._speed
                    self._frames_delivered += 1
                    delivered = self._frames_delivered
                # 媒体时间按视频自身帧周期前进，与播放速率无关，且单调递增：
                # 慢放不改变阈值对应的画面时间，循环回绕也不会让时间倒退。
                media_ts = media_base + delivered / max(1.0, float(native_fps))
                frame = Frame(
                    bgr=bgr,
                    width=int(w),
                    height=int(h),
                    ts=time.time(),
                    media_ts=media_ts,
                )
                # CAP_PROP_POS_FRAMES 指向“下一帧”，减 1 得到本帧序号；
                # 顺序播放、seek、循环回绕都由它统一反映，不用自己数。
                pos = int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                index = max(0, pos - 1)
                with self._lock:
                    self._frame = frame
                    self._width, self._height = int(w), int(h)
                    self._frame_index = index
                    if total > 0:
                        self._progress = max(0.0, min(1.0, float(pos) / total))

                # 播放速率每帧重取，UI 改速率后下一帧立即生效。
                interval = 1.0 / max(0.1, fps * speed)
                elapsed = time.perf_counter() - started
                self._stop.wait(max(0.0, interval - elapsed))
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
            self._logger.exception("示例视频解码失败: %s", self._path)
        finally:
            capture.release()
