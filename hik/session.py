from __future__ import annotations

from pathlib import Path

from app.models import CameraConfig

from .decode import FrameBuffer
from .errors import SessionError
from .sdk_context import HikSdkContext
from .session_actor import SessionActor
from .state_machine import SessionState

__all__ = ["HikSession", "SessionState"]


class HikSession:
    def __init__(
        self,
        *,
        sdk_ctx: HikSdkContext,
        camera: CameraConfig,
        decode_max_fps: int = 15,
        logger=None,
    ) -> None:
        # 会话逻辑由 Actor 串行执行，避免跨线程并发 SDK 调用
        self._actor = SessionActor(
            sdk_ctx=sdk_ctx,
            camera=camera,
            decode_max_fps=decode_max_fps,
            logger=logger,
        )

    @property
    def camera(self) -> CameraConfig:
        return self._actor.camera

    def update_camera(self, camera: CameraConfig) -> None:
        self._actor.update_camera(camera)

    def update_decode_max_fps(self, decode_max_fps: int) -> None:
        self._actor.update_decode_max_fps(decode_max_fps)

    @property
    def state(self) -> SessionState:
        return self._actor.state

    @property
    def last_error(self) -> SessionError | None:
        return self._actor.last_error

    @property
    def frame_buffer(self) -> FrameBuffer:
        return self._actor.frame_buffer

    def last_activity_ts(self) -> float | None:
        return self._actor.last_activity_ts()

    def connect(self) -> None:
        self._actor.connect()

    def start_preview_decode(self) -> None:
        self.start_preview(target_hwnd=0)

    def start_preview(self, *, target_hwnd: int | None = None) -> None:
        self._actor.start_preview(target_hwnd=target_hwnd)

    def stop(self) -> None:
        self._actor.stop()

    def reset(self) -> None:
        self._actor.reset()

    def snapshot(self, output_path: Path) -> Path:
        return self._actor.snapshot(output_path)

    def shutdown(self) -> None:
        self._actor.shutdown()
