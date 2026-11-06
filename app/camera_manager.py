"""相机会话管理。

本版本只负责海康相机连接、预览、重连和最新帧缓存，不再按相机 ID 绑定业务
检测逻辑。业务侧通过 ``get_session(...).frame_buffer`` 读取最新帧。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from hik.sdk_context import HikSdkContext
from hik.session import HikSession

from .config_service import ConfigService
from .models import AppConfig, CameraConfig
from .reconnect_policy import PolicyAction, ReconnectSupervisor


class CameraManager:
    def __init__(
        self,
        *,
        sdk_ctx: HikSdkContext,
        cfg_service: ConfigService,
        app_cfg: AppConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self._sdk_ctx = sdk_ctx
        self._cfg_service = cfg_service
        self._app_cfg = app_cfg
        self._logger = logger or logging.getLogger("oilcheck.camera_manager")

        self._lock = threading.Lock()
        self._sessions: dict[str, HikSession] = {}
        self._reconnect_supervisor = ReconnectSupervisor(
            app_offline_timeout_s=self._app_cfg.offline_timeout_s,
            logger=self._logger,
        )
        self._cameras: dict[str, CameraConfig] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="oilcheck-cam"
        )

        self.reload_from_disk()

    def reload_from_disk(self) -> None:
        cameras = self._cfg_service.load_cameras()
        self.set_cameras(cameras)

    def update_app_config(self, app_cfg: AppConfig) -> None:
        with self._lock:
            self._app_cfg = app_cfg
            sessions = list(self._sessions.values())

        self._reconnect_supervisor.update_app_config(
            app_offline_timeout_s=app_cfg.offline_timeout_s,
        )
        for session in sessions:
            session.update_decode_max_fps(app_cfg.decode_max_fps)

    def set_cameras(self, cameras: list[CameraConfig]) -> None:
        removed_sessions: list[HikSession] = []
        with self._lock:
            self._cameras = {c.id: c for c in cameras if c.id}
            active_cameras = {
                cam_id: cam for cam_id, cam in self._cameras.items() if cam.enabled
            }

            removed_ids = set(self._sessions).difference(active_cameras)
            for cam_id in removed_ids:
                removed_sessions.append(self._sessions.pop(cam_id))
                self._reconnect_supervisor.forget(cam_id)

            for cam_id, cam in active_cameras.items():
                if cam_id in self._sessions:
                    self._sessions[cam_id].update_camera(cam)
                else:
                    self._sessions[cam_id] = HikSession(
                        sdk_ctx=self._sdk_ctx,
                        camera=cam,
                        decode_max_fps=self._app_cfg.decode_max_fps,
                        logger=self._logger.getChild(cam_id),
                    )
                self._reconnect_supervisor.reset(cam_id)

        for session in removed_sessions:
            self._executor.submit(session.shutdown)

    def cameras(self) -> list[CameraConfig]:
        with self._lock:
            return list(self._cameras.values())

    def get_session(self, cam_id: str) -> HikSession | None:
        with self._lock:
            return self._sessions.get(cam_id)

    def start_async(
        self, cam_id: str, *, target_hwnd: int | None = None
    ) -> Future[None]:
        session = self.get_session(cam_id)
        if session is None:
            f: Future[None] = Future()
            f.set_exception(KeyError(f"Unknown camera: {cam_id}"))
            return f

        self._logger.info("收到启动请求 (相机ID=%s, 目标HWND=%s)", cam_id, target_hwnd)
        return self._executor.submit(self._start_worker, session, target_hwnd)

    def stop_async(self, cam_id: str) -> Future[None]:
        session = self.get_session(cam_id)
        if session is None:
            f: Future[None] = Future()
            f.set_exception(KeyError(f"Unknown camera: {cam_id}"))
            return f
        self._logger.info("收到停止请求 (相机ID=%s)", cam_id)
        return self._executor.submit(session.stop)

    def test_camera_async(self, camera: CameraConfig) -> Future[None]:
        self._logger.info(
            "收到连接测试请求 (id=%s, ip=%s, port=%s, user=%s)",
            camera.id,
            camera.ip,
            camera.port,
            camera.username,
        )
        return self._executor.submit(self._test_camera_worker, camera)

    def latest_frame(self, cam_id: str) -> Any | None:
        session = self.get_session(cam_id)
        if session is None or session.state.value != "previewing":
            return None
        return session.frame_buffer.get()

    def _start_worker(self, session: HikSession, target_hwnd: int | None) -> None:
        cam_id = session.camera.id
        try:
            self._sdk_ctx.initialize()
            session.start_preview(target_hwnd=target_hwnd)
            self._reconnect_supervisor.reset(cam_id)
        except Exception:
            self._logger.exception("启动失败 (%s)", cam_id)
            raise

    def _test_camera_worker(self, camera: CameraConfig) -> None:
        cam_id = camera.id
        with self._lock:
            decode_max_fps = self._app_cfg.decode_max_fps
        session = HikSession(
            sdk_ctx=self._sdk_ctx,
            camera=camera,
            decode_max_fps=decode_max_fps,
            logger=self._logger.getChild(f"test.{cam_id}"),
        )
        try:
            self._sdk_ctx.initialize()
            session.connect()
            self._logger.info("连接测试成功 (%s)", cam_id)
        except Exception:
            self._logger.exception("连接测试失败 (%s)", cam_id)
            raise
        finally:
            try:
                session.shutdown()
            except Exception:
                pass

    def tick(self) -> None:
        """周期健康检查（由 UI 定时器调用）。"""
        now = time.time()
        with self._lock:
            sessions = list(self._sessions.values())
            cameras = self._cameras.copy()

        for session in sessions:
            cam = cameras.get(session.camera.id)
            if cam is None:
                continue

            decision = self._reconnect_supervisor.evaluate(
                cam_id=cam.id,
                reconnect_cfg=cam.reconnect,
                state=session.state,
                last_activity_ts=session.last_activity_ts(),
                error=session.last_error,
                now=now,
            )
            if decision is None:
                continue

            if decision.action == PolicyAction.STOP:
                self._logger.warning("重连策略：停止 (%s): %s", cam.id, decision.reason)
                self._executor.submit(session.stop)
                continue

            if decision.action == PolicyAction.RECONNECT:
                self._logger.warning(
                    "重连策略：执行重连 (%s): %s", cam.id, decision.reason
                )
                self._executor.submit(self._reconnect_worker, cam.id)

    def _reconnect_worker(self, cam_id: str) -> None:
        session = self.get_session(cam_id)
        if session is None:
            return
        try:
            session.stop()
        except Exception:
            pass
        try:
            self._start_worker(session, None)
        except Exception as e:
            self._logger.error("重连失败 (%s): %s", cam_id, e)

    def shutdown(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            try:
                session.shutdown()
            except Exception:
                pass
        self._executor.shutdown(wait=True, cancel_futures=True)
