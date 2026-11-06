from __future__ import annotations

import ctypes
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.actors.dispatcher import ActorDispatcher
from app.actors.message import (
    CallbackData,
    Command,
    Connect,
    DecodedData,
    Message,
    Reset,
    Snapshot,
    StartPreview,
    Stop,
)
from app.config_service import ConfigService
from app.models import CameraConfig
from lib.PlayCtrl import PLAYM4_PORT

from .decode import FrameBuffer
from .errors import SessionError
from .sdk_context import HikSdkContext, SdkLease
from .state_machine import SessionEvent, SessionState, SessionStateMachine


class SessionActor:
    """基于 Actor 的会话执行器，用于串行化 SDK/PlayCtrl 调用。"""

    def __init__(
        self,
        *,
        sdk_ctx: HikSdkContext,
        camera: CameraConfig,
        decode_max_fps: int = 15,
        logger: logging.Logger | None = None,
        queue_size: int = 1000,
    ) -> None:
        self._sdk_ctx = sdk_ctx
        self._camera = camera
        self._decode_max_fps = max(1, int(decode_max_fps))
        self._logger = logger or logging.getLogger(f"oilcheck.session.{camera.id}")

        self._camera_lock = threading.Lock()
        self._decode_fps_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._activity_lock = threading.Lock()

        self._state_machine = SessionStateMachine(logger=self._logger)
        self._frame_buffer = FrameBuffer()
        self._last_activity_ts: float | None = None

        # Actor 线程串行处理 SDK/PlayCtrl 调用，避免并发冲突
        self._dispatcher = ActorDispatcher(
            name=f"session-{camera.id}",
            handler=self._handle_message,
            logger=self._logger,
            queue_size=queue_size,
            on_stop=self._on_dispatcher_stop,
        )
        self._closed = False
        self._actor_registered = False

        self._session_id = f"{camera.id}-{uuid.uuid4().hex[:8]}"
        self._lease: SdkLease | None = None

        self._user_id = -1
        self._realplay_handle = -1
        self._play_port = PLAYM4_PORT(-1)

        self._preview_mode: str | None = None
        self._render_hwnd: int = 0
        self._device_info = None

        self._real_data_cb = None
        self._dec_cb = None

        self._cv2: Any | None = None
        self._np: Any | None = None

        self._last_dec_perf = 0.0
        self._last_dec_enqueue = 0.0
        self._last_playctrl_error_ts = 0.0
        self._last_queue_drop_ts = 0.0

    @property
    def camera(self) -> CameraConfig:
        with self._camera_lock:
            return self._camera

    def update_camera(self, camera: CameraConfig) -> None:
        with self._camera_lock:
            self._camera = camera

    def update_decode_max_fps(self, decode_max_fps: int) -> None:
        with self._decode_fps_lock:
            self._decode_max_fps = max(1, int(decode_max_fps))

    @property
    def state(self) -> SessionState:
        with self._state_lock:
            return self._state_machine.state

    @property
    def last_error(self) -> SessionError | None:
        with self._state_lock:
            return self._state_machine.last_error

    @property
    def frame_buffer(self) -> FrameBuffer:
        return self._frame_buffer

    def last_activity_ts(self) -> float | None:
        with self._activity_lock:
            return self._last_activity_ts

    def connect(self) -> None:
        self._call_command(Connect())

    def start_preview(self, *, target_hwnd: int | None = None) -> None:
        self._call_command(StartPreview(target_hwnd=target_hwnd))

    def stop(self) -> None:
        self._call_command(Stop())

    def reset(self) -> None:
        self._call_command(Reset())

    def snapshot(self, output_path: Path) -> Path:
        return self._call_command(Snapshot(output_path=output_path))

    def shutdown(self) -> None:
        if self._closed:
            return
        try:
            self.stop()
        finally:
            self._closed = True
            self._dispatcher.stop()
            if not self._dispatcher.join(timeout=5.0):
                self._logger.warning("Actor 关闭超时 (%s)", self._get_camera().id)

    def _ensure_started(self) -> None:
        if not self._dispatcher.is_running:
            # 首次启动时登记 Actor 线程，便于 SDK 关闭等待
            self._sdk_ctx.register_actor()
            self._actor_registered = True
            try:
                self._dispatcher.start()
            except Exception:
                self._on_dispatcher_stop()
                raise

    def _call_command(self, command: Command) -> Any:
        if self._closed:
            raise RuntimeError("Session actor is closed")
        self._ensure_started()
        # 通过 Future 等待 Actor 线程处理完毕
        if command.reply is None:
            command = replace(command, reply=Future())
        if not self._dispatcher.send(command, block=True):
            raise RuntimeError("Actor queue is full")
        return command.reply.result()

    def _handle_message(self, message: Message) -> None:
        if isinstance(message, Command):
            self._handle_command(message)
            return
        if isinstance(message, CallbackData):
            self._handle_callback_data(message)
            return
        if isinstance(message, DecodedData):
            self._handle_decoded_data(message)
            return
        self._logger.debug("未处理的 actor 消息: %s", type(message).__name__)

    def _handle_command(self, message: Command) -> None:
        reply = message.reply
        try:
            if isinstance(message, Connect):
                result = self._do_connect()
            elif isinstance(message, StartPreview):
                result = self._do_start_preview(message.target_hwnd)
            elif isinstance(message, Stop):
                result = self._do_stop()
            elif isinstance(message, Reset):
                result = self._do_reset()
            elif isinstance(message, Snapshot):
                result = self._do_snapshot(message.output_path)
            else:
                raise RuntimeError(f"未处理的命令: {type(message).__name__}")
        except Exception as exc:
            if reply is not None and not reply.done():
                reply.set_exception(exc)
            else:
                self._logger.exception("Actor 命令执行失败: %s", type(message).__name__)
            return
        if reply is not None and not reply.done():
            reply.set_result(result)

    def _get_camera(self) -> CameraConfig:
        with self._camera_lock:
            return self._camera

    def _transition(
        self, event: SessionEvent, *, error: SessionError | None = None
    ) -> SessionState:
        with self._state_lock:
            return self._state_machine.transition(event, error=error)

    def _set_error(self, event: SessionEvent, error: SessionError) -> None:
        try:
            self._transition(event, error=error)
        except Exception:
            self._logger.debug("迁移到错误状态失败 (event=%s)", event.value)

    def _can_start_preview(self) -> bool:
        with self._state_lock:
            return self._state_machine.can_start_preview()

    def _touch_activity(self) -> None:
        with self._activity_lock:
            self._last_activity_ts = time.time()

    def _decode_interval_s(self) -> float:
        with self._decode_fps_lock:
            return 1.0 / self._decode_max_fps

    def _ensure_lease(self) -> None:
        if self._lease is None:
            # 获取 SDK 会话租约，避免关闭时还有活跃会话
            self._lease = self._sdk_ctx.acquire_session(self._session_id)

    def _release_lease(self) -> None:
        if self._lease is None:
            return
        self._sdk_ctx.release_session(self._lease)
        self._lease = None

    def _do_connect(self) -> None:
        if self.state in (SessionState.CONNECTED, SessionState.PREVIEWING):
            return

        self._transition(SessionEvent.CONNECT_REQUEST)
        self._sdk_ctx.initialize()
        self._ensure_lease()

        cam = ConfigService.with_resolved_password(self._get_camera())
        # 调用 SDK 登录设备
        self._logger.info(
            "正在连接 (id=%s, IP=%s, 端口=%s, 用户=%s)",
            cam.id,
            cam.ip,
            cam.port,
            cam.username,
        )
        from lib.HCNetSDK import NET_DVR_DEVICEINFO_V30  # type: ignore[import-not-found]

        dev_ip = ctypes.create_string_buffer(cam.ip.encode("utf-8"))
        dev_user = ctypes.create_string_buffer(cam.username.encode("utf-8"))
        dev_pwd = ctypes.create_string_buffer(cam.password.encode("utf-8"))
        device_info = NET_DVR_DEVICEINFO_V30()

        user_id = int(
            self._sdk_ctx.sdk.NET_DVR_Login_V30(
                dev_ip, int(cam.port), dev_user, dev_pwd, ctypes.byref(device_info)
            )
        )
        if user_id < 0:
            err_code, err_msg = self._sdk_ctx.last_error_detail()
            error = SessionError.sdk_error(err_code, err_msg, fatal=False)
            self._set_error(SessionEvent.LOGIN_FAILED, error)
            self._release_lease()
            raise RuntimeError(
                f"登录失败 ({cam.ip}:{cam.port}), err={err_code} {err_msg}".strip()
            )

        self._user_id = user_id
        self._device_info = device_info
        self._transition(SessionEvent.LOGIN_OK)
        try:
            chan_num = int(device_info.byChanNum)
            start_chan = int(device_info.byStartChan)
            ip_chan_num = int(device_info.byIPChanNum)
        except Exception:
            chan_num = start_chan = ip_chan_num = -1
        self._logger.info(
            "登录成功 (user_id=%s) %s (start_chan=%s, chan_num=%s, ip_chan_num=%s)",
            self._user_id,
            cam.ip,
            start_chan,
            chan_num,
            ip_chan_num,
        )

    def _cleanup_preview_resources(self) -> None:
        if self._realplay_handle < 0 and self._play_port.value > -1:
            try:
                self._sdk_ctx.playctrl.PlayM4_Stop(self._play_port)
                self._sdk_ctx.playctrl.PlayM4_CloseStream(self._play_port)
                self._sdk_ctx.playctrl.PlayM4_FreePort(self._play_port)
            except Exception:
                pass
            self._play_port = PLAYM4_PORT(-1)

    def _playctrl_last_error(self) -> int | None:
        func = getattr(self._sdk_ctx.playctrl, "PlayM4_GetLastError", None)
        if func is None:
            return None
        try:
            return int(func(self._play_port))
        except Exception:
            return None

    def _do_start_preview(self, target_hwnd: int | None) -> None:
        if self.state == SessionState.PREVIEWING:
            return

        if self.state == SessionState.DISCONNECTED:
            self._do_connect()

        if not self._can_start_preview() and self.state != SessionState.CONNECTING:
            if self.state == SessionState.ERROR:
                raise RuntimeError(f"ERROR 状态下无法启动预览: {self.last_error}")
            raise RuntimeError(f"无法从 {self.state.value} 状态启动预览")

        self._transition(SessionEvent.START_PREVIEW)
        # 新会话从空白画面开始，避免显示上一次预览的缓存帧。
        self._frame_buffer.clear()

        cam = self._get_camera()
        from lib.HCNetSDK import (
            NET_DVR_PREVIEWINFO,
            NET_DVR_STREAMDATA,
            NET_DVR_SYSHEAD,
            REALDATACALLBACK,
        )  # type: ignore[import-not-found]

        hwnd = int(target_hwnd) if target_hwnd is not None else int(self._render_hwnd)
        # 优先使用直连渲染（需要有效 HWND），否则走解码回调
        use_direct = cam.preferred_mode == "direct" and hwnd != 0
        self._preview_mode = "direct" if use_direct else "decode"
        if use_direct:
            self._render_hwnd = hwnd

        requested_channel = int(cam.channel)
        channel = requested_channel
        try:
            if self._device_info is not None:
                start_chan = int(self._device_info.byStartChan)
                if start_chan > 1 and channel < start_chan and channel > 0:
                    channel = start_chan + channel - 1
        except Exception:
            pass

        self._logger.info(
            "开始预览 (id=%s, 通道=%s->%s, 码流=%s, 模式=%s, hwnd=%s)",
            cam.id,
            requested_channel,
            channel,
            cam.stream_type,
            self._preview_mode,
            hwnd if use_direct else 0,
        )

        self._touch_activity()
        self._last_dec_perf = 0.0
        self._last_dec_enqueue = 0.0

        if self._preview_mode == "decode":
            try:
                import cv2  # type: ignore[import-not-found]
                import numpy as np  # type: ignore[import-not-found]
            except ModuleNotFoundError as e:
                error = SessionError.decode_error(
                    "解码模式需要安装 `opencv-python` 和 `numpy`"
                )
                self._set_error(SessionEvent.ERROR, error)
                raise RuntimeError("解码模式需要安装 `opencv-python` 和 `numpy`") from e

            self._cv2 = cv2
            self._np = np

            from lib.PlayCtrl import DECCBFUNWIN  # type: ignore[import-not-found]

            def dec_callback(nPort, pBuf, nSize, pFrameInfo, nUser, nReserved2):
                return self._on_decoded_callback(
                    nPort, pBuf, nSize, pFrameInfo, nUser, nReserved2
                )

            self._dec_cb = DECCBFUNWIN(dec_callback)
        else:
            self._dec_cb = None

        # 获取 PlayCtrl 端口用于流解码
        if not self._sdk_ctx.playctrl.PlayM4_GetPort(ctypes.byref(self._play_port)):
            error = SessionError.playctrl_error(-1, "PlayM4_GetPort 失败")
            self._set_error(SessionEvent.PREVIEW_FAILED, error)
            raise RuntimeError("PlayM4_GetPort 失败")

        try:
            preview_info = NET_DVR_PREVIEWINFO()
            preview_info.hPlayWnd = 0  # 使用回调 + PlayCtrl 解码
            preview_info.lChannel = channel
            preview_info.dwStreamType = 0 if cam.stream_type == "main" else 1
            preview_info.dwLinkMode = 0  # TCP 传输
            preview_info.bBlocked = 1
            preview_info.byPreviewMode = 0

            def real_data_callback(lPlayHandle, dwDataType, pBuffer, dwBufSize, pUser):
                return self._on_real_data_callback(
                    lPlayHandle,
                    dwDataType,
                    pBuffer,
                    dwBufSize,
                    pUser,
                    NET_DVR_SYSHEAD,
                    NET_DVR_STREAMDATA,
                )

            self._real_data_cb = REALDATACALLBACK(real_data_callback)

            handle = int(
                self._sdk_ctx.sdk.NET_DVR_RealPlay_V40(
                    self._user_id,
                    ctypes.byref(preview_info),
                    self._real_data_cb,
                    None,
                )
            )
            if handle < 0:
                err_code, err_msg = self._sdk_ctx.last_error_detail()
                error = SessionError.sdk_error(err_code, err_msg, fatal=False)
                self._set_error(SessionEvent.PREVIEW_FAILED, error)
                raise RuntimeError(
                    f"NET_DVR_RealPlay_V40 失败, err={err_code} {err_msg}".strip()
                )

            self._realplay_handle = handle
            self._transition(SessionEvent.PREVIEW_OK)
            self._logger.info(
                "预览已启动 (handle=%s) %s 模式=%s",
                self._realplay_handle,
                cam.id,
                self._preview_mode,
            )
        except Exception:
            self._cleanup_preview_resources()
            self._real_data_cb = None
            self._dec_cb = None
            self._preview_mode = None
            raise

    def _on_real_data_callback(
        self,
        lPlayHandle,
        dwDataType,
        pBuffer,
        dwBufSize,
        pUser,
        NET_DVR_SYSHEAD,
        NET_DVR_STREAMDATA,
    ):
        if self._play_port.value < 0:
            return
        if self.state == SessionState.ERROR:
            return
        # SDK 回调线程：区分头数据与码流数据
        if dwDataType == NET_DVR_SYSHEAD:
            kind = "SYSHEAD"
        elif dwDataType == NET_DVR_STREAMDATA:
            kind = "STREAMDATA"
        else:
            return
        try:
            payload = ctypes.string_at(pBuffer, int(dwBufSize))
        except Exception:
            return
        self._enqueue_event(CallbackData(kind=kind, payload=payload))

    def _on_decoded_callback(self, nPort, pBuf, nSize, pFrameInfo, nUser, nReserved2):
        try:
            frame_info = pFrameInfo.contents
        except Exception:
            return
        if int(frame_info.nType) != 3:  # YV12 格式
            return

        now = time.perf_counter()
        # 控制解码回调频率，避免过高帧率拖慢 UI
        min_interval = self._decode_interval_s()
        if now - self._last_dec_enqueue < min_interval:
            return
        self._last_dec_enqueue = now

        width = int(frame_info.nWidth)
        height = int(frame_info.nHeight)
        yuv_size = width * height * 3 // 2
        try:
            payload = ctypes.string_at(pBuf, yuv_size)
        except Exception:
            return
        self._enqueue_event(
            DecodedData(
                payload=payload,
                width=width,
                height=height,
                frame_type=int(frame_info.nType),
            )
        )

    def _enqueue_event(self, message: Message) -> None:
        if not self._dispatcher.is_running:
            return
        if self._dispatcher.send(message, block=False):
            return
        now = time.time()
        if now - self._last_queue_drop_ts >= 1.0:
            self._last_queue_drop_ts = now
            self._logger.warning("Actor 队列已满，丢弃 %s", type(message).__name__)

    def _handle_callback_data(self, message: CallbackData) -> None:
        if self._play_port.value < 0:
            return
        if self.state == SessionState.ERROR:
            return

        self._touch_activity()

        if message.kind == "SYSHEAD":
            # 收到系统头后初始化 PlayCtrl 解码
            self._sdk_ctx.playctrl.PlayM4_SetStreamOpenMode(self._play_port, 0)
            ok = bool(
                self._sdk_ctx.playctrl.PlayM4_OpenStream(
                    self._play_port,
                    message.payload,
                    len(message.payload),
                    1024 * 1024,
                )
            )
            if not ok:
                error = SessionError.playctrl_error(
                    self._playctrl_last_error() or -1,
                    "PlayM4_OpenStream 失败",
                )
                self._set_error(SessionEvent.ERROR, error)
                self._logger.error(
                    "PlayM4_OpenStream 失败 (端口=%s, 头数据=%s, 错误=%s)",
                    int(self._play_port.value),
                    len(message.payload),
                    self._playctrl_last_error(),
                )
                return

            ok = bool(self._sdk_ctx.playctrl.PlayM4_SetDecodeEngine(self._play_port, 0))
            if not ok:
                self._logger.warning(
                    "PlayM4_SetDecodeEngine 失败 (端口=%s, 错误=%s)",
                    int(self._play_port.value),
                    self._playctrl_last_error(),
                )
            if self._preview_mode == "decode":
                # 解码模式走回调获取 YUV 数据
                ok = bool(
                    self._sdk_ctx.playctrl.PlayM4_SetDecCallBackExMend(
                        self._play_port, self._dec_cb, None, 0, None
                    )
                )
                if not ok:
                    error = SessionError.playctrl_error(
                        self._playctrl_last_error() or -1,
                        "PlayM4_SetDecCallBackExMend 失败",
                    )
                    self._set_error(SessionEvent.ERROR, error)
                    self._logger.error(
                        "PlayM4_SetDecCallBackExMend 失败 (端口=%s, 错误=%s)",
                        int(self._play_port.value),
                        self._playctrl_last_error(),
                    )
                    return

                ok = bool(self._sdk_ctx.playctrl.PlayM4_Play(self._play_port, 0))
                if not ok:
                    error = SessionError.playctrl_error(
                        self._playctrl_last_error() or -1,
                        "PlayM4_Play 失败 (decode)",
                    )
                    self._set_error(SessionEvent.ERROR, error)
                    self._logger.error(
                        "PlayM4_Play 失败 (模式=decode, 端口=%s, 错误=%s)",
                        int(self._play_port.value),
                        self._playctrl_last_error(),
                    )
                    return
            else:
                # 直连模式由 PlayCtrl 直接渲染到 HWND
                ok = bool(
                    self._sdk_ctx.playctrl.PlayM4_Play(
                        self._play_port, ctypes.c_void_p(self._render_hwnd)
                    )
                )
                if not ok:
                    error = SessionError.playctrl_error(
                        self._playctrl_last_error() or -1,
                        "PlayM4_Play 失败 (direct)",
                    )
                    self._set_error(SessionEvent.ERROR, error)
                    self._logger.error(
                        "PlayM4_Play 失败 (模式=direct, 端口=%s, hwnd=%s, 错误=%s)",
                        int(self._play_port.value),
                        int(self._render_hwnd),
                        self._playctrl_last_error(),
                    )
                    return
            return

        if message.kind == "STREAMDATA":
            # 持续喂给 PlayCtrl 解码
            ok = bool(
                self._sdk_ctx.playctrl.PlayM4_InputData(
                    self._play_port, message.payload, len(message.payload)
                )
            )
            if not ok:
                error = SessionError.playctrl_error(
                    self._playctrl_last_error() or -1, "PlayM4_InputData 失败"
                )
                self._set_error(SessionEvent.ERROR, error)
                now = time.time()
                if now - self._last_playctrl_error_ts >= 1.0:
                    self._last_playctrl_error_ts = now
                    self._logger.error(
                        "PlayM4_InputData 失败 (端口=%s, 大小=%s, 错误=%s)",
                        int(self._play_port.value),
                        len(message.payload),
                        self._playctrl_last_error(),
                    )

    def _handle_decoded_data(self, message: DecodedData) -> None:
        if message.frame_type != 3:
            return
        if self.state == SessionState.ERROR:
            return
        now = time.perf_counter()
        # 控制 UI 刷新频率
        min_interval = self._decode_interval_s()
        if now - self._last_dec_perf < min_interval:
            return
        self._last_dec_perf = now

        cv2 = self._cv2
        np = self._np
        if cv2 is None or np is None:
            return

        # YUV -> BGR，供 UI 绘制
        try:
            yuv_array = np.frombuffer(message.payload, dtype=np.uint8)
            yuv_frame = yuv_array.reshape((message.height * 3 // 2, message.width))
            bgr = cv2.cvtColor(yuv_frame, cv2.COLOR_YUV2BGR_YV12)
        except Exception:
            return

        self._frame_buffer.update_bgr(bgr, message.width, message.height)
        self._touch_activity()

    def _do_reset(self) -> None:
        if self.state != SessionState.DISCONNECTED:
            self._do_stop()
        try:
            self._transition(SessionEvent.RESET)
        except Exception:
            return

    def _do_stop(self) -> None:
        if self.state == SessionState.DISCONNECTED:
            self._frame_buffer.clear()
            self._release_lease()
            return

        self._transition(SessionEvent.STOP_REQUEST)

        # 释放 SDK 预览与 PlayCtrl 资源
        if self._realplay_handle >= 0:
            try:
                self._sdk_ctx.sdk.NET_DVR_StopRealPlay(self._realplay_handle)
            except Exception:
                pass
            self._realplay_handle = -1

        if self._play_port.value > -1:
            try:
                self._sdk_ctx.playctrl.PlayM4_Stop(self._play_port)
                self._sdk_ctx.playctrl.PlayM4_CloseStream(self._play_port)
                self._sdk_ctx.playctrl.PlayM4_FreePort(self._play_port)
            except Exception:
                pass
            self._play_port = PLAYM4_PORT(-1)

        if self._user_id >= 0:
            try:
                self._sdk_ctx.sdk.NET_DVR_Logout(self._user_id)
            except Exception:
                pass
            self._user_id = -1

        self._transition(SessionEvent.STOPPED)
        self._logger.info("会话已停止 (%s)", self._get_camera().id)
        self._frame_buffer.clear()
        self._preview_mode = None
        self._real_data_cb = None
        self._dec_cb = None
        self._device_info = None
        self._release_lease()

    def _do_snapshot(self, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []

        if self._play_port.value > -1:
            # 优先使用 PlayCtrl 直接导出 JPEG/BMP
            path_bytes = str(output_path).encode("utf-8")
            path_buf = ctypes.create_string_buffer(path_bytes)
            func_name = (
                "PlayM4_GetJPEG"
                if output_path.suffix.lower() in (".jpg", ".jpeg")
                else "PlayM4_GetBMP"
            )
            func = getattr(self._sdk_ctx.playctrl, func_name, None)
            if func is None:
                errors.append(f"{func_name} 不可用")
            else:
                ok = bool(func(self._play_port, path_buf))
                if ok:
                    return output_path
                errors.append(f"{func_name} 失败")

        frame = self._frame_buffer.get()
        if frame is None:
            raise RuntimeError("; ".join(errors) or "暂无画面")

        cv2 = self._cv2
        if cv2 is None:
            try:
                import cv2  # type: ignore[import-not-found]
            except ModuleNotFoundError as e:
                raise RuntimeError("截图需要安装 `opencv-python`") from e
            self._cv2 = cv2

        ok = bool(cv2.imwrite(str(output_path), frame.bgr))
        if not ok:
            raise RuntimeError(f"写入截图失败: {output_path}")
        return output_path

    def _on_dispatcher_stop(self) -> None:
        if not self._actor_registered:
            return
        try:
            self._sdk_ctx.release_actor()
        finally:
            self._actor_registered = False
