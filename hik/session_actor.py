from __future__ import annotations

import ctypes
import logging
import threading
import time
import uuid
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FuturesTimeoutError
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

# 单条 Actor 命令的等待上限。SDK 登录在网络不可达时可阻塞数秒；抓图与
# 停止不应超过这个量级。超过就返回错误，不让调用方无限等。
COMMAND_TIMEOUT_S = 20.0

# 解码节流的容差。见 ``SessionActor._decode_on_schedule``：判定用
# ``now - last >= interval × 该值``，留一点余量吸收解码时间戳抖动。
_DECODE_GATE_TOLERANCE = 0.9


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
        # 带超时等待：线程若卡在阻塞型 SDK 调用（登录/登出/抓图）里，
        # 无超时的 result() 会把调用方的线程池 worker 永久占住，退出时
        # executor.shutdown(wait=True) 一起把进程钉死。超时后仍让 Actor
        # 继续处理（不取消），只是调用方先拿到错误。
        try:
            return command.reply.result(timeout=COMMAND_TIMEOUT_S)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"Actor 命令超时（{COMMAND_TIMEOUT_S:.0f}s）: "
                f"{type(command).__name__}"
            ) from exc

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

    def _decode_on_schedule(self, last: float, now: float) -> tuple[bool, float]:
        """解码节流：按「下一个应解码时刻」放行，而不是「距上一帧够不够久」。

        原先两道闸（``_on_decoded_callback`` 与 ``_handle_decoded_data``）
        各自用 ``now - last < 1/decode_max_fps`` 判定。相机主码流 25fps、
        ``decode_max_fps`` 也是 25 时，帧间隔正好卡在 40ms 边界：网络与
        PlayM4 的抖动让每第二帧差一点点、被判「早到」丢掉，丢掉之后下一帧
        隔了 80ms 又能过 —— 于是**恒定丢一半**（实测 45%）。这与推理速度
        无关（实测检测+管线 6.4ms/帧，预算有 10 倍余量）。

        改为按节拍推进后，早到的那帧把 ``last`` 留在原处、下一帧继续比，
        慢一拍的帧仍能补上，不会因为「每次都差 1ms」被系统性丢光；同时保留
        10% 的容差，避免解码时间戳的微抖动把一整拍推后。

        返回 ``(是否放行, 新的 last)``。``last <= 0`` 表示还没开始，直接放行。
        """

        interval = self._decode_interval_s()
        if last <= 0 or now - last >= interval * _DECODE_GATE_TOLERANCE:
            # 按整拍推进，而不是直接用 now：否则每次抖动都会把节拍往后拖。
            return True, (last + interval if last > 0 else now)
        return False, last

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
        # 进入 CONNECTING 之后的任何异常都必须把状态带回去。
        #
        # 原先 253-302 行的 initialize()/acquire_session()/登录若抛错，异常
        # 只被 _handle_command 塞进 reply、会话却停在 CONNECTING：
        # ReconnectPolicy 对非 PREVIEWING/ERROR 一律返回 NONE（永不重连），
        # 运维再点「开始」时 _do_start_preview 放行 CONNECTING、随后
        # (CONNECTING, START_PREVIEW) 在迁移表里查不到，界面永远报非法迁移。
        try:
            self._do_connect_inner()
        except Exception as exc:  # noqa: BLE001
            if self.state == SessionState.CONNECTING:
                self._set_error(
                    SessionEvent.ERROR,
                    SessionError.sdk_error(-1, f"连接阶段异常: {exc}", fatal=False),
                )
                self._release_lease()
            raise

    def _do_connect_inner(self) -> None:
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

        if self.state == SessionState.ERROR:
            # ERROR 是任何一次登录/开流失败后的状态，原先这里直接抛
            # 「ERROR 状态下无法启动预览」，界面按「开始」只会再抛一次同样
            # 的错，没有恢复入口。先复位到 DISCONNECTED 再走正常连接流程。
            self._logger.info("从 ERROR 状态复位后重新开始预览")
            self._reset_to_disconnected()

        if not self._can_start_preview() and self.state != SessionState.CONNECTING:
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
        # 控制解码回调频率，避免过高帧率拖慢 UI。
        #
        # 这是**唯一**一道解码节流：``_on_decoded_callback`` 在 SDK 解码
        # 线程上按节拍决定要不要把这一帧拷出来，``_handle_decoded_data``
        # 不再重复判定。原先两道闸各自用「距上一帧够不够久」把关，相机
        # 25fps + ``decode_max_fps=25`` 时帧间隔恰好压在 40ms 边界上，
        # 抖动让每第二帧被判早到丢掉 —— 恒定丢一半（实测 45%）。
        allowed, self._last_dec_enqueue = self._decode_on_schedule(
            self._last_dec_enqueue, now
        )
        if not allowed:
            return

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
        # 这里**不再**做帧率节流。``_on_decoded_callback`` 在 SDK 解码线程上
        # 已经按 ``decode_max_fps`` 的节拍决定放不放行；这里再做一道独立
        # 判定只会在两道闸相位错开时把本来该收的帧丢掉（现场表现：相机
        # 25fps 只到判定侧 8~12fps，日志里每小时上千条「视频时间戳异常」）。
        # 剩下的工作是 YUV→BGR，实测 2560×1440 只要 0.3ms，够快，不需要
        # 第二道闸。

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

    def _reset_to_disconnected(self) -> None:
        """把会话带回 DISCONNECTED，无论当前处于什么状态。

        只用于「用户主动重试」路径：ERROR 的迁移表里没有 CONNECT_REQUEST，
        必须先 RESET；ERROR + STOP_REQUEST 也可达，但那条路径会多走一遍
        资源释放。这里优先走 RESET，失败再退回 stop。
        """

        try:
            self._transition(SessionEvent.RESET)
            self._release_lease()
            return
        except Exception:
            pass
        try:
            self._do_stop()
        except Exception:
            self._logger.exception("复位会话失败")

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
        """把当前画面写成本地图片。

        只走 ``FrameBuffer`` 的 BGR 路径。原先还优先调用
        ``PlayM4_GetJPEG/GetBMP``，但有两个独立缺陷：

        - 这两个导出函数的实际 ABI 是 4 个参数
          ``(port, *buf, size, *out_size)``，而 sdk_context 里只声明了
          ``(port, *buf)``——后两个寄存器取 libffi 残值，``*out_size``
          那一步是对野指针的写（实测段错误，与现场「点截图后程序消失」
          一致）；
        - 这套 API 从不写文件，即使签名改对，拿到 ``True`` 也只会返回
          一个磁盘上不存在的路径，UI 却报「截图成功」。

        ``FrameBuffer`` 每帧都有最新 BGR 画面，路径本身是对的，直接用它。
        """

        frame = self._frame_buffer.get()
        if frame is None:
            raise RuntimeError("暂无画面，无法截图")

        cv2 = self._cv2
        if cv2 is None:
            try:
                import cv2  # type: ignore[import-not-found]
            except ModuleNotFoundError as e:
                raise RuntimeError("截图需要安装 `opencv-python`") from e
            self._cv2 = cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ok = bool(cv2.imwrite(str(output_path), frame.bgr))
        if not ok or not output_path.exists():
            raise RuntimeError(f"写入截图失败: {output_path}")
        return output_path

    def _on_dispatcher_stop(self) -> None:
        if not self._actor_registered:
            return
        try:
            self._sdk_ctx.release_actor()
        finally:
            self._actor_registered = False
