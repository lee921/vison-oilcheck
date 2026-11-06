from __future__ import annotations

import ctypes
import logging
import os
import platform
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from app.paths import runtime_root


class HikSdkError(RuntimeError):
    pass


@dataclass(frozen=True)
class SdkLease:
    """SDK 活跃会话的租约。"""

    session_id: str


class HikSdkContext:
    """全局 HCNetSDK/PlayCtrl 加载器，负责 NET_DVR_Init/Cleanup 生命周期。"""

    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        sdk_dir: Path | None = None,
        log_dir: Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._repo_root = (repo_root or runtime_root()).resolve()
        self._sdk_dir = sdk_dir
        self._log_dir = log_dir
        self._logger = logger or logging.getLogger("oilcheck.hik")

        self._init_lock = threading.Lock()
        self._session_cond = threading.Condition(self._init_lock)
        self._dll_dirs: list[os._AddedDllDirectory] = []  # type: ignore[attr-defined]
        self._sdk: ctypes.CDLL | None = None
        self._playctrl: ctypes.CDLL | None = None
        self._initialized = False
        self._encoding: str = "utf-8"
        self._active_sessions: dict[str, SdkLease] = {}
        self._active_actors = 0
        self._closed = False

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    @property
    def sdk_dir(self) -> Path:
        if self._sdk_dir is not None:
            return self._sdk_dir
        sys_name = platform.system()
        if sys_name == "Windows":
            return self._repo_root / "lib" / "win"
        return self._repo_root / "lib" / "linux"

    @property
    def log_dir(self) -> Path:
        if self._log_dir is not None:
            return self._log_dir
        return self._repo_root / "logs" / "hik_sdk"

    @property
    def sdk(self) -> ctypes.CDLL:
        if self._sdk is None:
            raise HikSdkError("HCNetSDK 尚未加载")
        return self._sdk

    @property
    def playctrl(self) -> ctypes.CDLL:
        if self._playctrl is None:
            raise HikSdkError("PlayCtrl 尚未加载")
        return self._playctrl

    def initialize(self) -> None:
        with self._init_lock:
            if self._initialized:
                return

            sdk_dir = self.sdk_dir
            log_dir = self.log_dir
            log_dir.mkdir(parents=True, exist_ok=True)

            self._logger.info(
                "正在初始化 HCNetSDK (sdk_dir=%s, log_dir=%s)", sdk_dir, log_dir
            )
            needs_cleanup = False
            try:
                sys_name = platform.system()
                if sys_name == "Windows":
                    self._dll_dirs.append(os.add_dll_directory(str(sdk_dir)))
                    com_dir = sdk_dir / "HCNetSDKCom"
                    if com_dir.exists():
                        self._dll_dirs.append(os.add_dll_directory(str(com_dir)))

                sdk_path = sdk_dir / (
                    "HCNetSDK.dll" if sys_name == "Windows" else "libhcnetsdk.so"
                )
                play_path = sdk_dir / (
                    "PlayCtrl.dll" if sys_name == "Windows" else "libPlayCtrl.so"
                )
                self._logger.info(
                    "正在加载动态库 (sdk=%s, playctrl=%s)", sdk_path, play_path
                )

                if not sdk_path.exists():
                    raise HikSdkError(f"缺少 SDK 动态库: {sdk_path}")
                if not play_path.exists():
                    raise HikSdkError(f"缺少 PlayCtrl 动态库: {play_path}")

                # 加载 SDK 与播放库动态库
                self._sdk = ctypes.CDLL(str(sdk_path))
                self._playctrl = ctypes.CDLL(str(play_path))

                # 导入 ctypes 结构与常量（来自 vendored 的 `lib/`）
                from lib.HCNetSDK import NET_DVR_LOCAL_SDK_PATH  # type: ignore[import-not-found]
                from lib.PlayCtrl import DECCBFUNWIN, PLAYM4_PORT  # type: ignore[import-not-found]

                encoding = "gbk" if sys_name == "Windows" else "utf-8"
                self._encoding = encoding

                try:
                    self._sdk.NET_DVR_GetErrorMsg.restype = ctypes.c_char_p
                    self._sdk.NET_DVR_GetErrorMsg.argtypes = [
                        ctypes.POINTER(ctypes.c_long)
                    ]
                except Exception:
                    pass
                self._configure_playctrl_signatures(PLAYM4_PORT, DECCBFUNWIN)

                # SDK 路径必须指向包含 `HCNetSDK.dll` 与 `HCNetSDKCom/` 的目录
                # 注意不要直接传入 `HCNetSDKCom/` 子目录
                sdk_path_cfg = NET_DVR_LOCAL_SDK_PATH()
                sdk_path_cfg.sPath = str(sdk_dir).encode(encoding)
                self._logger.info("设置 SDK 初始化路径(type=2)为 %s", sdk_dir)
                ok = bool(self.sdk.NET_DVR_SetSDKInitCfg(2, ctypes.byref(sdk_path_cfg)))
                if not ok:
                    err_code, err_msg = self.last_error_detail()
                    self._logger.warning(
                        "NET_DVR_SetSDKInitCfg(SDK_PATH) 失败, err=%s %s",
                        err_code,
                        err_msg,
                    )

                # 配置 OpenSSL 依赖路径
                if sys_name == "Windows":
                    crypto_path = (sdk_dir / "libcrypto-1_1-x64.dll").resolve()
                    ssl_path = (sdk_dir / "libssl-1_1-x64.dll").resolve()
                    if crypto_path.exists():
                        self.sdk.NET_DVR_SetSDKInitCfg(
                            3,
                            ctypes.create_string_buffer(
                                str(crypto_path).encode(encoding)
                            ),
                        )
                    if ssl_path.exists():
                        self.sdk.NET_DVR_SetSDKInitCfg(
                            4,
                            ctypes.create_string_buffer(str(ssl_path).encode(encoding)),
                        )
                else:
                    crypto_path = self._first_existing_path(
                        sdk_dir / "libcrypto.so.3",
                        sdk_dir / "libcrypto.so.1.1",
                    )
                    ssl_path = self._first_existing_path(
                        sdk_dir / "libssl.so.3",
                        sdk_dir / "libssl.so.1.1",
                    )
                    if crypto_path.exists():
                        self.sdk.NET_DVR_SetSDKInitCfg(
                            3,
                            ctypes.create_string_buffer(
                                str(crypto_path).encode(encoding)
                            ),
                        )
                    if ssl_path.exists():
                        self.sdk.NET_DVR_SetSDKInitCfg(
                            4,
                            ctypes.create_string_buffer(str(ssl_path).encode(encoding)),
                        )

                # 初始化 SDK 核心能力
                if not self.sdk.NET_DVR_Init():
                    raise HikSdkError(f"NET_DVR_Init 失败, err={self.last_error()}")
                needs_cleanup = True

                # 设置本地配置以避免录像分段（保持历史工具行为一致）
                # 预览不强依赖，但可以减少差异
                try:
                    from lib.HCNetSDK import NET_DVR_LOCAL_GENERAL_CFG  # type: ignore[import-not-found]

                    general_cfg = NET_DVR_LOCAL_GENERAL_CFG()
                    general_cfg.byNotSplitRecordFile = 1
                    ok = bool(
                        self.sdk.NET_DVR_SetSDKLocalCfg(17, ctypes.byref(general_cfg))
                    )
                    if ok:
                        self._logger.info(
                            "已应用 SDK 本地配置 (GENERAL.byNotSplitRecordFile=1)"
                        )
                    else:
                        err_code, err_msg = self.last_error_detail()
                        self._logger.warning(
                            "NET_DVR_SetSDKLocalCfg(GENERAL) 失败, err=%s %s",
                            err_code,
                            err_msg,
                        )
                except Exception as e:
                    self._logger.debug("已跳过 SDK 本地配置设置: %s", e, exc_info=True)

                # 启用 SDK 内部日志
                log_path = str(log_dir) + ("\\" if sys_name == "Windows" else "/")
                self.sdk.NET_DVR_SetLogToFile(3, log_path.encode(encoding), False)

                self._initialized = True
                self._logger.info("HCNetSDK 初始化成功 (sdk_dir=%s)", sdk_dir)
            except Exception:
                self._logger.exception("HCNetSDK 初始化失败")
                if needs_cleanup and self._sdk is not None:
                    try:
                        self._sdk.NET_DVR_Cleanup()
                    except Exception:
                        pass
                self._initialized = False
                self._sdk = None
                self._playctrl = None
                self._cleanup_dll_dirs()
                raise

    def cleanup(self) -> None:
        with self._init_lock:
            if not self._initialized or self._sdk is None:
                return
            try:
                self._sdk.NET_DVR_Cleanup()
            finally:
                self._initialized = False
                self._logger.info("HCNetSDK 已清理")

    def __enter__(self) -> HikSdkContext:
        """上下文进入：初始化 SDK。"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """上下文退出：关闭 SDK。"""
        self.close(timeout=None)

    def acquire_session(self, session_id: str) -> SdkLease:
        """
        获取 SDK 活跃会话租约。

        Args:
            session_id: 会话唯一标识

        Returns:
            SdkLease 对象

        Raises:
            HikSdkError: SDK 已关闭时抛出
        """
        with self._session_cond:
            if self._closed:
                raise HikSdkError("无法获取会话：SDK 上下文已关闭")
            lease = SdkLease(session_id=session_id)
            self._active_sessions[session_id] = lease
            self._logger.debug(
                "会话租约已获取: %s (活跃=%d)", session_id, len(self._active_sessions)
            )
            return lease

    def release_session(self, lease: SdkLease) -> None:
        """
        释放会话租约。

        Args:
            lease: 待释放的租约
        """
        with self._session_cond:
            if lease.session_id in self._active_sessions:
                del self._active_sessions[lease.session_id]
                self._logger.debug(
                    "会话租约已释放: %s (活跃=%d)",
                    lease.session_id,
                    len(self._active_sessions),
                )
                self._session_cond.notify_all()

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """
        等待所有会话释放。

        Args:
            timeout: 最大等待秒数，None 表示不限

        Returns:
            全部释放返回 True，超时返回 False
        """
        with self._session_cond:
            deadline = None if timeout is None else time.time() + timeout
            while self._active_sessions or self._active_actors > 0:
                if timeout is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        self._logger.warning(
                            "等待会话释放超时 (active=%d, actors=%d)",
                            len(self._active_sessions),
                            self._active_actors,
                        )
                        return False
                    self._session_cond.wait(remaining)
                else:
                    self._session_cond.wait()
            return True

    def register_actor(self) -> None:
        """登记活跃的会话 Actor 线程。"""
        with self._session_cond:
            if self._closed:
                raise HikSdkError("无法注册 actor：SDK 上下文已关闭")
            self._active_actors += 1
            self._logger.debug("会话 actor 已注册 (actors=%d)", self._active_actors)

    def release_actor(self) -> None:
        """释放会话 Actor 线程计数。"""
        with self._session_cond:
            if self._active_actors > 0:
                self._active_actors -= 1
                self._logger.debug("会话 actor 已释放 (actors=%d)", self._active_actors)
            self._session_cond.notify_all()

    def close(self, timeout: float | None = None) -> None:
        """
        关闭 SDK 上下文，并等待会话释放。

        Args:
            timeout: 最大等待时间，None 表示不限
        """
        # 标记为关闭，阻止新会话进入
        with self._session_cond:
            if self._closed:
                return
            self._closed = True

        # 等待所有会话释放
        if not self.wait_for_idle(timeout):
            self._logger.warning(
                "关闭 SDK 时仍有 %d 个活动会话和 %d 个 actor",
                len(self._active_sessions),
                self._active_actors,
            )

        # 清理 DLL 目录与 SDK 资源
        self.cleanup()
        with self._init_lock:
            self._cleanup_dll_dirs()

    def _cleanup_dll_dirs(self) -> None:
        """关闭 DLL 目录句柄。"""
        for dll_dir in self._dll_dirs:
            try:
                # os._AddedDllDirectory 在 Python 3.8+ 提供 close
                close_method = getattr(dll_dir, "close", None)
                if close_method:
                    close_method()
            except Exception as e:
                self._logger.debug("关闭 DLL 目录句柄失败: %s", e)
        self._dll_dirs.clear()

    def _configure_playctrl_signatures(
        self, playm4_port_type: type[ctypes.c_int], decode_callback_type: type
    ) -> None:
        """配置 PlayCtrl 的端口相关 ABI。"""

        if self._playctrl is None:
            return

        bool_type = ctypes.c_int
        signatures = {
            "PlayM4_GetPort": (
                bool_type,
                [ctypes.POINTER(playm4_port_type)],
            ),
            "PlayM4_FreePort": (bool_type, [playm4_port_type]),
            "PlayM4_SetStreamOpenMode": (
                bool_type,
                [playm4_port_type, ctypes.c_uint],
            ),
            "PlayM4_OpenStream": (
                bool_type,
                [playm4_port_type, ctypes.c_char_p, ctypes.c_uint, ctypes.c_uint],
            ),
            "PlayM4_SetDecodeEngine": (
                bool_type,
                [playm4_port_type, ctypes.c_uint],
            ),
            "PlayM4_SetDecCallBackExMend": (
                bool_type,
                [
                    playm4_port_type,
                    decode_callback_type,
                    ctypes.c_void_p,
                    ctypes.c_uint,
                    ctypes.c_void_p,
                ],
            ),
            "PlayM4_Play": (
                bool_type,
                [playm4_port_type, ctypes.c_void_p],
            ),
            "PlayM4_InputData": (
                bool_type,
                [playm4_port_type, ctypes.c_char_p, ctypes.c_uint],
            ),
            "PlayM4_Stop": (bool_type, [playm4_port_type]),
            "PlayM4_CloseStream": (bool_type, [playm4_port_type]),
            "PlayM4_GetLastError": (ctypes.c_uint, [playm4_port_type]),
            "PlayM4_GetJPEG": (
                bool_type,
                [playm4_port_type, ctypes.c_char_p],
            ),
            "PlayM4_GetBMP": (
                bool_type,
                [playm4_port_type, ctypes.c_char_p],
            ),
        }

        for name, (restype, argtypes) in signatures.items():
            func = getattr(self._playctrl, name, None)
            if func is None:
                continue
            func.restype = restype
            func.argtypes = argtypes

    @staticmethod
    def _first_existing_path(*paths: Path) -> Path:
        for path in paths:
            resolved = path.resolve()
            if resolved.exists():
                return resolved
        return paths[0].resolve()

    def last_error(self) -> int:
        if self._sdk is None:
            return -1
        try:
            return int(self._sdk.NET_DVR_GetLastError())
        except Exception:
            return -1

    def last_error_detail(self) -> tuple[int, str]:
        """返回最近一次 SDK 调用的错误码与错误信息。"""
        if self._sdk is None:
            return -1, ""

        try:
            err = ctypes.c_long(0)
            msg = self._sdk.NET_DVR_GetErrorMsg(ctypes.byref(err))
            if msg:
                try:
                    msg_text = msg.decode(self._encoding, errors="replace")
                except Exception:
                    msg_text = str(msg)
            else:
                msg_text = ""
            return int(err.value), msg_text
        except Exception:
            return self.last_error(), ""
