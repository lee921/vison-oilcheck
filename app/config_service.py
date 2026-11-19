from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from .models import AppConfig, CameraConfig
from .paths import config_root, data_root

logger = logging.getLogger("oilcheck.config")


class CamerasConfigError(RuntimeError):
    """``cameras.json`` 存在但读不出来（语法错误、编码不对、顶层不是数组）。

    抛异常而不是返回空列表：设备页拿到 ``[]`` 会以为是「用户删光了所有
    相机」，下一次增删改就把 ``[]`` 写回磁盘，现场两台相机的 IP、账号、
    口令全丢。调用方要么让用户先修文件，要么明确提示读取失败。
    """

    def __init__(self, path: Path, error: BaseException) -> None:
        super().__init__(f"摄像头配置无法读取（{path}）：{error}")
        self.path = path
        self.error = error


def read_config_text(path: Path) -> str:
    """按 UTF-8（含 BOM）读取配置文本，失败时回退 GB18030。

    中文 Windows 记事本「另存为 ANSI」产出的是 GBK；带 BOM 的 UTF-8
    也常见。两种都不能让程序崩在启动阶段（那时日志系统还没就绪）。
    """

    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gb18030")


class ConfigService:
    """读写仓库 ``config/`` 下的 app.json / cameras.json。

    配置目录（``config_dir``）与运行数据目录（``data_root``）是两个概念：
    前者固定在仓库里，UI 改的值、代码读的值、仓库里的文件永远是同一份；
    后者只放日志、截图、事件、训练录制。
    """

    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        storage_root: Path | None = None,
    ) -> None:
        if repo_root is None:
            self._config_dir = config_root()
            self._data_root = (storage_root or data_root()).resolve()
        else:
            # 显式传入 repo_root 时沿用旧语义：它同时是配置根与数据根。
            # 测试与离线工具靠这个指向临时目录。
            self._config_dir = (Path(repo_root) / "config").resolve()
            self._data_root = Path(repo_root).resolve()

    @property
    def repo_root(self) -> Path:
        """运行数据目录。保留旧名字，避免 UI 侧大改。"""

        return self._data_root

    @property
    def data_root(self) -> Path:
        return self._data_root

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    @property
    def app_config_path(self) -> Path:
        return self.config_dir / "app.json"

    @property
    def cameras_path(self) -> Path:
        return self.config_dir / "cameras.json"

    def ensure_defaults(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # 旧配置提醒**不在这里**打印：``ensure_defaults`` 在 ``setup_logging``
        # 之前被调用，此时 logger 还没有 handler，Python 的 lastResort 会把它
        # 直接甩到控制台，而日志文件里反而没有——现场看到的是控制台上一句
        # 孤零零的「发现旧配置」，把真正关心的启动信息盖掉了。
        # 改由 :meth:`orphaned_config_warning` 取文本、``main`` 在日志就绪后
        # 写进日志。

        if not self.app_config_path.exists():
            self.save_app_config(AppConfig(layout=1, snapshot_dir="snapshots"))

        if not self.cameras_path.exists():
            self.save_cameras(
                [
                    CameraConfig.from_dict(
                        {
                            "id": "cam_6ca47a",
                            "name": "1号位",
                            "enabled": True,
                            "ip": "192.168.1.64",
                            "port": 8000,
                            "username": "admin",
                            "password": "env:OILCHECK_CAM_PASSWORD",
                            "channel": 1,
                            "stream_type": "main",
                            "preferred_mode": "decode",
                            "reconnect": {
                                "enable": True,
                                "max_backoff_s": 30,
                                "offline_timeout_s": 10,
                            },
                        }
                    )
                ]
            )

    def orphaned_config_warning(self) -> str:
        """数据目录里若还留着旧配置，返回一条提醒文本；没有则返回空串。

        旧版本把 app.json / cameras.json / oilcheck.json 写在数据目录，
        升级后程序只读仓库 config/，数据目录那份不再生效。现场最容易踩的坑
        是：相机 IP、账号、ROI 都在旧文件里，升级后读的是仓库里的模板默认值，
        表现为「相机突然连不上」「ROI 全变了」，但配置文件看上去「还在」。

        只返回文本、不直接打日志：调用方（``main``）在日志系统就绪之后再写，
        这样它会进日志文件，也由调用方决定是否放到控制台。
        """

        orphan_dir = self.data_root / "config"
        try:
            if orphan_dir.resolve() == self.config_dir.resolve():
                return ""
        except OSError:
            return ""
        stale = [
            name
            for name in ("app.json", "cameras.json", "oilcheck.json")
            if (orphan_dir / name).exists()
        ]
        if not stale:
            return ""
        return (
            f"数据目录里发现旧配置，已不再生效：{orphan_dir}"
            f"（{', '.join(stale)}）。配置现在只读写仓库 config/："
            f"{self.config_dir}。若其中的相机/ROI 需要保留，"
            f"请把对应文件拷到 {self.config_dir} 后重启。"
        )

    def load_app_config(self, *, create_if_missing: bool = False) -> AppConfig:
        if create_if_missing:
            self.ensure_defaults()

        if not self.app_config_path.exists():
            return AppConfig()

        try:
            # utf-8-sig 兼容 Windows 记事本另存产生的 BOM；编码本身不是
            # UTF-8（中文记事本「另存为 ANSI」= GBK）时依次回退 gb18030。
            # 原先这里只写 encoding="utf-8"，UnicodeDecodeError（ValueError
            # 子类）不在 except 里，会一路冒到 main.py 的 setup_logging 之前，
            # 表现为「窗口闪一下就没」且日志里什么都没有。
            text = read_config_text(self.app_config_path)
            data = json.loads(text)
        except (OSError, ValueError) as e:
            logger.error("加载应用配置失败 (%s): %s", self.app_config_path, e)
            return AppConfig()

        if not isinstance(data, dict):
            return AppConfig()
        # from_dict 已对每个字段做严格解析（非法值退回默认并记 warning），
        # 这里不再可能因字段类型抛异常。
        return AppConfig.from_dict(dict(data))

    @staticmethod
    def _read_json(path: Path) -> Any:
        return json.loads(read_config_text(path))

    def save_app_config(self, cfg: AppConfig) -> None:
        self._write_json_atomic(self.app_config_path, cfg.to_dict())

    def load_cameras(self, *, create_if_missing: bool = False) -> list[CameraConfig]:
        if create_if_missing:
            self.ensure_defaults()

        if not self.cameras_path.exists():
            return []

        try:
            data = self._read_json(self.cameras_path)
        except (OSError, ValueError) as e:
            logger.error("加载摄像头配置失败 (%s): %s", self.cameras_path, e)
            # 解析失败必须让调用方知道：``[]`` 会被设备页当成「用户删光了
            # 所有相机」，下一次增删改直接写回 ``[]``，两台相机的 IP、账号、
            # 口令全部丢失（R6）。
            raise CamerasConfigError(self.cameras_path, e) from e

        if not isinstance(data, list):
            raise CamerasConfigError(
                self.cameras_path, ValueError("顶层不是数组")
            )

        cameras: list[CameraConfig] = []
        for item in data:
            if isinstance(item, dict):
                cam = CameraConfig.from_dict(item)
                if cam.id:
                    cameras.append(cam)
        return cameras

    def save_cameras(self, cameras: Iterable[CameraConfig]) -> None:
        payload = [c.to_dict() for c in cameras]
        self._write_json_atomic(self.cameras_path, payload)

    @staticmethod
    def _write_json_atomic(path: Path, payload: object) -> None:
        # 通过临时文件 + 原子替换，避免写入中断导致配置损坏
        path.parent.mkdir(parents=True, exist_ok=True)

        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            Path(tmp_name).replace(path)
        finally:
            try:
                os.remove(tmp_name)
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.debug("删除临时文件失败 (%s): %s", tmp_name, e)

    @staticmethod
    def resolve_password(value: str) -> str:
        value = value.strip()
        # 支持 env: 前缀从环境变量读取密码
        if value.lower().startswith("env:"):
            var_name = value[4:].strip()
            if not var_name:
                raise ValueError("密码配置写成了 ``env:`` 但没给变量名")
            resolved = os.environ.get(var_name)
            if not resolved:
                # 用空口令去登录会得到海康的「用户名或密码错误」，现场会去
                # 查 IP 和账号，联想不到是环境变量没设（桌面启动器 / systemd
                # 不继承 shell 的环境变量时很常见）。
                raise ValueError(
                    f"环境变量 {var_name} 未设置或为空（相机密码引用自 env:{var_name}）"
                )
            return resolved
        return value

    @staticmethod
    def with_resolved_password(cam: CameraConfig) -> CameraConfig:
        return replace(cam, password=ConfigService.resolve_password(cam.password))
