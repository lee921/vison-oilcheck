from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from .models import AppConfig, CameraConfig
from .paths import data_root, runtime_root

logger = logging.getLogger("oilcheck.config")


class ConfigService:
    def __init__(self, repo_root: Path | None = None) -> None:
        self._repo_root = (repo_root or data_root()).resolve()

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    @property
    def config_dir(self) -> Path:
        return self._repo_root / "config"

    @property
    def app_config_path(self) -> Path:
        return self.config_dir / "app.json"

    @property
    def cameras_path(self) -> Path:
        return self.config_dir / "cameras.json"

    def ensure_defaults(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)

        self._copy_missing_templates()

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

    def _copy_missing_templates(self) -> None:
        template_dir = runtime_root() / "config"
        if (
            not template_dir.exists()
            or template_dir.resolve() == self.config_dir.resolve()
        ):
            return

        for name in ("app.json", "cameras.json", "oilcheck.json"):
            self._copy_if_missing(template_dir / name, self.config_dir / name)

    @staticmethod
    def _copy_if_missing(src: Path, dst: Path) -> None:
        if dst.exists() or not src.exists():
            return
        try:
            shutil.copy2(src, dst)
        except Exception as e:
            logger.warning("复制默认配置失败 (%s -> %s): %s", src, dst, e)

    def load_app_config(self, *, create_if_missing: bool = False) -> AppConfig:
        if create_if_missing:
            self.ensure_defaults()

        if not self.app_config_path.exists():
            return AppConfig()

        try:
            with self.app_config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("加载应用配置失败 (%s): %s", self.app_config_path, e)
            return AppConfig()

        if not isinstance(data, dict):
            return AppConfig()
        return AppConfig.from_dict(dict(data))

    def save_app_config(self, cfg: AppConfig) -> None:
        self._write_json_atomic(self.app_config_path, cfg.to_dict())

    def load_cameras(self, *, create_if_missing: bool = False) -> list[CameraConfig]:
        if create_if_missing:
            self.ensure_defaults()

        if not self.cameras_path.exists():
            return []

        try:
            with self.cameras_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("加载摄像头配置失败 (%s): %s", self.cameras_path, e)
            return []

        if not isinstance(data, list):
            return []

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
            return os.environ.get(var_name, "")
        return value

    @staticmethod
    def with_resolved_password(cam: CameraConfig) -> CameraConfig:
        return replace(cam, password=ConfigService.resolve_password(cam.password))
