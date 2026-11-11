from __future__ import annotations

import os
import sys
from pathlib import Path


def find_repo_root(start: Path) -> Path:
    """向上查找直到发现 `pyproject.toml` 的仓库根目录。"""
    start = start.resolve()
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").exists():
            return path
    return start


def runtime_root() -> Path:
    """返回运行时根目录；支持 PyInstaller 的 `sys._MEIPASS`。"""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return find_repo_root(Path(__file__).resolve().parent)


def resolve_runtime_file(raw_path: str, *, repo_root: Path | None = None) -> str:
    """按运行目录、仓库目录、当前目录顺序解析模型等运行时文件。"""
    path = Path(str(raw_path).strip())
    if path.is_absolute():
        return str(path)

    bases = [runtime_root()]
    if repo_root is not None:
        bases.append(repo_root)
    bases.append(Path.cwd())

    for base in bases:
        candidate = base / path
        if candidate.exists():
            return str(candidate.resolve())
    return str(path)


def is_frozen() -> bool:
    """当运行于打包后的应用（如 PyInstaller）时返回 True。"""
    return bool(getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"))


def config_root() -> Path:
    """返回配置文件目录：仓库根下的 ``config/``。

    配置（app.json / cameras.json / oilcheck.json）固定放在仓库的 ``config/``，
    与代码同处一个检出里：UI 里改的值、判定实际读的值、仓库里的文件三者
    永远是同一份。日志、事件、截图、训练录制等运行数据仍写在
    :func:`data_root`，不随配置走。

    覆盖：设置 ``OILCHECK_CONFIG_ROOT``（支持绝对或相对路径）。测试与
    离线工具用它指向临时目录。
    """
    override = os.environ.get("OILCHECK_CONFIG_ROOT")
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    return runtime_root() / "config"


def user_data_root(app_name: str = "OilCheckData") -> Path:
    """返回用户级持久化数据目录。

    统一放在用户主目录下的 ``OilCheckData``，便于现场备份、迁移和查看；
    不再写入 ``~/.config`` 或系统应用数据目录。
    """
    return Path.home() / app_name


def data_root(app_name: str = "OilCheckData") -> Path:
    """返回日志/截图/事件/录制等运行数据的持久化根目录。

    注意：**配置不在这里**，见 :func:`config_root`。

    默认：用户级数据目录（开发运行也使用该路径）。
    覆盖：设置 `OILCHECK_DATA_ROOT`（支持绝对或相对路径）。
    兼容：旧项目的 `VISION_CRAFT_DATA_ROOT` 仍可作为后备变量。
    """
    override = os.environ.get("OILCHECK_DATA_ROOT") or os.environ.get(
        "VISION_CRAFT_DATA_ROOT"
    )
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    return user_data_root(app_name)
