from __future__ import annotations

import signal
import sys

from PySide6.QtWidgets import QApplication
from qfluentwidgets import Theme, setTheme

from app.camera_manager import CameraManager
from app.config_service import ConfigService
from app.logging_service import FILE_ONLY, LogStore, reconfigure_log_dir, setup_logging
from app.models import AppConfig
from app.oilcheck.config import OilCheckConfig, OilCheckConfigService
from app.oilcheck.events import EventStore
from app.oilcheck.service import OilCheckService
from app.paths import data_root, runtime_root
from hik.sdk_context import HikSdkContext
from ui.main_window import MainWindow

APP_NAME = "oilcheck"


def main() -> int:
    app_root = runtime_root()
    storage_root = data_root()
    # 配置固定在仓库的 config/（见 app.paths.config_root），运行数据仍在
    # storage_root：日志、事件、截图、训练录制不随代码检出走。
    cfg_service = ConfigService(storage_root=storage_root)
    cfg_service.ensure_defaults()

    # 先按默认日志目录把日志系统装起来，再读 app.json。
    #
    # app.json 里的 log_dir 才是最终目录，但读取它本身可能失败（GBK 编码、
    # BOM、字段类型不对）。原先「先读配置、再 setup_logging」的顺序下，
    # 读取抛异常时进程在日志就绪之前就退出：双击启动只见窗口闪一下，
    # 日志目录里连一条记录都没有，现场无从排查。
    log_store = LogStore()
    logger = setup_logging(log_dir=(storage_root / "logs").resolve(), store=log_store)

    try:
        app_cfg = cfg_service.load_app_config()
    except Exception:
        # load_app_config 内部已把可预期错误降级为默认配置；这里是最后兜底。
        logger.exception("加载应用配置失败，使用默认配置启动")
        app_cfg = AppConfig()

    log_dir = (storage_root / app_cfg.log_dir).resolve()
    if log_dir != (storage_root / "logs").resolve():
        # 配置里的目录与默认值不同：日志系统此时已装配在默认目录上，
        # 换目录需要重建 handler。直接复用同一 LogStore 重建即可。
        logger = reconfigure_log_dir(logger, log_dir, log_store)

    # 旧配置提醒放到日志就绪之后再打，并且**只进日志文件**：``ensure_defaults()``
    # 在 setup_logging 之前跑，那时打出去只会被 Python 的 lastResort 甩到
    # 控制台；日志就绪后也不该占启动输出——现场启动时想看的是 Modbus 连接
    # 与模型加载。这条提示在日志页与日志文件里都能查到。
    orphaned = cfg_service.orphaned_config_warning()
    if orphaned:
        logger.warning(orphaned, extra={FILE_ONLY: True})

    app = QApplication(sys.argv)
    setTheme(Theme.DARK)

    # 不在启动时初始化海康 SDK：无相机/无 SDK 环境也能进入 UI 配置与演示流程。
    sdk_ctx = HikSdkContext(
        repo_root=app_root, log_dir=log_dir / "hik_sdk", logger=logger
    )
    manager = CameraManager(
        sdk_ctx=sdk_ctx, cfg_service=cfg_service, app_cfg=app_cfg, logger=logger
    )

    oil_cfg_service = OilCheckConfigService()
    oil_cfg_service.ensure_defaults()
    try:
        oil_cfg = oil_cfg_service.load()
    except Exception:
        # load() 现在对解析失败抛异常（不再静默退回「零工位」的空配置）。
        # 启动流程不能因此中断：退回默认配置并明确记一条 error，日志文件
        # 里能看到是哪个文件、哪一行坏掉了。
        logger.exception(
            "加载检测配置失败，使用默认配置启动（请检查 %s）",
            oil_cfg_service.config_path,
        )
        oil_cfg = OilCheckConfig()

    event_store = EventStore(storage_root / "events", logger_=logger.getChild("events"))
    event_store.load_recent(days=7)

    oil_service = OilCheckService(
        manager=manager,
        config_service=oil_cfg_service,
        config=oil_cfg,
        event_store=event_store,
        data_root=storage_root,
        logger_=logger,
    )
    oil_service.start()

    default_excepthook = sys.excepthook

    def _excepthook(exc_type, exc, tb) -> None:  # noqa: ANN001
        if exc_type is KeyboardInterrupt:
            try:
                logger.info("收到 KeyboardInterrupt，正在退出...")
            except Exception:
                pass
            try:
                app.quit()
            except Exception:
                pass
            return
        default_excepthook(exc_type, exc, tb)

    sys.excepthook = _excepthook

    def _on_signal(signum, frame) -> None:  # noqa: ANN001, ARG001
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = str(signum)
        logger.info("收到 %s，正在退出...", name)
        app.quit()

    # SIGTERM / SIGHUP 都要接：现场用 kill、systemd stop、关闭 SSH 终端
    # 终止进程都是这两者。原先只接 SIGINT，硬杀时 aboutToQuit 不触发，
    # _cleanup → oil_service.shutdown() → recorder.stop() 全都不执行，
    # 正在写的录制分段不会 release writer，mp4 没有 moov atom（实测只剩
    # 44 字节、无法播放），丢的是整个最多 5 分钟的分段。
    for _sig in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, _sig, None)
        if signum is None:
            continue
        try:
            signal.signal(signum, _on_signal)
        except (ValueError, OSError):  # 非主线程/平台不支持
            pass

    win = MainWindow(
        cfg_service=cfg_service,
        app_cfg=app_cfg,
        manager=manager,
        log_store=log_store,
        oilcheck_cfg_service=oil_cfg_service,
        oilcheck_cfg=oil_cfg,
        oilcheck_service=oil_service,
        event_store=event_store,
    )

    def _cleanup() -> None:
        try:
            oil_service.shutdown()
        except Exception:
            logger.exception("关闭检测服务失败")
        try:
            manager.shutdown()
        except Exception:
            logger.exception("关闭相机管理器失败")
        try:
            sdk_ctx.close(timeout=3.0)
        except Exception:
            logger.exception("关闭海康 SDK 失败")

    app.aboutToQuit.connect(_cleanup)
    win.showFullScreen()
    try:
        return app.exec()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0) from None
