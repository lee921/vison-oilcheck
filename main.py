from __future__ import annotations

import signal
import sys

from PySide6.QtWidgets import QApplication
from qfluentwidgets import Theme, setTheme

from app.camera_manager import CameraManager
from app.config_service import ConfigService
from app.logging_service import LogStore, setup_logging
from app.oilcheck.config import OilCheckConfigService
from app.oilcheck.events import EventStore
from app.oilcheck.service import OilCheckService
from app.paths import data_root, runtime_root
from hik.sdk_context import HikSdkContext
from ui.main_window import MainWindow

APP_NAME = "oilcheck"


def main() -> int:
    app_root = runtime_root()
    storage_root = data_root()
    cfg_service = ConfigService(repo_root=storage_root)
    cfg_service.ensure_defaults()
    app_cfg = cfg_service.load_app_config()

    log_store = LogStore()
    log_dir = (storage_root / app_cfg.log_dir).resolve()
    logger = setup_logging(log_dir=log_dir, store=log_store)

    app = QApplication(sys.argv)
    setTheme(Theme.DARK)

    # 不在启动时初始化海康 SDK：无相机/无 SDK 环境也能进入 UI 配置与演示流程。
    sdk_ctx = HikSdkContext(
        repo_root=app_root, log_dir=log_dir / "hik_sdk", logger=logger
    )
    manager = CameraManager(
        sdk_ctx=sdk_ctx, cfg_service=cfg_service, app_cfg=app_cfg, logger=logger
    )

    oil_cfg_service = OilCheckConfigService(storage_root)
    oil_cfg_service.ensure_defaults()
    oil_cfg = oil_cfg_service.load()

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

    def _on_sigint(signum, frame) -> None:  # noqa: ANN001, ARG001
        logger.info("收到 SIGINT，正在退出...")
        app.quit()

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except Exception:
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
