from __future__ import annotations

from PySide6.QtCore import QTimer
from qfluentwidgets import FluentIcon, FluentWindow, NavigationItemPosition

from app.camera_manager import CameraManager
from app.config_service import ConfigService
from app.logging_service import LogStore
from app.models import AppConfig
from app.oilcheck.config import OilCheckConfig, OilCheckConfigService
from app.oilcheck.events import EventStore
from app.oilcheck.service import OilCheckService
from ui.pages.device_page import DevicePage
from ui.pages.logs_page import LogsPage
from ui.pages.monitor_page import MonitorPage
from ui.pages.recording_page import RecordingPage
from ui.pages.review_page import ReviewPage
from ui.pages.settings_page import SettingsPage


class MainWindow(FluentWindow):
    def __init__(
        self,
        *,
        cfg_service: ConfigService,
        app_cfg: AppConfig,
        manager: CameraManager,
        log_store: LogStore,
        oilcheck_cfg_service: OilCheckConfigService,
        oilcheck_cfg: OilCheckConfig,
        oilcheck_service: OilCheckService,
        event_store: EventStore,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("丝杆取料与二次涂油视觉检测系统")
        self.resize(1440, 900)

        self._cfg_service = cfg_service
        self._app_cfg = app_cfg
        self._manager = manager
        self._oilcheck_cfg_service = oilcheck_cfg_service
        self._oilcheck_cfg = oilcheck_cfg
        self._oilcheck_service = oilcheck_service
        self._event_store = event_store

        self._monitor_page = MonitorPage(
            service=self._oilcheck_service,
            manager=self._manager,
            cfg_service=self._cfg_service,
            app_cfg=self._app_cfg,
        )
        self._review_page = ReviewPage(event_store=self._event_store)
        self._recording_page = RecordingPage(
            service=self._oilcheck_service,
            manager=self._manager,
        )
        self._device_page = DevicePage(
            manager=self._manager,
            cfg_service=self._cfg_service,
            on_saved=self._on_cameras_saved,
        )
        self._settings_page = SettingsPage(
            cfg_service=self._cfg_service,
            app_cfg=self._app_cfg,
            manager=self._manager,
            oilcheck_cfg_service=self._oilcheck_cfg_service,
            oilcheck_cfg=self._oilcheck_cfg,
            on_saved=self._on_settings_saved,
            on_oilcheck_saved=self._on_oilcheck_saved,
        )
        log_dir = (self._cfg_service.repo_root / self._app_cfg.log_dir).resolve()
        self._logs_page = LogsPage(store=log_store, log_dir=log_dir)

        self.addSubInterface(
            self._monitor_page,
            FluentIcon.CAMERA,
            "实时监控",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._review_page,
            FluentIcon.HISTORY,
            "事件复核",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._recording_page,
            FluentIcon.VIDEO,
            "训练录制",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._device_page,
            FluentIcon.APPLICATION,
            "设备",
            position=NavigationItemPosition.TOP,
        )
        self.addSubInterface(
            self._settings_page,
            FluentIcon.SETTING,
            "设置",
            position=NavigationItemPosition.BOTTOM,
        )
        self.addSubInterface(
            self._logs_page,
            FluentIcon.DOCUMENT,
            "日志",
            position=NavigationItemPosition.BOTTOM,
        )

        self._monitor_page.reviewRequested.connect(self._open_review)
        self.switchTo(self._monitor_page)

        self._health_timer = QTimer(self)
        self._health_timer.setInterval(1000)
        self._health_timer.timeout.connect(self._manager.tick)
        self._health_timer.start()

    def _open_review(self, event_id: str) -> None:
        self._review_page.select_event(event_id)
        self.switchTo(self._review_page)

    def _on_cameras_saved(self) -> None:
        self._recording_page.reload_cameras()
        self._monitor_page.refresh_stations()

    def _on_settings_saved(self, app_cfg: AppConfig) -> None:
        self._app_cfg = app_cfg
        self._manager.update_app_config(app_cfg)
        self._monitor_page.set_app_config(app_cfg)
        self._settings_page.set_app_config(app_cfg)
        self._logs_page.set_log_dir(
            (self._cfg_service.repo_root / app_cfg.log_dir).resolve()
        )

    def _on_oilcheck_saved(self, cfg: OilCheckConfig) -> None:
        self._oilcheck_cfg = cfg
        self._oilcheck_service.reload(cfg)
        self._monitor_page.refresh_stations()
        self._recording_page.reload_cameras()
        self._settings_page.set_oilcheck_config(cfg)
