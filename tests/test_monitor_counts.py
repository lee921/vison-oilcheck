"""监控页计数与画面源下拉。"""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path

from app.oilcheck.config import PipelineConfig, StationConfig
from app.oilcheck.pipeline import StationPipeline
from app.oilcheck.types import BBox, Detection, FrameDetections, TrackSnapshot
from ui.pages.monitor_page import screws_in_roi, work_area_counts

WIDTH = 1000
HEIGHT = 1000


def _station() -> StationConfig:
    data = {
        "station_id": "ST01",
        "name": "测试工位",
        "camera_id": "cam_test",
        "enabled": True,
        "tray_roi": {"kind": "rect", "points": [[0.40, 0.02], [0.60, 0.52]]},
        # 工作追踪区：x 100–600、y 300–950
        "work_roi": {"kind": "rect", "points": [[0.10, 0.30], [0.60, 0.95]]},
        # 涂油区：x 100–400、y 500–900
        "oiling_roi": {"kind": "rect", "points": [[0.10, 0.50], [0.40, 0.90]]},
        "detector": {
            "kind": "null",
            "screw_label": "screw",
            "hand_label": "hand",
            "brush_label": "brush",
            "tray_label": "oil_station",
        },
    }
    station = StationConfig.from_dict(data)
    assert station is not None
    return station


def _track(track_id: int, label: str, cx: float, cy: float, *, observed=True):
    return TrackSnapshot(
        track_id=track_id,
        label=label,
        box=BBox(cx - 20, cy - 20, cx + 20, cy + 20),
        confidence=0.9,
        observed=observed,
        missing_s=0.0,
        last_seen_ts=1.0,
    )


def _snapshot(station, *, tracks=(), detections=(), raw_mode=False):
    pipeline = StationPipeline(
        station=station,
        pipeline_cfg=PipelineConfig(),
        on_event=None,
        cycle_id="T",
        detector_name="test",
    )
    snapshot = pipeline.process(
        detections=FrameDetections(
            ts=1.0,
            frame_size=(WIDTH, HEIGHT),
            detections=tuple(detections),
            source="test",
        )
    )
    # 直接替换成受控数据，避免经由管线时被 ROI 过滤掉一半。
    return replace(
        snapshot,
        tracks=tuple(tracks),
        detections=tuple(detections),
        stats={**snapshot.stats, "raw_mode": raw_mode},
    )


def _screw(cx: float, cy: float) -> Detection:
    return Detection(
        label="screw", confidence=0.9, box=BBox(cx - 10, cy - 40, cx + 10, cy + 40)
    )


def test_only_targets_inside_work_area_are_counted():
    station = _station()
    snapshot = _snapshot(
        station,
        tracks=(
            _track(1, "hand", 300, 600),  # 区内
            _track(2, "hand", 800, 750),  # 区外（x > 600）
            _track(3, "brush", 350, 620),  # 区内
        ),
        detections=(
            _screw(300, 600),  # 区内
            _screw(800, 600),  # 区外
            _screw(500, 100),  # 区外（y < 300，但可能在涂油盘区内）
        ),
    )
    counts, suffix = work_area_counts(station, snapshot)
    assert counts["hand"] == 1
    assert counts["brush"] == 1
    assert counts["screw"] == 1
    assert suffix == ""


def test_screws_are_not_fallen_back_to_full_frame_counts():
    """丝杆不建轨迹：绝不能退化成「全画面检测总数」。

    这是监控页曾经显示「丝杆 18 个」的原因——跟踪数为 0 时兜底用了
    label_counts（全画面），与“区内”完全不是一回事。
    """

    station = _station()
    outside = tuple(_screw(50.0 + i * 20.0, 100.0) for i in range(15))
    snapshot = _snapshot(station, detections=outside)
    counts, _ = work_area_counts(station, snapshot)
    assert counts["screw"] == 0, "区外丝杆不应计入区内"


def test_unobserved_tracks_are_not_counted():
    """丢失中的轨迹（matched=False）不算在区内。"""

    station = _station()
    snapshot = _snapshot(
        station,
        tracks=(
            _track(1, "hand", 300, 600, observed=True),
            _track(2, "hand", 320, 620, observed=False),
        ),
    )
    counts, _ = work_area_counts(station, snapshot)
    assert counts["hand"] == 1


def test_raw_mode_counts_full_frame_and_says_so():
    station = _station()
    snapshot = _snapshot(station, detections=(_screw(50.0, 100.0),), raw_mode=True)
    counts, suffix = work_area_counts(station, snapshot)
    assert counts["screw"] == 1
    assert "raw_mode" in suffix


def test_screws_in_oiling_roi_counts_only_screws_inside():
    """涂油区卡片显示区内丝杆数：只数中心落在涂油区里的，不数手/刷具。"""

    station = _station()
    roi = station.oiling_roi
    inside = roi.pixel_bbox(WIDTH, HEIGHT)
    snapshot = _snapshot(
        station,
        tracks=(_track(1, "hand", 300, 600),),
        detections=(
            _screw(inside.x1 + inside.width * 0.5, inside.y1 + inside.height * 0.5),
            _screw(inside.x1 + inside.width * 0.2, inside.y1 + inside.height * 0.2),
            _screw(500, 600),  # 工作追踪区内，但在涂油区外（x > 400）
        ),
    )
    assert screws_in_roi(station, snapshot, roi) == 2
    assert screws_in_roi(station, None, roi) == 0


# ----------------------------------------------------------------------
# 画面源下拉：配置里的示例视频清单变了要跟着重建
# ----------------------------------------------------------------------
def test_source_combo_follows_demo_video_changes():
    """服务层重载配置后，下拉要能出现新登记的示例视频。

    清单只在页面构建时读一次的话，运行中补上 ``demo.videos`` 后界面还是旧的，
    现场会以为「配置改了没用」。
    """

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:  # pragma: no cover - 无 Qt 环境
        return

    from app.models import AppConfig
    from app.oilcheck.config import DemoVideo, OilCheckConfig
    from app.oilcheck.events import EventStore
    from app.oilcheck.types import RecordingStatus
    from ui.pages.monitor_page import MonitorPage

    class _Service:
        def __init__(self) -> None:
            self.event_store = EventStore(Path(tempfile.mkdtemp()))
            self.videos: tuple[DemoVideo, ...] = ()
            self.config = OilCheckConfig(stations=(_station(),))

        def demo_videos(self):
            return self.videos

        def stations(self):
            return self.config.stations

        def station_snapshot(self, station_id):  # noqa: ANN001, ARG002
            return None

        def camera_state(self, camera_id):  # noqa: ANN001, ARG002
            return "missing"

        def active_video(self, station_id):  # noqa: ANN001, ARG002
            return None

        def demo_speed(self):
            return 1.0

        def latest_frame(self, camera_id):  # noqa: ANN001, ARG002
            return None

        def events(self, *_args, **_kwargs):
            return ()

        def recent_events(self, *_args, **_kwargs):
            return ()

        def recording_status(self, camera_id):
            return RecordingStatus(camera_id=camera_id, active=False)

        def video_status(self, station_id):  # noqa: ANN001, ARG002
            return {}

    class _Manager:
        def get_session(self, camera_id):  # noqa: ANN001, ARG002
            return None

    app = QApplication.instance() or QApplication([])
    page = MonitorPage(
        service=_Service(),
        manager=_Manager(),
        cfg_service=None,
        app_cfg=AppConfig(),
    )
    try:
        assert page._source_combo.count() == 1
        # 模拟服务层重载出两条示例视频，下一次刷新就应看到它们。
        page._service.videos = (
            DemoVideo(name="示例①", path="data/sample_video/a.mp4"),
            DemoVideo(name="示例②", path="data/sample_video/b.mp4"),
        )
        page._sync_sources()
        assert page._source_combo.count() == 3
        assert page._source_combo.itemText(1) == "示例①"

        # 清单没变时不要反复清空重建（会打断用户当前选择）。
        page._source_combo.setCurrentIndex(2)
        page._sync_sources()
        assert page._source_combo.currentIndex() == 2
    finally:
        page.deleteLater()
        app.processEvents()


def test_task_activity_label_shows_concrete_step():
    """任务卡要显示具体操作，而不是笼统的「进行中」。

    「进行中」看不出这一件是刚取完料还是已经在刷；卡片上按阶段细分，
    并把「取料完成」与「涂油进行中」区分开（两者都是 PICKED 阶段，
    区别在刷头接触累计是否大于 0）。
    """

    from app.oilcheck.types import TaskResult, TaskSnapshot, TaskStage

    base = TaskSnapshot(
        task_id="TASK-000001",
        created_at=0.0,
        stage=TaskStage.PICKED,
        stage_since=0.0,
        final_result=TaskResult.PROCESSING,
        brush_hold_target_s=1.0,
    )

    assert replace(base, stage=TaskStage.WAITING_PICK).activity_label == "等待取料"
    assert replace(base, brush_hold_s=0.0).activity_label == "取料完成"
    assert replace(base, brush_hold_s=0.42).activity_label == "涂油进行中"
    assert (
        replace(
            base,
            stage=TaskStage.OILED,
            final_result=TaskResult.COMPLETED,
            brush_hold_s=1.0,
        ).activity_label
        == "涂油完成"
    )
    # 漏涂仍显示 NG 文案，不能被「涂油完成」盖掉
    assert (
        replace(base, final_result=TaskResult.NG, brush_hold_s=0.2).activity_label
        == "NG（漏涂）"
    )
    # 等待取料 / 已完成 这两种状态不该出现笼统的「进行中」
    for stage in (TaskStage.WAITING_PICK, TaskStage.OILED):
        assert replace(base, stage=stage).activity_label != "进行中"


# ----------------------------------------------------------------------
# 事件表：最近事件分页浏览
# ----------------------------------------------------------------------
def test_event_table_pages_through_event_history():
    """事件表默认一页 50 条，翻页能看到最多 5000 条历史。

    之前只取 20 条且没有翻页入口，现场想回看稍早的记录只能去复核页翻。
    """

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:  # pragma: no cover - 无 Qt 环境
        return

    from app.models import AppConfig
    from app.oilcheck.config import OilCheckConfig
    from app.oilcheck.events import EVENT_HISTORY_LIMIT, EventStore
    from app.oilcheck.types import EventRecord, EventResult, EventType, RecordingStatus
    from ui.pages.monitor_page import EVENT_PAGE_SIZE, MonitorPage

    class _Service:
        def __init__(self) -> None:
            self.event_store = EventStore(Path(tempfile.mkdtemp()))
            self.config = OilCheckConfig(stations=(_station(),))
            ts = 1_700_000_000.0
            for _ in range(120):
                ts += 1.0
                self.event_store.add(
                    EventRecord.create(
                        timestamp_start=ts - 1.0,
                        timestamp_end=ts,
                        station_id="ST01",
                        camera_id="cam_test",
                        event_type=EventType.PICK,
                        result=EventResult.CONFIRMED,
                        task_id="TASK-000001",
                    )
                )

        def demo_videos(self):
            return ()

        def stations(self):
            return self.config.stations

        def station_snapshot(self, station_id):  # noqa: ANN001, ARG002
            return None

        def camera_state(self, camera_id):  # noqa: ANN001, ARG002
            return "missing"

        def active_video(self, station_id):  # noqa: ANN001, ARG002
            return None

        def demo_speed(self):
            return 1.0

        def latest_frame(self, camera_id):  # noqa: ANN001, ARG002
            return None

        def recent_events(self, limit=100, **_kwargs):
            return self.event_store.recent(limit)

        def recording_status(self, camera_id):
            return RecordingStatus(camera_id=camera_id, active=False)

        def video_status(self, station_id):  # noqa: ANN001, ARG002
            return {}

    class _Manager:
        def get_session(self, camera_id):  # noqa: ANN001, ARG002
            return None

    app = QApplication.instance() or QApplication([])
    page = MonitorPage(
        service=_Service(),
        manager=_Manager(),
        cfg_service=None,
        app_cfg=AppConfig(),
    )
    try:
        page._update_events("ST01")
        assert EVENT_PAGE_SIZE == 50
        assert page._event_table.rowCount() == EVENT_PAGE_SIZE
        assert page._btn_page_prev.isEnabled() is False
        assert page._btn_page_next.isEnabled() is True

        page._change_event_page(1)
        assert page._event_page == 1
        assert page._event_table.rowCount() == EVENT_PAGE_SIZE

        page._change_event_page(1)
        assert page._event_page == 2
        # 120 条，第三页只有 20 条，且不能再往后翻。
        assert page._event_table.rowCount() == 20
        assert page._btn_page_next.isEnabled() is False
        assert EVENT_HISTORY_LIMIT >= 5000
    finally:
        page.deleteLater()
        app.processEvents()
