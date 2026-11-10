"""监控页「工作追踪区」卡片计数：只算区内的目标。"""

from __future__ import annotations

from dataclasses import replace

from app.oilcheck.config import PipelineConfig, StationConfig
from app.oilcheck.pipeline import StationPipeline
from app.oilcheck.types import BBox, Detection, FrameDetections, TrackSnapshot
from ui.pages.monitor_page import work_area_counts

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
