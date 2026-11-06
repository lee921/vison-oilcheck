from __future__ import annotations

from itertools import count

from app.oilcheck.config import PipelineConfig, StationConfig, default_slot_layouts
from app.oilcheck.pipeline import StationPipeline
from app.oilcheck.roi import Roi
from app.oilcheck.tracker import ObjectTracker, TrayTracker
from app.oilcheck.types import BBox, Detection, FrameDetections, SlotOccupancy

WIDTH = 1000
HEIGHT = 1000


def _rect(x1, y1, x2, y2) -> Roi:
    roi = Roi.from_dict({"kind": "rect", "points": [[x1, y1], [x2, y2]]})
    assert roi is not None
    return roi


def make_station() -> StationConfig:
    station = StationConfig.from_dict(
        {
            "station_id": "ST01",
            "name": "移动盘工位",
            "camera_id": "cam_test",
            "enabled": True,
            "slot_mode": "tray",
            "pick_roi": {"kind": "rect", "points": [[0.10, 0.05], [0.90, 0.50]]},
            "oiling_roi": {"kind": "rect", "points": [[0.10, 0.65], [0.60, 0.95]]},
            "slot_layout": [layout.to_dict() for layout in default_slot_layouts(4)],
        }
    )
    assert station is not None
    return station


def make_pipeline() -> StationPipeline:
    station = make_station()
    return StationPipeline(
        station=station,
        pipeline_cfg=PipelineConfig(
            slot_occupied_confirm_s=0.1,
            slot_empty_confirm_s=0.2,
            event_cooldown_s=0.0,
        ),
        detector_name="test",
    )


def tray(box: BBox) -> Detection:
    return Detection(label="tray", confidence=0.95, box=box)


def screws_for(station: StationConfig, tray_box: BBox) -> tuple[Detection, ...]:
    items = []
    for layout in station.slot_layout:
        box = layout.pixel_bbox(tray_box)
        cx, cy = box.center
        items.append(
            Detection(
                label="rod",
                confidence=0.9,
                box=BBox(cx - 12, cy - 40, cx + 12, cy + 40),
            )
        )
    return tuple(items)


def feed(pipeline: StationPipeline, ts: float, *detections: Detection):
    return pipeline.process(
        detections=FrameDetections(
            ts=ts,
            frame_size=(WIDTH, HEIGHT),
            detections=tuple(detections),
            source="test",
        )
    )


def test_slot_state_survives_vertical_tray_motion():
    pipeline = make_pipeline()
    station = pipeline.station
    ts_iter = count(1.0, 0.1)

    first = BBox(200, 100, 800, 400)
    for _ in range(4):
        feed(pipeline, next(ts_iter), tray(first), *screws_for(station, first))
    assert all(slot.occupancy == SlotOccupancy.OCCUPIED for slot in pipeline.slots)

    # 盘体整体下移 60px，槽位 ROI 随盘体移动，但占用状态不能跳变。
    second = BBox(200, 160, 800, 460)
    for _ in range(6):
        snapshot = feed(
            pipeline, next(ts_iter), tray(second), *screws_for(station, second)
        )
        assert all(slot.occupancy == SlotOccupancy.OCCUPIED for slot in snapshot.slots)


def test_tray_tracker_reports_missing_and_reliability():
    tracker = TrayTracker(max_distance=0.2, max_missing_s=0.5, min_confidence=0.3)
    first = tracker.update(
        ts=1.0,
        detections=(tray(BBox(200, 100, 800, 400)),),
        frame_size=(WIDTH, HEIGHT),
    )
    assert first.matched and first.reliable and first.track_id == 1

    moved = tracker.update(
        ts=1.1,
        detections=(tray(BBox(200, 150, 800, 450)),),
        frame_size=(WIDTH, HEIGHT),
    )
    assert moved.track_id == 1
    assert moved.box is not None and moved.box.y1 == 150

    brief = tracker.update(ts=1.3, detections=(), frame_size=(WIDTH, HEIGHT))
    assert not brief.matched and not brief.reliable
    assert brief.box is not None

    lost = tracker.update(ts=2.0, detections=(), frame_size=(WIDTH, HEIGHT))
    assert lost.box is None


def test_object_tracker_keeps_hand_id_and_tolerates_brief_loss():
    tracker = ObjectTracker(
        max_distance=0.15, max_missing_s=0.4, min_hits=1, max_active=4
    )
    ids = set()
    ts = 1.0
    for step in range(6):
        ts += 0.1
        output = tracker.update(
            ts=ts,
            detections=(
                Detection(
                    label="hand",
                    confidence=0.9,
                    box=BBox(300 + step * 20, 300, 400 + step * 20, 400),
                ),
            ),
            frame_size=(WIDTH, HEIGHT),
        )
        ids.update(track.track_id for track in output.active)
    assert ids == {1}

    brief = tracker.update(ts=ts + 0.2, detections=(), frame_size=(WIDTH, HEIGHT))
    assert brief.active and brief.active[0].missing_s > 0

    lost = tracker.update(ts=ts + 0.6, detections=(), frame_size=(WIDTH, HEIGHT))
    assert not lost.active and lost.lost
