from __future__ import annotations

from app.oilcheck.tracker import ObjectTracker, TrayTracker
from app.oilcheck.types import BBox, Detection

WIDTH = 1000
HEIGHT = 1000


def tray(box: BBox) -> Detection:
    return Detection(label="oil_tray", confidence=0.95, box=box)


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
