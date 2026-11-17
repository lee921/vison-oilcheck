from __future__ import annotations

from app.oilcheck.tracker import ObjectTracker
from app.oilcheck.types import BBox, Detection

WIDTH = 1000
HEIGHT = 1000


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
