from __future__ import annotations

from app.oilcheck.tracker import ObjectTracker
from app.oilcheck.types import BBox, Detection


def _screw(cx: float, cy: float = 350.0) -> Detection:
    return Detection(
        label="screw",
        confidence=0.9,
        box=BBox(cx - 20, cy - 60, cx + 20, cy + 60),
        angle=1.5707963267948966,
    )


def test_tracker_keeps_id_for_continuously_moving_screw():
    tracker = ObjectTracker(
        max_distance=0.1,
        max_missing_s=0.5,
        min_hits=2,
        max_active=8,
    )
    ids = set()
    ts = 1.0
    for step in range(10):
        ts += 0.1
        output = tracker.update(
            ts=ts,
            detections=(_screw(300 + step * 30),),
            frame_size=(1000, 1000),
        )
        assert len(output.active) == 1
        ids.add(output.active[0].track_id)

    assert ids == {1}
    assert output.active[0].angle is not None


def test_tracker_tolerates_brief_miss_and_reports_loss():
    tracker = ObjectTracker(
        max_distance=0.1,
        max_missing_s=0.4,
        min_hits=1,
        max_active=8,
    )
    tracker.update(ts=1.0, detections=(_screw(400),), frame_size=(1000, 1000))

    brief = tracker.update(ts=1.2, detections=(), frame_size=(1000, 1000))
    assert len(brief.active) == 1
    assert brief.active[0].missing_s > 0

    lost = tracker.update(ts=1.6, detections=(), frame_size=(1000, 1000))
    assert not lost.active
    assert len(lost.lost) == 1


def test_tracker_tracks_two_screws_independently():
    tracker = ObjectTracker(
        max_distance=0.1,
        max_missing_s=0.5,
        min_hits=1,
        max_active=8,
    )
    output = tracker.update(
        ts=1.0,
        detections=(_screw(300), _screw(600)),
        frame_size=(1000, 1000),
    )
    first_ids = {track.track_id for track in output.active}
    assert len(first_ids) == 2

    output = tracker.update(
        ts=1.1,
        detections=(_screw(330), _screw(630)),
        frame_size=(1000, 1000),
    )
    assert {track.track_id for track in output.active} == first_ids
