"""手-丝杆关联测试。"""

from __future__ import annotations

from app.oilcheck.association import (
    HandScrewAssociator,
    ScrewCandidate,
    brush_work_point,
)
from app.oilcheck.config import DetectorConfig, PipelineConfig
from app.oilcheck.tracker import Track
from app.oilcheck.types import BBox

WIDTH = 1000
HEIGHT = 1000


def hand(track_id: int, cx: float, cy: float, matched: bool = True) -> Track:
    track = Track(
        track_id=track_id,
        box=BBox(cx - 45, cy - 45, cx + 45, cy + 45),
        last_ts=1.0,
        last_seen_ts=1.0,
    )
    track.matched = matched
    return track


def screw(identity: str, cx: float, cy: float = 200.0) -> ScrewCandidate:
    return ScrewCandidate(
        identity=identity,
        box=BBox(cx - 10, cy - 50, cx + 10, cy + 50),
    )


def make_assoc(**overrides) -> HandScrewAssociator:
    values = {
        "hand_screw_max_distance": 0.10,
        "pick_hand_distance": 0.05,
    }
    values.update(overrides)
    return HandScrewAssociator(cfg=PipelineConfig(**values))


def assign(assoc, *, hands, screws, frame_size=(WIDTH, HEIGHT)):
    return assoc.assign(hands=hands, screws=screws, frame_size=frame_size)


def test_single_hand_links_by_distance():
    assoc = make_assoc()
    result = assign(assoc, hands=(hand(1, 300, 200),), screws=(screw("A", 300),))
    assert len(result.links) == 1
    assert result.links[0].identity == "A"
    assert result.links[0].hand_track_id == 1
    assert result.by_hand(1) is not None
    assert result.identity_to_hand() == {"A": 1}


def test_link_requires_distance_within_gate():
    """超出 hand_screw_max_distance 的目标不建关联，即使不算太远。"""

    assoc = make_assoc(hand_screw_max_distance=0.05)
    # 归一化中心距离约 0.085 > 0.05
    result = assign(assoc, hands=(hand(1, 300, 200),), screws=(screw("A", 420),))
    assert result.links == ()


def test_two_hands_competing_for_one_screw_only_one_wins():
    assoc = make_assoc()
    result = assign(
        assoc,
        hands=(hand(1, 290, 200), hand(2, 312, 200)),
        screws=(screw("A", 300),),
    )
    assert len(result.links) == 1
    assert len({link.hand_track_id for link in result.links}) == 1


def test_two_hands_two_screws_are_independent():
    assoc = make_assoc()
    result = assign(
        assoc,
        hands=(hand(1, 300, 200), hand(2, 700, 200)),
        screws=(screw("A", 300), screw("B", 700)),
    )
    mapping = {link.identity: link.hand_track_id for link in result.links}
    assert mapping == {"A": 1, "B": 2}


def test_high_cost_returns_unmatched_instead_of_forcing():
    """代价整体偏高时应返回未匹配，不能为凑满分配强行配对。"""

    assoc = make_assoc(hand_screw_max_distance=0.05)
    result = assign(
        assoc,
        hands=(hand(1, 100, 100), hand(2, 900, 900)),
        screws=(screw("A", 500), screw("B", 520)),
    )
    assert result.links == ()


def test_unmatched_hand_is_not_reported():
    assoc = make_assoc()
    result = assign(
        assoc,
        hands=(hand(1, 300, 200), hand(2, 50, 950)),
        screws=(screw("A", 300),),
    )
    assert {link.hand_track_id for link in result.links} == {1}


def test_missing_hand_track_is_ignored():
    assoc = make_assoc()
    result = assign(
        assoc,
        hands=(hand(1, 300, 200, matched=False),),
        screws=(screw("A", 300),),
    )
    assert result.links == ()


def test_empty_inputs_return_no_links():
    assoc = make_assoc()
    assert assign(assoc, hands=(), screws=(screw("A", 300),)).links == ()
    assert assign(assoc, hands=(hand(1, 300, 200),), screws=()).links == ()


def test_assignment_is_recomputed_each_frame():
    """逐帧重算：手移到另一根丝杆身边时，关联跟着变。"""

    assoc = make_assoc()
    first = assign(
        assoc,
        hands=(hand(1, 300, 200),),
        screws=(screw("A", 300), screw("B", 330)),
    )
    assert first.links[0].identity == "A"

    second = assign(
        assoc,
        hands=(hand(1, 480, 200),),
        screws=(screw("A", 300), screw("B", 470)),
    )
    assert second.links[0].identity == "B"


def test_brush_work_point_uses_keypoint_then_offset():
    detector = DetectorConfig(brush_point_offset_x=0.5, brush_point_offset_y=0.9)
    plain = Track(
        track_id=1,
        box=BBox(390, 290, 410, 310),
        last_ts=1.0,
        last_seen_ts=1.0,
    )
    point = brush_work_point(plain, detector)
    assert point == (400.0, 300.0 + 0.4 * 20.0)

    keypoint = Track(
        track_id=2, box=BBox(390, 290, 410, 310), last_ts=1.0, last_seen_ts=1.0,
        point=(555.0, 666.0),
    )
    assert brush_work_point(keypoint, detector) == (555.0, 666.0)
