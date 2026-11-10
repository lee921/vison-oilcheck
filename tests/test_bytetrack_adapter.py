"""ByteTrack 适配器测试：契约兼容、快速目标保 ID、门控与回退。"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from app.oilcheck.bytetrack_adapter import ByteTrackObjectTracker
from app.oilcheck.types import BBox, Detection

pytest.importorskip("ultralytics")


def _det(cx: float, cy: float, size: float = 20.0, conf: float = 0.9) -> Detection:
    half = size / 2
    return Detection(
        label="hand",
        confidence=conf,
        box=BBox(cx - half, cy - half, cx + half, cy + half),
    )


def _feed(tracker, frames, *, step_s: float = 0.1):
    out = None
    for index, dets in enumerate(frames):
        out = tracker.update(
            ts=1.0 + index * step_s, detections=dets, frame_size=(1000, 1000)
        )
    return out


def test_fast_target_keeps_id():
    """裸 ByteTrack 只按 IoU 关联，快目标会丢 ID；适配器的中心距离代价要保住它。"""

    tracker = ByteTrackObjectTracker(max_distance=0.15)
    frames = [(_det(100 + i * 120, 300),) for i in range(6)]
    ids = set()
    for index, dets in enumerate(frames):
        out = tracker.update(
            ts=1.0 + index * 0.1, detections=dets, frame_size=(1000, 1000)
        )
        assert len(out.active) == 1
        ids.add(out.active[0].track_id)
    assert len(ids) == 1


def test_two_targets_stay_independent():
    tracker = ByteTrackObjectTracker(max_distance=0.15)
    frames = []
    for i in range(5):
        frames.append((_det(100 + i * 100, 300), _det(600 + i * 60, 500, size=40)))
    seen = []
    for index, dets in enumerate(frames):
        out = tracker.update(
            ts=1.0 + index * 0.1, detections=dets, frame_size=(1000, 1000)
        )
        seen.append({t.track_id for t in out.active})
    assert seen[-1] == seen[-2] == {min(seen[0]), max(seen[0])}
    assert len(seen[-1]) == 2


def test_brief_miss_stays_active_then_reports_loss():
    tracker = ByteTrackObjectTracker(max_distance=0.15, max_missing_s=0.4)
    _feed(tracker, [(_det(400, 300),)])

    brief = tracker.update(
        ts=1.4, detections=(), frame_size=(1000, 1000)
    )
    assert len(brief.active) == 1
    assert brief.active[0].matched is False
    assert brief.active[0].missing_s > 0

    lost = tracker.update(ts=2.0, detections=(), frame_size=(1000, 1000))
    assert not lost.active
    assert len(lost.lost) == 1


def test_empty_frame_is_safe():
    tracker = ByteTrackObjectTracker(max_distance=0.15)
    out = tracker.update(ts=1.0, detections=(), frame_size=(1000, 1000))
    assert out.active == ()
    assert out.lost == ()


def test_angle_point_and_confidence_come_from_detection():
    tracker = ByteTrackObjectTracker(max_distance=0.15)
    det = Detection(
        label="brush",
        confidence=0.77,
        box=BBox(90, 290, 110, 310),
        angle=0.5,
        point=(100.0, 305.0),
    )
    track = None
    for index in range(2):
        out = tracker.update(
            ts=1.0 + index * 0.1, detections=(det,), frame_size=(1000, 1000)
        )
        track = out.active[0]
    assert track.confidence == pytest.approx(0.77)
    assert track.angle == pytest.approx(0.5)
    assert track.point == pytest.approx((100.0, 305.0))
    assert track.label if hasattr(track, "label") else True


def test_id_base_namespacing_and_reset():
    hand = ByteTrackObjectTracker(id_base=100_000)
    brush = ByteTrackObjectTracker(id_base=200_000)
    for index in range(2):
        h = hand.update(
            ts=1.0 + index * 0.1, detections=(_det(100, 100),), frame_size=(1000, 1000)
        )
        b = brush.update(
            ts=1.0 + index * 0.1, detections=(_det(500, 500),), frame_size=(1000, 1000)
        )
    assert h.active[0].track_id >= 100_000
    assert b.active[0].track_id >= 200_000
    assert h.active[0].track_id != b.active[0].track_id

    hand.reset()
    after = hand.update(
        ts=9.0, detections=(_det(100, 100),), frame_size=(1000, 1000)
    )
    assert after.active[0].track_id >= 100_000


def test_distance_gate_rejects_far_match():
    """超出中心距离上限的目标不应被接到旧轨迹上（旧轨迹 id 不得延续）。"""

    tracker = ByteTrackObjectTracker(max_distance=0.05)
    _feed(tracker, [(_det(100, 300),)] * 2, step_s=0.1)
    first = tracker.active_tracks()[0].track_id

    far_frames = [
        (_det(900, 900),),
        (_det(900, 900),),
        (_det(900, 900),),
    ]
    matched_ids: set[int] = set()
    for index, dets in enumerate(far_frames):
        out = tracker.update(
            ts=1.5 + index * 0.1, detections=dets, frame_size=(1000, 1000)
        )
        matched_ids = {t.track_id for t in out.active if t.matched}
    # 远处的目标必须拿到新 id，不能复用旧轨迹 id（旧轨迹仍在 active 但未匹配）。
    assert matched_ids
    assert first not in matched_ids


def test_max_active_cap():
    tracker = ByteTrackObjectTracker(max_distance=0.2, max_active=2)
    dets = tuple(_det(200 + i * 300, 300) for i in range(4))
    for index in range(3):
        out = tracker.update(
            ts=1.0 + index * 0.1, detections=dets, frame_size=(1000, 1000)
        )
    assert len(out.active) <= 2


def test_low_confidence_second_stage_keeps_id():
    """第二阶段低置信度关联是源码纯 IoU，改写 get_dists 不应破坏它。"""

    tracker = ByteTrackObjectTracker(
        max_distance=0.15, high_thresh=0.3, low_thresh=0.1
    )
    frames = [
        (_det(100, 300, conf=0.9),),
        (_det(100, 300, conf=0.9),),
        (_det(104, 300, conf=0.15),),
        (_det(108, 300, conf=0.15),),
        (_det(112, 300, conf=0.9),),
    ]
    ids = set()
    for index, dets in enumerate(frames):
        out = tracker.update(
            ts=1.0 + index * 0.1, detections=dets, frame_size=(1000, 1000)
        )
        if out.active:
            ids.add(out.active[0].track_id)
    assert len(ids) == 1


def test_object_backend_fallback_keeps_contract():
    tracker = ByteTrackObjectTracker(
        id_base=300_000, backend="object", max_distance=0.15
    )
    assert tracker.backend == "object"
    out = _feed(tracker, [(_det(100, 300),)] * 2)
    assert len(out.active) == 1
    assert out.active[0].track_id >= 300_000
    assert tracker.get(out.active[0].track_id) is not None


def test_get_returns_unmatched_track():
    tracker = ByteTrackObjectTracker(max_distance=0.15, max_missing_s=0.5)
    _feed(tracker, [(_det(100, 300),)] * 2)
    track_id = tracker.active_tracks()[0].track_id

    out = tracker.update(ts=1.3, detections=(), frame_size=(1000, 1000))
    assert track_id in {t.track_id for t in out.active}
    track = tracker.get(track_id)
    assert track is not None
    assert track.matched is False


# ----------------------------------------------------------------------
# lap 缺失兜底
# ----------------------------------------------------------------------
def _block_lap(monkeypatch) -> None:
    """模拟「没装 lap」：清缓存并让后续 import 抛 ImportError。"""

    import sys

    class _Block:
        def find_spec(self, name, path=None, target=None):
            if name == "lap" or name.startswith("lap."):
                raise ImportError("No module named 'lap' (blocked by test)")
            return None

    monkeypatch.delitem(sys.modules, "lap", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_Block(), *sys.meta_path])


def test_lap_fallback_produces_optimal_assignment(monkeypatch):
    """没有 lap 时注入的 lapjv 与真实实现语义一致：最小代价一一匹配。"""

    import sys

    from app.oilcheck.bytetrack_adapter import _ensure_lap_available

    _block_lap(monkeypatch)
    _ensure_lap_available()
    lap = sys.modules["lap"]
    assert getattr(lap, "__version__", "").endswith("+oilcheck-fallback")

    # 3x3 代价矩阵：最优分配是 (0→1, 1→0, 2→2)，总代价 1+1+1=3。
    cost = np.array(
        [[4.0, 1.0, 9.0], [1.0, 5.0, 8.0], [7.0, 6.0, 1.0]], dtype=np.float64
    )
    total, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=1e9)
    assert total == pytest.approx(3.0)
    assert list(x) == [1, 0, 2]
    assert list(y) == [1, 0, 2]


def test_lap_fallback_respects_cost_limit(monkeypatch):
    """超过 cost_limit 的匹配必须留作未匹配（返回 -1）。"""

    import sys

    from app.oilcheck.bytetrack_adapter import _ensure_lap_available

    _block_lap(monkeypatch)
    _ensure_lap_available()
    lap = sys.modules["lap"]

    cost = np.array([[0.1, 9.0], [9.0, 0.2]], dtype=np.float64)
    _, x, y = lap.lapjv(cost, extend_cost=True, cost_limit=0.9)
    assert list(x) == [0, 1]
    assert list(y) == [0, 1]

    # 门限收紧到 0.05 后两个都不该匹配。
    _, x2, y2 = lap.lapjv(cost, extend_cost=True, cost_limit=0.05)
    assert list(x2) == [-1, -1]
    assert list(y2) == [-1, -1]


def test_bytetrack_works_without_lap(monkeypatch):
    """lap 缺失时仍走 ByteTrack（而不是回退 ObjectTracker），且 ID 稳定。"""

    from app.oilcheck import bytetrack_adapter

    _block_lap(monkeypatch)
    monkeypatch.setattr(
        bytetrack_adapter, "logger", logging.getLogger("test.bytetrack")
    )
    tracker = bytetrack_adapter.ByteTrackObjectTracker(
        max_distance=0.15, backend="auto"
    )
    assert tracker.backend == "bytetrack"

    ids = set()
    for i in range(10):
        cx = 100.0 + i * 60.0
        out = tracker.update(
            ts=i / 15.0,
            detections=(_det(cx, 500.0),),
            frame_size=(2560, 1440),
        )
        ids |= {t.track_id for t in out.active if t.matched}
    assert len(ids) == 1, f"快速移动时 ID 应保持，实际 {sorted(ids)}"
