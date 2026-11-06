from __future__ import annotations

import json

from app.oilcheck.config import OilCheckConfig, StationConfig
from app.oilcheck.roi import Roi
from app.paths import runtime_root


def test_repo_oilcheck_template_is_valid():
    path = runtime_root() / "config" / "oilcheck.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = OilCheckConfig.from_dict(data)

    assert cfg.version == "1.0"
    assert cfg.stations
    station = cfg.stations[0]
    assert station.station_id == "ST01"
    assert station.camera_id
    assert station.slot_mode in {"fixed", "dynamic", "tray"}
    if station.slot_mode == "fixed":
        assert len(station.slots) == 4
    else:
        assert station.slots == ()
    assert len(station.slot_layout) == 4
    assert station.tray_roi is not None
    assert station.pick_roi.kind == "rect"
    assert station.oiling_roi.kind == "rect"
    assert station.detector.kind in {"null", "simulated", "yolo"}
    assert station.detector.brush_label
    assert station.detector.tray_label
    assert cfg.pipeline.max_fps >= 1
    assert cfg.pipeline.brush_min_duration_s >= 0
    assert cfg.pipeline.slot_empty_confirm_s >= 0
    assert cfg.recording.segment_seconds >= 10


def test_invalid_roi_is_rejected():
    assert Roi.from_dict({"kind": "rect", "points": [[0.2, 0.2]]}) is None
    assert (
        Roi.from_dict({"kind": "polygon", "points": [[0.1, 0.1], [0.2, 0.2]]}) is None
    )


def test_station_defaults_to_tray_mode_and_fixed_requires_slots():
    base = {
        "station_id": "ST01",
        "name": "x",
        "camera_id": "cam",
        "pick_roi": {"kind": "rect", "points": [[0.1, 0.1], [0.4, 0.4]]},
        "oiling_roi": {"kind": "rect", "points": [[0.5, 0.5], [0.8, 0.8]]},
        "slots": [],
    }
    station = StationConfig.from_dict(base)
    assert station is not None
    assert station.slot_mode == "tray"
    assert len(station.slot_layout) == 4
    assert StationConfig.from_dict({**base, "camera_id": ""}) is None

    fixed = {
        **base,
        "slot_mode": "fixed",
        "slots": [
            {
                "index": 1,
                "name": "1号位",
                "roi": {"kind": "rect", "points": [[0.1, 0.1], [0.2, 0.3]]},
            }
        ],
    }
    restored = StationConfig.from_dict(fixed)
    assert restored is not None
    assert len(restored.slots) == 1
    assert StationConfig.from_dict({**fixed, "slots": []}) is None
