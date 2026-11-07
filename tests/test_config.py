from __future__ import annotations

import json

from app.oilcheck.config import OilCheckConfig, PipelineConfig, StationConfig
from app.oilcheck.roi import Roi
from app.paths import runtime_root


def test_repo_oilcheck_template_is_valid():
    path = runtime_root() / "config" / "oilcheck.json"
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = OilCheckConfig.from_dict(data)

    assert cfg.version == "2.1"
    assert cfg.stations
    station = cfg.stations[0]
    assert station.station_id == "ST01"
    assert station.camera_id
    assert station.tray_roi.kind == "rect"
    assert station.hand_roi.kind == "rect"
    assert station.brush_roi.kind == "rect"
    assert station.exclude_rois == ()
    assert station.detector.kind in {"null", "yolo"}
    assert station.detector.screw_label
    assert station.detector.brush_label
    assert station.detector.tray_label
    assert cfg.pipeline.max_fps >= 1
    assert cfg.pipeline.brush_min_duration_s >= 0
    assert 0.0 <= cfg.pipeline.coverage_required <= 1.0
    assert cfg.recording.segment_seconds >= 10


def test_invalid_roi_is_rejected():
    assert Roi.from_dict({"kind": "rect", "points": [[0.2, 0.2]]}) is None
    assert (
        Roi.from_dict({"kind": "polygon", "points": [[0.1, 0.1], [0.2, 0.2]]}) is None
    )


def test_legacy_station_config_is_migrated():
    """旧版 pick_roi/oiling_roi/slots 配置应自动迁移，不要求用户改文件。"""

    legacy = {
        "station_id": "ST01",
        "name": "x",
        "camera_id": "cam",
        "pick_roi": {"kind": "rect", "points": [[0.1, 0.1], [0.4, 0.4]]},
        "oiling_roi": {"kind": "rect", "points": [[0.5, 0.5], [0.8, 0.8]]},
        "slot_mode": "tray",
        "slot_layout": [{"index": 1, "x1": 0, "y1": 0, "x2": 1, "y2": 1}],
        "slots": [],
        "detector": {"kind": "null", "rod_label": "rod", "brush_label": "brush"},
    }
    station = StationConfig.from_dict(legacy)
    assert station is not None
    assert station.tray_roi.points == ((0.1, 0.1), (0.4, 0.4))
    assert station.hand_roi.points == ((0.5, 0.5), (0.8, 0.8))
    assert station.brush_roi == station.hand_roi
    assert station.detector.screw_label == "rod"
    assert station.detector.brush_label == "brush"

    assert StationConfig.from_dict({**legacy, "camera_id": ""}) is None
    assert (
        StationConfig.from_dict(
            {
                "station_id": "ST01",
                "camera_id": "cam",
                "hand_roi": {"kind": "rect", "points": [[0.1, 0.1], [0.2, 0.2]]},
            }
        )
        is None
    )


def test_exclude_rois_are_parsed():
    station = StationConfig.from_dict(
        {
            "station_id": "ST01",
            "camera_id": "cam",
            "tray_roi": {"kind": "rect", "points": [[0.0, 0.0], [0.5, 0.5]]},
            "hand_roi": {"kind": "rect", "points": [[0.5, 0.5], [1.0, 1.0]]},
            "exclude_rois": [
                {"kind": "rect", "points": [[0.8, 0.0], [1.0, 0.2]]},
            ],
        }
    )
    assert station is not None
    assert len(station.exclude_rois) == 1


def test_legacy_pipeline_keys_are_migrated():
    cfg = PipelineConfig.from_dict(
        {
            "track_max_distance": 0.11,
            "track_max_missing_s": 0.9,
            "track_min_hits": 3,
            "brush_part_distance": 0.07,
            "occlusion_timeout_s": 1.5,
            "operation_max_active": 6,
        }
    )
    assert cfg.screw_track_max_distance == 0.11
    assert cfg.screw_track_max_missing_s == 0.9
    assert cfg.screw_track_min_hits == 3
    assert cfg.brush_contact_distance == 0.07
    assert cfg.pick_occlusion_timeout_s == 1.5
    assert cfg.task_max_active == 6


def test_pipeline_defaults_are_coherent():
    cfg = PipelineConfig()
    assert cfg.pick_comotion_min_s > 0
    assert cfg.pick_confirm_s > 0
    assert cfg.brush_contact_distance < cfg.pick_hand_distance * 2
    assert cfg.coverage_segments >= 1


def test_legacy_runtime_config_is_migrated_on_load(tmp_path):
    """服务器上已存在的旧配置，加载时应补上新字段并能接上模型。"""

    import json

    from app.oilcheck.config import OilCheckConfigService

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    legacy = {
        "version": "2.0",
        "stations": [
            {
                "station_id": "ST01",
                "name": "1号取料工位",
                "camera_id": "cam_6ca47a",
                "enabled": True,
                "tray_roi": {"kind": "rect", "points": [[0.28, 0.08], [0.72, 0.42]]},
                "hand_roi": {"kind": "rect", "points": [[0.1, 0.4], [0.48, 0.96]]},
                "brush_roi": {"kind": "rect", "points": [[0.35, 0.41], [0.88, 1.0]]},
                "exclude_rois": [],
                "detector": {
                    "kind": "null",
                    "model_path": "",
                    "brush_head_label": "brush_head",
                    "tray_label": "oil_tray",
                },
            }
        ],
    }
    path = config_dir / "oilcheck.json"
    path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    service = OilCheckConfigService(tmp_path)
    cfg = service.load()
    station = cfg.enabled_stations()[0]

    assert cfg.version == "2.1"
    assert station.detector.kind == "yolo"
    assert station.detector.model_path.endswith("best.pt")
    assert station.detector.brush_label == "brush"
    assert station.detector.tray_label == "oil_station"
    assert station.detector.imgsz == 0
    assert station.detector.device == "auto"
    assert cfg.demo.videos, "旧配置缺少 demo 段，迁移后应补上示例视频"
    # 现场调过的 ROI 必须保留
    assert station.brush_roi.points == ((0.35, 0.41), (0.88, 1.0))
    # 原文件已备份
    assert (config_dir / "oilcheck.json.bak").exists()


def test_migration_is_idempotent(tmp_path):
    """重复加载不应反复改写配置。"""

    import json

    from app.oilcheck.config import OilCheckConfigService

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "oilcheck.json").write_text(
        json.dumps({"version": "2.1", "stations": []}), encoding="utf-8"
    )
    service = OilCheckConfigService(tmp_path)
    service.load()
    assert not (config_dir / "oilcheck.json.bak").exists()


def test_explicit_cpu_device_is_kept(tmp_path):
    """显式配置的 cpu 不应被迁移覆盖成 auto。"""

    import json

    from app.oilcheck.config import OilCheckConfigService

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "oilcheck.json").write_text(
        json.dumps(
            {
                "version": "2.0",
                "stations": [
                    {
                        "station_id": "ST01",
                        "camera_id": "cam",
                        "tray_roi": {"kind": "rect", "points": [[0.1, 0.1], [0.4, 0.4]]},
                        "hand_roi": {"kind": "rect", "points": [[0.5, 0.5], [0.8, 0.8]]},
                        "detector": {"kind": "yolo", "device": "cpu",
                                     "model_path": "x.pt"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cfg = OilCheckConfigService(tmp_path).load()
    assert cfg.enabled_stations()[0].detector.device == "cpu"
