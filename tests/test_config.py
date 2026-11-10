from __future__ import annotations

import json

from app.oilcheck.config import (
    OilCheckConfig,
    PipelineConfig,
    StationConfig,
    strip_json_comments,
)
from app.oilcheck.roi import Roi
from app.paths import runtime_root


def test_repo_oilcheck_template_is_valid():
    path = runtime_root() / "config" / "oilcheck.json"
    assert path.exists()
    # 模板带注释（JSONC），用容忍注释的读取方式。
    data = json.loads(strip_json_comments(path.read_text(encoding="utf-8")))
    cfg = OilCheckConfig.from_dict(data)

    assert cfg.version == "2.5"
    assert data["version"] == "2.5"
    assert cfg.stations
    station = cfg.stations[0]
    assert station.station_id == "ST01"
    assert station.camera_id
    assert station.tray_roi.kind == "rect"
    assert station.work_roi.kind == "rect"
    assert station.oiling_roi.kind == "rect"
    assert station.detector.kind in {"null", "yolo"}
    assert station.detector.screw_label
    assert station.detector.brush_label
    assert station.detector.tray_label
    assert cfg.pipeline.max_fps >= 1
    assert cfg.pipeline.brush_hold_s > 0
    # 2.5 跟踪稳定性参数：锚点绝对步长上限、关联代价阈值语义。
    # 上限不得低于基准步长，否则取料瞬间锚点跨不过去会永久卡住。
    assert cfg.pipeline.screw_anchor_max_step_abs >= cfg.pipeline.screw_anchor_max_step
    assert cfg.pipeline.bytetrack_match_thresh >= 1.0
    # 检测置信度必须低于 ByteTrack 高置信度阈值，否则第二阶段收不到输入。
    assert station.detector.confidence <= cfg.pipeline.bytetrack_high_thresh
    assert cfg.recording.segment_seconds >= 10
    # 模板里不应再残留槽位/遮挡区或已废弃的判定参数。
    raw_station = data["stations"][0]
    for key in ("slot_rois", "occlusion_rois", "slot_auto_count", "slot_follow_tray"):
        assert key not in raw_station
    for key in (
        "slot_occupied_confirm_s",
        "brush_hand_distance",
        "coverage_required",
        "association_sustain_s",
        "screw_track_max_distance",
    ):
        assert key not in data["pipeline"]


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
    # 旧手持区迁移为工作追踪区；旧配置没有涂油区概念，沿用工作追踪区。
    assert station.work_roi.points == ((0.5, 0.5), (0.8, 0.8))
    assert station.oiling_roi.points == ((0.5, 0.5), (0.8, 0.8))
    assert not hasattr(station, "hand_roi")
    assert not hasattr(station, "brush_roi")
    assert station.detector.screw_label == "rod"
    assert station.detector.brush_label == "brush"

    assert StationConfig.from_dict({**legacy, "camera_id": ""}) is None
    # 没有涂油盘区就无法判断离盘，配置无效。
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


def test_work_roi_defaults_to_full_frame_when_missing():
    station = StationConfig.from_dict(
        {
            "station_id": "ST01",
            "camera_id": "cam",
            "tray_roi": {"kind": "rect", "points": [[0.0, 0.0], [0.5, 0.5]]},
        }
    )
    assert station is not None
    assert station.work_roi.points == ((0.0, 0.0), (1.0, 1.0))
    # 没有涂油区时回退到工作追踪区。
    assert station.oiling_roi.points == station.work_roi.points


def test_legacy_hand_and_brush_rois_merge_into_work_roi():
    station = StationConfig.from_dict(
        {
            "station_id": "ST01",
            "camera_id": "cam",
            "tray_roi": {"kind": "rect", "points": [[0.0, 0.0], [0.5, 0.5]]},
            "hand_roi": {"kind": "rect", "points": [[0.1, 0.4], [0.48, 0.96]]},
            "brush_roi": {"kind": "rect", "points": [[0.349, 0.412], [0.876, 1.0]]},
        }
    )
    assert station is not None
    assert station.work_roi.points == ((0.1, 0.4), (0.876, 1.0))


def test_legacy_pipeline_keys_are_migrated():
    cfg = PipelineConfig.from_dict(
        {
            "brush_part_distance": 0.07,
            "brush_min_duration_s": 0.8,
            "operation_max_active": 6,
        }
    )
    assert cfg.brush_contact_distance == 0.07
    # 旧「最短刷动时长」映射到新的「刷子靠近判完成时长」。
    assert cfg.brush_hold_s == 0.8
    assert cfg.task_max_active == 6


def test_pipeline_defaults_are_coherent():
    cfg = PipelineConfig()
    assert cfg.pick_confirm_s > 0
    assert cfg.brush_hold_s > 0
    assert cfg.brush_contact_distance < cfg.pick_hand_distance * 2
    for removed in (
        "slot_occupied_confirm_s",
        "brush_hand_distance",
        "coverage_required",
        "coverage_segments",
        "pick_occlusion_timeout_s",
        "screw_track_max_distance",
        "association_sustain_s",
        "identity_timeout_s",
        "oiling_wait_timeout_s",
    ):
        assert not hasattr(cfg, removed), removed


def test_legacy_runtime_config_is_migrated_on_load(tmp_path):
    """服务器上已存在的旧配置，加载时应补上新字段并能接上模型。"""

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

    from app.oilcheck.config import OilCheckConfigService

    service = OilCheckConfigService(tmp_path)
    cfg = service.load()
    station = cfg.enabled_stations()[0]

    assert cfg.version == "2.5"
    assert station.detector.kind == "yolo"
    assert station.detector.model_path.endswith("best.pt")
    assert station.detector.brush_label == "brush"
    assert station.detector.tray_label == "oil_station"
    assert station.detector.imgsz == 0
    assert station.detector.device == "auto"
    assert cfg.demo.videos, "旧配置缺少 demo 段，迁移后应补上示例视频"
    # 现场调过的 ROI 必须保留：旧手持区/刷涂区取并集后作为工作追踪区
    assert station.work_roi.points == ((0.1, 0.4), (0.88, 1.0))
    # 涂油盘区是现场调过的值，必须原样保留。
    assert station.tray_roi.points == ((0.28, 0.08), (0.72, 0.42))
    # 旧配置没有涂油区：迁移后沿用工作追踪区
    assert station.oiling_roi.points == station.work_roi.points
    # 原文件已备份
    assert (config_dir / "oilcheck.json.bak").exists()


def test_v23_config_is_cleaned_up_to_v24(tmp_path):
    """2.3 配置里的槽位键、刷-手参数、覆盖度量字段应在加载时清掉。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    legacy = {
        "version": "2.3",
        "pipeline": {
            "pick_confirm_s": 0.4,
            "brush_contact_distance": 0.055,
            "brush_min_duration_s": 1.4,
            "brush_hold_s": 1.4,
            "slot_occupied_confirm_s": 0.2,
            "slot_empty_confirm_s": 0.25,
            "slot_layout_anchor_ratio": 0.5,
            "brush_hand_distance": 0.12,
            "brush_require_hand_link": True,
            "coverage_segments": 5,
            "coverage_required": 0.6,
            "association_sustain_s": 0.1,
            "association_lock_confidence": 0.6,
            "screw_track_min_hits": 2,
            "identity_timeout_s": 2.0,
            "oiling_wait_timeout_s": 0.0,
        },
        "stations": [
            {
                "station_id": "ST01",
                "camera_id": "cam",
                "tray_roi": {"kind": "rect", "points": [[0.5, 0.1], [0.72, 0.44]]},
                "work_roi": {"kind": "rect", "points": [[0.35, 0.0], [0.99, 0.95]]},
                "detector": {"kind": "yolo", "model_path": "best.pt"},
                "slot_rois": [
                    {"slot_id": "slot_1", "name": "1号槽位",
                     "roi": {"kind": "rect", "points": [[0.5, 0.1], [0.6, 0.2]]}}
                ],
                "occlusion_rois": [],
                "slot_auto_count": 4,
                "slot_follow_tray": True,
            }
        ],
    }
    path = config_dir / "oilcheck.json"
    path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    from app.oilcheck.config import OilCheckConfigService

    service = OilCheckConfigService(tmp_path)
    cfg = service.load()
    station = cfg.enabled_stations()[0]

    assert cfg.version == "2.5"
    # 用户调过的 ROI 与判定参数保留
    assert station.tray_roi.points == ((0.5, 0.1), (0.72, 0.44))
    assert station.work_roi.points == ((0.35, 0.0), (0.99, 0.95))
    assert cfg.pipeline.pick_confirm_s == 0.4
    assert cfg.pipeline.brush_contact_distance == 0.055
    assert cfg.pipeline.brush_hold_s == 1.4
    # 槽位/遮挡区补成涂油区，工作追踪区原样保留
    assert station.oiling_roi.points == station.work_roi.points

    # 模板带注释（JSONC），用容忍注释的读取方式。
    data = json.loads(strip_json_comments(path.read_text(encoding="utf-8")))
    raw_station = data["stations"][0]
    for key in ("slot_rois", "occlusion_rois", "slot_auto_count", "slot_follow_tray"):
        assert key not in raw_station
    for key in (
        "slot_occupied_confirm_s",
        "slot_empty_confirm_s",
        "slot_layout_anchor_ratio",
        "brush_hand_distance",
        "brush_require_hand_link",
        "coverage_segments",
        "coverage_required",
        "association_sustain_s",
        "association_lock_confidence",
        "screw_track_min_hits",
        "identity_timeout_s",
        "oiling_wait_timeout_s",
    ):
        assert key not in data["pipeline"]
    assert data["stations"][0]["oiling_roi"]["kind"] == "rect"


def test_migration_is_idempotent(tmp_path):
    """重复加载不应反复改写配置。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "oilcheck.json").write_text(
        json.dumps({"version": "2.5", "stations": []}), encoding="utf-8"
    )
    from app.oilcheck.config import OilCheckConfigService

    service = OilCheckConfigService(tmp_path)
    service.load()
    assert not (config_dir / "oilcheck.json.bak").exists()


def test_explicit_cpu_device_is_kept(tmp_path):
    """显式配置的 cpu 不应被迁移覆盖成 auto。"""

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
                        "work_roi": {"kind": "rect", "points": [[0.5, 0.5], [0.8, 0.8]]},
                        "detector": {"kind": "yolo", "device": "cpu",
                                     "model_path": "x.pt"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    from app.oilcheck.config import OilCheckConfigService

    cfg = OilCheckConfigService(tmp_path).load()
    assert cfg.enabled_stations()[0].detector.device == "cpu"


def test_demo_speed_defaults_and_roundtrips():
    from app.oilcheck.config import DemoConfig

    cfg = DemoConfig.from_dict({})
    assert cfg.speed == 1.0
    assert DemoConfig.from_dict({"speed": 0.5}).speed == 0.5
    # 非法值回退默认，越界值夹到允许范围。
    assert DemoConfig.from_dict({"speed": "x"}).speed == 1.0
    assert DemoConfig.from_dict({"speed": 0}).speed == 1.0
    assert DemoConfig.from_dict({"speed": 99}).speed == 4.0
    assert DemoConfig().to_dict()["speed"] == 1.0


def test_legacy_config_gets_demo_speed(tmp_path):
    """旧配置没有 demo.speed 时迁移补齐，已有的值保持不动。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "oilcheck.json").write_text(
        json.dumps(
            {
                "version": "2.3",
                "demo": {"loop": True, "fps_limit": 15, "videos": []},
                "stations": [],
            }
        ),
        encoding="utf-8",
    )
    from app.oilcheck.config import OilCheckConfigService

    service = OilCheckConfigService(tmp_path)
    service.load()
    data = json.loads(
        strip_json_comments((config_dir / "oilcheck.json").read_text(encoding="utf-8"))
    )
    assert data["demo"]["speed"] == 1.0

    data["demo"]["speed"] = 0.5
    (config_dir / "oilcheck.json").write_text(json.dumps(data), encoding="utf-8")
    service.load()
    data = json.loads(
        strip_json_comments((config_dir / "oilcheck.json").read_text(encoding="utf-8"))
    )
    assert data["demo"]["speed"] == 0.5


# ----------------------------------------------------------------------
# 配置注释（JSONC）
# ----------------------------------------------------------------------
def test_strip_json_comments_keeps_strings_intact():
    """只去掉字符串之外的注释，值里的 // 不能被当成注释。"""

    text = """{
      // 行注释
      "url": "http://example.com/a",   // 尾注释
      "note": "含 // 和 /* 的值",
      /* 块注释
         跨多行 */
      "n": 1
    }"""
    parsed = json.loads(strip_json_comments(text))
    assert parsed == {
        "url": "http://example.com/a",
        "note": "含 // 和 /* 的值",
        "n": 1,
    }


def test_repo_template_is_commented():
    """仓库模板每个参数都有注释，便于现场改配置。"""

    path = runtime_root() / "config" / "oilcheck.json"
    text = path.read_text(encoding="utf-8")
    assert "//" in text, "模板应带注释"
    # 注释不破坏可解析性
    assert json.loads(strip_json_comments(text))["version"] == "2.5"


def test_save_preserves_comments(tmp_path):
    """UI 保存配置后注释不能丢。"""

    from app.oilcheck.config import CONFIG_COMMENTS, OilCheckConfigService

    svc = OilCheckConfigService(tmp_path)
    cfg = svc.load()
    svc.save(cfg)

    text = svc.config_path.read_text(encoding="utf-8")
    assert "//" in text
    assert len(CONFIG_COMMENTS) > 50
    # 保存后的文件仍可正常读回，且内容与保存前一致
    assert json.loads(strip_json_comments(text)) == cfg.to_dict()
    assert svc.load().to_dict() == cfg.to_dict()


def test_comment_table_covers_every_config_key():
    """注释表应覆盖实际出现的全部配置键（防止新增参数忘了写注释）。"""

    import re as _re

    from app.oilcheck.config import CONFIG_COMMENTS

    path = runtime_root() / "config" / "oilcheck.json"
    data = json.loads(strip_json_comments(path.read_text(encoding="utf-8")))

    missing: list[str] = []

    def walk(value, path_str: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{path_str}.{key}" if path_str else key
                wildcard = _re.sub(r"\[\d+\]", "[]", child)
                if child not in CONFIG_COMMENTS and wildcard not in CONFIG_COMMENTS:
                    missing.append(child)
                walk(item, child)
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            walk(value[0], f"{path_str}[]")

    walk(data, "")
    assert not missing, f"以下配置项缺少注释: {missing}"
