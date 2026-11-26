from __future__ import annotations

import json
import shutil
import time
from dataclasses import replace
from pathlib import Path

import pytest

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

    assert cfg.version == "2.12"
    assert data["version"] == "2.12"
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

    service = OilCheckConfigService(tmp_path / "config")
    cfg = service.load()
    station = cfg.enabled_stations()[0]

    assert cfg.version == "2.12"
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

    service = OilCheckConfigService(tmp_path / "config")
    cfg = service.load()
    station = cfg.enabled_stations()[0]

    assert cfg.version == "2.12"
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
    """重复加载不应反复改写配置。

    第一次加载会把文件里缺的键补齐（让文件与生效参数一致），之后每次加载
    都不应再改动文件。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    path = config_dir / "oilcheck.json"
    path.write_text(
        json.dumps({"version": "2.5", "stations": []}), encoding="utf-8"
    )
    from app.oilcheck.config import OilCheckConfigService

    OilCheckConfigService(tmp_path / "config").load()
    first = path.read_bytes()
    # 缺键已补齐
    assert (config_dir / "oilcheck.json.bak").exists()

    OilCheckConfigService(tmp_path / "config").load()
    assert path.read_bytes() == first, "第二次加载不应再改写配置"
    OilCheckConfigService(tmp_path / "config").load()
    assert path.read_bytes() == first, "第三次加载不应再改写配置"


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

    cfg = OilCheckConfigService(tmp_path / "config").load()
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

    service = OilCheckConfigService(tmp_path / "config")
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


def _write_config(tmp_path: Path, mutate) -> Path:  # noqa: ANN001
    """按仓库模板造一份运行配置，交给 ``mutate`` 改动后写进临时数据目录。"""

    path = runtime_root() / "config" / "oilcheck.json"
    data = json.loads(strip_json_comments(path.read_text(encoding="utf-8")))
    mutate(data)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    target = config_dir / "oilcheck.json"
    target.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return target


def test_missing_demo_section_is_filled_with_defaults(tmp_path):
    """demo 整段缺失时补出默认段（含 sample_dir），示例视频照样能扫出来。

    2.12 起视频清单不再写进配置：demo 段缺不缺都不影响扫描目录，所以这里
    断言的是「段被补出来、扫描目录是默认值」，不再断言某份清单。
    """

    from app.oilcheck.config import OilCheckConfigService, load_json_with_comments

    _write_config(tmp_path, lambda data: data.pop("demo"))
    cfg = OilCheckConfigService(tmp_path / "config").load()

    assert cfg.demo.sample_dir == "data/sample_video"
    data = load_json_with_comments(tmp_path / "config" / "oilcheck.json")
    assert set(data["demo"]) == {"loop", "fps_limit", "speed", "sample_dir"}
    assert "videos" not in data["demo"]


def test_tracking_threshold_migrates_only_from_old_default(tmp_path):
    """2.7 把仍是旧默认 0.30 的高置信度阈值提到 0.45，现场调过的值保留。"""

    from app.oilcheck.config import OilCheckConfigService

    def _old_default(data) -> None:  # noqa: ANN001
        data["version"] = "2.6"
        data["pipeline"]["bytetrack_high_thresh"] = 0.30
        data["pipeline"].pop("brush_spawn_max_distance", None)

    path = _write_config(tmp_path, _old_default)
    cfg = OilCheckConfigService(path.parent).load()
    assert cfg.version == "2.12"
    assert cfg.pipeline.bytetrack_high_thresh == 0.45
    # 新键补默认值（关闭），不会因为迁移被打开
    assert cfg.pipeline.brush_spawn_max_distance == 0.0

    def _tuned(data) -> None:  # noqa: ANN001
        data["version"] = "2.6"
        data["pipeline"]["bytetrack_high_thresh"] = 0.36

    tuned_path = _write_config(tmp_path / "tuned", _tuned)
    tuned = OilCheckConfigService(tuned_path.parent).load()
    assert tuned.pipeline.bytetrack_high_thresh == 0.36


def test_legacy_videos_list_is_removed_on_load(tmp_path):
    """2.12 起 ``demo.videos`` 手工清单删除：示例视频改为扫描目录。

    留着清单会让现场以为「登记了才出现」：拷进目录却不改配置时下拉里没有，
    改了配置又和目录内容对不上。这里要求升级后键被清掉，且不因清理而丢
    其它 demo 参数。
    """

    from app.oilcheck.config import OilCheckConfigService, load_json_with_comments

    def _make_legacy(data) -> None:  # noqa: ANN001
        data["version"] = "2.11"
        data["demo"]["videos"] = [
            {"name": "老名字", "path": "data/sample_video/cam_x_20260101_000000.mp4"}
        ]
        data["demo"]["speed"] = 0.5

    path = _write_config(tmp_path, _make_legacy)
    cfg = OilCheckConfigService(tmp_path / "config").load()

    assert cfg.version == "2.12"
    assert cfg.demo.speed == 0.5, "清理清单不能连带丢现场调过的参数"
    data = load_json_with_comments(path)
    assert "videos" not in data["demo"]
    assert data["demo"]["sample_dir"] == "data/sample_video"


def _write_sample_videos(root: Path, names: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"")


def test_demo_videos_are_scanned_from_sample_dir(tmp_path):
    """``data/sample_video`` 下的录像自动出现在清单里，不需要登记。"""

    from app.oilcheck.config import DemoConfig

    sample = tmp_path / "data" / "sample_video"
    _write_sample_videos(
        sample,
        [
            "cam_6ca47a_20261126_110047.mp4",
            "cam_6ca47a_20261125_092817.mp4",
            # 元数据不是视频，不收起
            "cam_6ca47a_20261126_110047.json",
            # 其它常见容器一并认
            "hand_take.mov",
        ],
    )
    cfg = DemoConfig()
    videos = cfg.videos(base_dir=tmp_path)

    assert [v.name for v in videos] == [
        "20261125_092817",
        "20261126_110047",
        "hand_take",  # 命名不规范的文件用文件名
    ]
    # 路径写成相对仓库根，日志/配置里看到的就是现场目录结构
    assert videos[0].path == "data/sample_video/cam_6ca47a_20261125_092817.mp4"
    assert cfg.video("20261126_110047", base_dir=tmp_path) is not None
    assert cfg.video("不存在", base_dir=tmp_path) is None


def test_demo_scan_follows_new_files_without_config_change(tmp_path):
    """新录像拷进目录后立刻出现；目录不存在时干净地返回空。"""

    from app.oilcheck.config import DemoConfig

    cfg = DemoConfig()
    # 目录还没有：不抛异常，只是没有视频可选。
    assert cfg.videos(base_dir=tmp_path) == ()

    sample = tmp_path / "data" / "sample_video"
    _write_sample_videos(sample, ["cam_6ca47a_20261126_101415.mp4"])
    assert [v.name for v in cfg.videos(base_dir=tmp_path)] == ["20261126_101415"]

    _write_sample_videos(sample, ["cam_6ca47a_20261126_110047.mp4"])
    assert [v.name for v in cfg.videos(base_dir=tmp_path)] == [
        "20261126_101415",
        "20261126_110047",
    ]


def test_demo_sample_dir_can_be_overridden(tmp_path):
    """``demo.sample_dir`` 支持相对仓库根与绝对路径。"""

    from app.oilcheck.config import DemoConfig

    other = tmp_path / "other_videos"
    _write_sample_videos(other, ["clip.mp4"])
    assert DemoConfig.from_dict({"sample_dir": str(other)}).videos(base_dir=tmp_path)
    assert DemoConfig.from_dict({"sample_dir": "other_videos"}).videos(
        base_dir=tmp_path
    )
    # 空串按默认目录
    assert DemoConfig.from_dict({"sample_dir": "  "}).sample_dir == "data/sample_video"


def test_legacy_screw_label_is_migrated(tmp_path):
    """旧类别名 rod 必须纠正成模型实际输出的 screw。

    模型输出 hand / brush / oil_station / screw。配置里若还是 1.0 时代的
    ``rod``，这类检测会被判定侧当作未配置类别整批丢掉——画面上的框照常
    显示（用的模型输出标签），但取料/涂油一个事件都产生不了。迁移只纠正
    ``tray_label`` / ``brush_label`` 时，服务器上就会是这个症状。
    """

    from app.oilcheck.config import OilCheckConfigService

    def _legacy_rod(data) -> None:  # noqa: ANN001
        data["stations"][0]["detector"]["screw_label"] = "rod"

    _write_config(tmp_path, _legacy_rod)
    cfg = OilCheckConfigService(tmp_path / "config").load()
    assert cfg.stations[0].detector.screw_label == "screw"

    # 用户自己改过的、非旧默认值的类别名要保持不动。
    tmp2 = tmp_path / "keep"
    tmp2.mkdir()

    def _custom(data) -> None:  # noqa: ANN001
        data["stations"][0]["detector"]["screw_label"] = "my_screw"

    _write_config(tmp2, _custom)
    cfg = OilCheckConfigService(tmp2 / "config").load()
    assert cfg.stations[0].detector.screw_label == "my_screw"


def test_legacy_rod_label_field_is_migrated(tmp_path):
    """1.0 配置只有 rod_label 字段，迁移后要变成 screw 而不是 rod。"""

    from app.oilcheck.config import OilCheckConfigService

    legacy = {
        "version": "1.0",
        "stations": [
            {
                "station_id": "ST01",
                "name": "1号取料工位",
                "camera_id": "cam_6ca47a",
                "enabled": True,
                "tray_roi": {"kind": "rect", "points": [[0.54, 0.01], [0.66, 0.39]]},
                "hand_roi": {"kind": "rect", "points": [[0.36, 0.008], [0.879, 0.998]]},
                "detector": {"kind": "yolo", "rod_label": "rod"},
            }
        ],
    }
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "oilcheck.json").write_text(
        json.dumps(legacy, ensure_ascii=False), encoding="utf-8"
    )

    cfg = OilCheckConfigService(tmp_path / "config").load()
    assert cfg.stations[0].detector.screw_label == "screw"


def test_orphaned_data_config_is_reported(tmp_path):
    """数据目录里遗留旧配置时要点名，别让现场以为配置还在生效。

    只返回文本、不直接打日志：``ensure_defaults`` 跑在 ``setup_logging``
    之前，那时打出去只会被 lastResort 甩到控制台、进不了日志文件。
    """

    from app.config_service import ConfigService

    data_dir = tmp_path / "data"
    (data_dir / "config").mkdir(parents=True)
    (data_dir / "config" / "cameras.json").write_text("[]", encoding="utf-8")

    # repo_root 显式传入时配置目录是 <repo>/config，与 data_dir/config 不同，
    # 正好复刻「升级前配置在数据目录」的现场布局。
    service = ConfigService(repo_root=tmp_path / "repo")
    service._data_root = data_dir  # 运行数据根指向带旧配置的目录

    service.ensure_defaults()
    message = service.orphaned_config_warning()
    assert "已不再生效" in message
    assert "cameras.json" in message


def test_orphaned_config_warning_is_silent_when_absent(tmp_path):
    """没有遗留旧配置时不产生任何文案，调用方也就不会打日志。"""

    from app.config_service import ConfigService

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    service = ConfigService(repo_root=tmp_path / "repo")
    service._data_root = data_dir
    service.ensure_defaults()
    assert service.orphaned_config_warning() == ""


def test_orphaned_warning_does_not_hit_console_before_logging(tmp_path, capsys):
    """启动早期（日志未就绪）不许往 stderr 甩旧配置告警。

    现场看到的启动输出必须是 Modbus 连接信息，不是这句配置迁移提示。
    """

    from app.config_service import ConfigService

    data_dir = tmp_path / "data"
    (data_dir / "config").mkdir(parents=True)
    (data_dir / "config" / "app.json").write_text("{}", encoding="utf-8")
    service = ConfigService(repo_root=tmp_path / "repo")
    service._data_root = data_dir

    service.ensure_defaults()

    captured = capsys.readouterr()
    assert "已不再生效" not in captured.out
    assert "已不再生效" not in captured.err


def test_config_dir_is_the_repo_config_by_default(monkeypatch, tmp_path):
    """不传参时读写仓库的 config/，不是数据目录下的副本。

    UI 保存的位置、判定读取的位置、仓库里的文件必须是同一份；否则现场会
    遇到「改了配置不生效」「两份文件不一致」这类问题。
    """

    from app.oilcheck.config import OilCheckConfigService
    from app.paths import config_root, runtime_root

    monkeypatch.delenv("OILCHECK_CONFIG_ROOT", raising=False)
    service = OilCheckConfigService()
    assert service.config_dir == runtime_root() / "config"
    assert service.config_path == config_root() / "oilcheck.json"

    monkeypatch.setenv("OILCHECK_CONFIG_ROOT", str(tmp_path / "cfg"))
    overridden = OilCheckConfigService()
    assert overridden.config_dir == (tmp_path / "cfg").resolve()


def test_ui_save_and_pipeline_read_the_same_file(monkeypatch, tmp_path):
    """UI 保存的值 = 仓库 config 文件里的值 = 判定管线手里的值。"""

    from app.oilcheck.config import OilCheckConfigService, load_json_with_comments
    from app.oilcheck.pipeline import StationPipeline
    from app.oilcheck.roi import Roi

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    template = runtime_root() / "config" / "oilcheck.json"
    shutil.copy2(template, config_dir / "oilcheck.json")

    service = OilCheckConfigService(config_dir)
    cfg = service.load()
    station = cfg.stations[0]

    new_roi = Roi.from_dict({"kind": "rect", "points": [[0.11, 0.12], [0.31, 0.62]]})
    assert new_roi is not None
    updated = replace(station, tray_roi=new_roi)
    service.save(
        replace(
            cfg,
            stations=tuple(
                updated if s.station_id == station.station_id else s
                for s in cfg.stations
            ),
        )
    )

    # 文件里
    on_disk = load_json_with_comments(service.config_path)["stations"][0]
    assert on_disk["tray_roi"]["points"] == [[0.11, 0.12], [0.31, 0.62]]

    # 管线里（reload 用的就是 service.load() 的结果）
    pipeline = StationPipeline(
        station=service.load().stations[0],
        pipeline_cfg=cfg.pipeline,
        detector_name="null",
    )
    assert pipeline.station.tray_roi.points == ((0.11, 0.12), (0.31, 0.62))


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
    assert json.loads(strip_json_comments(text))["version"] == "2.12"


def test_save_preserves_comments(tmp_path):
    """UI 保存配置后注释不能丢。"""

    from app.oilcheck.config import CONFIG_COMMENTS, OilCheckConfigService

    svc = OilCheckConfigService(tmp_path / "config")
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


# ----------------------------------------------------------------------
# 运行配置与生效参数对齐
# ----------------------------------------------------------------------
def test_missing_keys_are_written_back_to_the_config_file(tmp_path):
    """代码里有默认值、文件里缺的键，启动时要补写回文件。

    缺键的参数走代码默认值却不出现在文件里，现场对着配置排查时会以为
    「还是老参数」——例如 pick_sample_every_frames / pick_history_samples。
    """

    from app.oilcheck.config import OilCheckConfigService, load_json_with_comments

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    path = config_dir / "oilcheck.json"
    path.write_text(
        json.dumps(
            {
                "version": "2.5",
                "pipeline": {"max_fps": 15},
                "stations": [
                    {
                        "station_id": "ST01",
                        "name": "1号取料工位",
                        "camera_id": "cam",
                        "enabled": True,
                        "tray_roi": {"kind": "rect", "points": [[0.5, 0.1], [0.7, 0.4]]},
                        "work_roi": {"kind": "rect", "points": [[0.3, 0.0], [0.9, 0.9]]},
                        "detector": {"kind": "yolo", "model_path": "x.pt"},
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cfg = OilCheckConfigService(tmp_path / "config").load()
    data = load_json_with_comments(path)

    # 文件里的键集合 == 实际生效的参数集合（工位按 station_id 对齐）。
    assert set(data["pipeline"]) == set(cfg.pipeline.to_dict())
    assert set(data["recording"]) == set(cfg.recording.to_dict())
    assert set(data["demo"]) == set(cfg.demo.to_dict())
    assert set(data["stations"][0]) == set(cfg.stations[0].to_dict())
    assert set(data["stations"][0]["detector"]) == set(
        cfg.stations[0].detector.to_dict()
    )
    # 补进去的值就是实际生效的值。
    assert data["pipeline"]["pick_sample_every_frames"] == (
        cfg.pipeline.pick_sample_every_frames
    )
    assert data["pipeline"]["pick_history_samples"] == (
        cfg.pipeline.pick_history_samples
    )


def test_key_sync_never_touches_field_values_or_rois(tmp_path):
    """补键只补缺失的键：现场调过的 ROI 与其他值原样保留。"""

    from app.oilcheck.config import OilCheckConfigService, load_json_with_comments

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    path = config_dir / "oilcheck.json"
    original = {
        "version": "2.5",
        "pipeline": {"max_fps": 12, "pick_confirm_s": 0.45},
        "stations": [
            {
                "station_id": "ST01",
                "name": "1号取料工位",
                "camera_id": "cam",
                "enabled": True,
                # 现场标定值（6 位小数），绝不能被模板或默认值覆盖。
                "tray_roi": {"kind": "rect", "points": [[0.515893, 0.006849], [0.678671, 0.398973]]},
                "work_roi": {"kind": "rect", "points": [[0.344446, 0.003425], [0.932952, 0.998288]]},
                "oiling_roi": {"kind": "rect", "points": [[0.344446, 0.446918], [0.932952, 0.994863]]},
                "detector": {"kind": "yolo", "model_path": "x.pt", "confidence": 0.33},
            }
        ],
    }
    path.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")

    OilCheckConfigService(tmp_path / "config").load()
    data = load_json_with_comments(path)

    station = data["stations"][0]
    for key in ("tray_roi", "work_roi", "oiling_roi"):
        assert station[key] == original["stations"][0][key], key
    assert data["pipeline"]["max_fps"] == 12
    assert data["pipeline"]["pick_confirm_s"] == 0.45
    assert station["detector"]["confidence"] == 0.33


def test_unknown_keys_are_kept(tmp_path):
    """当前版本不认识的键只提示，不删除。"""

    from app.oilcheck.config import OilCheckConfigService, load_json_with_comments

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    path = config_dir / "oilcheck.json"
    path.write_text(
        json.dumps(
            {
                "version": "2.5",
                "pipeline": {"max_fps": 15, "field_tuned_note": "现场手记"},
                "stations": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    OilCheckConfigService(tmp_path / "config").load()
    data = load_json_with_comments(path)
    assert data["pipeline"]["field_tuned_note"] == "现场手记"


def test_key_sync_pairs_stations_by_id_not_index(tmp_path):
    """工位数组按 station_id 对齐：不合法的工位被解析跳过时不能补错对象。"""

    from app.oilcheck.config import OilCheckConfigService, load_json_with_comments

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    path = config_dir / "oilcheck.json"
    path.write_text(
        json.dumps(
            {
                "version": "2.5",
                "stations": [
                    # 缺 tray_roi，解析时被跳过，但必须原样留在文件里。
                    {"station_id": "BROKEN", "camera_id": "cam0"},
                    {
                        "station_id": "ST01",
                        "camera_id": "cam",
                        "tray_roi": {"kind": "rect", "points": [[0.5, 0.1], [0.7, 0.4]]},
                        "work_roi": {"kind": "rect", "points": [[0.3, 0.0], [0.9, 0.9]]},
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    OilCheckConfigService(tmp_path / "config").load()
    data = load_json_with_comments(path)
    broken, station = data["stations"]

    assert broken["station_id"] == "BROKEN"
    assert "tray_roi" not in broken, "不能把 ST01 的键补到坏工位上"
    assert "detector" in station
    assert set(station["detector"]) >= {"kind", "model_path", "confidence"}


def test_repo_template_matches_comment_table_rendering():
    """仓库模板必须由注释表渲染得出。

    运行配置写回时按注释表加注释，模板若手改过就会和运行配置产生纯注释
    差异，`diff` 时看不出哪些是真正调过的参数。
    """

    from app.oilcheck.config import (
        CONFIG_COMMENTS,
        add_json_comments,
        load_json_with_comments,
    )

    path = runtime_root() / "config" / "oilcheck.json"
    rendered = add_json_comments(load_json_with_comments(path), CONFIG_COMMENTS)
    assert path.read_text(encoding="utf-8").rstrip("\n") == rendered


def test_runtime_config_and_template_differ_only_in_values(tmp_path):
    """按模板生成的运行配置，与模板的差异只能是值，不能是缺键或缺注释。"""

    from app.oilcheck.config import (
        CONFIG_COMMENTS,
        OilCheckConfigService,
        add_json_comments,
        load_json_with_comments,
    )

    template = load_json_with_comments(runtime_root() / "config" / "oilcheck.json")
    expected = add_json_comments(template, CONFIG_COMMENTS).splitlines()

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "oilcheck.json").write_text(
        add_json_comments(template, CONFIG_COMMENTS) + "\n", encoding="utf-8"
    )
    OilCheckConfigService(tmp_path / "config").load()

    actual = (config_dir / "oilcheck.json").read_text(encoding="utf-8").splitlines()
    assert actual == expected


# ----------------------------------------------------------------------
# 生效配置文件的位置与外部修改
# ----------------------------------------------------------------------
def test_source_info_points_at_the_data_directory_file(tmp_path):
    """``source_info`` 必须报出数据目录里的那份，而不是仓库模板。

    现场最常见的误判是「改了仓库 config/oilcheck.json 却不生效」——程序
    只读数据目录那份，仓库里的仅在首次运行时复制一次。
    """

    from app.oilcheck.config import OilCheckConfigService
    from app.paths import runtime_root

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "oilcheck.json").write_text(
        json.dumps({"version": "2.5", "stations": []}), encoding="utf-8"
    )

    service = OilCheckConfigService(tmp_path / "config")
    service.load()
    info = service.source_info()

    assert info["path"] == str(config_dir / "oilcheck.json")
    assert info["config_dir"] == str(config_dir.resolve())
    assert info["exists"] is True
    assert info["mtime"] is not None
    assert info["template"] == str(runtime_root() / "config" / "oilcheck.json")
    # 指向临时目录时，与仓库模板绝不是同一个文件。
    assert info["same_as_template"] is False


def test_source_info_reports_missing_file(tmp_path):
    from app.oilcheck.config import OilCheckConfigService

    info = OilCheckConfigService(tmp_path / "config").source_info()
    assert info["exists"] is False
    assert info["mtime"] is None


# ----------------------------------------------------------------------
# PLC 警示灯配置
# ----------------------------------------------------------------------
def test_plc_config_defaults_are_off_and_read_only():
    """装满代码不等于会写 PLC：默认关闭且只读。"""

    from app.oilcheck.config import PlcConfig

    cfg = PlcConfig()
    assert cfg.enabled is False
    assert cfg.dry_run is True
    assert cfg.host == "192.168.1.88"
    assert cfg.port == 502
    assert cfg.register_base == 0
    assert cfg.steps == {"pick": 2, "oiling": 4, "result": 5}


def test_plc_config_roundtrips():
    from app.oilcheck.config import PlcConfig

    cfg = PlcConfig.from_dict(
        {
            "enabled": True,
            "dry_run": False,
            "host": "10.0.0.5",
            "hold_s": 2.5,
        }
    )
    assert cfg.enabled is True
    assert cfg.dry_run is False
    assert cfg.host == "10.0.0.5"
    assert cfg.hold_s == 2.5
    assert cfg.steps == {"pick": 2, "oiling": 4, "result": 5}
    assert PlcConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()


def test_plc_config_rejects_bad_values():
    from app.oilcheck.config import PlcConfig

    cfg = PlcConfig.from_dict(
        {
            "hold_s": "x",
            "reconnect_backoff_ms": "nope",
            "steps": {"pick": 99, "oiling": 0},
        }
    )
    assert cfg.hold_s == 2.0
    assert cfg.reconnect_backoff_ms == (200, 500, 1000, 2000, 5000)
    # 越界工步号回退默认，不写进配置
    assert cfg.steps == {"pick": 2, "oiling": 4, "result": 5}


def test_legacy_config_gets_plc_section(tmp_path):
    """2.7 的旧配置升级后要出现 plc 段，且默认关闭只读。"""

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "oilcheck.json").write_text(
        json.dumps({"version": "2.7", "demo": {"videos": []}, "stations": []}),
        encoding="utf-8",
    )
    from app.oilcheck.config import OilCheckConfigService

    service = OilCheckConfigService(tmp_path / "config")
    cfg = service.load()
    data = json.loads(
        strip_json_comments((config_dir / "oilcheck.json").read_text(encoding="utf-8"))
    )

    assert data["version"] == cfg.version
    assert "plc" in data
    assert data["plc"]["enabled"] is False
    assert data["plc"]["dry_run"] is True
    assert data["plc"]["host"] == "192.168.1.88"


def test_migration_removes_obsolete_plc_byte_order(tmp_path):
    """2.10 起删掉 plc.byte_order：它从未接进写入路径，改它无效。

    旧配置里存在这个键时必须被清掉，否则「当前版本不认识的键」会一直挂在
    文件里，并且每次加载都留一条警告。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "oilcheck.json").write_text(
        json.dumps(
            {
                "version": "2.9",
                "plc": {"enabled": False, "byte_order": "msb", "steps": {"pick": 2, "oiling": 4}},
                "demo": {"videos": []},
                "stations": [],
            }
        ),
        encoding="utf-8",
    )
    from app.oilcheck.config import OilCheckConfigService

    OilCheckConfigService(config_dir).load()
    data = json.loads(
        strip_json_comments((config_dir / "oilcheck.json").read_text(encoding="utf-8"))
    )
    assert "byte_order" not in data["plc"]
    assert data["version"] == "2.12"


def test_migration_removes_obsolete_plc_result_hold_s(tmp_path):
    """2.11 起删掉 plc.result_hold_s：结果灯改为常亮到下一个结论。

    这个时长已经没有消费方，留着只会让现场以为改它有效。
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "oilcheck.json").write_text(
        json.dumps(
            {
                "version": "2.10",
                "plc": {
                    "enabled": False,
                    "result_hold_s": 8.0,
                    "steps": {"pick": 2, "oiling": 4, "result": 5},
                },
                "demo": {"videos": []},
                "stations": [],
            }
        ),
        encoding="utf-8",
    )
    from app.oilcheck.config import OilCheckConfigService

    cfg = OilCheckConfigService(config_dir).load()
    assert not hasattr(cfg.plc, "result_hold_s")
    data = json.loads(
        strip_json_comments((config_dir / "oilcheck.json").read_text(encoding="utf-8"))
    )
    assert "result_hold_s" not in data["plc"]
    assert data["version"] == "2.12"


def test_console_filter_drops_file_only_records(tmp_path):
    """标了 FILE_ONLY 的记录只进日志文件，不上控制台。"""

    import logging

    from app.logging_service import FILE_ONLY, LogStore, setup_logging

    store = LogStore()
    logger = setup_logging(log_dir=tmp_path / "logs", store=store)
    try:
        logger.warning("只该进文件的提示", extra={FILE_ONLY: True})
        logger.warning("该上控制台的提示")
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    snapshot = "\n".join(store.snapshot())
    # 内存日志（日志页）两条都在
    assert "只该进文件的提示" in snapshot
    assert "该上控制台的提示" in snapshot

    # 控制台 handler 的过滤器挡住 FILE_ONLY
    console = logging.StreamHandler()
    from app.logging_service import _ConsoleFilter

    console.addFilter(_ConsoleFilter())
    record = logging.LogRecord(
        "oilcheck", logging.WARNING, __file__, 1, "x", None, None
    )
    record.__dict__[FILE_ONLY] = True
    # Handler.filter 放行时返回 LogRecord 本身（不是 True），拦截时返回 False
    assert not console.filter(record)
    del record.__dict__[FILE_ONLY]
    assert console.filter(record)


def test_log_file_rotates_across_midnight(tmp_path):
    """跨午夜必须真的切出新文件，且 rolloverAt 前进。

    原实现用 ``namer`` 把轮转目标改写成 app-<日期>.log，启动当天第一次
    午夜轮转的目标名与 baseFilename 完全相同，doRollover 命中「目标已存在」
    直接 return：既不关流也不推进 rolloverAt，此后日志全堆在启动日文件里。
    """

    from logging.handlers import TimedRotatingFileHandler

    from app.logging_service import LogStore, setup_logging

    store = LogStore()
    logger = setup_logging(log_dir=tmp_path / "logs", store=store)
    try:
        handler = next(
            h for h in logger.handlers if isinstance(h, TimedRotatingFileHandler)
        )
        logger.info("午夜前的一条")
        # 强制轮转：白天调用时 computeRollover 算出的下一个午夜不变，所以
        # 这里只断言「文件真的切走了」，不断言 rolloverAt 数值。
        handler.doRollover()
        logger.info("午夜后的一条")
        handler.flush()
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    files = sorted(p.name for p in (tmp_path / "logs").glob("app-*.log*"))
    # 轮转产出一个「baseFilename.日期」的备份文件，且备份里是午夜前的内容。
    # 原实现（namer 改写目标名）下目标与 baseFilename 相同，doRollover
    # 直接 return，根本不会产生这个备份文件。
    backups = [name for name in files if ".log." in name]
    assert backups, files
    base = tmp_path / "logs" / f"app-{time.strftime('%Y-%m-%d')}.log"
    rotated = tmp_path / "logs" / backups[0]
    assert "午夜前的一条" in rotated.read_text(encoding="utf-8")
    # 新写的内容留在 baseFilename 里
    assert "午夜后的一条" in base.read_text(encoding="utf-8")


def test_config_text_accepts_bom_and_gbk(tmp_path):
    """带 BOM 的 UTF-8 与 GBK 编码的配置都要能读。"""

    import json

    from app.config_service import ConfigService, read_config_text

    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + json.dumps({"name": "工位"}).encode("utf-8"))
    assert json.loads(read_config_text(bom)) == {"name": "工位"}

    gbk = tmp_path / "gbk.json"
    gbk.write_bytes(json.dumps({"name": "一号工位"}, ensure_ascii=False).encode("gbk"))
    assert json.loads(read_config_text(gbk)) == {"name": "一号工位"}

    # 端到端：app.json 为 GBK 时不再抛 UnicodeDecodeError
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "app.json").write_bytes(
        json.dumps({"ui_fps": 20, "log_dir": "日志"}, ensure_ascii=False).encode("gbk")
    )
    service = ConfigService(repo_root=tmp_path)
    cfg = service.load_app_config()
    assert cfg.ui_fps == 20
    assert cfg.log_dir == "日志"


def test_app_config_bad_field_types_fall_back_to_defaults(tmp_path):
    """字段类型非法时退回默认值，不能抛异常（那会崩在日志就绪之前）。"""

    from app.config_service import ConfigService

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "app.json").write_text(
        '{"ui_fps": "abc", "layout": 1e400, "decode_max_fps": null}',
        encoding="utf-8",
    )
    cfg = ConfigService(repo_root=tmp_path).load_app_config()
    assert cfg.ui_fps == 15
    assert cfg.layout == 4
    assert cfg.decode_max_fps == 15


def test_cameras_parse_failure_raises_instead_of_empty_list(tmp_path):
    """cameras.json 读不出来必须抛错，不能返回 []。

    返回 [] 会被设备页当成「相机被删光了」，下一次增删改把 [] 写回磁盘，
    两台相机的 IP/账号/口令全部丢失。
    """

    from app.config_service import CamerasConfigError, ConfigService

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "cameras.json").write_text('[{"id": "cam1",}]', encoding="utf-8")
    service = ConfigService(repo_root=tmp_path)
    with pytest.raises(CamerasConfigError):
        service.load_cameras()


def test_pipeline_from_dict_round_trips_every_writable_field():
    """每个 to_dict 导出的键，写入非默认值后都必须生效。

    字段声明、to_dict、from_dict 是三份各自维护的列表，漏一处就「改了
    配置不生效」。这里按 to_dict 全量校验，新增字段漏写会直接失败。
    """

    from app.oilcheck.config import PipelineConfig

    default = PipelineConfig()
    base = default.to_dict()
    # 白名单字段：合法取值不是「原值加后缀」，单独给一个合法替代值。
    enum_probes = {
        "hand_tracker_backend": "object",
        "brush_tracker_backend": "bytetrack",
    }
    mismatches: list[tuple[str, object, object]] = []
    for key, value in base.items():
        if key in enum_probes:
            probe: object = enum_probes[key]
        elif isinstance(value, bool):
            probe = not value
        elif isinstance(value, int):
            probe = value + 1
        elif isinstance(value, float):
            probe = value + 0.01
        elif isinstance(value, str):
            probe = value + "x"
        else:
            continue
        got = PipelineConfig.from_dict({key: probe}).to_dict()[key]
        if got != probe:
            mismatches.append((key, probe, got))
    assert mismatches == []


@pytest.mark.parametrize(
    "payload",
    [
        {"pick_confirm_s": float("nan")},
        {"brush_hold_s": float("nan")},
        {"brush_contact_distance": float("nan")},
        {"oiling_timeout_s": float("nan")},
    ],
)
def test_pipeline_nan_thresholds_fall_back_to_defaults(payload):
    """NaN 不能静默夹成 0.0（那会让一组阈值同时失效）。"""

    from app.oilcheck.config import PipelineConfig

    (key,), = [tuple(payload)]
    cfg = PipelineConfig.from_dict(payload)
    assert getattr(cfg, key) == getattr(PipelineConfig(), key)


def test_pipeline_overflow_number_falls_back():
    """1e400 会被 json 解析成 inf，int(inf) 抛 OverflowError。"""

    from app.oilcheck.config import PipelineConfig

    assert PipelineConfig.from_dict({"max_fps": 1e400}).max_fps == 15


def test_plc_string_booleans_are_parsed():
    """``"false"`` 必须解析成 False，否则 PLC 线程会被意外启用。"""

    from app.oilcheck.config import PlcConfig, StationConfig

    plc = PlcConfig.from_dict({"enabled": "false", "dry_run": "false"})
    assert plc.enabled is False
    assert plc.dry_run is False
    station = StationConfig.from_dict(
        {
            "station_id": "s",
            "camera_id": "c",
            "tray_roi": {"kind": "rect", "points": [[0, 0], [1, 1]]},
            "enabled": "false",
        }
    )
    assert station is not None and station.enabled is False


def test_plc_step_order_is_validated():
    """进度灯占前 4 盏，结果灯在其后；oiling 必须 = pick + 2。

    位 = 灯号：pick=1 只在「漏涂时灯1 本来就该红」时可用；oiling≠pick+2
    会让绿灯位数不单调；进度灯超过第 4 盏会占掉结果灯的位置；结果灯与
    进度灯重叠或超出灯数都不允许。
    """

    from app.oilcheck.config import PlcConfig

    default = {"pick": 2, "oiling": 4, "result": 5}
    for steps in (
        {"pick": 0, "oiling": 2},  # pick 从 1 起
        {"pick": 3, "oiling": 3},  # oiling != pick+2
        {"pick": 5, "oiling": 2},  # 倒序
        {"pick": 2, "oiling": 5},  # oiling != pick+2
        {"pick": 7, "oiling": 9},  # 超出灯数
        {"pick": 4, "oiling": 6},  # 进度灯占掉第 5 盏
        {"pick": 2, "oiling": 4, "result": 3},  # 结果灯与进度灯重叠
    ):
        assert PlcConfig.from_dict({"steps": steps}).steps == default, steps
    # 合法组合原样保留（pick=3/oiling=5 不行——进度灯只占前 4 盏）
    assert PlcConfig.from_dict({"steps": {"pick": 1, "oiling": 3}}).steps == {
        "pick": 1,
        "oiling": 3,
        "result": 5,
    }
    # 结果灯可以关掉，也可以往后挪
    assert PlcConfig.from_dict(
        {"steps": {"pick": 2, "oiling": 4, "result": 0}}
    ).steps == {"pick": 2, "oiling": 4, "result": 0}
    assert PlcConfig.from_dict(
        {"steps": {"pick": 2, "oiling": 4, "result": 8}}
    ).steps == {"pick": 2, "oiling": 4, "result": 8}


def test_migration_keeps_station_with_only_pick_roi(tmp_path):
    """只写 pick_roi 的 2.0 工位，迁移后必须转成 tray_roi 而不是被丢弃。"""

    import json

    from app.oilcheck.config import OilCheckConfigService

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "oilcheck.json").write_text(
        json.dumps(
            {
                "version": "2.0",
                "stations": [
                    {
                        "station_id": "s1",
                        "camera_id": "cam1",
                        "pick_roi": {"kind": "rect", "points": [[0.1, 0.1], [0.5, 0.5]]},
                        "hand_roi": {"kind": "rect", "points": [[0, 0], [1, 1]]},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cfg = OilCheckConfigService(config_dir=cfg_dir).load()
    assert [s.station_id for s in cfg.stations] == ["s1"]
    assert cfg.stations[0].tray_roi.to_dict()["points"] == [[0.1, 0.1], [0.5, 0.5]]


def test_env_password_missing_raises(monkeypatch):
    """``env:`` 引用的变量缺失时必须报错，不能静默用空口令登录。"""

    from app.config_service import ConfigService

    monkeypatch.delenv("OILCHECK_TEST_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="OILCHECK_TEST_PASSWORD"):
        ConfigService.resolve_password("env:OILCHECK_TEST_PASSWORD")
    with pytest.raises(ValueError, match="变量名"):
        ConfigService.resolve_password("env:")
    monkeypatch.setenv("OILCHECK_TEST_PASSWORD", "s3cret")
    assert ConfigService.resolve_password("env:OILCHECK_TEST_PASSWORD") == "s3cret"
    # 明文口令不受影响
    assert ConfigService.resolve_password("plain") == "plain"
