from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

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

    assert cfg.version == "2.6"
    assert data["version"] == "2.6"
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

    assert cfg.version == "2.6"
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

    assert cfg.version == "2.6"
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


def test_missing_demo_section_is_filled_with_template_videos(tmp_path):
    """demo 整段缺失时补回模板清单，不能补成空列表。

    补成 ``videos: []`` 的文件看起来一切正常（demo 段在、视频文件也在），
    监控页下拉里却一条示例都没有，且不报任何错——现场最难排查的一种。
    """

    from app.oilcheck.config import OilCheckConfigService

    _write_config(tmp_path, lambda data: data.pop("demo"))
    cfg = OilCheckConfigService(tmp_path / "config").load()

    assert [video.name for video in cfg.demo.videos] == [
        "示例① 105708",
        "示例② 140911",
    ]


def test_missing_videos_key_is_filled_with_template_videos(tmp_path):
    """demo 段在、videos 键缺失（早期版本保存出来的形状）也要补回清单。"""

    from app.oilcheck.config import OilCheckConfigService

    _write_config(tmp_path, lambda data: data["demo"].pop("videos"))
    cfg = OilCheckConfigService(tmp_path / "config").load()

    assert cfg.demo.videos, "缺 videos 键不应被补成空清单"


def test_legacy_empty_videos_list_is_refilled_once(tmp_path):
    """2.0 版本保存出来的空清单要修回来，且只改一次。

    早期版本没有 demo 段时，UI 保存会把 ``videos: []`` 写进文件；之后按版本号
    补 demo 的分支不再触发，服务器上从此一条示例视频都没有。升级到 2.6 时
    补回模板清单——这份空清单不是用户主动清空的。
    """

    from app.oilcheck.config import OilCheckConfigService, load_json_with_comments

    def _make_legacy(data) -> None:  # noqa: ANN001
        data["version"] = "2.5"
        data["demo"]["videos"] = []

    path = _write_config(tmp_path, _make_legacy)
    cfg = OilCheckConfigService(tmp_path / "config").load()
    assert cfg.demo.videos, "旧版本遗留的空清单应补回模板示例视频"

    # 用户之后主动清空（已是 2.6）必须保持为空，不能每次启动又加回来。
    data = load_json_with_comments(path)
    data["demo"]["videos"] = []
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    cfg = OilCheckConfigService(tmp_path / "config").load()
    assert cfg.demo.videos == ()
    assert load_json_with_comments(path)["demo"]["videos"] == []


def test_empty_videos_list_is_kept(tmp_path):
    """显式写成 ``videos: []`` 是用户主动清空，迁移不能改回去。"""

    from app.oilcheck.config import OilCheckConfigService, load_json_with_comments

    path = _write_config(tmp_path, lambda data: data["demo"].__setitem__("videos", []))
    cfg = OilCheckConfigService(tmp_path / "config").load()

    assert cfg.demo.videos == ()
    assert load_json_with_comments(path)["demo"]["videos"] == []


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
    assert json.loads(strip_json_comments(text))["version"] == "2.6"


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
