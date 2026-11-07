"""取料/二次涂油业务配置。

配置结构对应《涂油工位视觉检测项目实现文档》第 3、10 节：

- 每个工位配置涂油盘、手持操作、刷涂三个有效 ROI，以及需要完全排除的
  物料盒 ROI；
- 不配置槽位，也不做槽位绑定；
- 判定参数按“取料事件 / 跟踪 / 刷涂 / 任务”分组。

配置文件默认位于数据目录的 ``config/oilcheck.json``。仓库
``config/oilcheck.json`` 只作为首次运行的模板，实际运行配置由
:class:`OilCheckConfigService` 复制并维护。旧版配置中的 ``pick_roi``、
``oiling_roi``、``slot_layout`` 等字段在读取时会自动迁移为新的 ROI。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.paths import runtime_root

from .roi import Roi

logger = logging.getLogger("oilcheck.config")

# 配置结构版本。升级时 ensure_defaults 会把新增字段合并进已有配置文件。
_CONFIG_VERSION = "2.1"

# 仓库自带的权重（相对仓库根解析）。
_DEFAULT_MODEL_PATH = "model/detect/exp/weights/best.pt"

# 2.1 新增的检测器字段及其默认值，迁移时补齐。
_DETECTOR_DEFAULTS_2_1: dict[str, Any] = {
    "iou": 0.7,
    "max_det": 100,
    "imgsz": 0,
    "device": "auto",
    "half": False,
    "raw_mode": True,
    "brush_label": "brush",
    "tray_label": "oil_station",
}


@dataclass(frozen=True)
class DetectorConfig:
    """检测器配置。

    ``kind`` 支持：

    - ``null``：不产生检测结果，用于无模型时的框架自检。
    - ``yolo``：Ultralytics YOLO 权重（普通 Detect 输出）。

    类别名与训练导出模型保持一致（``best.pt`` 的 4 个类别）：

    - ``0=hand``、``1=brush``、``2=oil_station``、``3=screw``。

    ``imgsz=0`` 表示自适应：优先取权重 checkpoint 中训练时的 ``imgsz``，
    取不到再回退 640，避免用 640 推理 1024 训练的模型而丢精度。

    ``raw_mode=True`` 时流水线只做“把这一帧的全部检测结果交出来”，跳过
    ROI 过滤、跟踪、取料/刷涂状态机与事件产出，用于先确认模型标签是否齐全。
    默认关闭，按需在配置里打开；关闭时走完整的 SOP 判定流程。
    """

    kind: str = "null"
    model_path: str = ""
    confidence: float = 0.5
    iou: float = 0.7
    max_det: int = 100
    imgsz: int = 0
    # 空串/auto = 有 CUDA 用 cuda:0，否则 cpu；可显式写 cpu / cuda:0 / 0。
    device: str = ""
    half: bool = False
    raw_mode: bool = False
    screw_label: str = "screw"
    hand_label: str = "hand"
    brush_label: str = "brush"
    tray_label: str = "oil_station"
    # 模型只输出刷具框、没有关键点时，用框内相对坐标估计工作端。
    brush_point_offset_x: float = 0.5
    brush_point_offset_y: float = 0.9

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DetectorConfig:
        data = dict(data or {})
        d = cls()
        kind = str(data.get("kind", "null")).strip().lower()
        if kind not in {"null", "yolo"}:
            kind = "null"

        def _f(key: str, default: float, lo: float, hi: float) -> float:
            try:
                return max(lo, min(hi, float(data.get(key, default))))
            except (TypeError, ValueError):
                return default

        def _i(key: str, default: int, lo: int, hi: int) -> int:
            try:
                return max(lo, min(hi, int(data.get(key, default))))
            except (TypeError, ValueError):
                return default

        def _b(key: str, default: bool) -> bool:
            value = data.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        return cls(
            kind=kind,
            model_path=str(data.get("model_path", "")),
            confidence=_f("confidence", d.confidence, 0.0, 1.0),
            iou=_f("iou", d.iou, 0.0, 1.0),
            max_det=_i("max_det", d.max_det, 1, 1000),
            # 0 表示自适应训练尺寸；非 0 时会被向上取整到 stride 的整数倍。
            imgsz=_i("imgsz", d.imgsz, 0, 8192),
            device=str(data.get("device") or d.device).strip(),
            half=_b("half", d.half),
            raw_mode=_b("raw_mode", d.raw_mode),
            # 旧配置：rod_label → screw_label，brush_head_label → brush_label。
            screw_label=str(
                data.get("screw_label") or data.get("rod_label") or d.screw_label
            ),
            hand_label=str(data.get("hand_label") or d.hand_label),
            brush_label=str(
                data.get("brush_label")
                or data.get("brush_head_label")
                or d.brush_label
            ),
            tray_label=str(data.get("tray_label") or d.tray_label),
            brush_point_offset_x=_f(
                "brush_point_offset_x", d.brush_point_offset_x, 0.0, 1.0
            ),
            brush_point_offset_y=_f(
                "brush_point_offset_y", d.brush_point_offset_y, 0.0, 1.0
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "model_path": self.model_path,
            "confidence": self.confidence,
            "iou": self.iou,
            "max_det": self.max_det,
            "imgsz": self.imgsz,
            "device": self.device,
            "half": self.half,
            "raw_mode": self.raw_mode,
            "screw_label": self.screw_label,
            "hand_label": self.hand_label,
            "brush_label": self.brush_label,
            "tray_label": self.tray_label,
            "brush_point_offset_x": self.brush_point_offset_x,
            "brush_point_offset_y": self.brush_point_offset_y,
        }


@dataclass(frozen=True)
class PipelineConfig:
    """时序事件判定参数。所有时间单位为秒，距离为按画面对角线归一化。"""

    max_fps: int = 15

    # 取料事件（文档 5）
    pick_hand_distance: float = 0.05
    pick_motion_min: float = 0.010
    pick_relative_max: float = 0.030
    pick_comotion_min_s: float = 0.25
    pick_confirm_s: float = 0.30
    pick_abort_s: float = 0.50
    pick_occlusion_timeout_s: float = 2.0
    pick_ambiguity_margin: float = 0.02
    pick_window_s: float = 2.0

    # 目标跟踪（文档 6）
    screw_track_max_distance: float = 0.08
    screw_track_max_missing_s: float = 0.70
    screw_track_min_hits: int = 2
    screw_track_max_active: int = 16
    hand_track_max_distance: float = 0.15
    hand_track_max_missing_s: float = 0.80
    hand_track_min_hits: int = 1
    brush_track_max_distance: float = 0.15
    brush_track_max_missing_s: float = 0.80
    brush_track_min_hits: int = 1

    # 刷涂行为（文档 7）
    brush_contact_distance: float = 0.060
    brush_min_duration_s: float = 1.0
    brush_max_gap_s: float = 0.60
    brush_min_motion: float = 0.15
    brush_require_hand: bool = False
    coverage_segments: int = 5
    coverage_required: float = 0.60

    # 业务身份与任务（文档 6、8、10）
    identity_timeout_s: float = 2.0
    identity_ambiguity_margin: float = 0.02
    holding_lost_timeout_s: float = 2.0
    oiling_wait_timeout_s: float = 0.0
    single_active_task: bool = True
    task_max_active: int = 4
    completed_retain_s: float = 10.0

    # 涂油盘定位（只作为取料参考区域）
    tray_max_distance: float = 0.20
    tray_max_missing_s: float = 1.0
    tray_min_confidence: float = 0.30

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PipelineConfig:
        data = dict(data or {})
        d = cls()

        def _f(key: str, default: float, lo: float = 0.0) -> float:
            try:
                value = float(data.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(lo, value)

        def _b(key: str, default: bool) -> bool:
            value = data.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        def _i(key: str, default: int, lo: int, hi: int) -> int:
            try:
                value = int(data.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(lo, min(hi, value))

        def _legacy(new_key: str, *old_keys: str, default: float) -> float:
            """新字段缺失时回退到旧配置字段，便于现场配置平滑迁移。"""

            if new_key in data:
                return _f(new_key, default)
            for old_key in old_keys:
                if old_key in data:
                    return _f(old_key, default)
            return default

        return cls(
            max_fps=_i("max_fps", d.max_fps, 1, 60),
            pick_hand_distance=_legacy(
                "pick_hand_distance",
                "slot_hand_margin",
                "hand_proximity_margin",
                default=d.pick_hand_distance,
            ),
            pick_motion_min=_f("pick_motion_min", d.pick_motion_min, 0.0),
            pick_relative_max=_f("pick_relative_max", d.pick_relative_max, 0.0),
            pick_comotion_min_s=_f(
                "pick_comotion_min_s", d.pick_comotion_min_s, 0.0
            ),
            pick_confirm_s=_f("pick_confirm_s", d.pick_confirm_s, 0.0),
            pick_abort_s=_f("pick_abort_s", d.pick_abort_s, 0.0),
            pick_occlusion_timeout_s=_legacy(
                "pick_occlusion_timeout_s",
                "occlusion_timeout_s",
                default=d.pick_occlusion_timeout_s,
            ),
            pick_ambiguity_margin=_legacy(
                "pick_ambiguity_margin",
                "identity_ambiguity_margin",
                default=d.pick_ambiguity_margin,
            ),
            pick_window_s=_f("pick_window_s", d.pick_window_s, 0.0),
            screw_track_max_distance=_legacy(
                "screw_track_max_distance",
                "track_max_distance",
                default=d.screw_track_max_distance,
            ),
            screw_track_max_missing_s=_legacy(
                "screw_track_max_missing_s",
                "track_max_missing_s",
                default=d.screw_track_max_missing_s,
            ),
            screw_track_min_hits=_i(
                "screw_track_min_hits",
                _int_or(data.get("track_min_hits"), d.screw_track_min_hits),
                1,
                20,
            ),
            screw_track_max_active=_i(
                "screw_track_max_active",
                _int_or(data.get("track_max_active"), d.screw_track_max_active),
                1,
                64,
            ),
            hand_track_max_distance=_f(
                "hand_track_max_distance", d.hand_track_max_distance, 0.005
            ),
            hand_track_max_missing_s=_f(
                "hand_track_max_missing_s", d.hand_track_max_missing_s, 0.0
            ),
            hand_track_min_hits=_i(
                "hand_track_min_hits", d.hand_track_min_hits, 1, 20
            ),
            brush_track_max_distance=_f(
                "brush_track_max_distance", d.brush_track_max_distance, 0.005
            ),
            brush_track_max_missing_s=_f(
                "brush_track_max_missing_s", d.brush_track_max_missing_s, 0.0
            ),
            brush_track_min_hits=_i(
                "brush_track_min_hits", d.brush_track_min_hits, 1, 20
            ),
            brush_contact_distance=_legacy(
                "brush_contact_distance",
                "brush_part_distance",
                default=d.brush_contact_distance,
            ),
            brush_min_duration_s=_legacy(
                "brush_min_duration_s",
                "oiling_min_duration_s",
                default=d.brush_min_duration_s,
            ),
            brush_max_gap_s=_legacy(
                "brush_max_gap_s",
                "oiling_max_gap_s",
                default=d.brush_max_gap_s,
            ),
            brush_min_motion=_legacy(
                "brush_min_motion",
                "oiling_min_motion",
                default=d.brush_min_motion,
            ),
            brush_require_hand=_b("brush_require_hand", d.brush_require_hand),
            coverage_segments=_i(
                "coverage_segments", d.coverage_segments, 1, 16
            ),
            coverage_required=max(
                0.0,
                min(1.0, _f("coverage_required", d.coverage_required, 0.0)),
            ),
            identity_timeout_s=_f(
                "identity_timeout_s", d.identity_timeout_s, 0.0
            ),
            identity_ambiguity_margin=_f(
                "identity_ambiguity_margin", d.identity_ambiguity_margin, 0.0
            ),
            holding_lost_timeout_s=_f(
                "holding_lost_timeout_s", d.holding_lost_timeout_s, 0.0
            ),
            oiling_wait_timeout_s=_f(
                "oiling_wait_timeout_s", d.oiling_wait_timeout_s, 0.0
            ),
            single_active_task=_b(
                "single_active_task", d.single_active_task
            ),
            task_max_active=_i(
                "task_max_active",
                int(data.get("operation_max_active", d.task_max_active))
                if "task_max_active" not in data
                else d.task_max_active,
                1,
                64,
            ),
            completed_retain_s=_f(
                "completed_retain_s", d.completed_retain_s, 0.0
            ),
            tray_max_distance=_f(
                "tray_max_distance", d.tray_max_distance, 0.01
            ),
            tray_max_missing_s=_f(
                "tray_max_missing_s", d.tray_max_missing_s, 0.0
            ),
            tray_min_confidence=max(
                0.0,
                min(1.0, _f("tray_min_confidence", d.tray_min_confidence, 0.0)),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_fps": self.max_fps,
            "pick_hand_distance": self.pick_hand_distance,
            "pick_motion_min": self.pick_motion_min,
            "pick_relative_max": self.pick_relative_max,
            "pick_comotion_min_s": self.pick_comotion_min_s,
            "pick_confirm_s": self.pick_confirm_s,
            "pick_abort_s": self.pick_abort_s,
            "pick_occlusion_timeout_s": self.pick_occlusion_timeout_s,
            "pick_ambiguity_margin": self.pick_ambiguity_margin,
            "pick_window_s": self.pick_window_s,
            "screw_track_max_distance": self.screw_track_max_distance,
            "screw_track_max_missing_s": self.screw_track_max_missing_s,
            "screw_track_min_hits": self.screw_track_min_hits,
            "screw_track_max_active": self.screw_track_max_active,
            "hand_track_max_distance": self.hand_track_max_distance,
            "hand_track_max_missing_s": self.hand_track_max_missing_s,
            "hand_track_min_hits": self.hand_track_min_hits,
            "brush_track_max_distance": self.brush_track_max_distance,
            "brush_track_max_missing_s": self.brush_track_max_missing_s,
            "brush_track_min_hits": self.brush_track_min_hits,
            "brush_contact_distance": self.brush_contact_distance,
            "brush_min_duration_s": self.brush_min_duration_s,
            "brush_max_gap_s": self.brush_max_gap_s,
            "brush_min_motion": self.brush_min_motion,
            "brush_require_hand": self.brush_require_hand,
            "coverage_segments": self.coverage_segments,
            "coverage_required": self.coverage_required,
            "identity_timeout_s": self.identity_timeout_s,
            "identity_ambiguity_margin": self.identity_ambiguity_margin,
            "holding_lost_timeout_s": self.holding_lost_timeout_s,
            "oiling_wait_timeout_s": self.oiling_wait_timeout_s,
            "single_active_task": self.single_active_task,
            "task_max_active": self.task_max_active,
            "completed_retain_s": self.completed_retain_s,
            "tray_max_distance": self.tray_max_distance,
            "tray_max_missing_s": self.tray_max_missing_s,
            "tray_min_confidence": self.tray_min_confidence,
        }


@dataclass(frozen=True)
class RecordingConfig:
    """训练视频录制参数。"""

    training_dir: str = "recordings/training"
    fps: int = 15
    segment_seconds: float = 600.0
    codec: str = "mp4v"
    auto_start: bool = False
    event_clips_enabled: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RecordingConfig:
        data = dict(data or {})
        d = cls()

        def _f(key: str, default: float) -> float:
            try:
                return float(data.get(key, default))
            except (TypeError, ValueError):
                return default

        def _i(key: str, default: int) -> int:
            try:
                return int(data.get(key, default))
            except (TypeError, ValueError):
                return default

        def _b(key: str, default: bool) -> bool:
            value = data.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        return cls(
            training_dir=str(data.get("training_dir") or d.training_dir),
            fps=max(1, min(60, _i("fps", d.fps))),
            segment_seconds=max(10.0, _f("segment_seconds", d.segment_seconds)),
            codec=str(data.get("codec") or d.codec),
            auto_start=_b("auto_start", d.auto_start),
            event_clips_enabled=_b("event_clips_enabled", d.event_clips_enabled),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_dir": self.training_dir,
            "fps": self.fps,
            "segment_seconds": self.segment_seconds,
            "codec": self.codec,
            "auto_start": self.auto_start,
            "event_clips_enabled": self.event_clips_enabled,
        }


@dataclass(frozen=True)
class DemoVideo:
    """一段示例视频：显示名 + 相对仓库根或绝对路径。"""

    name: str
    path: str

    @classmethod
    def from_dict(cls, data: Any) -> DemoVideo | None:
        if isinstance(data, str):
            name, path = "", data
        elif isinstance(data, dict):
            path = str(data.get("path", "")).strip()
            name = str(data.get("name", "")).strip()
        else:
            return None
        if not path:
            return None
        if not name:
            name = Path(path).stem
        return cls(name=name, path=path)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path}


@dataclass(frozen=True)
class DemoConfig:
    """演示视频源配置。

    演示模式不再使用脚本伪造检测，而是直接回放 ``data/sample_video`` 下的
    真实录像：画面来自视频文件，检测结果来自训练好的模型。
    """

    loop: bool = True
    fps_limit: int = 15
    videos: tuple[DemoVideo, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DemoConfig:
        data = dict(data or {})
        d = cls()
        loop = data.get("loop", d.loop)
        if isinstance(loop, str):
            loop = loop.strip().lower() in {"1", "true", "yes", "on"}
        try:
            fps_limit = int(data.get("fps_limit", d.fps_limit))
        except (TypeError, ValueError):
            fps_limit = d.fps_limit

        videos: list[DemoVideo] = []
        for item in data.get("videos", ()) or ():
            video = DemoVideo.from_dict(item)
            if video is not None:
                videos.append(video)
        return cls(
            loop=bool(loop),
            fps_limit=max(1, min(60, fps_limit)),
            videos=tuple(videos),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop": self.loop,
            "fps_limit": self.fps_limit,
            "videos": [video.to_dict() for video in self.videos],
        }

    def video(self, name: str) -> DemoVideo | None:
        for video in self.videos:
            if video.name == name:
                return video
        return None


@dataclass(frozen=True)
class StationConfig:
    """一个工位的有效 ROI 与检测配置。

    ``exclude_rois`` 内的区域完全屏蔽推理（物料盒等）。ROI 使用归一化坐标，
    现场调整后无需改代码。
    """

    station_id: str
    name: str
    camera_id: str
    enabled: bool
    tray_roi: Roi
    hand_roi: Roi
    brush_roi: Roi
    exclude_rois: tuple[Roi, ...] = ()
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StationConfig | None:
        # 兼容旧配置：pick_roi → tray_roi，oiling_roi → hand_roi。
        tray_roi = Roi.from_dict(data.get("tray_roi") or data.get("pick_roi"))
        hand_roi = Roi.from_dict(data.get("hand_roi") or data.get("oiling_roi"))
        if tray_roi is None or hand_roi is None:
            return None
        # 刷涂区默认取人工涂油操作台范围（画面右下方），
        # 与涂油盘区（tray_roi / oil_station）分开配置。
        brush_roi = Roi.from_dict(data.get("brush_roi")) or hand_roi

        exclude_rois: list[Roi] = []
        raw_exclude = data.get("exclude_rois")
        if raw_exclude is None:
            raw_exclude = data.get("material_box_rois")
        if raw_exclude is None and data.get("material_box_roi") is not None:
            raw_exclude = [data.get("material_box_roi")]
        if isinstance(raw_exclude, dict):
            raw_exclude = [raw_exclude]
        for item in raw_exclude or ():
            roi = Roi.from_dict(item)
            if roi is not None:
                exclude_rois.append(roi)

        station_id = str(data.get("station_id", "")).strip()
        camera_id = str(data.get("camera_id", "")).strip()
        if not station_id or not camera_id:
            return None

        return cls(
            station_id=station_id,
            name=str(data.get("name") or station_id),
            camera_id=camera_id,
            enabled=bool(data.get("enabled", True)),
            tray_roi=tray_roi,
            hand_roi=hand_roi,
            brush_roi=brush_roi,
            exclude_rois=tuple(exclude_rois),
            detector=DetectorConfig.from_dict(data.get("detector")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "name": self.name,
            "camera_id": self.camera_id,
            "enabled": self.enabled,
            "tray_roi": self.tray_roi.to_dict(),
            "hand_roi": self.hand_roi.to_dict(),
            "brush_roi": self.brush_roi.to_dict(),
            "exclude_rois": [roi.to_dict() for roi in self.exclude_rois],
            "detector": self.detector.to_dict(),
        }


@dataclass(frozen=True)
class OilCheckConfig:
    version: str = "2.1"
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    stations: tuple[StationConfig, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OilCheckConfig:
        stations: list[StationConfig] = []
        for item in data.get("stations", ()):
            if not isinstance(item, dict):
                continue
            station = StationConfig.from_dict(item)
            if station is not None:
                stations.append(station)
        return cls(
            version=str(data.get("version", _CONFIG_VERSION)),
            pipeline=PipelineConfig.from_dict(data.get("pipeline")),
            recording=RecordingConfig.from_dict(data.get("recording")),
            demo=DemoConfig.from_dict(data.get("demo")),
            stations=tuple(stations),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pipeline": self.pipeline.to_dict(),
            "recording": self.recording.to_dict(),
            "demo": self.demo.to_dict(),
            "stations": [station.to_dict() for station in self.stations],
        }

    def station(self, station_id: str) -> StationConfig | None:
        for station in self.stations:
            if station.station_id == station_id:
                return station
        return None

    def enabled_stations(self) -> tuple[StationConfig, ...]:
        return tuple(s for s in self.stations if s.enabled)


class OilCheckConfigService:
    """读写 ``<数据目录>/config/oilcheck.json``。"""

    def __init__(self, data_root: Path) -> None:
        self._data_root = Path(data_root).resolve()

    @property
    def config_dir(self) -> Path:
        return self._data_root / "config"

    @property
    def config_path(self) -> Path:
        return self.config_dir / "oilcheck.json"

    def ensure_defaults(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            self._migrate_existing()
            return
        template = runtime_root() / "config" / "oilcheck.json"
        if template.exists() and template.resolve() != self.config_path.resolve():
            try:
                shutil.copy2(template, self.config_path)
                logger.info("已从模板复制检测配置: %s", self.config_path)
                return
            except OSError as exc:
                logger.warning("复制检测配置模板失败: %s", exc)
        self.save(OilCheckConfig())

    def _migrate_existing(self) -> None:
        """把新增配置项合并进已存在的配置文件。

        已有配置不会被模板覆盖（现场 ROI 等不能丢），但旧版本缺少
        ``demo`` 段、``detector`` 仍是 ``null`` 时，服务器拉取新代码后
        会出现“接入不了模型 / 没有示例视频”的假故障。这里按版本号
        做一次幂等合并，并在改动前备份原文件。
        """

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("检测配置无法解析，跳过迁移 (%s): %s", self.config_path, exc)
            return
        if not isinstance(data, dict):
            return

        old_version = str(data.get("version", "2.0"))
        try:
            outdated = tuple(int(x) for x in old_version.split(".")) < (2, 1)
        except ValueError:
            outdated = True

        changed: list[str] = []

        # 1) demo 段：仅对旧版本补上模板里的示例视频清单。
        #    已是当前版本但缺 demo（例如用户主动删掉），尊重用户选择不再加回。
        if outdated and not isinstance(data.get("demo"), dict):
            template = runtime_root() / "config" / "oilcheck.json"
            demo = None
            if template.exists():
                try:
                    tpl = json.loads(template.read_text(encoding="utf-8"))
                    demo = tpl.get("demo")
                except (OSError, json.JSONDecodeError):
                    demo = None
            # 模板里也没有示例视频时不要凭空写一个空段，否则每次启动都会改文件。
            if isinstance(demo, dict) and demo:
                data["demo"] = demo
                changed.append("demo")

        # 2) detector：旧配置的 null / 空路径不会自动用上仓库权重
        for station in data.get("stations", ()) or ():
            if not isinstance(station, dict):
                continue
            detector = station.get("detector")
            if not isinstance(detector, dict):
                station["detector"] = detector = {}
            kind = str(detector.get("kind", "null")).strip().lower()
            if outdated and kind in {"", "null"}:
                detector["kind"] = "yolo"
                changed.append("detector.kind")
            if outdated and not str(detector.get("model_path", "")).strip():
                detector["model_path"] = _DEFAULT_MODEL_PATH
                changed.append("detector.model_path")
            for key, value in _DETECTOR_DEFAULTS_2_1.items():
                if key not in detector:
                    detector[key] = value
                    changed.append(f"detector.{key}")

            # 旧类别名与训练模型（0=hand 1=brush 2=oil_station 3=screw）
            # 不一致，模型只输出 oil_station，配置成 oil_tray 会一个盘体都匹配不到。
            if outdated:
                legacy_labels = {
                    "tray_label": "oil_tray",
                    "brush_label": "brush_head",
                }
                for key, legacy in legacy_labels.items():
                    if str(detector.get(key, "")).strip() == legacy:
                        detector[key] = _DETECTOR_DEFAULTS_2_1[key]
                        changed.append(f"detector.{key}")

        if not changed:
            return

        backup = self.config_path.with_suffix(".json.bak")
        try:
            shutil.copy2(self.config_path, backup)
        except OSError as exc:
            logger.warning("备份检测配置失败: %s", exc)

        data["version"] = _CONFIG_VERSION
        try:
            _write_json_atomic(self.config_path, data)
        except OSError as exc:
            logger.error("写回迁移后的检测配置失败: %s", exc)
            return
        logger.warning(
            "检测配置已升级到 %s，补充字段: %s（原文件备份为 %s）",
            _CONFIG_VERSION,
            ", ".join(dict.fromkeys(changed)),
            backup.name,
        )

    def load(self, *, create_if_missing: bool = True) -> OilCheckConfig:
        if create_if_missing:
            self.ensure_defaults()
        if not self.config_path.exists():
            return OilCheckConfig()
        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("加载检测配置失败 (%s): %s", self.config_path, exc)
            return OilCheckConfig()
        if not isinstance(data, dict):
            return OilCheckConfig()
        return OilCheckConfig.from_dict(data)

    def save(self, cfg: OilCheckConfig) -> None:
        _write_json_atomic(self.config_path, cfg.to_dict())


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_name).replace(path)
    finally:
        try:
            os.remove(tmp_name)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("删除临时文件失败 (%s): %s", tmp_name, exc)
