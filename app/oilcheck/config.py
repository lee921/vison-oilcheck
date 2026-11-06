"""取料/二次涂油业务配置。

配置文件默认位于数据目录的 ``config/oilcheck.json``。仓库 ``config/oilcheck.json``
只作为首次运行的模板，实际运行配置由 :class:`OilCheckConfigService` 复制并维护。
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.paths import runtime_root

from .roi import Roi
from .types import BBox

logger = logging.getLogger("oilcheck.config")


@dataclass(frozen=True)
class SlotLayout:
    """槽位相对机器涂油盘边界的局部归一化几何。

    ``x1/y1/x2/y2`` 均为相对 tray 检测框的 0~1 比例；盘体上下移动时随盘体
    一起平移，不需要重新标定绝对 ROI。
    """

    index: int
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlotLayout | None:
        try:
            index = int(data.get("index", 0))
            x1 = _clamp01(float(data.get("x1", 0.0)))
            y1 = _clamp01(float(data.get("y1", 0.0)))
            x2 = _clamp01(float(data.get("x2", 0.0)))
            y2 = _clamp01(float(data.get("y2", 0.0)))
        except (TypeError, ValueError):
            return None
        if index < 1:
            return None
        left, right = sorted((x1, x2))
        top, bottom = sorted((y1, y2))
        if right - left <= 1e-6 or bottom - top <= 1e-6:
            return None
        return cls(
            index=index,
            name=str(data.get("name") or f"{index}号槽"),
            x1=left,
            y1=top,
            x2=right,
            y2=bottom,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
        }

    def pixel_bbox(self, tray_box: BBox) -> BBox:
        width = max(1.0, tray_box.width)
        height = max(1.0, tray_box.height)
        return BBox(
            tray_box.x1 + self.x1 * width,
            tray_box.y1 + self.y1 * height,
            tray_box.x1 + self.x2 * width,
            tray_box.y1 + self.y2 * height,
        )


def default_slot_layouts(count: int = 4) -> tuple[SlotLayout, ...]:
    """默认 4 槽横向均分布局；现场应按实际盘体几何调整。"""
    count = max(1, int(count))
    step = 1.0 / count
    margin = min(0.02, step * 0.15)
    return tuple(
        SlotLayout(
            index=index + 1,
            name=f"{index + 1}号槽",
            x1=index * step + margin,
            y1=0.15,
            x2=(index + 1) * step - margin,
            y2=0.90,
        )
        for index in range(count)
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class SlotConfig:
    """一个取料位置。设备最多 4 根丝杆，对应 1~4 号位。"""

    index: int
    name: str
    roi: Roi

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SlotConfig | None:
        roi = Roi.from_dict(data.get("roi"))
        if roi is None:
            return None
        try:
            index = int(data.get("index", 0))
        except (TypeError, ValueError):
            return None
        if index < 1:
            return None
        return cls(
            index=index,
            name=str(data.get("name") or f"{index}号位"),
            roi=roi,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "name": self.name, "roi": self.roi.to_dict()}


@dataclass(frozen=True)
class DetectorConfig:
    """检测器配置。

    ``kind`` 支持：

    - ``null``：不产生检测结果，用于无模型的第一阶段框架运行。
    - ``simulated``：按时间脚本生成手部/丝杆检测，用于 UI 与流程联调。
    - ``yolo``：预留给外部训练项目导出的模型，本版本不实现推理。
    """

    kind: str = "null"
    model_path: str = ""
    confidence: float = 0.5
    hand_label: str = "hand"
    rod_label: str = "rod"
    brush_label: str = "brush"
    tray_label: str = "tray"
    cycle_seconds: float = 18.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DetectorConfig:
        data = dict(data or {})
        kind = str(data.get("kind", "null")).strip().lower()
        if kind not in {"null", "simulated", "yolo"}:
            kind = "null"
        try:
            confidence = float(data.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5
        try:
            cycle_seconds = float(data.get("cycle_seconds", 18.0))
        except (TypeError, ValueError):
            cycle_seconds = 18.0
        return cls(
            kind=kind,
            model_path=str(data.get("model_path", "")),
            confidence=max(0.0, min(1.0, confidence)),
            hand_label=str(data.get("hand_label", "hand") or "hand"),
            rod_label=str(data.get("rod_label") or data.get("screw_label") or "rod"),
            brush_label=str(data.get("brush_label", "brush") or "brush"),
            tray_label=str(data.get("tray_label", "tray") or "tray"),
            cycle_seconds=max(4.0, cycle_seconds),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "model_path": self.model_path,
            "confidence": self.confidence,
            "hand_label": self.hand_label,
            "rod_label": self.rod_label,
            "brush_label": self.brush_label,
            "tray_label": self.tray_label,
            "cycle_seconds": self.cycle_seconds,
        }


@dataclass(frozen=True)
class PipelineConfig:
    """时序事件判定参数。所有时间单位为秒。"""

    max_fps: int = 15
    pick_confirm_s: float = 0.6
    pick_abort_s: float = 0.5
    occlusion_timeout_s: float = 2.0
    pick_require_hand: bool = True
    hand_proximity_margin: float = 0.03
    oiling_min_duration_s: float = 1.5
    oiling_max_gap_s: float = 0.8
    oiling_min_motion: float = 0.04
    oiling_require_rod: bool = True
    oiling_wait_timeout_s: float = 0.0
    event_cooldown_s: float = 1.0
    part_reset_absent_s: float = 0.5
    # 动态跟踪（物料盘连续移动）参数
    track_max_distance: float = 0.08
    track_max_missing_s: float = 0.7
    track_min_hits: int = 2
    track_max_active: int = 8
    track_hand_distance: float = 0.06
    track_pick_window_s: float = 1.5
    completed_retain_s: float = 10.0
    # 机器涂油盘与动态槽位
    tray_max_distance: float = 0.20
    tray_max_missing_s: float = 1.0
    tray_min_confidence: float = 0.30
    tray_motion_speed: float = 0.05
    slot_occupied_confirm_s: float = 0.25
    slot_empty_confirm_s: float = 0.60
    slot_occlusion_unknown_s: float = 2.0
    slot_hand_occlusion_ratio: float = 0.15
    slot_hand_margin: float = 0.03
    # 取料与操作身份
    pick_window_s: float = 2.0
    hand_track_max_distance: float = 0.15
    hand_track_max_missing_s: float = 0.8
    hand_track_min_hits: int = 1
    brush_track_max_distance: float = 0.15
    brush_track_max_missing_s: float = 0.8
    brush_track_min_hits: int = 1
    # 二次涂油：刷子相对物料的运动证据
    brush_part_distance: float = 0.08
    brush_min_duration_s: float = 1.0
    brush_max_gap_s: float = 0.6
    brush_min_motion: float = 0.15
    brush_require_hand: bool = True
    identity_timeout_s: float = 2.0
    identity_ambiguity_margin: float = 0.02
    operation_max_active: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PipelineConfig:
        data = dict(data or {})
        d = cls()

        def _f(key: str, default: float) -> float:
            try:
                return float(data.get(key, default))
            except (TypeError, ValueError):
                return default

        def _b(key: str, default: bool) -> bool:
            value = data.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        def _i(key: str, default: int) -> int:
            try:
                return int(data.get(key, default))
            except (TypeError, ValueError):
                return default

        return cls(
            max_fps=max(1, min(60, _i("max_fps", d.max_fps))),
            pick_confirm_s=max(0.0, _f("pick_confirm_s", d.pick_confirm_s)),
            pick_abort_s=max(0.0, _f("pick_abort_s", d.pick_abort_s)),
            occlusion_timeout_s=max(
                0.0, _f("occlusion_timeout_s", d.occlusion_timeout_s)
            ),
            pick_require_hand=_b("pick_require_hand", d.pick_require_hand),
            hand_proximity_margin=max(
                0.0, min(0.5, _f("hand_proximity_margin", d.hand_proximity_margin))
            ),
            oiling_min_duration_s=max(
                0.0, _f("oiling_min_duration_s", d.oiling_min_duration_s)
            ),
            oiling_max_gap_s=max(0.0, _f("oiling_max_gap_s", d.oiling_max_gap_s)),
            oiling_min_motion=max(0.0, _f("oiling_min_motion", d.oiling_min_motion)),
            oiling_require_rod=_b("oiling_require_rod", d.oiling_require_rod),
            oiling_wait_timeout_s=max(
                0.0, _f("oiling_wait_timeout_s", d.oiling_wait_timeout_s)
            ),
            event_cooldown_s=max(0.0, _f("event_cooldown_s", d.event_cooldown_s)),
            part_reset_absent_s=max(
                0.0, _f("part_reset_absent_s", d.part_reset_absent_s)
            ),
            track_max_distance=max(
                0.005, min(0.5, _f("track_max_distance", d.track_max_distance))
            ),
            track_max_missing_s=max(
                0.0, _f("track_max_missing_s", d.track_max_missing_s)
            ),
            track_min_hits=max(1, min(20, _i("track_min_hits", d.track_min_hits))),
            track_max_active=max(
                1, min(64, _i("track_max_active", d.track_max_active))
            ),
            track_hand_distance=max(
                0.005, min(0.5, _f("track_hand_distance", d.track_hand_distance))
            ),
            track_pick_window_s=max(
                0.0, _f("track_pick_window_s", d.track_pick_window_s)
            ),
            completed_retain_s=max(0.0, _f("completed_retain_s", d.completed_retain_s)),
            tray_max_distance=max(
                0.01, min(0.5, _f("tray_max_distance", d.tray_max_distance))
            ),
            tray_max_missing_s=max(0.0, _f("tray_max_missing_s", d.tray_max_missing_s)),
            tray_min_confidence=max(
                0.0, min(1.0, _f("tray_min_confidence", d.tray_min_confidence))
            ),
            tray_motion_speed=max(
                0.001, min(0.5, _f("tray_motion_speed", d.tray_motion_speed))
            ),
            slot_occupied_confirm_s=max(
                0.0, _f("slot_occupied_confirm_s", d.slot_occupied_confirm_s)
            ),
            slot_empty_confirm_s=max(
                0.0, _f("slot_empty_confirm_s", d.slot_empty_confirm_s)
            ),
            slot_occlusion_unknown_s=max(
                0.0,
                _f("slot_occlusion_unknown_s", d.slot_occlusion_unknown_s),
            ),
            slot_hand_occlusion_ratio=max(
                0.0,
                min(1.0, _f("slot_hand_occlusion_ratio", d.slot_hand_occlusion_ratio)),
            ),
            slot_hand_margin=max(
                0.0, min(0.5, _f("slot_hand_margin", d.slot_hand_margin))
            ),
            pick_window_s=max(0.0, _f("pick_window_s", d.pick_window_s)),
            hand_track_max_distance=max(
                0.01,
                min(0.5, _f("hand_track_max_distance", d.hand_track_max_distance)),
            ),
            hand_track_max_missing_s=max(
                0.0, _f("hand_track_max_missing_s", d.hand_track_max_missing_s)
            ),
            hand_track_min_hits=max(
                1, min(20, _i("hand_track_min_hits", d.hand_track_min_hits))
            ),
            brush_track_max_distance=max(
                0.01,
                min(0.5, _f("brush_track_max_distance", d.brush_track_max_distance)),
            ),
            brush_track_max_missing_s=max(
                0.0, _f("brush_track_max_missing_s", d.brush_track_max_missing_s)
            ),
            brush_track_min_hits=max(
                1, min(20, _i("brush_track_min_hits", d.brush_track_min_hits))
            ),
            brush_part_distance=max(
                0.005, min(0.5, _f("brush_part_distance", d.brush_part_distance))
            ),
            brush_min_duration_s=max(
                0.0,
                _f(
                    "brush_min_duration_s",
                    _f("oiling_min_duration_s", d.brush_min_duration_s),
                ),
            ),
            brush_max_gap_s=max(
                0.0,
                _f("brush_max_gap_s", _f("oiling_max_gap_s", d.brush_max_gap_s)),
            ),
            brush_min_motion=max(0.0, _f("brush_min_motion", d.brush_min_motion)),
            brush_require_hand=_b("brush_require_hand", d.brush_require_hand),
            identity_timeout_s=max(0.0, _f("identity_timeout_s", d.identity_timeout_s)),
            identity_ambiguity_margin=max(
                0.0,
                min(0.2, _f("identity_ambiguity_margin", d.identity_ambiguity_margin)),
            ),
            operation_max_active=max(
                1, min(64, _i("operation_max_active", d.operation_max_active))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_fps": self.max_fps,
            "pick_confirm_s": self.pick_confirm_s,
            "pick_abort_s": self.pick_abort_s,
            "occlusion_timeout_s": self.occlusion_timeout_s,
            "pick_require_hand": self.pick_require_hand,
            "hand_proximity_margin": self.hand_proximity_margin,
            "oiling_min_duration_s": self.oiling_min_duration_s,
            "oiling_max_gap_s": self.oiling_max_gap_s,
            "oiling_min_motion": self.oiling_min_motion,
            "oiling_require_rod": self.oiling_require_rod,
            "oiling_wait_timeout_s": self.oiling_wait_timeout_s,
            "event_cooldown_s": self.event_cooldown_s,
            "part_reset_absent_s": self.part_reset_absent_s,
            "track_max_distance": self.track_max_distance,
            "track_max_missing_s": self.track_max_missing_s,
            "track_min_hits": self.track_min_hits,
            "track_max_active": self.track_max_active,
            "track_hand_distance": self.track_hand_distance,
            "track_pick_window_s": self.track_pick_window_s,
            "completed_retain_s": self.completed_retain_s,
            "tray_max_distance": self.tray_max_distance,
            "tray_max_missing_s": self.tray_max_missing_s,
            "tray_min_confidence": self.tray_min_confidence,
            "tray_motion_speed": self.tray_motion_speed,
            "slot_occupied_confirm_s": self.slot_occupied_confirm_s,
            "slot_empty_confirm_s": self.slot_empty_confirm_s,
            "slot_occlusion_unknown_s": self.slot_occlusion_unknown_s,
            "slot_hand_occlusion_ratio": self.slot_hand_occlusion_ratio,
            "slot_hand_margin": self.slot_hand_margin,
            "pick_window_s": self.pick_window_s,
            "hand_track_max_distance": self.hand_track_max_distance,
            "hand_track_max_missing_s": self.hand_track_max_missing_s,
            "hand_track_min_hits": self.hand_track_min_hits,
            "brush_track_max_distance": self.brush_track_max_distance,
            "brush_track_max_missing_s": self.brush_track_max_missing_s,
            "brush_track_min_hits": self.brush_track_min_hits,
            "brush_part_distance": self.brush_part_distance,
            "brush_min_duration_s": self.brush_min_duration_s,
            "brush_max_gap_s": self.brush_max_gap_s,
            "brush_min_motion": self.brush_min_motion,
            "brush_require_hand": self.brush_require_hand,
            "identity_timeout_s": self.identity_timeout_s,
            "identity_ambiguity_margin": self.identity_ambiguity_margin,
            "operation_max_active": self.operation_max_active,
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
class StationConfig:
    station_id: str
    name: str
    camera_id: str
    enabled: bool
    pick_roi: Roi
    oiling_roi: Roi
    slots: tuple[SlotConfig, ...]
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    slot_mode: str = "tray"
    tray_roi: Roi | None = None
    slot_layout: tuple[SlotLayout, ...] = field(default_factory=default_slot_layouts)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StationConfig | None:
        pick_roi = Roi.from_dict(data.get("pick_roi"))
        oiling_roi = Roi.from_dict(data.get("oiling_roi"))
        if pick_roi is None or oiling_roi is None:
            return None
        tray_roi = Roi.from_dict(data.get("tray_roi")) or pick_roi

        slots: list[SlotConfig] = []
        for item in data.get("slots", ()):
            if not isinstance(item, dict):
                continue
            slot = SlotConfig.from_dict(item)
            if slot is not None:
                slots.append(slot)
        slots.sort(key=lambda s: s.index)

        layouts: list[SlotLayout] = []
        raw_layout = data.get("slot_layout")
        if isinstance(raw_layout, list):
            for item in raw_layout:
                if not isinstance(item, dict):
                    continue
                layout = SlotLayout.from_dict(item)
                if layout is not None:
                    layouts.append(layout)
        if not layouts:
            try:
                count = int(data.get("slot_count", 4))
            except (TypeError, ValueError):
                count = 4
            layouts = list(default_slot_layouts(max(1, min(16, count))))
        layouts.sort(key=lambda item: item.index)

        station_id = str(data.get("station_id", "")).strip()
        camera_id = str(data.get("camera_id", "")).strip()
        slot_mode = str(data.get("slot_mode", "tray")).strip().lower()
        if slot_mode not in {"fixed", "dynamic", "tray"}:
            slot_mode = "tray"
        if not station_id or not camera_id:
            return None
        if slot_mode == "fixed" and not slots:
            return None

        return cls(
            station_id=station_id,
            name=str(data.get("name") or station_id),
            camera_id=camera_id,
            enabled=bool(data.get("enabled", True)),
            pick_roi=pick_roi,
            oiling_roi=oiling_roi,
            slots=tuple(slots),
            detector=DetectorConfig.from_dict(data.get("detector")),
            slot_mode=slot_mode,
            tray_roi=tray_roi,
            slot_layout=tuple(layouts),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "name": self.name,
            "camera_id": self.camera_id,
            "enabled": self.enabled,
            "pick_roi": self.pick_roi.to_dict(),
            "oiling_roi": self.oiling_roi.to_dict(),
            "tray_roi": (self.tray_roi or self.pick_roi).to_dict(),
            "slot_layout": [layout.to_dict() for layout in self.slot_layout],
            "slots": [slot.to_dict() for slot in self.slots],
            "detector": self.detector.to_dict(),
            "slot_mode": self.slot_mode,
        }


@dataclass(frozen=True)
class OilCheckConfig:
    version: str = "1.0"
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
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
            version=str(data.get("version", "1.0")),
            pipeline=PipelineConfig.from_dict(data.get("pipeline")),
            recording=RecordingConfig.from_dict(data.get("recording")),
            stations=tuple(stations),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pipeline": self.pipeline.to_dict(),
            "recording": self.recording.to_dict(),
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


def iter_station_ids(stations: Iterable[StationConfig]) -> tuple[str, ...]:
    return tuple(station.station_id for station in stations)
