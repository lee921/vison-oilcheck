"""检测器接口与第一阶段可用的实现。

模型训练在另一个项目中完成。本项目的职责是：

1. 定义稳定的检测结果契约（:class:`FrameDetections`）；
2. 提供 ``null`` 与 ``simulated`` 两种实现，保证无模型时 UI/状态机可运行；
3. 提供 YOLO 适配器骨架，后续接入外部项目导出的权重。

检测器只负责“这一帧里有什么”，不做跨帧判断；跨帧证据与状态流转在
``pipeline.py`` 中完成。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .config import DetectorConfig, StationConfig, default_slot_layouts
from .roi import Roi
from .types import BBox, Detection, FrameDetections


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray, ts: float) -> FrameDetections:
        """返回该帧的检测结果。实现必须线程安全地只读访问自身状态。"""

    def close(self) -> None:
        """释放模型等资源。"""


class NullDetector:
    """不产生任何检测结果。

    第一阶段未接入模型时使用。状态机会稳定停留在“待取料”，不会伪造事件，
    适合先验证相机、ROI、录制和 UI。
    """

    name = "null"

    def detect(self, frame: np.ndarray, ts: float) -> FrameDetections:
        h, w = frame.shape[:2]
        return FrameDetections(
            ts=float(ts), frame_size=(int(w), int(h)), detections=(), source=self.name
        )

    def close(self) -> None:
        return None


class SimulatedDetector:
    """演示检测器：机器涂油盘上下移动、4 个槽位、取料与刷子涂油。

    每个周期的脚本：

    1. 手部接近一个 OCCUPIED 槽位并取走丝杆；
    2. 手部把丝杆带到人工涂油区；
    3. 刷子在丝杆附近做往复运动，提供“相对物料运动”证据；
    4. 其他槽位保持 OCCUPIED，模拟机器继续处理其他物料。

    演示数据不是真实现场结果，仅用于验证盘体/槽位/操作状态机和 UI。
    """

    name = "simulated"

    def __init__(self, station: StationConfig, cfg: DetectorConfig) -> None:
        self._station = station
        self._cfg = cfg
        self._start_ts: float | None = None
        self._cycle = max(12.0, float(cfg.cycle_seconds))
        self._cycle_index = -1
        self._operation_slot = 1

    def close(self) -> None:
        return None

    def detect(self, frame: np.ndarray, ts: float) -> FrameDetections:
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return FrameDetections(ts=ts, frame_size=(w, h), source=self.name)

        now = float(ts)
        if self._start_ts is None:
            self._start_ts = now
        elapsed = max(0.0, now - self._start_ts)
        cycle_index = int(elapsed // self._cycle)
        phase = (elapsed % self._cycle) / self._cycle
        if cycle_index != self._cycle_index:
            self._cycle_index = cycle_index
            self._operation_slot = (cycle_index % 4) + 1

        tray_box = self._tray_box(w=w, h=h, elapsed=elapsed)
        layouts = self._station.slot_layout or default_slot_layouts(4)
        detections: list[Detection] = [
            Detection(
                label=self._cfg.tray_label,
                confidence=0.92,
                box=tray_box,
            )
        ]

        pick_phase = 0.22
        oil_start = 0.32
        oil_end = 0.64
        leave_phase = 0.78
        reload_phase = 0.96

        operation_layout = None
        for layout in layouts:
            if layout.index == self._operation_slot:
                operation_layout = layout
                break
        if operation_layout is None:
            operation_layout = layouts[0]

        slot_boxes = {layout.index: layout.pixel_bbox(tray_box) for layout in layouts}
        picked = phase >= pick_phase and phase < reload_phase
        for layout in layouts:
            empty = layout.index == operation_layout.index and picked
            if empty:
                continue
            slot_box = slot_boxes[layout.index]
            detections.append(
                Detection(
                    label=self._cfg.rod_label,
                    confidence=0.90,
                    box=_shrink_box(slot_box, 0.55),
                )
            )

        slot_center = slot_boxes[operation_layout.index].center
        oil_box = self._station.oiling_roi.pixel_bbox(w, h)
        oil_center = oil_box.center
        hand_center: tuple[float, float] | None = None
        held_center: tuple[float, float] | None = None

        if 0.04 <= phase < pick_phase:
            # 手部从画面下方接近目标槽位，手框覆盖槽位造成短时遮挡。
            t = _smoothstep((phase - 0.04) / max(1e-6, pick_phase - 0.04))
            start = (slot_center[0], min(h * 0.98, slot_center[1] + h * 0.22))
            hand_center = (
                _lerp(start[0], slot_center[0], t),
                _lerp(start[1], slot_center[1], t),
            )
        elif pick_phase <= phase < oil_start:
            # 取走丝杆并移动到人工涂油区。
            t = _smoothstep((phase - pick_phase) / max(1e-6, oil_start - pick_phase))
            hand_center = (
                _lerp(slot_center[0], oil_center[0], t),
                _lerp(slot_center[1], oil_center[1], t),
            )
            held_center = hand_center
        elif oil_start <= phase < oil_end:
            # 在涂油区小幅移动，手中丝杆始终跟随。
            angle = (phase - oil_start) / max(1e-6, oil_end - oil_start)
            offset = math.sin(angle * math.pi * 2.0) * oil_box.width * 0.04
            hand_center = (oil_center[0] + offset, oil_center[1])
            held_center = hand_center
        elif oil_end <= phase < leave_phase:
            # 涂完后手部离开涂油区，不产生新的涂油证据。
            t = _smoothstep((phase - oil_end) / max(1e-6, leave_phase - oil_end))
            hand_center = (
                _lerp(oil_center[0], slot_center[0], t),
                _lerp(oil_center[1], slot_center[1], t),
            )
            held_center = hand_center

        if hand_center is not None:
            detections.append(
                Detection(
                    label=self._cfg.hand_label,
                    confidence=0.90,
                    box=_box_around(hand_center, w, h),
                )
            )
        if held_center is not None and phase < leave_phase:
            detections.append(
                Detection(
                    label=self._cfg.rod_label,
                    confidence=0.86,
                    box=_held_rod_box(held_center, w, h),
                )
            )

        if oil_start + 0.04 <= phase < oil_end - 0.04 and held_center is not None:
            # 刷子相对丝杆做往复运动：横向和纵向都有位移，提供相对运动证据。
            angle = (phase - oil_start) / max(1e-6, oil_end - oil_start)
            swing = math.sin(angle * math.pi * 6.0)
            amp_x = max(12.0, w * 0.035)
            amp_y = max(8.0, h * 0.018)
            brush_center = (
                held_center[0] + swing * amp_x,
                held_center[1] + math.cos(angle * math.pi * 6.0) * amp_y,
            )
            detections.append(
                Detection(
                    label=self._cfg.brush_label,
                    confidence=0.87,
                    box=_brush_box(brush_center, w, h),
                )
            )

        return FrameDetections(
            ts=now,
            frame_size=(int(w), int(h)),
            detections=tuple(detections),
            source=self.name,
        )

    def _tray_box(self, *, w: int, h: int, elapsed: float) -> BBox:
        base = (self._station.tray_roi or self._station.pick_roi).pixel_bbox(w, h)
        tray_w = max(40.0, base.width * 0.86)
        tray_h = max(30.0, base.height * 0.72)
        cx = base.center[0]
        cy = base.center[1] + math.sin(elapsed * math.pi / 3.0) * base.height * 0.10
        x1 = max(0.0, min(float(w) - tray_w, cx - tray_w * 0.5))
        y1 = max(0.0, min(float(h) - tray_h, cy - tray_h * 0.5))
        return BBox(x1, y1, x1 + tray_w, y1 + tray_h)


class YoloDetector:
    """外部训练项目导出权重的适配器骨架。

    本版本不把 ``ultralytics`` 作为依赖，调用时会给出明确错误。接入步骤见
    ``docs/04-视觉检测管线.md``。
    """

    name = "yolo"

    def __init__(self, station: StationConfig, cfg: DetectorConfig) -> None:
        self._station = station
        self._cfg = cfg
        self._model = None

        if not cfg.model_path:
            raise RuntimeError("detector.kind=yolo 但未配置 model_path")
        model_path = Path(cfg.model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"检测模型不存在: {model_path}")

        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "当前环境未安装 ultralytics。请先按 docs/04-视觉检测管线.md "
                "安装推理依赖，或将 detector.kind 设为 null/simulated。"
            ) from exc

        self._model = YOLO(str(model_path))

    def detect(self, frame: np.ndarray, ts: float) -> FrameDetections:
        h, w = frame.shape[:2]
        if self._model is None:
            return FrameDetections(ts=ts, frame_size=(w, h), source=self.name)

        results = self._model.predict(
            frame,
            conf=float(self._cfg.confidence),
            verbose=False,
        )
        detections: list[Detection] = []
        if results:
            result = results[0]
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    try:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    except (IndexError, TypeError, ValueError):
                        continue
                    label = str(names.get(cls_id, cls_id))
                    detections.append(
                        Detection(
                            label=label,
                            confidence=conf,
                            box=BBox(x1, y1, x2, y2),
                        )
                    )

        return FrameDetections(
            ts=float(ts),
            frame_size=(int(w), int(h)),
            detections=tuple(detections),
            source=self.name,
        )

    def close(self) -> None:
        self._model = None


def create_detector(
    station: StationConfig,
    override_kind: str | None = None,
) -> Detector:
    """按配置创建检测器。``override_kind`` 用于 UI 临时切换演示模式。"""
    kind = (override_kind or station.detector.kind or "null").strip().lower()
    if kind == "simulated":
        return SimulatedDetector(station, station.detector)
    if kind == "yolo":
        return YoloDetector(station, station.detector)
    return NullDetector()


def _box_around(center: tuple[float, float], width: int, height: int) -> BBox:
    size = max(48.0, min(width, height) * 0.16)
    cx, cy = center
    return BBox(cx - size * 0.5, cy - size * 0.5, cx + size * 0.5, cy + size * 0.5)


def _held_rod_box(center: tuple[float, float], width: int, height: int) -> BBox:
    rw = max(24.0, width * 0.10)
    rh = max(12.0, height * 0.045)
    cx, cy = center
    return BBox(cx - rw * 0.5, cy - rh * 0.5, cx + rw * 0.5, cy + rh * 0.5)


def _brush_box(center: tuple[float, float], width: int, height: int) -> BBox:
    size = max(14.0, min(width, height) * 0.025)
    cx, cy = center
    return BBox(
        cx - size * 0.5,
        cy - size * 0.5,
        cx + size * 0.5,
        cy + size * 0.5,
    )


def _shrink_box(box: BBox, ratio: float) -> BBox:
    cx, cy = box.center
    half_w = box.width * ratio * 0.5
    half_h = box.height * ratio * 0.5
    return BBox(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def now_ts() -> float:
    return time.time()


def roi_inside_station(
    roi: Roi, station: StationConfig, width: int, height: int
) -> bool:
    """调试辅助：判断 ROI 是否落在工位取料区内。"""
    return station.pick_roi.intersects_box(roi.pixel_bbox(width, height), width, height)
