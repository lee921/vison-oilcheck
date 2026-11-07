"""检测器接口与第一阶段可用的实现。

模型训练在另一个项目中完成。本项目的职责是：

1. 定义稳定的检测结果契约（:class:`FrameDetections`）；
2. 提供 ``null`` 与 ``simulated`` 两种实现，保证无模型时 UI/状态机可运行；
3. 提供 YOLO 适配器骨架（Detect / OBB / Keypoint），后续接入外部项目导出的权重。

类别按实现文档第 4 节：``screw``、``hand``、``brush_head`` 为核心，
``brush_rod``、``oil_tray`` 为辅助。检测器只负责“这一帧里有什么”，
跨帧证据与状态流转在 ``pipeline.py`` 中完成。
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .config import DetectorConfig, StationConfig
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

    第一阶段未接入模型时使用。状态机会稳定停留在“等待取料”，不会伪造事件，
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
    """演示检测器：模拟“取出一根丝杆 → 手持 → 刷头刷涂”的完整周期。

    每个周期的脚本（按 ``cycle_seconds`` 归一化）：

    1. 涂油盘内有多根丝杆，手从画面下方接近其中一根；
    2. 手与丝杆共同移动到手持操作区；
    3. 刷头进入丝杆邻域，沿丝杆轴向做往复运动，覆盖全部区段；
    4. 刷头离开，丝杆仍被手持；手返回，下一周期重新上料。

    演示数据不是真实现场结果，仅用于验证状态机与 UI。
    """

    name = "simulated"

    def __init__(self, station: StationConfig, cfg: DetectorConfig) -> None:
        self._station = station
        self._cfg = cfg
        self._start_ts: float | None = None
        self._cycle = max(12.0, float(cfg.cycle_seconds))
        self._cycle_index = -1
        self._operation_index = 0

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
            self._operation_index = cycle_index % 4

        tray_box = self._tray_box(w=w, h=h, elapsed=elapsed)
        detections: list[Detection] = [
            Detection(label=self._cfg.tray_label, confidence=0.92, box=tray_box)
        ]

        approach_end = 0.26
        move_end = 0.40
        brush_start = 0.44
        brush_end = 0.68
        release_phase = 0.80
        reload_phase = 0.94

        screw_boxes = self._tray_screw_boxes(tray_box, w=w, h=h)
        picked = self._operation_index
        tray_screws_visible = phase < approach_end or phase >= reload_phase
        if tray_screws_visible:
            for box in screw_boxes:
                detections.append(
                    Detection(
                        label=self._cfg.screw_label,
                        confidence=0.90,
                        box=box,
                        angle=math.pi * 0.5,
                    )
                )
        else:
            for index, box in enumerate(screw_boxes):
                if index != picked:
                    detections.append(
                        Detection(
                            label=self._cfg.screw_label,
                            confidence=0.90,
                            box=box,
                            angle=math.pi * 0.5,
                        )
                    )

        target_center = screw_boxes[picked].center
        hand_box = self._station.hand_roi.pixel_bbox(w, h)
        hand_center = hand_box.center
        start_center = (target_center[0], min(h * 0.98, target_center[1] + h * 0.25))

        hand_center_now: tuple[float, float] | None = None
        held_center: tuple[float, float] | None = None

        if 0.02 <= phase < approach_end:
            t = _smoothstep(
                (phase - 0.02) / max(1e-6, approach_end - 0.02)
            )
            hand_center_now = (
                _lerp(start_center[0], target_center[0], t),
                _lerp(start_center[1], target_center[1], t),
            )
        elif approach_end <= phase < move_end:
            t = _smoothstep(
                (phase - approach_end) / max(1e-6, move_end - approach_end)
            )
            hand_center_now = (
                _lerp(target_center[0], hand_center[0], t),
                _lerp(target_center[1], hand_center[1], t),
            )
            held_center = hand_center_now
        elif move_end <= phase < release_phase:
            # 手持丝杆在操作区内轻微移动；刷涂时可以更稳定地保持中心。
            angle = (phase - move_end) / max(1e-6, release_phase - move_end)
            offset = math.sin(angle * math.pi * 2.0) * hand_box.width * 0.05
            hand_center_now = (hand_center[0] + offset, hand_center[1])
            held_center = hand_center_now
        elif release_phase <= phase < reload_phase:
            t = _smoothstep(
                (phase - release_phase) / max(1e-6, reload_phase - release_phase)
            )
            hand_center_now = (
                _lerp(hand_center[0], start_center[0], t),
                _lerp(hand_center[1], start_center[1], t),
            )

        if hand_center_now is not None:
            detections.append(
                Detection(
                    label=self._cfg.hand_label,
                    confidence=0.90,
                    box=_box_around(hand_center_now, w, h, ratio=0.09),
                )
            )

        if held_center is not None:
            screw_box = _held_screw_box(held_center, w, h)
            detections.append(
                Detection(
                    label=self._cfg.screw_label,
                    confidence=0.88,
                    box=screw_box,
                    angle=0.0,
                )
            )

        if brush_start <= phase < brush_end and held_center is not None:
            t = (phase - brush_start) / max(1e-6, brush_end - brush_start)
            # 刷头工作端沿丝杆轴向来回移动，覆盖整根丝杆。
            screw_box = _held_screw_box(held_center, w, h)
            half_len = screw_box.width * 0.46
            work_point = (
                held_center[0] + math.sin(t * math.pi * 8.0) * half_len,
                held_center[1],
            )
            detections.append(
                Detection(
                    label=self._cfg.brush_head_label,
                    confidence=0.87,
                    box=_brush_head_box(work_point, w),
                    point=work_point,
                )
            )
            if self._cfg.brush_rod_label:
                detections.append(
                    Detection(
                        label=self._cfg.brush_rod_label,
                        confidence=0.80,
                        box=BBox(
                            work_point[0],
                            work_point[1] - max(6.0, h * 0.01),
                            work_point[0] + max(70.0, w * 0.06),
                            work_point[1] + max(6.0, h * 0.01),
                        ),
                        angle=0.0,
                    )
                )

        return FrameDetections(
            ts=now,
            frame_size=(int(w), int(h)),
            detections=tuple(detections),
            source=self.name,
        )

    def _tray_box(self, *, w: int, h: int, elapsed: float) -> BBox:
        base = self._station.tray_roi.pixel_bbox(w, h)
        tray_w = max(60.0, base.width * 0.88)
        tray_h = max(40.0, base.height * 0.74)
        cx = base.center[0]
        cy = base.center[1] + math.sin(elapsed * math.pi / 3.0) * base.height * 0.08
        x1 = max(0.0, min(float(w) - tray_w, cx - tray_w * 0.5))
        y1 = max(0.0, min(float(h) - tray_h, cy - tray_h * 0.5))
        return BBox(x1, y1, x1 + tray_w, y1 + tray_h)

    def _tray_screw_boxes(self, tray: BBox, *, w: int, h: int) -> list[BBox]:
        boxes: list[BBox] = []
        for ratio in (0.18, 0.40, 0.62, 0.84):
            cx = tray.x1 + tray.width * ratio
            cy = tray.center[1]
            rw = max(8.0, w * 0.009)
            rh = max(36.0, tray.height * 0.42)
            boxes.append(BBox(cx - rw, cy - rh * 0.5, cx + rw, cy + rh * 0.5))
        return boxes


class YoloDetector:
    """外部训练项目导出权重的适配器。

    支持 Ultralytics 的普通检测框、OBB 与关键点输出：

    - 普通框：``angle`` 为空，流水线用框的长边估计丝杆主轴；
    - OBB：读取长轴角度；
    - 关键点：每个实例的第一个关键点作为 ``point``（刷头工作端）。
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
            points = _keypoint_points(result)
            obb = getattr(result, "obb", None)
            if obb is not None and len(obb) > 0:
                detections.extend(_detections_from_obb(obb, names, points))
            else:
                boxes = getattr(result, "boxes", None)
                if boxes is not None:
                    detections.extend(_detections_from_boxes(boxes, names, points))

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


def _keypoint_points(result: Any) -> list[tuple[float, float] | None]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return []
    xy = getattr(keypoints, "xy", None)
    if xy is None:
        return []
    try:
        data = xy.tolist()
    except (AttributeError, TypeError, ValueError):
        return []
    points: list[tuple[float, float] | None] = []
    for instance in data:
        if not instance:
            points.append(None)
            continue
        first = instance[0]
        try:
            points.append((float(first[0]), float(first[1])))
        except (IndexError, TypeError, ValueError):
            points.append(None)
    return points


def _label(names: Any, cls_id: int) -> str:
    try:
        return str(names.get(cls_id, cls_id))
    except AttributeError:
        return str(cls_id)


def _detections_from_boxes(
    boxes: Any,
    names: Any,
    points: list[tuple[float, float] | None],
) -> list[Detection]:
    items: list[Detection] = []
    for index, box in enumerate(boxes):
        try:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        except (IndexError, TypeError, ValueError):
            continue
        items.append(
            Detection(
                label=_label(names, cls_id),
                confidence=conf,
                box=BBox(x1, y1, x2, y2),
                point=points[index] if index < len(points) else None,
            )
        )
    return items


def _detections_from_obb(
    obb: Any,
    names: Any,
    points: list[tuple[float, float] | None],
) -> list[Detection]:
    items: list[Detection] = []
    try:
        xywhr = obb.xywhr.tolist()
        cls_ids = obb.cls.tolist()
        confs = obb.conf.tolist()
    except (AttributeError, TypeError, ValueError):
        return items
    for index, (cx, cy, width, height, angle) in enumerate(xywhr):
        try:
            cls_id = int(cls_ids[index])
            conf = float(confs[index])
        except (IndexError, TypeError, ValueError):
            continue
        box = _rotated_bbox(float(cx), float(cy), float(width), float(height), float(angle))
        items.append(
            Detection(
                label=_label(names, cls_id),
                confidence=conf,
                box=box,
                angle=float(angle),
                point=points[index] if index < len(points) else None,
            )
        )
    return items


def _rotated_bbox(
    cx: float, cy: float, width: float, height: float, angle: float
) -> BBox:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    hw, hh = width * 0.5, height * 0.5
    xs: list[float] = []
    ys: list[float] = []
    for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        xs.append(cx + dx * cos_a - dy * sin_a)
        ys.append(cy + dx * sin_a + dy * cos_a)
    return BBox(min(xs), min(ys), max(xs), max(ys))


def _box_around(
    center: tuple[float, float],
    width: int,
    height: int,
    *,
    ratio: float = 0.16,
) -> BBox:
    size = max(48.0, min(width, height) * ratio)
    cx, cy = center
    return BBox(cx - size * 0.5, cy - size * 0.5, cx + size * 0.5, cy + size * 0.5)


def _held_screw_box(
    center: tuple[float, float], width: int, height: int
) -> BBox:
    rw = max(70.0, width * 0.09)
    rh = max(10.0, height * 0.025)
    cx, cy = center
    return BBox(cx - rw * 0.5, cy - rh * 0.5, cx + rw * 0.5, cy + rh * 0.5)


def _brush_head_box(center: tuple[float, float], width: int) -> BBox:
    size = max(14.0, width * 0.018)
    cx, cy = center
    return BBox(
        cx - size * 0.5,
        cy - size * 0.5,
        cx + size * 0.5,
        cy + size * 0.5,
    )


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
    """调试辅助：判断 ROI 是否落在工位有效推理区域内。"""
    return station.tray_roi.intersects_box(roi.pixel_bbox(width, height), width, height)
