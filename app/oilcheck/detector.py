"""检测器接口与 YOLO 实现。

模型训练在另一个项目中完成，本项目只消费导出权重：

- 类别与训练导出一致：``0=hand``、``1=brush``、``2=oil_station``、``3=screw``；
- 推理尺寸默认自适应训练时的 ``imgsz``（见 :func:`_resolve_imgsz`）；
- 只回答“这一帧里有什么”，跨帧证据与状态流转在 ``pipeline.py`` 中完成。

``null`` 实现用于无模型环境，保证 UI 与框架可运行且不伪造任何检测。
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from app.paths import resolve_runtime_file, runtime_root

from .config import DetectorConfig, StationConfig
from .types import BBox, Detection, FrameDetections

logger = logging.getLogger("oilcheck.detector")

# Ultralytics 的 FPN 要求输入边长为 stride 的整数倍。
_MODEL_STRIDE = 32
_FALLBACK_IMGSZ = 640


@runtime_checkable
class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray, ts: float) -> FrameDetections:
        """返回该帧的检测结果。实现必须线程安全地只读访问自身状态。"""

    def close(self) -> None:
        """释放模型等资源。"""


class NullDetector:
    """不产生任何检测结果。

    未接入模型时使用。状态机会稳定停留在“等待取料”，不伪造事件，
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


class YoloDetector:
    """Ultralytics YOLO 权重适配器（普通 Detect 输出）。

    - 权重路径支持相对仓库根目录（``model/detect/exp/weights/best.pt``）；
    - ``imgsz=0`` 时自适应 checkpoint 中训练时的输入尺寸；
    - ``device`` 为空或 ``auto`` 时自动选：有 CUDA 用 ``cuda:0``，否则 ``cpu``；
    - 类别名原样输出（``hand`` / ``brush`` / ``oil_station`` / ``screw``），
      与配置文件中的标签字段一一对应。
    """

    name = "yolo"

    def __init__(self, station: StationConfig, cfg: DetectorConfig) -> None:
        self._station = station
        self._cfg = cfg
        self._model: Any = None
        self._imgsz = _FALLBACK_IMGSZ
        self._names: dict[int, str] = {}

        if not cfg.model_path:
            raise RuntimeError("detector.kind=yolo 但未配置 model_path")
        model_path = Path(
            resolve_runtime_file(cfg.model_path, repo_root=runtime_root())
        )
        if not model_path.exists():
            raise FileNotFoundError(f"检测模型不存在: {model_path}")

        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "当前环境未安装 ultralytics。请先执行 `uv sync` 安装推理依赖，"
                "或将 detector.kind 设为 null。"
            ) from exc

        self._model = YOLO(str(model_path))
        self._imgsz = _resolve_imgsz(self._model, cfg.imgsz)
        self._names = _model_names(self._model)
        self._device = _resolve_device(cfg.device)
        logger.info(
            "检测模型已加载: %s（imgsz=%d, device=%s, half=%s, 类别=%s）",
            model_path,
            self._imgsz,
            self._device,
            bool(cfg.half and self._device.startswith("cuda")),
            ",".join(f"{k}={v}" for k, v in sorted(self._names.items())),
        )

    @property
    def imgsz(self) -> int:
        return self._imgsz

    @property
    def device(self) -> str:
        return self._device

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    def detect(self, frame: np.ndarray, ts: float) -> FrameDetections:
        h, w = frame.shape[:2]
        if self._model is None:
            return FrameDetections(ts=ts, frame_size=(w, h), source=self.name)

        kwargs: dict[str, Any] = {
            "conf": float(self._cfg.confidence),
            "iou": float(self._cfg.iou),
            "max_det": int(self._cfg.max_det),
            "imgsz": int(self._imgsz),
            "device": self._device,
            "verbose": False,
        }
        # Ultralytics 8.4 起对 half 给出弃用告警；只在 CUDA 上真正开启时传。
        if self._cfg.half and self._device.startswith("cuda"):
            kwargs["half"] = True
        results = self._model.predict(frame, **kwargs)
        detections: list[Detection] = []
        if results:
            result = results[0]
            names = getattr(result, "names", None) or self._names
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                detections = _detections_from_boxes(boxes, names)

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
    """按配置创建检测器。``override_kind`` 用于 UI 临时切换（不写回配置）。"""
    kind = (override_kind or station.detector.kind or "null").strip().lower()
    if kind == "yolo":
        return YoloDetector(station, station.detector)
    return NullDetector()


def _resolve_imgsz(model: Any, configured: int) -> int:
    """确定推理输入尺寸。

    优先级：显式配置 > checkpoint 训练尺寸 > 640 兜底。
    结果向上取整到 stride 的整数倍，Ultralytics 才能正常构建张量。
    """

    value = configured
    if value <= 0:
        args = getattr(getattr(model, "model", None), "args", None)
        if isinstance(args, dict):
            raw = args.get("imgsz")
        elif args is not None:
            raw = getattr(args, "imgsz", None)
        else:
            raw = None
        if isinstance(raw, (list, tuple)) and raw:
            try:
                value = max(int(v) for v in raw)
            except (TypeError, ValueError):
                value = 0
        elif isinstance(raw, (int, float)):
            value = int(raw)
        elif isinstance(raw, str) and raw.strip().isdigit():
            value = int(raw.strip())
        else:
            value = 0

    if value <= 0:
        return _FALLBACK_IMGSZ
    return int(math.ceil(value / _MODEL_STRIDE) * _MODEL_STRIDE)


def _resolve_device(configured: str) -> str:
    """确定推理设备。

    ``auto`` / 空串表示跟随硬件：有 CUDA 用 ``cuda:0``，否则 ``cpu``。
    显式配置（``cpu`` / ``cuda:0`` / ``0``）原样使用，便于现场固定行为。
    """

    value = (configured or "").strip()
    if value and value.lower() not in {"auto", "default"}:
        return value

    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def _model_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        try:
            return {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError):
            return {}
    if isinstance(names, (list, tuple)):
        return {index: str(name) for index, name in enumerate(names)}
    return {}


def _label(names: Any, cls_id: int) -> str:
    try:
        return str(names.get(cls_id, cls_id))
    except AttributeError:
        return str(cls_id)


def _detections_from_boxes(boxes: Any, names: Any) -> list[Detection]:
    items: list[Detection] = []
    for box in boxes:
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
            )
        )
    return items
