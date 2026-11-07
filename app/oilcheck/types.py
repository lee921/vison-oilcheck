"""取料与二次涂油检测的领域模型。

本模块只放纯数据定义，不依赖 Qt、相机 SDK 或具体检测模型，方便单元测试与
后续替换推理实现。

模型对应《涂油工位视觉检测项目实现文档》：

- 视觉 Track ID 与业务任务 ID 分离；
- 单根丝杆任务（:class:`TaskSnapshot`）是唯一业务对象；
- 取料、手持、刷涂分别有独立状态字段，最终结果为 OK / NG /
  PENDING_CONFIRMATION / PROCESSING。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PickStatus(str, Enum):
    """取料状态（文档 5.1 / 10）。"""

    CANDIDATE = "CANDIDATE"
    CONFIRMED = "CONFIRMED"
    UNKNOWN = "UNKNOWN"


PICK_STATUS_LABELS: dict[PickStatus, str] = {
    PickStatus.CANDIDATE: "取料候选",
    PickStatus.CONFIRMED: "取料已确认",
    PickStatus.UNKNOWN: "取料未知",
}


class HoldingStatus(str, Enum):
    """手持丝杆关联状态（文档 6 / 10）。"""

    HOLDING = "HOLDING"
    LOST = "LOST"
    RELEASED = "RELEASED"


HOLDING_STATUS_LABELS: dict[HoldingStatus, str] = {
    HoldingStatus.HOLDING: "手持中",
    HoldingStatus.LOST: "已丢失",
    HoldingStatus.RELEASED: "已结束",
}


class OilingStatus(str, Enum):
    """二次涂油进程状态（文档 7 / 10）。"""

    WAITING = "WAITING"
    CONTACT = "CONTACT"
    BRUSHING = "BRUSHING"
    COMPLETED = "COMPLETED"


OILING_STATUS_LABELS: dict[OilingStatus, str] = {
    OilingStatus.WAITING: "等待刷涂",
    OilingStatus.CONTACT: "刷头接触",
    OilingStatus.BRUSHING: "有效刷涂",
    OilingStatus.COMPLETED: "刷涂完成",
}


class TaskStage(str, Enum):
    """业务状态机阶段（文档 8）。"""

    WAITING_PICK = "WAITING_PICK"
    PICK_CANDIDATE = "PICK_CANDIDATE"
    LEAVING_TRAY = "LEAVING_TRAY"
    PICK_CONFIRMED = "PICK_CONFIRMED"
    HOLDING = "HOLDING"
    BRUSH_CONTACT = "BRUSH_CONTACT"
    BRUSHING = "BRUSHING"
    OILING_COMPLETED = "OILING_COMPLETED"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    NG = "NG"


TASK_STAGE_LABELS: dict[TaskStage, str] = {
    TaskStage.WAITING_PICK: "等待取料",
    TaskStage.PICK_CANDIDATE: "取料候选",
    TaskStage.LEAVING_TRAY: "离开涂油盘",
    TaskStage.PICK_CONFIRMED: "取料确认",
    TaskStage.HOLDING: "手持跟踪",
    TaskStage.BRUSH_CONTACT: "刷头接触",
    TaskStage.BRUSHING: "二次涂油中",
    TaskStage.OILING_COMPLETED: "二次涂油完成",
    TaskStage.PENDING_CONFIRMATION: "待确认",
    TaskStage.NG: "不合格",
}


class TaskResult(str, Enum):
    """任务最终结果（文档 8 / 10）。"""

    PROCESSING = "PROCESSING"
    OK = "OK"
    NG = "NG"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"


TASK_RESULT_LABELS: dict[TaskResult, str] = {
    TaskResult.PROCESSING: "进行中",
    TaskResult.OK: "合格",
    TaskResult.NG: "不合格",
    TaskResult.PENDING_CONFIRMATION: "待确认",
}


class EventType(str, Enum):
    PICK = "PICK"
    BRUSH_CONTACT = "BRUSH_CONTACT"
    MANUAL_OILING = "MANUAL_OILING"
    TASK_RESULT = "TASK_RESULT"


EVENT_TYPE_LABELS: dict[EventType, str] = {
    EventType.PICK: "取料",
    EventType.BRUSH_CONTACT: "刷头接触",
    EventType.MANUAL_OILING: "人工二次涂油",
    EventType.TASK_RESULT: "任务结果",
}


class EventResult(str, Enum):
    CONFIRMED = "CONFIRMED"
    UNCONFIRMED = "UNCONFIRMED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


EVENT_RESULT_LABELS: dict[EventResult, str] = {
    EventResult.CONFIRMED: "已确认",
    EventResult.UNCONFIRMED: "未确认",
    EventResult.NEEDS_REVIEW: "待复核",
}


class ReviewResult(str, Enum):
    CONFIRMED_ABNORMAL = "CONFIRMED_ABNORMAL"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    UNDETERMINABLE = "UNDETERMINABLE"


REVIEW_RESULT_LABELS: dict[ReviewResult, str] = {
    ReviewResult.CONFIRMED_ABNORMAL: "确认异常",
    ReviewResult.FALSE_POSITIVE: "误报",
    ReviewResult.UNDETERMINABLE: "无法判断",
}


@dataclass(frozen=True)
class BBox:
    """原始图像像素坐标下的检测框。"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    @property
    def size(self) -> float:
        return max(self.width, self.height)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    def intersects(self, other: BBox) -> bool:
        return not (
            self.x2 <= other.x1
            or other.x2 <= self.x1
            or self.y2 <= other.y1
            or other.y2 <= self.y1
        )

    def intersection_area(self, other: BBox) -> float:
        w = min(self.x2, other.x2) - max(self.x1, other.x1)
        h = min(self.y2, other.y2) - max(self.y1, other.y1)
        if w <= 0 or h <= 0:
            return 0.0
        return w * h

    def iou(self, other: BBox) -> float:
        inter = self.intersection_area(other)
        union = self.width * self.height + other.width * other.height - inter
        return 0.0 if union <= 0 else inter / union

    def long_axis(self) -> tuple[float, float]:
        """由水平框估计主轴方向（单位向量）。

        有 OBB 时检测器应在 :class:`Detection` 中提供 ``angle``，这里只作为
        没有角度信息时的兜底。
        """

        angle = 0.0 if self.width >= self.height else 1.5707963267948966
        return (1.0, 0.0) if angle == 0.0 else (0.0, 1.0)


@dataclass(frozen=True)
class Detection:
    """一帧中的单个目标检测结果。

    ``angle`` 为 OBB 长轴角度（弧度，0 表示水平向右），可选。
    ``point`` 为关键点（例如刷头工作端），可选；没有时由检测框推算。
    """

    label: str
    confidence: float
    box: BBox
    track_id: int | None = None
    angle: float | None = None
    point: tuple[float, float] | None = None


@dataclass(frozen=True)
class FrameDetections:
    """检测器对一帧的完整输出。"""

    ts: float
    frame_size: tuple[int, int]
    detections: tuple[Detection, ...] = ()
    source: str = ""

    def by_label(self, label: str) -> tuple[Detection, ...]:
        return tuple(d for d in self.detections if d.label == label)


@dataclass(frozen=True)
class TraySnapshot:
    """涂油盘定位结果（只作为取料参考区域，不识别槽位）。"""

    track_id: int | None
    box: BBox | None
    confidence: float | None
    motion: str
    reliable: bool
    visible: bool
    missing_s: float = 0.0
    last_seen_ts: float | None = None


@dataclass(frozen=True)
class TrackSnapshot:
    """丝杆/手部/刷头的跟踪快照。"""

    track_id: int
    label: str
    box: BBox
    confidence: float
    observed: bool
    missing_s: float
    last_seen_ts: float
    angle: float | None = None
    point: tuple[float, float] | None = None


@dataclass(frozen=True)
class TaskSnapshot:
    """单根丝杆任务（文档 10 的 ScrewTask），供 UI 展示。"""

    task_id: str
    created_at: float
    stage: TaskStage
    stage_since: float
    pick_status: PickStatus
    holding_status: HoldingStatus
    oiling_status: OilingStatus
    final_result: TaskResult
    screw_track_id: int | None = None
    hand_track_id: int | None = None
    brush_track_id: int | None = None
    contact_intervals: tuple[tuple[float, float], ...] = ()
    axis_segments: tuple[bool, ...] = ()
    coverage: float = 0.0
    effective_brush_duration_s: float = 0.0
    brush_motion: float = 0.0
    confidence: float | None = None
    reason_code: str | None = None
    evidence: tuple[str, ...] = ()
    picked_at: float | None = None
    confirmed_at: float | None = None
    completed_at: float | None = None
    last_seen_ts: float | None = None
    timeline: tuple[tuple[str, float], ...] = ()

    @property
    def stage_label(self) -> str:
        return TASK_STAGE_LABELS.get(self.stage, self.stage.value)

    @property
    def result_label(self) -> str:
        return TASK_RESULT_LABELS.get(self.final_result, self.final_result.value)

    @property
    def pick_label(self) -> str:
        return PICK_STATUS_LABELS.get(self.pick_status, self.pick_status.value)

    @property
    def holding_label(self) -> str:
        return HOLDING_STATUS_LABELS.get(
            self.holding_status, self.holding_status.value
        )

    @property
    def oiling_label(self) -> str:
        return OILING_STATUS_LABELS.get(
            self.oiling_status, self.oiling_status.value
        )

    @property
    def coverage_percent(self) -> int:
        return int(round(max(0.0, min(1.0, self.coverage)) * 100))


@dataclass(frozen=True)
class StationSnapshot:
    """单个工位的实时快照。"""

    station_id: str
    station_name: str
    camera_id: str
    mode: str
    online: bool
    tasks: tuple[TaskSnapshot, ...]
    tray: TraySnapshot | None = None
    tracks: tuple[TrackSnapshot, ...] = ()
    detections: tuple[Detection, ...] = ()
    frame_size: tuple[int, int] = (0, 0)
    updated_at: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def operations(self) -> tuple[TaskSnapshot, ...]:
        """兼容旧 UI/统计命名。"""

        return self.tasks

    @property
    def materials(self) -> tuple[TaskSnapshot, ...]:
        return self.tasks


@dataclass(frozen=True)
class RecordingStatus:
    """训练视频录制状态。"""

    camera_id: str
    active: bool
    started_at: float | None = None
    segment_path: str | None = None
    frames_written: int = 0
    frames_dropped: int = 0
    last_error: str = ""


@dataclass(frozen=True)
class EventRecord:
    """一条流程事件，字段与实现文档第 8、10 节对应。"""

    event_id: str
    timestamp_start: float
    timestamp_end: float
    station_id: str
    camera_id: str
    event_type: EventType
    result: EventResult
    task_id: str
    cycle_id: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    clip_reference: str | None = None
    review_result: ReviewResult | None = None
    review_note: str = ""
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        *,
        timestamp_start: float,
        timestamp_end: float,
        station_id: str,
        camera_id: str,
        event_type: EventType,
        result: EventResult,
        task_id: str,
        cycle_id: str | None = None,
        evidence: dict[str, Any] | None = None,
        clip_reference: str | None = None,
    ) -> EventRecord:
        return cls(
            event_id=uuid.uuid4().hex,
            timestamp_start=float(timestamp_start),
            timestamp_end=float(timestamp_end),
            station_id=str(station_id),
            camera_id=str(camera_id),
            event_type=event_type,
            result=result,
            task_id=str(task_id),
            cycle_id=cycle_id,
            evidence=dict(evidence or {}),
            clip_reference=clip_reference,
        )

    @property
    def event_label(self) -> str:
        return EVENT_TYPE_LABELS.get(self.event_type, self.event_type.value)

    @property
    def result_label(self) -> str:
        return EVENT_RESULT_LABELS.get(self.result, self.result.value)

    @property
    def review_label(self) -> str:
        if self.review_result is None:
            return "未复核"
        return REVIEW_RESULT_LABELS.get(self.review_result, self.review_result.value)

    @property
    def logical_part_id(self) -> str:
        """兼容旧字段：业务物料即任务。"""

        return self.task_id

    def with_review(
        self,
        review_result: ReviewResult,
        *,
        note: str = "",
    ) -> EventRecord:
        return EventRecord(
            event_id=self.event_id,
            timestamp_start=self.timestamp_start,
            timestamp_end=self.timestamp_end,
            station_id=self.station_id,
            camera_id=self.camera_id,
            event_type=self.event_type,
            result=self.result,
            task_id=self.task_id,
            cycle_id=self.cycle_id,
            evidence=dict(self.evidence),
            clip_reference=self.clip_reference,
            review_result=review_result,
            review_note=str(note),
            created_at=self.created_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp_start": _iso(self.timestamp_start),
            "timestamp_end": _iso(self.timestamp_end),
            "station_id": self.station_id,
            "camera_id": self.camera_id,
            "cycle_id": self.cycle_id,
            "task_id": self.task_id,
            "event_type": self.event_type.value,
            "result": self.result.value,
            "evidence": self.evidence,
            "clip_reference": self.clip_reference,
            "review_result": (
                None if self.review_result is None else self.review_result.value
            ),
            "review_note": self.review_note,
            "created_at": _iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EventRecord:
        review_raw = data.get("review_result")
        review = None
        if review_raw:
            try:
                review = ReviewResult(str(review_raw))
            except ValueError:
                review = None
        # 兼容旧事件文件：task_id 缺失时回退到 logical_part_id。
        task_id = str(
            data.get("task_id")
            or data.get("logical_part_id")
            or ""
        )
        return cls(
            event_id=str(data.get("event_id", "")),
            timestamp_start=_parse_ts(data.get("timestamp_start")),
            timestamp_end=_parse_ts(data.get("timestamp_end")),
            station_id=str(data.get("station_id", "")),
            camera_id=str(data.get("camera_id", "")),
            event_type=_enum_or(EventType, data.get("event_type"), EventType.PICK),
            result=_enum_or(EventResult, data.get("result"), EventResult.NEEDS_REVIEW),
            task_id=task_id,
            cycle_id=(
                None if data.get("cycle_id") in (None, "") else str(data["cycle_id"])
            ),
            evidence=dict(data.get("evidence") or {}),
            clip_reference=(
                None
                if data.get("clip_reference") in (None, "")
                else str(data["clip_reference"])
            ),
            review_result=review,
            review_note=str(data.get("review_note", "")),
            created_at=_parse_ts(data.get("created_at")),
        )


def format_timestamp(ts: float | None) -> str:
    if not ts:
        return "-"
    return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")


def _iso(ts: float | None) -> str | None:
    if not ts:
        return None
    return (
        datetime.fromtimestamp(float(ts))
        .astimezone()
        .isoformat(timespec="milliseconds")
    )


def _parse_ts(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return 0.0
    return 0.0


def _enum_or(enum_cls: type[Enum], value: Any, default: Any) -> Any:
    try:
        return enum_cls(str(value))
    except (ValueError, TypeError):
        return default
