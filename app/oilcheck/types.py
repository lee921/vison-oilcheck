"""取料与二次涂油检测的领域模型。

本模块只放纯数据定义，不依赖 Qt、相机 SDK 或具体检测模型，方便单元测试与
后续替换推理实现。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class SlotOccupancy(str, Enum):
    """机器涂油盘单个槽位的占用状态（设计文档 5.3）。"""

    OCCUPIED = "OCCUPIED"
    EMPTY = "EMPTY"
    UNKNOWN = "UNKNOWN"


SLOT_OCCUPANCY_LABELS: dict[SlotOccupancy, str] = {
    SlotOccupancy.OCCUPIED: "有物料",
    SlotOccupancy.EMPTY: "空槽",
    SlotOccupancy.UNKNOWN: "未知",
}


class Visibility(str, Enum):
    """槽位当前是否具备可靠可见证据。"""

    VISIBLE = "VISIBLE"
    OCCLUDED = "OCCLUDED"
    UNRELIABLE = "UNRELIABLE"


VISIBILITY_LABELS: dict[Visibility, str] = {
    Visibility.VISIBLE: "可见",
    Visibility.OCCLUDED: "遮挡",
    Visibility.UNRELIABLE: "定位不可靠",
}


class OperationState(str, Enum):
    """逐根逻辑物料的二次涂油状态（设计文档 7.2）。"""

    NOT_STARTED = "NOT_STARTED"
    PICKED = "PICKED"
    OILING_CANDIDATE = "OILING_CANDIDATE"
    PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
    IDENTITY_UNCERTAIN = "IDENTITY_UNCERTAIN"
    OILING_CONFIRMED = "OILING_CONFIRMED"


OPERATION_STATE_LABELS: dict[OperationState, str] = {
    OperationState.NOT_STARTED: "未开始",
    OperationState.PICKED: "已取料",
    OperationState.OILING_CANDIDATE: "涂油候选",
    OperationState.PENDING_CONFIRMATION: "待确认",
    OperationState.IDENTITY_UNCERTAIN: "身份不确定",
    OperationState.OILING_CONFIRMED: "二次涂油已确认",
}

UNCONFIRMED_STATES = {
    OperationState.PENDING_CONFIRMATION,
    OperationState.IDENTITY_UNCERTAIN,
}


class EventType(str, Enum):
    PICK = "PICK"
    MANUAL_OILING = "MANUAL_OILING"


EVENT_TYPE_LABELS: dict[EventType, str] = {
    EventType.PICK: "取料",
    EventType.MANUAL_OILING: "人工二次涂油",
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


@dataclass(frozen=True)
class Detection:
    """一帧中的单个目标检测结果。"""

    label: str
    confidence: float
    box: BBox
    track_id: int | None = None


@dataclass(frozen=True)
class SyntheticFrame:
    """无相机演示模式下的合成帧，仅用于驱动状态机与 UI。"""

    bgr: Any
    width: int
    height: int
    ts: float


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
class SlotSnapshot:
    """机器涂油盘单个槽位的实时状态。"""

    slot_index: int
    slot_name: str
    occupancy: SlotOccupancy
    visibility: Visibility
    confidence: float | None
    last_change_ts: float
    reason: str
    box: BBox | None = None
    last_observed_ts: float | None = None

    @property
    def occupancy_label(self) -> str:
        return SLOT_OCCUPANCY_LABELS.get(self.occupancy, self.occupancy.value)

    @property
    def visibility_label(self) -> str:
        return VISIBILITY_LABELS.get(self.visibility, self.visibility.value)


@dataclass(frozen=True)
class TraySnapshot:
    """机器涂油盘定位结果。"""

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
    """手部或刷子的跟踪快照。"""

    track_id: int
    label: str
    box: BBox
    confidence: float
    observed: bool
    missing_s: float
    last_seen_ts: float


@dataclass(frozen=True)
class OperationSnapshot:
    """一条人工取料/二次涂油操作记录，供 UI 展示。"""

    operation_id: str
    logical_part_id: str
    source_slot: int
    slot_name: str
    state: OperationState
    state_since: float
    pickup_status: str = "CONFIRMED"
    identity_status: str = "ASSOCIATED"
    confidence: float | None = None
    reason_code: str | None = None
    evidence: tuple[str, ...] = ()
    hand_track_id: int | None = None
    brush_track_id: int | None = None
    picked_at: float | None = None
    last_seen_ts: float | None = None

    @property
    def slot_index(self) -> int:
        return self.source_slot

    @property
    def state_label(self) -> str:
        return OPERATION_STATE_LABELS.get(self.state, self.state.value)


@dataclass(frozen=True)
class StationSnapshot:
    """单个工位的实时快照。"""

    station_id: str
    station_name: str
    camera_id: str
    mode: str
    online: bool
    operations: tuple[OperationSnapshot, ...]
    slots: tuple[SlotSnapshot, ...] = ()
    tray: TraySnapshot | None = None
    tracks: tuple[TrackSnapshot, ...] = ()
    detections: tuple[Detection, ...] = ()
    frame_size: tuple[int, int] = (0, 0)
    updated_at: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def materials(self) -> tuple[OperationSnapshot, ...]:
        """兼容旧 UI/统计代码：物料即人工操作记录。"""
        return self.operations


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
    """一条流程事件，字段与方案文档第 8 节对应。"""

    event_id: str
    timestamp_start: float
    timestamp_end: float
    station_id: str
    camera_id: str
    event_type: EventType
    result: EventResult
    logical_part_id: str
    slot_index: int
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
        logical_part_id: str,
        slot_index: int,
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
            logical_part_id=str(logical_part_id),
            slot_index=int(slot_index),
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
            logical_part_id=self.logical_part_id,
            slot_index=self.slot_index,
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
            "logical_part_id": self.logical_part_id,
            "slot_index": self.slot_index,
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
        return cls(
            event_id=str(data.get("event_id", "")),
            timestamp_start=_parse_ts(data.get("timestamp_start")),
            timestamp_end=_parse_ts(data.get("timestamp_end")),
            station_id=str(data.get("station_id", "")),
            camera_id=str(data.get("camera_id", "")),
            event_type=_enum_or(EventType, data.get("event_type"), EventType.PICK),
            result=_enum_or(EventResult, data.get("result"), EventResult.NEEDS_REVIEW),
            logical_part_id=str(data.get("logical_part_id", "")),
            slot_index=int(data.get("slot_index", 0) or 0),
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
