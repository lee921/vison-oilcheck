"""丝杆取料与人工二次涂油检测业务包。"""

from .config import (
    DetectorConfig,
    OilCheckConfig,
    OilCheckConfigService,
    PipelineConfig,
    RecordingConfig,
    SlotConfig,
    SlotLayout,
    StationConfig,
)
from .detector import Detector, NullDetector, SimulatedDetector, create_detector
from .events import EventStore
from .pipeline import StationPipeline
from .recorder import TrainingRecorder, list_segments
from .service import OilCheckService
from .tracker import ObjectTracker, RodTracker, Track, TrayTracker
from .types import (
    Detection,
    EventRecord,
    EventResult,
    EventType,
    FrameDetections,
    OperationSnapshot,
    OperationState,
    RecordingStatus,
    ReviewResult,
    SlotOccupancy,
    SlotSnapshot,
    StationSnapshot,
    TraySnapshot,
    Visibility,
)

__all__ = [
    "Detection",
    "Detector",
    "DetectorConfig",
    "EventRecord",
    "EventResult",
    "EventStore",
    "EventType",
    "FrameDetections",
    "NullDetector",
    "ObjectTracker",
    "OilCheckConfig",
    "OilCheckConfigService",
    "OilCheckService",
    "OperationSnapshot",
    "OperationState",
    "PipelineConfig",
    "RecordingConfig",
    "RecordingStatus",
    "RodTracker",
    "ReviewResult",
    "SimulatedDetector",
    "SlotConfig",
    "SlotLayout",
    "SlotOccupancy",
    "SlotSnapshot",
    "StationConfig",
    "StationPipeline",
    "StationSnapshot",
    "Track",
    "TrainingRecorder",
    "TraySnapshot",
    "TrayTracker",
    "Visibility",
    "create_detector",
    "list_segments",
]
