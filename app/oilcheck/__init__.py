"""丝杆取料与人工二次涂油检测业务包。"""

from .config import (
    DetectorConfig,
    OilCheckConfig,
    OilCheckConfigService,
    PipelineConfig,
    RecordingConfig,
    StationConfig,
)
from .detector import Detector, NullDetector, SimulatedDetector, create_detector
from .events import EventStore
from .pipeline import StationPipeline
from .recorder import TrainingRecorder, list_segments
from .service import OilCheckService
from .tracker import ObjectTracker, Track, TrayTracker
from .types import (
    Detection,
    EventRecord,
    EventResult,
    EventType,
    FrameDetections,
    HoldingStatus,
    OilingStatus,
    PickStatus,
    RecordingStatus,
    ReviewResult,
    StationSnapshot,
    TaskResult,
    TaskSnapshot,
    TaskStage,
    TraySnapshot,
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
    "HoldingStatus",
    "NullDetector",
    "ObjectTracker",
    "OilCheckConfig",
    "OilCheckConfigService",
    "OilCheckService",
    "OilingStatus",
    "PickStatus",
    "PipelineConfig",
    "RecordingConfig",
    "RecordingStatus",
    "ReviewResult",
    "SimulatedDetector",
    "StationConfig",
    "StationPipeline",
    "StationSnapshot",
    "TaskResult",
    "TaskSnapshot",
    "TaskStage",
    "Track",
    "TrainingRecorder",
    "TraySnapshot",
    "TrayTracker",
    "create_detector",
    "list_segments",
]
