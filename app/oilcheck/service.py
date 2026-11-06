"""业务服务层：把相机帧、检测器、状态机、事件存储与录制串起来。

线程模型：

- 一个轮询线程按配置帧率读取各相机最新帧；
- 每个工位一个单线程执行器执行检测 + 状态机，忙时丢帧，避免排队延迟；
- 录制器自带写盘线程；事件写盘在状态机线程内完成（量小、需要可靠落盘）。

UI 只读取快照与事件，不直接触碰检测线程。
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import OilCheckConfig, OilCheckConfigService, StationConfig
from .detector import Detector, create_detector
from .events import EventStore
from .pipeline import StationPipeline
from .recorder import TrainingRecorder, delete_segments, list_segments
from .types import (
    EventRecord,
    FrameDetections,
    RecordingStatus,
    StationSnapshot,
    SyntheticFrame,
)

logger = logging.getLogger("oilcheck.service")


@dataclass
class _StationRuntime:
    station: StationConfig
    detector: Detector
    pipeline: StationPipeline
    detector_kind: str
    executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="oilcheck-station"
        )
    )
    busy: bool = False
    last_frame_ts: float = 0.0
    last_snapshot: StationSnapshot | None = None
    last_error: str = ""
    closed: bool = False


class OilCheckService:
    def __init__(
        self,
        *,
        manager: Any,
        config_service: OilCheckConfigService,
        config: OilCheckConfig,
        event_store: EventStore,
        data_root: Path,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._manager = manager
        self._config_service = config_service
        self._config = config
        self._event_store = event_store
        self._data_root = Path(data_root)
        self._logger = (logger_ or logger).getChild("oilcheck")
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._runtimes: dict[str, _StationRuntime] = {}
        self._recorders: dict[str, TrainingRecorder] = {}
        self._demo_frames: dict[str, SyntheticFrame] = {}
        self._detector_overrides: dict[str, str] = {}
        self._cycle_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self._rebuild_runtimes_locked()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def config(self) -> OilCheckConfig:
        return self._config

    @property
    def cycle_id(self) -> str:
        return self._cycle_id

    @property
    def event_store(self) -> EventStore:
        return self._event_store

    @property
    def recording_root(self) -> Path:
        path = Path(self._config.recording.training_dir)
        return path if path.is_absolute() else (self._data_root / path).resolve()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="oilcheck-service", daemon=True
            )
            thread = self._thread
        thread.start()
        self._logger.info(
            "取料检测服务已启动（工位=%d，周期=%s）",
            len(self._runtimes),
            self._cycle_id,
        )
        if self._config.recording.auto_start:
            for station in self.stations():
                try:
                    self.start_training_recording(station.camera_id)
                except Exception:
                    self._logger.exception(
                        "自动开始录制失败 (相机=%s)", station.camera_id
                    )

    def shutdown(self) -> None:
        self._stop_event.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

        with self._lock:
            recorders = list(self._recorders.values())
            self._recorders.clear()
            runtimes = list(self._runtimes.values())
            self._runtimes.clear()
            self._demo_frames.clear()

        for recorder in recorders:
            try:
                recorder.stop()
            except Exception:
                pass
        for runtime in runtimes:
            self._shutdown_runtime(runtime)
        self._logger.info("取料检测服务已停止")

    def reload(self, config: OilCheckConfig) -> None:
        """应用新的配置并重建工位运行时（录制保持不变）。"""
        old = list(self._runtimes.values())
        with self._lock:
            self._config = config
            self._rebuild_runtimes_locked()
        for runtime in old:
            self._shutdown_runtime(runtime)
        self._logger.info("检测配置已重新加载（工位=%d）", len(self._runtimes))

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    def stations(self) -> tuple[StationConfig, ...]:
        return tuple(s for s in self._config.stations if s.enabled)

    def station(self, station_id: str) -> StationConfig | None:
        return self._config.station(station_id)

    def station_snapshot(self, station_id: str) -> StationSnapshot | None:
        with self._lock:
            runtime = self._runtimes.get(station_id)
        if runtime is None:
            station = self.station(station_id)
            if station is None:
                return None
            return StationSnapshot(
                station_id=station.station_id,
                station_name=station.name,
                camera_id=station.camera_id,
                mode=station.detector.kind,
                online=False,
                operations=(),
            )
        snapshot = runtime.last_snapshot or runtime.pipeline.snapshot()
        session = self._manager.get_session(snapshot.camera_id)
        online = False
        camera_state = "missing"
        if session is not None:
            camera_state = session.state.value
            last_ts = session.frame_buffer.last_ts()
            online = (
                camera_state == "previewing"
                and last_ts is not None
                and (time.time() - last_ts < 5.0)
            )
        if not online and runtime.detector_kind == "simulated":
            with self._lock:
                demo = self._demo_frames.get(snapshot.camera_id)
            if demo is not None and time.time() - demo.ts < 2.0:
                online = True
                camera_state = "demo"
        stats = dict(snapshot.stats)
        stats["camera_state"] = camera_state
        if runtime.last_error:
            stats["last_error"] = runtime.last_error
        stats["detector_override"] = self._detector_overrides.get(station_id, "")
        return replace(
            snapshot,
            online=online,
            mode=runtime.detector_kind,
            stats=stats,
        )

    def latest_frame(self, camera_id: str) -> Any | None:
        session = self._manager.get_session(camera_id)
        # 只在预览状态下返回相机帧；停止后 frame_buffer 可能已被清空，
        # 这里再做一次状态保护，避免 UI 把残留帧当作实时画面。
        if session is not None and session.state.value == "previewing":
            frame = session.frame_buffer.get()
            if frame is not None:
                return frame
        with self._lock:
            return self._demo_frames.get(camera_id)

    def camera_state(self, camera_id: str) -> str:
        session = self._manager.get_session(camera_id)
        return "missing" if session is None else session.state.value

    def clear_events(self) -> int:
        """清空当前所有事件与复核记录（不影响录制视频）。"""
        return self._event_store.clear()

    def recent_events(
        self,
        limit: int = 100,
        *,
        station_id: str | None = None,
        pending_only: bool = False,
    ) -> list[EventRecord]:
        return self._event_store.recent(
            limit, station_id=station_id, pending_only=pending_only
        )

    def count_materials(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            snapshot = runtime.last_snapshot or runtime.pipeline.snapshot()
            for material in snapshot.materials:
                key = material.state.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # 操作接口
    # ------------------------------------------------------------------
    def reset_station(self, station_id: str) -> bool:
        with self._lock:
            runtime = self._runtimes.get(station_id)
        if runtime is None:
            return False
        try:
            runtime.executor.submit(runtime.pipeline.reset)
        except RuntimeError:
            return False
        self._logger.info("已请求重置工位流程状态: %s", station_id)
        return True

    def set_station_detector_kind(
        self, station_id: str, kind: str
    ) -> StationSnapshot | None:
        """临时切换某个工位的检测器（演示联调用），不写回配置文件。"""
        with self._lock:
            station = self._config.station(station_id)
            if station is None:
                return None
            old = self._runtimes.pop(station_id, None)
            if kind and kind.lower() != "default":
                self._detector_overrides[station_id] = kind.lower()
                effective = kind.lower()
            else:
                self._detector_overrides.pop(station_id, None)
                effective = station.detector.kind
            try:
                runtime = self._build_runtime(station, effective)
            except Exception:  # noqa: BLE001
                self._logger.exception("切换检测器失败 (工位=%s)", station_id)
                if old is not None:
                    self._runtimes[station_id] = old
                return None
            self._runtimes[station_id] = runtime
            if effective != "simulated":
                self._demo_frames.pop(station.camera_id, None)
        if old is not None:
            self._shutdown_runtime(old)
        self._logger.info("工位 %s 检测器切换为 %s", station_id, runtime.detector_kind)
        return runtime.pipeline.snapshot()

    def start_training_recording(self, camera_id: str) -> RecordingStatus:
        station = self._station_for_camera(camera_id)
        camera = self._camera_config(camera_id)
        with self._lock:
            recorder = self._recorders.get(camera_id)
            if recorder is None:
                recorder = TrainingRecorder(
                    camera_id=camera_id,
                    camera_name="" if camera is None else camera.name,
                    root=self.recording_root,
                    fps=self._config.recording.fps,
                    segment_seconds=self._config.recording.segment_seconds,
                    codec=self._config.recording.codec,
                    station_meta=_station_meta(station),
                    logger_=self._logger,
                )
                self._recorders[camera_id] = recorder
        recorder.start()
        session = self._manager.get_session(camera_id)
        if session is None or session.state.value != "previewing":
            self._logger.warning(
                "已开始录制，但相机当前未处于预览状态 (相机=%s)；启动预览后才会写入帧。",
                camera_id,
            )
        return recorder.status()

    def stop_training_recording(self, camera_id: str) -> RecordingStatus:
        with self._lock:
            recorder = self._recorders.get(camera_id)
        if recorder is None:
            return RecordingStatus(camera_id=camera_id, active=False)
        recorder.stop()
        return recorder.status()

    def recording_status(self, camera_id: str) -> RecordingStatus:
        with self._lock:
            recorder = self._recorders.get(camera_id)
        if recorder is None:
            return RecordingStatus(camera_id=camera_id, active=False)
        return recorder.status()

    def recording_segments(self, limit: int = 200) -> list[dict[str, Any]]:
        return list_segments(self.recording_root, limit=limit)

    def delete_recording_segments(self, paths: list[str]) -> tuple[int, list[str]]:
        """删除选中的录制分段；正在写入的分段会被跳过。"""
        active: set[str] = set()
        with self._lock:
            recorders = list(self._recorders.values())
        for recorder in recorders:
            status = recorder.status()
            if status.active and status.segment_path:
                active.add(status.segment_path)
        return delete_segments(self.recording_root, paths, active_paths=active)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop_event.is_set():
            interval = self._poll_interval()
            if self._stop_event.wait(interval):
                break
            with self._lock:
                runtimes = list(self._runtimes.values())
            if not runtimes:
                continue
            for runtime in runtimes:
                if runtime.closed:
                    continue
                self._poll_station(runtime)

    def _poll_interval(self) -> float:
        pipeline_fps = self._config.pipeline.max_fps
        recording_fps = 0
        with self._lock:
            active = [r for r in self._recorders.values() if r.active]
        if active:
            recording_fps = self._config.recording.fps
        fps = max(1, pipeline_fps, recording_fps)
        return max(0.02, min(0.2, 1.0 / fps))

    def _poll_station(self, runtime: _StationRuntime) -> None:
        camera_id = runtime.station.camera_id
        session = self._manager.get_session(camera_id)
        frame = None if session is None else session.frame_buffer.get()
        synthetic = False
        if frame is None:
            if runtime.detector_kind != "simulated":
                return
            # 无相机演示：合成帧只驱动状态机与 UI，绝不写入训练录制。
            frame = self._demo_frame(camera_id)
            synthetic = True
        if frame.ts <= runtime.last_frame_ts:
            return
        runtime.last_frame_ts = frame.ts

        with self._lock:
            recorder = self._recorders.get(camera_id)
        if recorder is not None and recorder.active and not synthetic:
            recorder.submit(frame.ts, frame.bgr)

        if runtime.busy:
            return
        runtime.busy = True
        try:
            runtime.executor.submit(self._process_frame, runtime, frame.bgr, frame.ts)
        except RuntimeError:
            runtime.busy = False

    def _demo_frame(self, camera_id: str) -> SyntheticFrame:
        now = time.time()
        with self._lock:
            existing = self._demo_frames.get(camera_id)
        if existing is None:
            bgr = np.full((720, 1280, 3), 30, dtype=np.uint8)
            bgr[::120, :, :] = 44
            bgr[:, ::160, :] = 44
            frame = SyntheticFrame(bgr=bgr, width=1280, height=720, ts=now)
        else:
            frame = SyntheticFrame(
                bgr=existing.bgr,
                width=existing.width,
                height=existing.height,
                ts=now,
            )
        with self._lock:
            self._demo_frames[camera_id] = frame
        return frame

    def _process_frame(self, runtime: _StationRuntime, bgr: Any, ts: float) -> None:
        started = time.perf_counter()
        try:
            detections: FrameDetections = runtime.detector.detect(bgr, ts)
            snapshot = runtime.pipeline.process(detections=detections)
            elapsed = (time.perf_counter() - started) * 1000.0
            stats = dict(snapshot.stats)
            stats["processing_ms"] = round(elapsed, 2)
            stats["detections"] = len(detections.detections)
            snapshot = replace(snapshot, stats=stats)
            runtime.last_snapshot = snapshot
            runtime.last_error = ""
        except Exception as exc:  # noqa: BLE001
            runtime.last_error = f"{type(exc).__name__}: {exc}"
            self._logger.exception(
                "工位处理失败 (工位=%s, 相机=%s)",
                runtime.station.station_id,
                runtime.station.camera_id,
            )
        finally:
            runtime.busy = False

    def _on_event(self, event: EventRecord) -> None:
        self._event_store.add(event)
        self._logger.debug("事件已入库: %s", event.event_id)

    def _build_runtime(self, station: StationConfig, kind: str) -> _StationRuntime:
        detector = create_detector(station, override_kind=kind)
        pipeline = StationPipeline(
            station=station,
            pipeline_cfg=self._config.pipeline,
            on_event=self._on_event,
            cycle_id=self._cycle_id,
            detector_name=kind,
            logger=self._logger.getChild(f"station.{station.station_id}"),
        )
        return _StationRuntime(
            station=station,
            detector=detector,
            pipeline=pipeline,
            detector_kind=kind,
            executor=ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"oilcheck-{station.station_id}",
            ),
        )

    def _rebuild_runtimes_locked(self) -> None:
        self._runtimes = {}
        for station in self._config.enabled_stations():
            kind = self._detector_overrides.get(station.station_id)
            if not kind:
                kind = station.detector.kind
            try:
                self._runtimes[station.station_id] = self._build_runtime(station, kind)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "工位初始化失败 (工位=%s): %s", station.station_id, exc
                )

    def _shutdown_runtime(self, runtime: _StationRuntime) -> None:
        runtime.closed = True
        try:
            runtime.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            runtime.detector.close()
        except Exception:
            pass

    def _station_for_camera(self, camera_id: str) -> StationConfig | None:
        for station in self._config.stations:
            if station.camera_id == camera_id:
                return station
        return None

    def _camera_config(self, camera_id: str) -> Any | None:
        try:
            for camera in self._manager.cameras():
                if camera.id == camera_id:
                    return camera
        except Exception:
            pass
        return None


def _station_meta(station: StationConfig | None) -> dict[str, Any]:
    if station is None:
        return {}
    return {
        "station_id": station.station_id,
        "station_name": station.name,
        "rois": {
            "pick_roi": station.pick_roi.to_dict(),
            "oiling_roi": station.oiling_roi.to_dict(),
            "tray_roi": (station.tray_roi or station.pick_roi).to_dict(),
            "slot_layout": [layout.to_dict() for layout in station.slot_layout],
            "slots": [slot.to_dict() for slot in station.slots],
        },
    }
