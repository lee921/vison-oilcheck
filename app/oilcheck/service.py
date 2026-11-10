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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from app.paths import resolve_runtime_file, runtime_root

from .config import DemoVideo, OilCheckConfig, OilCheckConfigService, StationConfig
from .detector import Detector, create_detector
from .events import EventStore
from .pipeline import StationPipeline
from .recorder import TrainingRecorder, delete_segments, list_segments
from .types import (
    EventRecord,
    FrameDetections,
    RecordingStatus,
    StationSnapshot,
)
from .video_source import VideoFileSource

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
    video_source: VideoFileSource | None = None
    busy: bool = False
    last_frame_ts: float = 0.0
    dropped_frames: int = 0
    # 示例回放当前帧序号；用于识别「循环回绕」，回绕时重置判定状态。
    last_video_frame_index: int = -1
    # 待执行的判定状态重置；由执行器线程在处理下一帧前执行，
    # 避免与正在跑 ``pipeline.process`` 的执行器线程并发调用。
    pending_pipeline_reset: bool = False
    # 最近 1 秒内完成处理的帧时间戳，用于计算实际检测帧率。
    processed_ts: deque[float] = field(default_factory=deque)
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
        self._detector_overrides: dict[str, str] = {}
        self._active_video: dict[str, DemoVideo] = {}
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
            self._active_video.clear()

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
                tasks=(),
            )
        snapshot = runtime.last_snapshot or runtime.pipeline.snapshot()
        online = False
        camera_state = "missing"
        video = runtime.video_source
        if video is not None:
            last_ts = video.last_ts()
            camera_state = "video" if video.playing else "video_stopped"
            online = (
                camera_state == "video"
                and last_ts is not None
                and (time.time() - last_ts < 5.0)
            )
        else:
            session = self._manager.get_session(snapshot.camera_id)
            if session is not None:
                camera_state = session.state.value
                last_ts = session.frame_buffer.last_ts()
                online = (
                    camera_state == "previewing"
                    and last_ts is not None
                    and (time.time() - last_ts < 5.0)
                )
        stats = dict(snapshot.stats)
        stats["camera_state"] = camera_state
        stats["dropped_frames"] = runtime.dropped_frames
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
        video = self._video_for_camera(camera_id)
        if video is not None:
            return video.get()
        session = self._manager.get_session(camera_id)
        # 只在预览状态下返回相机帧；停止后 frame_buffer 可能已被清空，
        # 这里再做一次状态保护，避免 UI 把残留帧当作实时画面。
        if session is not None and session.state.value == "previewing":
            return session.frame_buffer.get()
        return None

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

    def count_tasks(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            snapshot = runtime.last_snapshot or runtime.pipeline.snapshot()
            for task in snapshot.tasks:
                key = task.final_result.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    def count_materials(self) -> dict[str, int]:
        """兼容旧 UI 命名：物料统计即任务统计。"""

        return self.count_tasks()

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

    # ------------------------------------------------------------------
    # 示例视频源
    # ------------------------------------------------------------------
    def demo_videos(self) -> tuple[DemoVideo, ...]:
        return tuple(self._config.demo.videos)

    def demo_speed(self) -> float:
        """配置里的示例回放默认速率。"""

        return float(self._config.demo.speed)

    def active_video(self, station_id: str) -> DemoVideo | None:
        with self._lock:
            return self._active_video.get(station_id)

    def _make_video_source(
        self,
        video: DemoVideo,
        station_id: str,
        loop: bool,
        fps_limit: float,
        speed: float = 1.0,
    ) -> VideoFileSource | None:
        path = Path(resolve_runtime_file(video.path, repo_root=runtime_root()))
        if not path.exists():
            self._logger.error("示例视频不存在: %s", path)
            return None
        return VideoFileSource(
            path,
            loop=loop,
            fps_limit=fps_limit,
            speed=speed,
            logger_=self._logger.getChild(station_id),
        )

    def set_station_video(
        self, station_id: str, video_name: str | None
    ) -> bool:
        """把工位切换到示例视频（``None`` 表示回到实时相机）。

        只替换帧源，不重建检测器：切换视频不会重新加载模型。
        """

        with self._lock:
            runtime = self._runtimes.get(station_id)
            if runtime is None:
                return False
            loop = self._config.demo.loop
            fps_limit = float(self._config.demo.fps_limit)
            speed = float(self._config.demo.speed)

        old_source = runtime.video_source
        new_video: DemoVideo | None = None
        new_source: VideoFileSource | None = None

        if video_name:
            new_video = self._config.demo.video(video_name)
            if new_video is None:
                self._logger.warning(
                    "示例视频未在配置中登记: %s（工位=%s）", video_name, station_id
                )
                return False
            new_source = self._make_video_source(
                new_video, station_id, loop, fps_limit, speed
            )
            if new_source is None:
                return False

        runtime.video_source = None
        runtime.last_frame_ts = 0.0
        runtime.last_video_frame_index = -1
        runtime.pending_pipeline_reset = False
        if old_source is not None:
            try:
                old_source.stop()
            except Exception:
                pass

        if new_source is not None:
            try:
                new_source.start()
            except Exception:
                self._logger.exception("启动示例视频失败: %s", new_video)
                with self._lock:
                    self._active_video.pop(station_id, None)
                return False
            runtime.video_source = new_source
            with self._lock:
                self._active_video[station_id] = new_video
            self._logger.info(
                "工位 %s 切换到示例视频: %s", station_id, new_video.name
            )
        else:
            with self._lock:
                self._active_video.pop(station_id, None)
            self._logger.info("工位 %s 已切回实时相机", station_id)

        try:
            runtime.executor.submit(runtime.pipeline.reset)
        except RuntimeError:
            pass
        return True

    def set_station_video_speed(self, station_id: str, speed: float) -> bool:
        """调整示例回放速率（仅对该工位当前播放的视频生效）。"""

        with self._lock:
            runtime = self._runtimes.get(station_id)
        source = None if runtime is None else runtime.video_source
        if source is None:
            return False
        source.set_speed(speed)
        self._logger.info(
            "工位 %s 示例回放速率设为 %.2f×", station_id, source.speed
        )
        return True

    def video_status(self, station_id: str) -> dict[str, Any]:
        with self._lock:
            runtime = self._runtimes.get(station_id)
        if runtime is None or runtime.video_source is None:
            return {"active": False}
        status = runtime.video_source.status()
        status["active"] = True
        return status

    def set_station_detector_kind(
        self, station_id: str, kind: str
    ) -> StationSnapshot | None:
        """临时切换某个工位的检测器（联调用），不写回配置文件。"""
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
            # 视频源交接给新 runtime：先把旧 runtime 的引用摘掉，
            # 否则 _shutdown_runtime 会把刚接手解码线程停掉。
            if old is not None:
                runtime.video_source = old.video_source
                old.video_source = None
            self._runtimes[station_id] = runtime
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
        video = runtime.video_source
        synthetic = video is not None
        if video is not None:
            frame = video.get()
        else:
            session = self._manager.get_session(camera_id)
            frame = None if session is None else session.frame_buffer.get()
        if frame is None:
            return
        if frame.ts <= runtime.last_frame_ts:
            return
        runtime.last_frame_ts = frame.ts
        if video is not None:
            self._reset_on_video_rewind(runtime, video)

        with self._lock:
            recorder = self._recorders.get(camera_id)
        # 示例回放不写入训练录制：回放视频没有录制价值。
        if recorder is not None and recorder.active and not synthetic:
            recorder.submit(frame.ts, frame.bgr)

        if runtime.busy:
            runtime.dropped_frames += 1
            return
        runtime.busy = True
        # 判定用媒体时间：示例回放变速时它仍按视频自身帧周期前进，
        # 因此“盘内停留多久”“刷子靠近累计多久”与播放速度无关。
        # 实时相机没有媒体时间，回退到墙钟时间。
        pipelines_ts = frame.ts if frame.media_ts is None else frame.media_ts
        try:
            runtime.executor.submit(
                self._process_frame, runtime, frame.bgr, pipelines_ts
            )
        except RuntimeError:
            runtime.busy = False

    def _reset_on_video_rewind(
        self, runtime: _StationRuntime, video: VideoFileSource
    ) -> None:
        """示例回放循环回绕/跳转时重置判定状态。

        画面回到开头而管线还带着上一轮的最后几秒状态时，同一个取料动作会被
        算成两次（下一轮的同一时刻直接判出 PICK），事件数比真实件数多。
        回绕不是连续过程，与「切换视频」一样按新会话处理。
        """

        index = video.frame_index
        previous = runtime.last_video_frame_index
        runtime.last_video_frame_index = index
        if previous < 0 or index >= previous:
            return
        # 只置标记：``pipeline`` 归执行器线程独占，重置必须排到那一侧执行。
        runtime.pending_pipeline_reset = True
        self._logger.info(
            "示例回放回绕（帧 %s → %s），将重置判定状态 (工位=%s)",
            previous,
            index,
            runtime.station.station_id,
        )

    def _process_frame(self, runtime: _StationRuntime, bgr: Any, ts: float) -> None:
        started = time.perf_counter()
        try:
            if runtime.pending_pipeline_reset:
                runtime.pending_pipeline_reset = False
                runtime.pipeline.reset()
                runtime.last_snapshot = None
            detections: FrameDetections = runtime.detector.detect(bgr, ts)
            snapshot = runtime.pipeline.process(detections=detections)
            elapsed = (time.perf_counter() - started) * 1000.0
            now = time.monotonic()
            runtime.processed_ts.append(now)
            while runtime.processed_ts and now - runtime.processed_ts[0] > 1.0:
                runtime.processed_ts.popleft()

            stats = dict(snapshot.stats)
            stats["processing_ms"] = round(elapsed, 2)
            stats["process_fps"] = len(runtime.processed_ts)
            stats["detections"] = len(detections.detections)
            imgsz = getattr(runtime.detector, "imgsz", None)
            if imgsz:
                stats["imgsz"] = int(imgsz)
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
                runtime = self._build_runtime(station, kind)
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "工位初始化失败 (工位=%s): %s", station.station_id, exc
                )
                continue
            self._runtimes[station.station_id] = runtime
            # 配置重载后恢复该工位之前选中的示例视频。
            video = self._active_video.get(station.station_id)
            if video is not None:
                source = self._make_video_source(
                    video,
                    station.station_id,
                    self._config.demo.loop,
                    float(self._config.demo.fps_limit),
                    float(self._config.demo.speed),
                )
                if source is not None:
                    try:
                        source.start()
                        runtime.video_source = source
                    except Exception:
                        self._logger.exception(
                            "恢复示例视频失败: %s", video.name
                        )

    def _shutdown_runtime(self, runtime: _StationRuntime) -> None:
        runtime.closed = True
        source = runtime.video_source
        runtime.video_source = None
        if source is not None:
            try:
                source.stop()
            except Exception:
                pass
        try:
            runtime.executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            runtime.detector.close()
        except Exception:
            pass

    def _video_for_camera(self, camera_id: str) -> VideoFileSource | None:
        with self._lock:
            runtimes = list(self._runtimes.values())
        for runtime in runtimes:
            if runtime.station.camera_id == camera_id and runtime.video_source:
                return runtime.video_source
        return None

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
            "tray_roi": station.tray_roi.to_dict(),
            "work_roi": station.work_roi.to_dict(),
            "oiling_roi": station.oiling_roi.to_dict(),
        },
    }
