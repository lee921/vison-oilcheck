"""训练视频录制。

第一阶段不训练模型，但需要持续采集现场视频供另一个项目训练。录制器从相机
最新帧中按固定帧率取帧，写入带时间戳的 MP4 分段，并为每段生成 JSON 元数据
（相机、工位、ROI、分辨率、帧数、丢帧数）和全局 manifest。

录制线程与 UI/检测线程解耦，队列满时丢帧而不阻塞采集。
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .types import RecordingStatus

logger = logging.getLogger("oilcheck.recorder")


@dataclass(frozen=True)
class RecordingSegment:
    path: str
    metadata_path: str
    camera_id: str
    camera_name: str
    station_id: str
    station_name: str
    started_at: float
    ended_at: float
    fps: int
    width: int
    height: int
    frame_count: int
    dropped_frames: int
    codec: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "training_video",
            "status": "finished",
            "path": self.path,
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "station_id": self.station_id,
            "station_name": self.station_name,
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at),
            "started_at_epoch": self.started_at,
            "ended_at_epoch": self.ended_at,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "dropped_frames": self.dropped_frames,
            "codec": self.codec,
        }


@dataclass
class _SegmentState:
    path: Path
    started_at: float
    width: int
    height: int
    frame_count: int = 0
    dropped_at_start: int = 0
    writer: Any = None


class TrainingRecorder:
    """单个相机的训练视频录制器。"""

    def __init__(
        self,
        *,
        camera_id: str,
        camera_name: str,
        root: Path,
        fps: int = 15,
        segment_seconds: float = 600.0,
        codec: str = "mp4v",
        station_meta: dict[str, Any] | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self.camera_id = str(camera_id)
        self.camera_name = str(camera_name)
        self._root = Path(root)
        self._fps = max(1, min(60, int(fps)))
        self._segment_seconds = max(10.0, float(segment_seconds))
        self._codec = (str(codec) or "mp4v")[:4]
        self._station_meta = dict(station_meta or {})
        self._logger = logger_ or logger

        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[float, np.ndarray] | None] = queue.Queue(
            maxsize=max(4, self._fps * 3)
        )
        self._thread: threading.Thread | None = None
        self._active = False
        self._started_at: float | None = None
        self._segment: _SegmentState | None = None
        self._frames_written = 0
        self._frames_dropped = 0
        self._last_error = ""

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    def status(self) -> RecordingStatus:
        with self._lock:
            segment_path = (
                str(self._segment.path) if self._segment is not None else None
            )
            return RecordingStatus(
                camera_id=self.camera_id,
                active=self._active,
                started_at=self._started_at,
                segment_path=segment_path,
                frames_written=self._frames_written,
                frames_dropped=self._frames_dropped,
                last_error=self._last_error,
            )

    def start(self) -> bool:
        with self._lock:
            if self._active:
                return False
            if self._thread is not None and self._thread.is_alive():
                # 上一次停止尚未完成（写盘阻塞），拒绝重复启动避免分段状态竞争。
                return False
            self._active = True
            self._started_at = time.time()
            self._frames_written = 0
            self._frames_dropped = 0
            self._last_error = ""
            self._segment = None
            self._drain_queue_locked()
            thread = threading.Thread(
                target=self._run,
                name=f"training-recorder-{self.camera_id}",
                daemon=True,
            )
            self._thread = thread
        thread.start()
        self._logger.info(
            "开始训练视频录制 (相机=%s, 帧率=%s, 分段=%ss, 目录=%s)",
            self.camera_id,
            self._fps,
            self._segment_seconds,
            self._root,
        )
        return True

    def stop(self) -> None:
        with self._lock:
            if not self._active and self._thread is None:
                return
            self._active = False
            thread = self._thread
        self._put_sentinel()
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._logger.info(
            "停止训练视频录制 (相机=%s, 已写帧=%s, 丢帧=%s)",
            self.camera_id,
            self._frames_written,
            self._frames_dropped,
        )

    def submit(self, ts: float, frame: np.ndarray) -> None:
        if not self.active:
            return
        try:
            self._queue.put_nowait((float(ts), frame))
        except queue.Full:
            with self._lock:
                self._frames_dropped += 1
            # 丢弃旧帧，尽量保留最新画面，避免录制队列无限滞后。
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((float(ts), frame))
            except (queue.Empty, queue.Full):
                return

    # ------------------------------------------------------------------
    # 录制线程
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.5)
                except queue.Empty:
                    if not self.active:
                        break
                    continue
                if item is None:
                    break
                ts, frame = item
                if not self._write_frame(ts, frame):
                    break
        except Exception as exc:  # noqa: BLE001
            self._logger.exception("训练视频录制线程异常 (相机=%s)", self.camera_id)
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._finalize_segment()
            with self._lock:
                if self._thread is threading.current_thread():
                    self._active = False
                    self._thread = None
            self._drain_queue()

    def _write_frame(self, ts: float, frame: np.ndarray) -> bool:
        if frame is None or getattr(frame, "size", 0) == 0:
            return True
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return True

        segment = self._segment
        if segment is None or segment.width != w or segment.height != h:
            self._finalize_segment()
            segment = self._open_segment(ts, w, h)
            if segment is None:
                return False
        elif ts - segment.started_at >= self._segment_seconds:
            self._finalize_segment()
            segment = self._open_segment(ts, w, h)
            if segment is None:
                return False

        writer = segment.writer
        if writer is None:
            return False
        try:
            writer.write(frame)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_error = f"写入视频帧失败: {exc}"
            self._logger.exception("写入视频帧失败 (相机=%s)", self.camera_id)
            return False

        segment.frame_count += 1
        with self._lock:
            self._frames_written += 1
            self._segment = segment
        return True

    def _open_segment(self, ts: float, width: int, height: int) -> _SegmentState | None:
        day = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        out_dir = self._root / self.camera_id / day
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromtimestamp(ts).strftime("%Y%m%d_%H%M%S")
        path = _unique_path(out_dir / f"{self.camera_id}_{stamp}")

        fourcc = cv2.VideoWriter_fourcc(*self._codec)
        writer = cv2.VideoWriter(str(path), fourcc, float(self._fps), (width, height))
        if not writer.isOpened():
            writer.release()
            message = (
                f"无法创建视频文件（codec={self._codec}, path={path}）。"
                "请检查 OpenCV 编码支持或改用 segment_seconds 更小的配置。"
            )
            self._logger.error(message)
            with self._lock:
                self._last_error = message
            return None

        segment = _SegmentState(
            path=path,
            started_at=float(ts),
            width=int(width),
            height=int(height),
            writer=writer,
            dropped_at_start=self._frames_dropped,
        )
        with self._lock:
            self._segment = segment
        self._write_sidecar(segment, finished=False)
        self._logger.info(
            "录制分段开始 (相机=%s, 文件=%s, 分辨率=%sx%s)",
            self.camera_id,
            path,
            width,
            height,
        )
        return segment

    def _finalize_segment(self) -> None:
        with self._lock:
            segment = self._segment
            self._segment = None
            dropped = (
                0
                if segment is None
                else max(0, self._frames_dropped - segment.dropped_at_start)
            )
        if segment is None:
            return
        try:
            segment.writer.release()
        except Exception:
            pass
        segment.writer = None
        self._write_sidecar(segment, finished=True, dropped_frames=dropped)
        self._append_manifest(segment, dropped_frames=dropped)
        self._logger.info(
            "录制分段结束 (相机=%s, 文件=%s, 帧数=%s, 丢帧=%s)",
            self.camera_id,
            segment.path,
            segment.frame_count,
            dropped,
        )

    def _write_sidecar(
        self,
        segment: _SegmentState,
        *,
        finished: bool,
        dropped_frames: int = 0,
    ) -> None:
        meta = RecordingSegment(
            path=str(segment.path),
            metadata_path=str(segment.path.with_suffix(".json")),
            camera_id=self.camera_id,
            camera_name=self.camera_name,
            station_id=str(self._station_meta.get("station_id", "")),
            station_name=str(self._station_meta.get("station_name", "")),
            started_at=segment.started_at,
            ended_at=time.time() if finished else 0.0,
            fps=self._fps,
            width=segment.width,
            height=segment.height,
            frame_count=segment.frame_count,
            dropped_frames=dropped_frames,
            codec=self._codec,
        ).to_dict()
        meta["status"] = "finished" if finished else "recording"
        if not finished:
            meta["ended_at"] = None
        meta["rois"] = self._station_meta.get("rois", {})
        try:
            _write_json(segment.path.with_suffix(".json"), meta)
        except OSError:
            self._logger.exception("写入录制元数据失败: %s", segment.path)

    def _append_manifest(self, segment: _SegmentState, *, dropped_frames: int) -> None:
        record = {
            "type": "training_video",
            "path": str(segment.path),
            "metadata_path": str(segment.path.with_suffix(".json")),
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "station_id": str(self._station_meta.get("station_id", "")),
            "station_name": str(self._station_meta.get("station_name", "")),
            "started_at": _iso(segment.started_at),
            "ended_at": _iso(time.time()),
            "started_at_epoch": segment.started_at,
            "fps": self._fps,
            "width": segment.width,
            "height": segment.height,
            "frame_count": segment.frame_count,
            "dropped_frames": dropped_frames,
            "codec": self._codec,
        }
        self._root.mkdir(parents=True, exist_ok=True)
        try:
            with (self._root / "manifest.jsonl").open(
                "a", encoding="utf-8", newline="\n"
            ) as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            self._logger.exception("写入录制 manifest 失败")

    def _put_sentinel(self) -> None:
        for _ in range(self._queue.maxsize + 2):
            try:
                self._queue.put_nowait(None)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    continue

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def _drain_queue_locked(self) -> None:
        self._drain_queue()


def list_segments(root: Path, *, limit: int = 200) -> list[dict[str, Any]]:
    """读取录制目录下的分段元数据，按开始时间倒序。"""
    root = Path(root)
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for path in root.glob("**/*.json"):
        if path.name == "manifest.jsonl":
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or data.get("type") != "training_video":
            continue
        items.append(data)
    items.sort(key=lambda d: float(d.get("started_at_epoch", 0.0) or 0.0), reverse=True)
    return items[: max(1, int(limit))]


def _unique_path(base: Path) -> Path:
    candidate = base.with_suffix(".mp4")
    if not candidate.exists():
        return candidate
    for index in range(1, 1000):
        candidate = base.with_name(f"{base.name}_{index:03d}").with_suffix(".mp4")
        if not candidate.exists():
            return candidate
    return base.with_name(f"{base.name}_{int(time.time())}").with_suffix(".mp4")


def _write_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _iso(ts: float) -> str | None:
    if not ts:
        return None
    return (
        datetime.fromtimestamp(float(ts))
        .astimezone()
        .isoformat(timespec="milliseconds")
    )


def delete_segments(
    root: Path,
    paths: Iterable[str | Path],
    *,
    active_paths: Iterable[str] = (),
) -> tuple[int, list[str]]:
    """删除选中的录制分段（MP4 + 同名 JSON + manifest 条目）。

    ``active_paths`` 中的文件正在写入，跳过并返回原因。返回
    ``(成功删除数量, 错误信息列表)``。
    """
    root = Path(root)
    active = {str(Path(path)) for path in active_paths}
    deleted = 0
    errors: list[str] = []
    removed_paths: set[str] = set()

    for raw in paths:
        path = Path(raw)
        path_key = str(path)
        if path_key in active:
            errors.append(f"{path.name}: 正在录制，已跳过")
            continue

        failed = False
        for target in (path, path.with_suffix(".json")):
            if not target.exists():
                continue
            try:
                target.unlink()
            except OSError as exc:
                failed = True
                errors.append(f"{target.name}: {exc}")
        if not failed:
            deleted += 1
            removed_paths.add(path_key)

    manifest = root / "manifest.jsonl"
    if removed_paths and manifest.exists():
        try:
            kept: list[str] = []
            for line in manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                if str(data.get("path", "")) in removed_paths:
                    continue
                kept.append(line)
            temp = manifest.with_suffix(".jsonl.tmp")
            temp.write_text(
                "\n".join(kept) + ("\n" if kept else ""),
                encoding="utf-8",
            )
            temp.replace(manifest)
        except OSError as exc:
            errors.append(f"更新 manifest 失败: {exc}")

    return deleted, errors
