"""NG 事件的画面片段存储。

复核员看到一条漏涂 NG（``OILING_MISSING``）时，要能直接播放这一件前后的
画面，而不是去几小时长的训练录像里翻。设计说明见
``docs/14-NG事件片段存储设计.md``。

三件事：

1. :class:`_ClipBuffer` —— 每相机一个环形帧缓冲，只留最近 ``pre_seconds``；
2. :class:`EventClipService` —— 事件到来时**登记**要录哪一段，后续帧把
   「事件之后」的部分录满，再切出 mp4 并回填事件的 ``clip_reference``；
3. 纯逻辑、不碰 Qt 与相机 SDK，便于单测。

为什么必须先有环形缓冲：NG 是**事后**才成立的事实。``oiling_timeout``
要等取料后 ``oiling_timeout_s``（现场配置 5 秒）才判出，等收到事件再开始
录，前面那 5 秒已经过去了。

片段要覆盖**完整过程**：从取料前 ``clip_margin_seconds``（1.5 秒）到判出
NG 后 ``clip_margin_seconds``。也就是说缓冲里必须一直保留
``oiling_timeout_s + 2 × margin`` ≈ 8 秒的历史。

**缓冲存的是编码后的 JPEG，不是原始帧。** 2560×1440 的原始帧 11MB，
8 秒 ×15fps 要 1.3GB 内存，现场机器扛不住；编码成 JPEG（q=75）后每帧
约 1.7~2.2MB，同样时长约 210~265MB。代价是每帧多一次
``cv2.imencode``（原生分辨率约 11ms），比检测本身便宜、且与判定线程
解耦（在轮询线程里做）。

**写盘跑在独立线程。** ``imdecode + VideoWriter.write`` 实测原生分辨率
14~41ms/帧（取决于画面内容），一段 60 帧的片段就是 1~2.7 秒。这段开销
原先摊在服务轮询线程里，而轮询线程同时是**取帧线程**：写盘期间不取帧，
相机帧在 ``FrameBuffer`` 里被覆盖。现场日志里 1.2~1.7 秒的「视频时间
戳异常」有八成落在这类窗口内。现在 push/on_event/flush_idle 只收集帧与
登记待写片段，真正的解码写盘交给 :class:`_ClipWriter` 单线程完成。

JPEG 压不动时（现场画面几乎全静态）小段缓冲还能达到多条；这时按内存
上限丢最旧的帧并记一条 warning——**画面比配置时长更重要**，宁可少一点
缓冲，也不能 OOM 拖垮判定。
"""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("oilcheck.clips")

# 片段文件名里的原因短语，让复核员在文件管理器里也能认出来。
_REASON_SLUG = {
    "oiling_missing_timeout": "oiling_missing_timeout",
    "oiling_missing_superseded_by_pick": "oiling_missing_superseded",
}


@dataclass(frozen=True)
class ClipConfig:
    """片段存储的运行参数（由 ``RecordingConfig`` 派生）。"""

    enabled: bool = False
    root: Path = Path("clips")
    # 历史缓冲要保留多久。这一项只用于算容量，起点仍从事件里取。
    pre_seconds: float = 3.0
    # 漏涂超时窗（``pipeline.oiling_timeout_s``）。缓冲容量要把它算进去：
    # 事件起点（取料时刻）可能比事件到达时刻早整整一个超时窗。
    timeout_seconds: float = 10.0
    # 事件自带的起止之外，前后各再延长多久（秒）。
    # 事件里的起止是**判定**口径（例如漏涂的起止是「取料那一刻」到「判出那一刻」），
    # 比人手看到的完整动作窄：取料前手已经在盘上方伸过来，判出后还要放下件、
    # 抬头、去取下一件。留一段余量才能看清完整过程。
    margin_seconds: float = 1.5
    post_seconds: float = 2.0
    fps: int = 15
    codec: str = "mp4v"
    max_pending: int = 2
    retain_days: int = 30
    # 缓冲里 JPEG 编码的缩放与质量。0 = 不缩放。720p 的编码只要 2ms/帧、
    # 缓冲 45MB；原生分辨率更清楚但更贵。复核看动作，720p 通常够。
    buffer_width: int = 0
    jpeg_quality: int = 75
    # 环形缓冲的内存上限（MB）。超过就丢最旧的帧并记 warning。
    buffer_max_mb: int = 512
    # 事件之后多久没有新帧就强制收尾。相机断流时不能永远挂着一份半成品。
    finalize_timeout_s: float = 3.0


@dataclass(frozen=True)
class _Frame:
    """缓冲里的一帧：编码后的 JPEG + 两个时间 + 编码尺寸。"""

    ts: float
    data: np.ndarray
    # 判定时间轴上的时间；实时相机上等于 ``ts``。
    pipeline_ts: float
    # 编码后的 (宽, 高)。写片段时按它建 VideoWriter。
    width: int
    height: int

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)


class _ClipBuffer:
    """单相机的环形帧缓冲，外加「正在录」的片段登记。

    所有方法都在 :class:`EventClipService` 的锁里调用。
    """

    def __init__(self, capacity: int, *, max_bytes: int = 0) -> None:
        self._frames: deque[_Frame] = deque(maxlen=max(2, int(capacity)))
        self.pending: list[_PendingClip] = []
        self._max_bytes = max(0, int(max_bytes))
        self._bytes = 0
        # 已经因为内存上限丢过帧：只记一次，避免每帧刷屏。
        self.over_cap = False

    @property
    def frames(self) -> deque[_Frame]:
        return self._frames

    @property
    def nbytes(self) -> int:
        return self._bytes

    def push(self, frame: _Frame) -> None:
        if self._frames.maxlen is not None and len(self._frames) == self._frames.maxlen:
            # deque 会自己丢掉最旧的一帧，内存账得同步减掉。
            self._bytes -= self._frames[0].nbytes
        self._frames.append(frame)
        self._bytes += frame.nbytes
        # 内存上限：丢掉最旧的帧。宁可缓冲短一点，也不能 OOM 拖垮判定。
        while self._max_bytes and self._bytes > self._max_bytes and self._frames:
            self._bytes -= self._frames.popleft().nbytes
            self.over_cap = True

    def clear(self) -> None:
        self._frames.clear()
        self._bytes = 0

    def earliest(self) -> _Frame | None:
        return self._frames[0] if self._frames else None

    def latest(self) -> _Frame | None:
        return self._frames[-1] if self._frames else None

    def offset(self, window: int = 30) -> float:
        """``墙钟 − 判定时间`` 的当前偏移（最近若干帧的中位数）。

        实时相机上恒为 0（判定时间回退成墙钟）；示例回放时是真实的差值。
        切片需要把事件里**媒体时间**的起点换算成缓冲里**墙钟**的位置，
        靠它来换。
        """

        frames = list(self._frames)[-max(1, window) :]
        if not frames:
            return 0.0
        deltas = sorted(frame.ts - frame.pipeline_ts for frame in frames)
        return deltas[len(deltas) // 2]


@dataclass
class _PendingClip:
    """一段已经登记、正在攒「事件之后」帧的片段。"""

    event_id: str
    task_id: str
    event_type: str
    reason: str
    station_id: str
    camera_id: str
    # 起点（墙钟）与预计终点（墙钟）。
    start_ts: float
    end_ts: float
    registered_at: float
    # 已经收进这一段、待写盘的帧，与其中的写入进度。
    frames: list[_Frame] = field(default_factory=list)
    # 已经收进 ``frames`` 的最大帧时间戳，用来只取更新的帧。
    consumed_ts: float = 0.0
    truncated: bool = False
    # 最后一次收到帧的时间，用于「断流后强制收尾」。
    last_frame_ts: float = 0.0
    dropped: int = 0
    closed: bool = False
    # 已被判定要收尾（超并发上限 / 断流超时），排到写盘队列等写盘线程处理。
    # 判定线程与轮询线程只置标记，不在这里编码。
    close_requested: bool = False
    # 已经排进 ``EventClipService._write_queue``，避免重复入队。
    queued: bool = False


EventUpdater = Callable[[str, str], None]


@dataclass(frozen=True)
class _WriteJob:
    """一段待写片段交给写盘线程的完整信息（不可变，跨线程传递安全）。"""

    # 已经收集好、按时间排序的帧（含事件前的历史缓冲帧）。
    frames: tuple[_Frame, ...]
    # 输出目录、文件名主干与编码参数。
    root: Path
    codec: str
    fps: int
    # 文件名主干：``<task_id>_<reason_slug>``。时间戳由写盘线程按首帧补。
    name_hint: str
    # 收尾元数据。
    meta: dict[str, Any]
    # 写完后回填事件用的回调；None 表示不需要回填。
    notify: EventUpdater | None


class _ClipWriter:
    """单线程串行写片段的解码与编码。

    为什么必须独立线程：``cv2.imdecode + VideoWriter.write`` 实测原生
    2560×1440 是 14ms/帧（画面简单）到 41ms/帧（高频细节），一段 60 帧
    的片段要 0.9~2.5 秒。原先这段开销发生在服务轮询线程里，而轮询线程
    同时负责从 ``FrameBuffer`` 取帧——写盘期间相机的新帧直接被覆盖掉，
    表现为 1.2~1.7 秒的「视频时间戳异常」。
    """

    def __init__(
        self, *, on_finished: Callable[[_WriteJob, Path | None], None], logger_: logging.Logger
    ) -> None:
        self._on_finished = on_finished
        self._logger = logger_
        self._queue: queue.Queue[_WriteJob | None] = queue.Queue()
        # 队列的 task_done 通知：``drain`` 靠它等写完（测试与关停前用）。
        self._cond = threading.Condition()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, job: _WriteJob) -> None:
        with self._lock:
            if self._closed:
                return
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, name="oilcheck-clip-writer", daemon=True
                )
                self._thread.start()
        self._queue.put(job)

    def close(self, timeout: float = 5.0) -> None:
        """把队列里的片段写完再退出（最多等 ``timeout`` 秒）。"""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is None:
            return
        self._queue.put(None)
        thread.join(timeout=timeout)
        if thread.is_alive():
            self._logger.warning("片段写盘线程未在 %.0fs 内退出", timeout)
        # 哨兵项也要 task_done，否则 drain() 永远等下去。
        try:
            self._queue.task_done()
        except ValueError:
            pass
        with self._cond:
            self._cond.notify_all()

    def drain(self, timeout: float = 5.0) -> bool:
        """等队列里的片段写完（不关闭，后续还能继续提交）。返回是否排空。"""

        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while self._queue.unfinished_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def _run(self) -> None:
        while True:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                with self._lock:
                    if self._closed:
                        return
                continue
            if job is None:
                self._queue.task_done()
                with self._cond:
                    self._cond.notify_all()
                return
            path: Path | None = None
            try:
                path = self._write(job)
            except Exception:  # noqa: BLE001
                self._logger.exception("写事件片段失败 (%s)", job.meta.get("task_id"))
            try:
                self._on_finished(job, path)
            except Exception:  # noqa: BLE001
                self._logger.exception("收尾事件片段失败 (%s)", job.meta.get("event_id"))
            finally:
                self._queue.task_done()
                with self._cond:
                    self._cond.notify_all()

    def _write(self, job: _WriteJob) -> Path | None:
        if not job.frames:
            self._logger.warning(
                "事件 %s 的片段没有可写画面（缓冲不足），未生成文件",
                job.meta.get("event_id"),
            )
            return None
        first = job.frames[0]
        width, height = first.width, first.height
        if width <= 0 or height <= 0:
            return None
        day = datetime.fromtimestamp(first.ts).strftime("%Y-%m-%d")
        out_dir = job.root / day
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._logger.error("创建片段目录失败 %s: %s", out_dir, exc)
            return None
        stamp = datetime.fromtimestamp(first.ts).strftime("%Y%m%d_%H%M%S")
        name = f"{job.name_hint}_{stamp}.mp4"
        path = _unique_path(out_dir / name)
        fourcc = cv2.VideoWriter_fourcc(*job.codec)
        writer = cv2.VideoWriter(str(path), fourcc, float(job.fps), (width, height))
        if not writer.isOpened():
            writer.release()
            self._logger.error(
                "无法创建片段文件（codec=%s, path=%s）", job.codec, path
            )
            return None
        self._logger.info(
            "开始写事件片段 %s（%s，%d 帧）",
            path.name,
            job.meta.get("task_id"),
            len(job.frames),
        )
        written = 0
        try:
            for frame in job.frames:
                img = cv2.imdecode(frame.data, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                if img.shape[1] != frame.width or img.shape[0] != frame.height:
                    img = cv2.resize(img, (frame.width, frame.height))
                writer.write(img)
                written += 1
        finally:
            writer.release()
        meta = dict(job.meta)
        meta.update(
            {
                "path": str(path),
                "width": width,
                "height": height,
                "frame_count": written,
                "started_at_epoch": first.ts,
                "ended_at_epoch": job.frames[-1].ts,
            }
        )
        try:
            path.with_suffix(".json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:
            self._logger.warning("写入片段元数据失败 %s: %s", path, exc)
        self._logger.info(
            "事件片段已保存 %s（%d 帧%s）",
            path.name,
            written,
            "，开头被截断" if meta.get("truncated") else "",
        )
        return path


class EventClipService:
    """按相机维护环形缓冲，把 NG 事件前后的画面切成 mp4。"""

    def __init__(
        self,
        *,
        config: ClipConfig,
        on_clip_ready: EventUpdater | None = None,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._cfg = config
        self._on_clip_ready = on_clip_ready
        self._logger = logger_ or logger
        self._lock = threading.Lock()
        self._buffers: dict[str, _ClipBuffer] = {}
        # 写盘线程：解码 + 编码要 14~41ms/帧，放在轮询线程里会直接造成丢帧。
        self._writer = _ClipWriter(
            on_finished=self._on_clip_written, logger_=self._logger
        )
        # 开始时清一次过期片段，长期运行不会无限涨。
        if self._cfg.enabled and self._cfg.retain_days > 0:
            self.purge_old_clips(self._cfg.retain_days)
        if self._cfg.enabled:
            self._clean_stale_pending_files()

    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    @property
    def root(self) -> Path:
        return self._cfg.root

    # ------------------------------------------------------------------
    # 帧入口
    # ------------------------------------------------------------------
    def push(self, camera_id: str, *, bgr: np.ndarray, ts: float, pipeline_ts: float) -> None:
        """每帧调用一次。未启用时立即返回（零开销）。

        缓冲里存**编码后的 JPEG**：原始帧 11MB/帧，要覆盖「取料 → 等满
        ``oiling_timeout_s`` → 再余量」约 8 秒，按原始帧存要 1.3GB 内存。
        """

        if not self._cfg.enabled or bgr is None:
            return
        frame = self._encode(bgr, float(ts), float(pipeline_ts))
        if frame is None:
            return
        with self._lock:
            buffer = self._buffer_locked(camera_id)
            was_over = buffer.over_cap
            buffer.push(frame)
            if buffer.over_cap and not was_over:
                self._logger.warning(
                    "片段缓冲达到内存上限 %dMB（%s）：已丢最旧的帧，"
                    "片段开头可能比配置的时长短。可调小 clip_buffer_width "
                    "或 clip_jpeg_quality",
                    self._cfg.buffer_max_mb,
                    camera_id,
                )
            finished: list[_PendingClip] = []
            for pending in buffer.pending:
                pending.last_frame_ts = max(pending.last_frame_ts, frame.ts)
                if pending.close_requested:
                    finished.append(pending)
                    continue
                self._advance_locked(pending, finished)
            for clip in finished:
                buffer.pending.remove(clip)
                self._finalize_locked(clip)

    def _encode(self, bgr: np.ndarray, ts: float, pipeline_ts: float) -> _Frame | None:
        """把一帧编码成 JPEG 并缩放（可选）。失败时返回 None。"""

        img = bgr
        # ``buffer_width`` 是目标宽度；按比例缩，保持长宽比为偶数（编码器要求）。
        target = int(self._cfg.buffer_width)
        if target > 0 and img.shape[1] > target:
            scale = target / float(img.shape[1])
            size = (target - target % 2, max(2, int(round(img.shape[0] * scale / 2)) * 2))
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        params = [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(self._cfg.jpeg_quality)))]
        try:
            ok, data = cv2.imencode(".jpg", img, params)
        except Exception:  # noqa: BLE001
            self._logger.exception("片段缓冲编码失败 (%s)", ts)
            return None
        if not ok:
            return None
        height, width = img.shape[:2]
        return _Frame(
            ts=ts,
            data=data,
            pipeline_ts=pipeline_ts,
            width=int(width),
            height=int(height),
        )

    def flush_idle(self, camera_id: str | None = None) -> None:
        """把长时间没有新帧的片段强制收尾（由服务轮询线程定期调用）。

        不依赖「下一帧到来」：相机断流后就再也不会有帧了，只能靠时间来收。
        这里只把片段排进写盘队列，解码与编码在 :class:`_ClipWriter` 里做。
        """

        if not self._cfg.enabled:
            return
        now = time.monotonic()
        with self._lock:
            for key, buffer in self._buffers.items():
                if camera_id is not None and key != camera_id:
                    continue
                finished: list[_PendingClip] = []
                for pending in list(buffer.pending):
                    if pending.close_requested:
                        finished.append(pending)
                    elif now - pending.last_frame_ts > self._cfg.finalize_timeout_s:
                        finished.append(pending)
                for clip in finished:
                    buffer.pending.remove(clip)
                    self._finalize_locked(clip)

    def close(self) -> None:
        """服务退出时收尾全部在录片段，并等写盘队列排空。

        必须等：``kill``/关窗时队列里可能还有几段没写，直接返回会留下
        没写元数据、复核页打不开的 mp4。
        """

        with self._lock:
            for buffer in self._buffers.values():
                for pending in list(buffer.pending):
                    self._finalize_locked(pending)
                buffer.pending.clear()
                buffer.clear()
        self._writer.close()

    def reset(self, camera_id: str | None = None) -> None:
        """丢掉缓冲（切换视频 / 回绕 / 相机重连时调用）。"""

        with self._lock:
            if camera_id is None:
                for buffer in self._buffers.values():
                    buffer.clear()
            else:
                buffer = self._buffers.get(camera_id)
                if buffer is not None:
                    buffer.clear()

    # ------------------------------------------------------------------
    # 事件入口
    # ------------------------------------------------------------------
    def on_event(
        self,
        *,
        event_id: str,
        task_id: str,
        event_type: str,
        reason: str,
        station_id: str,
        camera_id: str,
        event_start: float,
        event_end: float,
        needs_review: bool,
    ) -> bool:
        """登记一段片段。返回是否真的排上了。

        区间口径（``event_start`` / ``event_end``）由事件自带：

        - 起点是**取料时刻**（漏涂就是这一件被取走的那一刻）；
        - 终点是**判出 NG 的时刻**。

        两者前后各放宽 ``margin_seconds``，得到「取料前 1.5 秒 → NG 后 1.5 秒」
        的完整过程。漏涂的终点是等满 ``oiling_timeout_s`` 才判出的，这
        一整个等待窗本来就属于「完整过程」的一部分，不能裁。
        """

        if not self._cfg.enabled or not needs_review:
            return False
        with self._lock:
            buffer = self._buffers.get(camera_id)
            if buffer is None or not buffer.frames:
                # 缓冲为空：还没开始预览/刚重置。不伪造画面，记一条便于排查。
                self._logger.debug(
                    "事件 %s 到达时 %s 的帧缓冲为空，跳过片段", event_id, camera_id
                )
                return False
            offset = buffer.offset()
            margin = max(0.0, self._cfg.margin_seconds)
            # 起点 = 取料前 margin 秒。``pre_seconds`` 只是缓冲容量，不进这里。
            start_ts = event_start + offset - margin
            # 终点 = 判出 NG 后 margin 秒，再叠加 ``post_seconds``（默认 2 秒：
            # 判出后工人往往立刻回头处理，这一段能看清后续动作）。
            anchor_end = max(event_end, event_start)
            end_ts = anchor_end + offset + margin + self._cfg.post_seconds
            # 起点早于缓冲里最早那帧时，直接从最早一帧开始（不伪造黑帧）。
            earliest = buffer.earliest()
            truncated = earliest is None or start_ts < earliest.ts
            if earliest is not None and start_ts < earliest.ts:
                start_ts = earliest.ts
            if start_ts >= end_ts:
                # 缓冲里最早的帧都比终点晚：没有可用画面。
                self._logger.debug(
                    "事件 %s 的区间 [%.2f, %.2f] 已在缓冲之外，跳过片段",
                    event_id,
                    start_ts,
                    end_ts,
                )
                return False
            pending = _PendingClip(
                event_id=event_id,
                task_id=task_id,
                event_type=event_type,
                reason=reason,
                station_id=station_id,
                camera_id=camera_id,
                start_ts=start_ts,
                end_ts=end_ts,
                registered_at=time.monotonic(),
                last_frame_ts=time.monotonic(),
                truncated=truncated,
            )
            buffer.pending.append(pending)
            limit = max(1, self._cfg.max_pending)
            # 只数「还在录」的那些：已请求收尾的还留在列表里等轮询线程落盘，
            # 不算并发。不这样区分的话 max_pending=1 时会反复把同一段标一轮。
            while len([q for q in buffer.pending if not q.close_requested]) > limit:
                old = next(q for q in buffer.pending if not q.close_requested)
                self._logger.info(
                    "片段并发数超限，提前收尾 %s", old.task_id or old.event_id
                )
                # 不在这里编码：判定线程不能等写盘。置标记，轮询线程收尾。
                old.close_requested = True
            # 先把已缓存的帧收进这段（不写盘）：写盘留给轮询线程，
            # 免得一段在**判定线程**里同步编码、卡住判定。
            self._advance_locked(pending, None)
            return True

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _buffer_locked(self, camera_id: str) -> _ClipBuffer:
        buffer = self._buffers.get(camera_id)
        if buffer is None:
            # 容量要覆盖「取料前 margin → 判出 NG → 再 margin」整段：
            # 事件起点可能远早于事件到达时刻（漏涂要等满超时窗）。
            history = (
                self._cfg.pre_seconds
                + self._cfg.margin_seconds
                + self._cfg.timeout_seconds
            )
            capacity = int(round(history * max(1, self._cfg.fps))) + 4
            buffer = _ClipBuffer(
                capacity, max_bytes=int(self._cfg.buffer_max_mb) * 1024 * 1024
            )
            self._buffers[camera_id] = buffer
        return buffer

    def _advance_locked(self, pending: _PendingClip, finished: list[_PendingClip]) -> None:
        """把缓冲里还没收过的帧收进这段；到终点就排进写盘队列。

        ``finished`` 允许为 ``None``：``on_event`` 里只想把已缓存的帧先收进
        这段（不排写盘、不判终点），写盘要等下一帧推进时才做——否则一段在
        当前帧就已经跨过终点的片段会在**判定线程**里同步编码，卡住判定。

        不论哪种路径，这里都**只收集帧、不写盘**：解码与编码在
        :class:`_ClipWriter` 线程里做（见模块文档）。
        """

        if finished is None:
            self._collect_locked(pending)
            return
        buffer = self._buffers.get(pending.camera_id)
        if buffer is None or pending.closed:
            return
        self._collect_locked(pending)
        # 收尾看的是「缓冲里已经出现过终点之后的帧」，不是「最后一帧过了终点」——
        # 超过终点的帧按设计不收（见 _collect_locked），所以只能看缓冲区。
        latest = buffer.latest()
        if latest is not None and latest.ts >= pending.end_ts:
            finished.append(pending)

    def _collect_locked(self, pending: _PendingClip) -> None:
        """把缓冲里落在 ``[start_ts, end_ts]`` 且尚未收过的帧收进这段。

        超过 ``end_ts`` 的帧**不收**：片段长度由事件口径决定，不能因为收尾
        晚了半拍就把后面无关的画面也带进来。
        """

        buffer = self._buffers.get(pending.camera_id)
        if buffer is None:
            return
        for frame in buffer.frames:
            if frame.ts < pending.start_ts or frame.ts <= pending.consumed_ts:
                continue
            if frame.ts > pending.end_ts:
                break
            pending.frames.append(frame)
            pending.consumed_ts = frame.ts

    def _finalize_locked(self, pending: _PendingClip) -> None:
        """收尾：把这段交给写盘线程（只入队，不做任何编解码）。"""

        if pending.closed:
            return
        pending.closed = True
        if pending.queued:
            return
        pending.queued = True
        slug = _REASON_SLUG.get(pending.reason, pending.reason or "event")
        # 文件名主干：任务号 + 原因，写盘线程再拼时间戳。放这里是为了让
        # 文件名里的原因与事件一致（原因短语只在本模块有映射）。
        task_label = pending.task_id or pending.event_id
        meta = {
            "type": "event_clip",
            "status": "finished",
            "event_id": pending.event_id,
            "event_type": pending.event_type,
            "reason": pending.reason,
            "task_id": pending.task_id,
            "station_id": pending.station_id,
            "camera_id": pending.camera_id,
            "fps": self._cfg.fps,
            "truncated": pending.truncated,
        }
        job = _WriteJob(
            frames=tuple(pending.frames),
            root=self._cfg.root,
            codec=self._cfg.codec,
            fps=self._cfg.fps,
            meta=meta,
            notify=self._on_clip_ready,
            name_hint=f"{task_label}_{slug}",
        )
        self._writer.submit(job)

    def _on_clip_written(self, job: _WriteJob, path: Path | None) -> None:
        """写盘线程回调：回填事件的片段路径（复核页据此播放）。"""

        if path is None or job.notify is None:
            return
        job.notify(str(job.meta.get("event_id", "")), str(path))

    # ------------------------------------------------------------------
    # 维护
    # ------------------------------------------------------------------
    def purge_old_clips(self, retain_days: int) -> int:
        """删掉超过保留天数的片段目录。返回删除的目录数。"""

        if retain_days <= 0 or not self._cfg.root.exists():
            return 0
        cutoff = time.time() - retain_days * 86400
        removed = 0
        for day_dir in sorted(self._cfg.root.iterdir()):
            if not day_dir.is_dir():
                continue
            try:
                day_ts = datetime.strptime(day_dir.name, "%Y-%m-%d").timestamp()
            except ValueError:
                continue
            if day_ts >= cutoff - 86400:
                continue
            try:
                shutil.rmtree(day_dir)
                removed += 1
            except OSError as exc:
                self._logger.warning("删除过期片段目录失败 %s: %s", day_dir, exc)
        if removed:
            self._logger.info("已清理 %d 个超过 %d 天的片段目录", removed, retain_days)
        return removed

    def _clean_stale_pending_files(self) -> None:
        """启动时删掉上次异常退出留下的 0 字节片段。

        ``kill -9`` 时 writer 不会 release，mp4 没有 moov atom、无法播放。
        留着只会让复核页点开报错，不如清掉。
        """

        if not self._cfg.root.exists():
            return
        count = 0
        for path in self._cfg.root.rglob("*.mp4"):
            try:
                if path.stat().st_size == 0:
                    path.unlink()
                    count += 1
            except OSError:
                continue
        if count:
            self._logger.info("已清理 %d 个无法播放的空片段文件", count)

    def pending_count(self, camera_id: str | None = None) -> int:
        with self._lock:
            if camera_id is not None:
                buffer = self._buffers.get(camera_id)
                return 0 if buffer is None else len(buffer.pending)
            return sum(len(buffer.pending) for buffer in self._buffers.values())

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """等写盘队列排空（含还没写完成的片段）。返回是否排空。

        生产代码不依赖它——轮询线程只负责登记。测试与「关停前确认已经
        落盘」用它，避免用 ``sleep`` 猜时间。
        """

        return self._writer.drain(timeout)


def _unique_path(base: Path) -> Path:
    """避免重名覆盖：``name.mp4`` → ``name_2.mp4``。"""

    if not base.exists():
        return base
    stem, suffix = base.stem, base.suffix
    for index in range(2, 1000):
        candidate = base.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    return base.with_name(f"{stem}_{int(time.time())}{suffix}")
