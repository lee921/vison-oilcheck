"""离线 A/B：比较跟踪参数/策略下的手部与刷头轨迹质量，以及判定结果是否变化。

用途是回答「换个关联策略或调个阈值，手/刷头的 ID 到底稳不稳」：

    uv run python tools/tracker_ab.py                       # 当前配置跑全部示例视频
    uv run python tools/tracker_ab.py --high-thresh 0.30,0.45,0.50
    uv run python tools/tracker_ab.py --brush-spawn 0.10
    uv run python tools/tracker_ab.py --downsample 3        # 每 3 帧取 1 帧，模拟低帧率
    uv run python tools/tracker_ab.py --video 2 --frames 300

只读：不写配置、不写事件库、不启动 GUI。检测只跑一遍（结果缓存在内存里），
每个变体各自建一套 ``StationPipeline`` 回放，判定走与线上完全相同的代码。

指标口径：

- ``PICK`` / ``OIL`` / ``MISS``：取料、人工二次涂油、漏涂事件数；
- ``ids``：该类别轨迹总数（碎片与「静默劫持」都会让它变少，不能单独看）；
- ``frag``：生命周期 ≤3 帧的轨迹数（真正的碎片）；
- ``big``：**同一 ID 相邻帧中心位移 >10% 画面对角线**的次数——这代表 ID 悄悄
  换成了另一个目标，比 ID 总数更能反映错误；
- ``match``：轨迹帧中 ``matched=True`` 的比例；
- ``/件``：每个作业周期里绑定过的手/刷 ID 数的最大值（业务口径）。

某个变体让事件数与基线不同时行尾标 `!`；``big`` 比基线多时标 `hijack?`。
"""

from __future__ import annotations

import argparse
import collections
import glob
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from app.oilcheck.config import OilCheckConfigService  # noqa: E402
from app.oilcheck.detector import create_detector  # noqa: E402
from app.oilcheck.pipeline import StationPipeline  # noqa: E402
from app.oilcheck.types import FrameDetections  # noqa: E402

# 同一 ID 相邻帧位移超过画面对角线的这个比例，判为「换了目标」。
JUMP_FRACTION = 0.10
# 生命周期不超过这么多帧算碎片。
FRAGMENT_FRAMES = 3


def _fps_limit(cfg) -> float:
    """示例视频按原始帧率抽帧，超过 demo.fps_limit 时降采样。"""

    limit = float(cfg.demo.fps_limit or 15.0)
    return min(15.0, max(1.0, limit))


def _cache_video(detector, path: Path, *, max_frames: int, fps: float):
    cap = cv2.VideoCapture(str(path))
    frames: list[FrameDetections] = []
    if not cap.isOpened():
        print(f"  无法打开视频: {path}")
        return frames
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        ts = len(frames) / fps
        dets = detector.detect(frame, ts)
        frames.append(
            FrameDetections(
                ts=ts,
                frame_size=(int(width), int(height)),
                detections=tuple(dets.detections),
                source="cached",
            )
        )
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames


def evaluate(frames, *, station, pipeline_cfg) -> dict:
    """跑一遍判定，统计跟踪与业务指标。"""

    pipeline = StationPipeline(
        station=station,
        pipeline_cfg=pipeline_cfg,
        detector_name="cached",
    )
    events: list[str] = []
    pipeline._on_event = lambda ev: events.append(str(ev.event_type))  # noqa: SLF001

    diag = 0.0
    life = {"hand": collections.Counter(), "brush": collections.Counter()}
    matched = {"hand": 0, "brush": 0}
    total = {"hand": 0, "brush": 0}
    jumps = {"hand": 0, "brush": 0}
    prev: dict[str, dict[int, tuple[float, float]]] = {"hand": {}, "brush": {}}
    binds = {"hand": collections.defaultdict(set), "brush": collections.defaultdict(set)}

    for frame in frames:
        diag = max(1.0, float(np.hypot(*frame.frame_size)))
        pipeline.process(detections=frame)
        for label, tracks in (
            ("hand", pipeline._hand_tracks),  # noqa: SLF001
            ("brush", pipeline._brush_tracks),  # noqa: SLF001
        ):
            current = {}
            for track in tracks:
                life[label][track.track_id] += 1
                matched[label] += int(track.matched)
                total[label] += 1
                current[track.track_id] = track.box.center
            for track_id, center in current.items():
                last = prev[label].get(track_id)
                if last is None:
                    continue
                distance = (
                    np.hypot(center[0] - last[0], center[1] - last[1]) / diag
                )
                if distance > JUMP_FRACTION:
                    jumps[label] += 1
            prev[label] = current
        # 业务口径：每个周期绑定过的手/刷 ID 数
        for cycle_id, cycle in pipeline._cycles.items():  # noqa: SLF001
            if cycle.hand_track_id is not None:
                binds["hand"][cycle_id].add(cycle.hand_track_id)
            if cycle.brush_track_id is not None:
                binds["brush"][cycle_id].add(cycle.brush_track_id)

    counts = collections.Counter(
        event.replace("EventType.", "") for event in events
    )
    return {
        "PICK": counts.get("PICK", 0),
        "OIL": counts.get("MANUAL_OILING", 0),
        "MISS": counts.get("OILING_MISSING", 0),
        "tasks": len(pipeline.tasks),
        "hand_ids": len(life["hand"]),
        "brush_ids": len(life["brush"]),
        "hand_frag": sum(1 for v in life["hand"].values() if v <= FRAGMENT_FRAMES),
        "brush_frag": sum(1 for v in life["brush"].values() if v <= FRAGMENT_FRAMES),
        "hand_big": jumps["hand"],
        "brush_big": jumps["brush"],
        "hand_match": matched["hand"] / max(1, total["hand"]),
        "brush_match": matched["brush"] / max(1, total["brush"]),
        "hand_per_task": max((len(v) for v in binds["hand"].values()), default=0),
        "brush_per_task": max((len(v) for v in binds["brush"].values()), default=0),
    }


def _fmt(name: str, row: dict, base: dict | None) -> str:
    flag = ""
    if base is not None:
        if (row["PICK"], row["OIL"], row["MISS"]) != (
            base["PICK"],
            base["OIL"],
            base["MISS"],
        ):
            flag += " !"
        if row["hand_big"] + row["brush_big"] > base["hand_big"] + base["brush_big"]:
            flag += " hijack?"
    return (
        f"  {name:<14} PICK={row['PICK']} OIL={row['OIL']} MISS={row['MISS']} "
        f"tasks={row['tasks']} | "
        f"手 ids={row['hand_ids']} frag={row['hand_frag']} big={row['hand_big']} "
        f"m={row['hand_match']:.3f} /件={row['hand_per_task']} | "
        f"刷 ids={row['brush_ids']} frag={row['brush_frag']} big={row['brush_big']} "
        f"m={row['brush_match']:.3f} /件={row['brush_per_task']}{flag}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video", type=int, default=0, help="只跑第 N 段示例视频（1 基），0=全部"
    )
    parser.add_argument("--frames", type=int, default=0, help="只跑前 N 帧，0=全部")
    parser.add_argument(
        "--downsample", type=int, default=1, help="每 N 帧取 1 帧，模拟低帧率"
    )
    parser.add_argument(
        "--high-thresh",
        default="",
        help="逗号分隔的 bytetrack_high_thresh 取值，例如 0.30,0.45",
    )
    parser.add_argument(
        "--brush-spawn",
        default="",
        help="逗号分隔的 brush_spawn_max_distance 取值，例如 0,0.10",
    )
    args = parser.parse_args()

    service = OilCheckConfigService()
    # 只读：不走 ensure_defaults()，避免分析动作触发配置迁移/写回。
    cfg = service.load(create_if_missing=False)
    stations = cfg.enabled_stations()
    if not stations:
        print("配置里没有启用的工位")
        return 1
    station = stations[0]
    detector = create_detector(station)
    if detector.name == "null":
        print("该工位检测器是 null，没有可比较的检测结果")
        return 1
    detector_open = getattr(detector, "close", None)

    variants: list[tuple[str, object]] = [("当前配置", cfg.pipeline)]
    for value in _values(args.high_thresh):
        variants.append(
            (
                f"high_thresh={value:g}",
                replace(cfg.pipeline, bytetrack_high_thresh=value),
            )
        )
    for value in _values(args.brush_spawn):
        variants.append(
            (
                f"brush_spawn={value:g}",
                replace(cfg.pipeline, brush_spawn_max_distance=value),
            )
        )

    fps = _fps_limit(cfg)
    videos = [
        Path(p)
        for p in sorted(glob.glob(str(REPO_ROOT / "data" / "sample_video" / "*.mp4")))
    ]
    if args.video:
        videos = videos[args.video - 1 : args.video]
    if not videos:
        print("data/sample_video 下没有示例视频")
        return 1

    step = max(1, int(args.downsample))
    print(f"配置版本 {cfg.version}，抽帧 {fps:g}fps，每 {step} 帧取 1 帧")
    for path in videos:
        print(f"\n=== {path.name}")
        frames = _cache_video(detector, path, max_frames=args.frames, fps=fps)
        if not frames:
            continue
        frames = frames[::step]
        print(f"  有效帧 {len(frames)}（{len(frames) / fps:.1f}s）")
        base_row = None
        for name, pipeline_cfg in variants:
            row = evaluate(frames, station=station, pipeline_cfg=pipeline_cfg)
            if base_row is None:
                base_row = row
            print(_fmt(name, row, base_row))
    if callable(detector_open):
        detector_open()
    return 0


def _values(text: str) -> list[float]:
    values: list[float] = []
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            values.append(float(item))
        except ValueError:
            print(f"忽略无法解析的取值: {item}")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
