"""离线验 PLC 工步序列：回放视频，把每一拍的 PLC 帧打成表格。

**不连 PLC**——用于在没有现场相机/PLC 时验证 4 工步映射，以及把序列打印
给 PLC 侧逐位核对。

用法：

    uv run python tools/plc_dryrun.py
    uv run python tools/plc_dryrun.py --video data/sample_video/xxx.mp4
    uv run python tools/plc_dryrun.py --limit 600 --hold 0

时间口径：判定用**媒体时间**（每帧前进 1/原生帧率），与示例回放一致；
完成保持窗按墙钟计时（主程序里也是这样），所以这里的 ``now`` 用处理帧的
墙钟时刻。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from app.oilcheck.config import OilCheckConfigService  # noqa: E402
from app.oilcheck.detector import create_detector  # noqa: E402
from app.oilcheck.pipeline import StationPipeline  # noqa: E402
from app.oilcheck.plc_io import PlcEvidence, PlcFrame, PlcStepMapper  # noqa: E402
from app.oilcheck.service import _pick_plc_task  # noqa: E402
from app.oilcheck.types import FrameDetections  # noqa: E402
from app.paths import resolve_runtime_file, runtime_root  # noqa: E402


def run_video(
    path: Path,
    pipeline: StationPipeline,
    mapper: PlcStepMapper,
    *,
    limit: int = 0,
) -> list[tuple[float, PlcFrame]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频: {path}")

    native_fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    if native_fps <= 0 or native_fps > 240:
        native_fps = 15.0
    dt = 1.0 / native_fps
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    detector = pipeline.detector_name  # 只是显示用

    print(f"\n=== {path.name} ===")
    print(f"  {width}x{height} @ {native_fps:.1f}fps，共 {total} 帧，检测器={detector}")

    media_ts = 0.0
    frames_out: list[tuple[float, PlcFrame]] = []
    last_registers: tuple[int, int] | None = None
    processed = 0

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        processed += 1
        media_ts += dt

        # 检测器返回 FrameDetections；这里用媒体时间，让判定与回放一致
        # （主程序里实时相机走墙钟、示例回放走媒体时间）。
        detected = pipeline.detect(frame, media_ts)  # type: ignore[attr-defined]
        snapshot = pipeline.process(
            detections=FrameDetections(
                ts=media_ts,
                frame_size=(width, height),
                detections=detected.detections,
            )
        )

        task, superseded = _pick_plc_task(tuple(snapshot.tasks))
        if task is None:
            evidence = PlcEvidence(online=True)
        else:
            evidence = PlcEvidence(
                online=True,
                task_id=task.task_id,
                stage=task.stage,
                final_result=task.final_result,
                brush_hold_s=task.brush_hold_s,
                superseded_task_id=(None if superseded is None else superseded.task_id),
                superseded_result=(
                    None if superseded is None else superseded.final_result
                ),
                superseded_brush_hold_s=(
                    0.0 if superseded is None else superseded.brush_hold_s
                ),
            )
        frame_out = mapper.update(evidence, now=time.time())
        frames_out.append((media_ts, frame_out))
        if frame_out.registers != last_registers:
            print(
                f"    媒体 {media_ts:7.2f}s  {frame_out.describe()}"
                f"  任务={frame_out.task_id or '-'}  {frame_out.reason}"
            )
            last_registers = frame_out.registers

        if limit and processed >= limit:
            print(f"    已达到 --limit {limit} 帧，停止")
            break

    capture.release()
    return frames_out


def _summary(frames: list[tuple[float, PlcFrame]]) -> None:
    """统计各工步停留时长，确认没有来回跳。"""

    if not frames:
        print("\n没有处理任何帧")
        return
    print("\n== 工步停留统计 ==")
    buckets: dict[str, float] = {}
    for index, (ts, frame) in enumerate(frames):
        end = frames[index + 1][0] if index + 1 < len(frames) else ts
        label = f"工步{frame.step_number} {frame.step_label}"
        buckets[label] = buckets.get(label, 0.0) + max(0.0, end - ts)
    for label, seconds in buckets.items():
        print(f"    {label:<22} {seconds:7.2f}s")
    transitions = 0
    prev = None
    for _ts, frame in frames:
        if frame.registers != prev:
            transitions += 1
            prev = frame.registers
    print(f"    共 {transitions} 次寄存器变化")


def main() -> int:
    parser = argparse.ArgumentParser(description="离线验 PLC 4 工步序列")
    parser.add_argument("--video", action="append", default=[], help="指定视频路径")
    parser.add_argument("--limit", type=int, default=0, help="每段视频最多处理帧数")
    parser.add_argument("--config", default="", help="配置目录（默认仓库 config/）")
    parser.add_argument(
        "--hold",
        type=float,
        default=None,
        help="覆盖完成保持窗（秒）；默认取配置里的 plc.hold_s",
    )
    args = parser.parse_args()

    cfg_service = OilCheckConfigService(Path(args.config) if args.config else None)
    cfg = cfg_service.load()
    station = cfg.enabled_stations()[0] if cfg.stations else None
    if station is None:
        raise SystemExit(f"配置里没有启用的工位: {cfg_service.config_path}")

    print(f"配置: {cfg_service.config_path}")
    print(
        f"PLC: {cfg.plc.host}:{cfg.plc.port} unit={cfg.plc.unit_id} "
        f"字节序={cfg.plc.byte_order} 工步={cfg.plc.steps} "
        f"保持窗={cfg.plc.hold_s if args.hold is None else args.hold}s"
    )
    print("本工具不连接 PLC，只打印每一拍应该写出的字节。")

    hold = cfg.plc.hold_s if args.hold is None else args.hold
    mapper = PlcStepMapper(hold_s=hold, steps=cfg.plc.steps)
    detector = create_detector(station)
    pipeline = StationPipeline(
        station=station,
        pipeline_cfg=cfg.pipeline,
        detector_name=detector.name,
    )
    # ``StationPipeline`` 本身不做检测（服务层里由执行器调检测器再喂进来），
    # 离线工具直接把检测器挂上去复用同一份帧循环。
    pipeline.detect = detector.detect  # type: ignore[attr-defined]

    videos: list[Path] = [Path(item) for item in args.video]
    if not videos:
        for video in cfg.demo.videos:
            videos.append(
                Path(resolve_runtime_file(video.path, repo_root=runtime_root()))
            )
    if not videos:
        raise SystemExit(
            "没有可用视频：请在 config/oilcheck.json 的 demo.videos 中登记"
        )

    all_frames: list[tuple[float, PlcFrame]] = []
    for path in videos:
        if not path.exists():
            print(f"跳过不存在的视频: {path}")
            continue
        mapper.reset()
        # pipeline 也必须重置：run_video 每段都从 media_ts=0.0 起算，
        # 第 2 段第一帧就小于第 1 段末尾的 _last_frame_ts，pipeline.process
        # 在时间戳倒退分支直接 return，此后每帧都倒退——第 2 段起处理 0 帧，
        # 工具却照常打印标题，docs/12 §8 的验收会假通过。
        pipeline.reset()
        all_frames.extend(run_video(path, pipeline, mapper, limit=args.limit))

    detector.close()
    _summary(all_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
