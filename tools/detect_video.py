"""离线检测示例视频，输出带框图与逐帧 JSONL。

用法：

    uv run python tools/detect_video.py                 # 跑配置里的全部示例视频
    uv run python tools/detect_video.py --limit 120     # 只跑前 120 帧
    uv run python tools/detect_video.py --video path.mp4

产物写到 ``data/detect_preview/``：``<名字>.mp4`` 带框，``<名字>.jsonl``
逐帧检测记录（类别、置信度、框坐标），终端打印每类总数与平均置信度。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from app.oilcheck.config import DetectorConfig, OilCheckConfigService  # noqa: E402
from app.oilcheck.detector import YoloDetector  # noqa: E402
from app.paths import resolve_runtime_file, runtime_root  # noqa: E402

COLORS = {
    "hand": (10, 159, 255),
    "brush": (95, 55, 255),
    "oil_station": (10, 214, 255),
    "screw": (75, 215, 50),
}


def _color(label: str) -> tuple[int, int, int]:
    if label in COLORS:
        return COLORS[label]
    index = sum(ord(ch) for ch in label) % 6
    return [(255, 210, 100), (250, 200, 90), (255, 45, 85), (90, 200, 250),
            (255, 154, 60), (210, 90, 191)][index]


def run_video(
    path: Path,
    detector: YoloDetector,
    out_dir: Path,
    *,
    limit: int = 0,
) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频: {path}")

    native_fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    if native_fps <= 0 or native_fps > 240:
        native_fps = 15.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stem = path.stem

    out_video = out_dir / f"{stem}.mp4"
    writer = cv2.VideoWriter(
        str(out_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(native_fps),
        (width, height),
    )
    jsonl_path = out_dir / f"{stem}.jsonl"
    jsonl = jsonl_path.open("w", encoding="utf-8")

    counts: Counter[str] = Counter()
    conf_sum: dict[str, float] = defaultdict(float)
    processed = 0
    elapsed_total = 0.0
    frames_without = 0

    print(f"\n=== {path.name} ===")
    print(f"  {width}x{height} @ {native_fps:.1f}fps, 共 {total} 帧, imgsz={detector.imgsz}")

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        processed += 1

        started = time.perf_counter()
        detections = detector.detect(frame, time.time())
        elapsed_total += time.perf_counter() - started

        if not detections.detections:
            frames_without += 1

        record = {
            "frame": processed,
            "ts": round(detections.ts, 3),
            "size": list(detections.frame_size),
            "detections": [
                {
                    "label": det.label,
                    "conf": round(det.confidence, 4),
                    "box": [
                        round(det.box.x1, 1),
                        round(det.box.y1, 1),
                        round(det.box.x2, 1),
                        round(det.box.y2, 1),
                    ],
                }
                for det in detections.detections
            ],
        }
        jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")

        for det in detections.detections:
            counts[det.label] += 1
            conf_sum[det.label] += det.confidence
            x1, y1, x2, y2 = det.box.as_int_tuple()
            color = _color(det.label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            text = f"{det.label} {det.confidence:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                frame, (x1, max(0, y1 - th - baseline - 6)), (x1 + tw + 6, y1), color, -1
            )
            cv2.putText(
                frame,
                text,
                (x1 + 3, max(th + 2, y1 - baseline - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        if limit and processed >= limit:
            break
        if processed % 50 == 0:
            print(f"  ... {processed}/{total or '?'} 帧", flush=True)

    capture.release()
    writer.release()
    jsonl.close()

    avg_ms = (elapsed_total / processed * 1000.0) if processed else 0.0
    print(f"  已处理 {processed} 帧，平均 {avg_ms:.1f} ms/帧（{1000/avg_ms:.1f} fps）")
    print(f"  无检测帧: {frames_without}/{processed}")
    print("  类别统计:")
    for label, count in counts.most_common():
        avg_conf = conf_sum[label] / count
        print(f"    {label:<12} {count:>6} 次  平均置信度 {avg_conf:.3f}")
    print(f"  带框视频: {out_video}")
    print(f"  逐帧记录: {jsonl_path}")

    return {
        "video": path.name,
        "frames": processed,
        "avg_ms": round(avg_ms, 2),
        "frames_without": frames_without,
        "counts": dict(counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="示例视频离线检测")
    parser.add_argument("--video", action="append", default=[], help="指定视频路径")
    parser.add_argument("--limit", type=int, default=0, help="每段视频最多处理帧数")
    parser.add_argument("--out", default="", help="输出目录")
    parser.add_argument(
        "--config", default="", help="配置目录（默认仓库 config/）"
    )
    args = parser.parse_args()

    cfg_service = OilCheckConfigService(Path(args.config) if args.config else None)
    cfg = cfg_service.load()
    station = cfg.enabled_stations()[0] if cfg.stations else None
    if station is None:
        raise SystemExit(f"配置里没有启用的工位: {cfg_service.config_path}")
    if station.detector.kind != "yolo":
        raise SystemExit(
            f"工位 {station.station_id} 的 detector.kind={station.detector.kind}，"
            f"需要设为 yolo 才能离线检测。配置: {cfg_service.config_path}"
        )

    print(f"配置: {cfg_service.config_path}")
    detector_cfg: DetectorConfig = station.detector
    detector = YoloDetector(station, detector_cfg)

    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "detect_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos: list[Path] = []
    for item in args.video:
        videos.append(Path(item))
    if not videos:
        for video in cfg.demo.videos:
            videos.append(Path(resolve_runtime_file(video.path, repo_root=runtime_root())))
    if not videos:
        raise SystemExit("没有可用视频：请在 config/oilcheck.json 的 demo.videos 中登记")

    results = []
    for path in videos:
        if not path.exists():
            print(f"跳过不存在的视频: {path}")
            continue
        results.append(run_video(path, detector, out_dir, limit=args.limit))

    detector.close()

    summary = out_dir / "summary.json"
    summary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n汇总已写入: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
