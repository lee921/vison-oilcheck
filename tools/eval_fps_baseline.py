"""按帧率把示例视频重采样/插帧后，用**线上同款判定**逐视频核对基准次数。

「程序能不能检测出」有两层，本脚本量的是第二层（线上判定）：

1. 独立清点（``tools/detect_operations.py``）：视频里实际发生了几次取料/
   几次涂油，是基准本身；
2. 线上判定（``StationPipeline`` + ``config/judge_*.json`` 二级判定）：
   监控页/事件库里会出现几次。

用法::

    uv run python tools/eval_fps_baseline.py                          # 20fps 目录对比基准
    uv run python tools/eval_fps_baseline.py --dir data/sample_video   # 原始 15fps
    uv run python tools/eval_fps_baseline.py --dir data/sample_video_20fps --limit 200

基准取 ``data/operations_preview/<视频名>.json``（15fps 原始录像上的人工核对
清点），按**文件名**配对：插帧后的录像沿用同一个文件名。

判定参数一律用当前 ``config/oilcheck.json``（``pipeline.max_fps`` 等），
与监控页完全一致。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from app.oilcheck.config import OilCheckConfigService  # noqa: E402
from app.oilcheck.detector import create_detector  # noqa: E402
from app.oilcheck.pipeline import StationPipeline  # noqa: E402
from app.oilcheck.service import _load_judge_file  # noqa: E402

DEMO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov"}


def load_baseline(path: Path) -> dict[str, dict[str, int]]:
    """基准：``<stem> -> {picks, oilings}``（来自独立清点脚本）。"""

    table: dict[str, dict[str, int]] = {}
    for item in sorted(path.glob("*.json")):
        if item.name == "summary.json":
            continue
        try:
            data = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if "pick_count" not in data or "oiling_count" not in data:
            continue
        table[Path(data.get("path") or item.stem).stem] = {
            "picks": int(data["pick_count"]),
            "oilings": int(data["oiling_count"]),
        }
    return table


def run_video(
    path: Path,
    *,
    pipeline: StationPipeline,
    detector,
    limit: int = 0,
) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    if fps <= 0 or fps > 240:
        fps = 15.0

    events: list = []
    pipeline._on_event = events.append  # noqa: SLF001
    n = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            n += 1
            if limit and n > limit:
                break
            pipeline.process(detections=detector.detect(bgr, n / fps))
    finally:
        capture.release()

    def _kind(event_type: str) -> list:
        return [e for e in events if str(getattr(e.event_type, "value", e.event_type)) == event_type]

    picks = _kind("PICK")
    oils = _kind("MANUAL_OILING")
    missing = _kind("OILING_MISSING")
    return {
        "video": path.name,
        "fps": round(fps, 3),
        "frames": n,
        "duration_s": round(n / fps, 2),
        "pick_count": len(picks),
        "oiling_count": len(oils),
        "missing_count": len(missing),
        "pick_ts": [round(float(e.timestamp_start), 2) for e in picks],
        "oiling_ts": [round(float(e.timestamp_start), 2) for e in oils],
        "missing_ts": [round(float(e.timestamp_start), 2) for e in missing],
        "pick_sources": [
            str((e.evidence or {}).get("pick_source", "")) for e in picks
        ],
        "pick_judge": [
            None if (e.evidence or {}).get("judge") is None
            else round(float(e.evidence["judge"].get("probability", 0.0)), 3)
            for e in picks
        ],
        "oiling_judge": [
            None if (e.evidence or {}).get("judge") is None
            else round(float(e.evidence["judge"].get("probability", 0.0)), 3)
            for e in oils
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="按帧率核对线上判定的次数基准")
    parser.add_argument(
        "--dir", default="data/sample_video_20fps", help="待测视频目录"
    )
    parser.add_argument(
        "--baseline",
        default="data/operations_preview",
        help="基准清点目录（detect_operations.py 的产物，15fps 原始录像）",
    )
    parser.add_argument("--limit", type=int, default=0, help="每段只跑前 N 帧")
    parser.add_argument("--video", default="", help="只跑文件名包含该片段的视频")
    parser.add_argument("--out", default="", help="把逐视频结果写成 JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)

    cfg_service = OilCheckConfigService()
    cfg = cfg_service.load(create_if_missing=False)
    station = cfg.enabled_stations()[0] if cfg.stations else None
    if station is None:
        raise SystemExit(f"配置里没有启用的工位: {cfg_service.config_path}")

    baseline_dir = REPO_ROOT / args.baseline
    baseline = load_baseline(baseline_dir)
    if not baseline:
        raise SystemExit(f"基准目录里没有清点结果: {baseline_dir}")

    video_dir = Path(args.dir)
    if not video_dir.is_absolute():
        video_dir = REPO_ROOT / video_dir
    videos = [
        p
        for p in sorted(video_dir.iterdir(), key=lambda p: p.name)
        if p.suffix.lower() in DEMO_SUFFIXES and p.is_file()
    ]
    if args.video:
        videos = [p for p in videos if args.video in p.name]
    if not videos:
        raise SystemExit(f"目录里没有视频: {video_dir}")

    # 二级判定：与监控进程同一条加载路径（缺文件则不启用）。
    config_dir = cfg_service.config_dir
    pick_judge = _load_judge_file(config_dir / "judge_pick.json")
    oiling_judge = _load_judge_file(config_dir / "judge_oiling.json")

    detector = create_detector(station)
    print(
        f"配置 {cfg_service.config_path}\n"
        f"判定 max_fps={cfg.pipeline.max_fps}，二级判定 "
        f"pick={'有' if pick_judge is not None else '无'} "
        f"oiling={'有' if oiling_judge is not None else '无'}\n"
        f"视频目录 {video_dir}；基准 {baseline_dir}\n"
    )

    results: list[dict] = []
    for path in videos:
        pipeline = StationPipeline(
            station=station,
            pipeline_cfg=cfg.pipeline,
            on_event=None,
            detector_name=getattr(detector, "name", "yolo"),
            cycle_id="FPS",
        )
        pipeline.attach_judges(pick_judge, oiling_judge)
        row = run_video(path, pipeline=pipeline, detector=detector, limit=args.limit)
        ref = baseline.get(path.stem)
        row["baseline"] = ref
        if ref is not None:
            row["pick_ok"] = row["pick_count"] == ref["picks"]
            row["oiling_ok"] = row["oiling_count"] == ref["oilings"]
            row["ok"] = bool(row["pick_ok"] and row["oiling_ok"])
            row["pick_diff"] = row["pick_count"] - ref["picks"]
            row["oiling_diff"] = row["oiling_count"] - ref["oilings"]
        results.append(row)
        print(
            f"  {path.stem}: {row['fps']:g}fps {row['frames']} 帧 "
            f"({row['duration_s']:g}s)  取料 {row['pick_count']} 涂油 "
            f"{row['oiling_count']} 漏涂NG {row['missing_count']}"
            + (
                f"  基准 {ref['picks']}/{ref['oilings']} "
                f"{'✓' if row['ok'] else '✗'}"
                if ref is not None
                else "  基准（无）"
            ),
            flush=True,
        )
        if row.get("pick_diff") or row.get("oiling_diff"):
            print(
                f"      取料时刻 {row['pick_ts']}（来源 {row['pick_sources']}）\n"
                f"      涂油时刻 {row['oiling_ts']}\n"
                f"      漏涂NG时刻 {row['missing_ts']}"
            )

    try:
        detector.close()
    except Exception:  # noqa: BLE001
        pass

    paired = [r for r in results if r.get("baseline") is not None]
    ok_rows = [r for r in paired if r["ok"]]
    print("\n" + "=" * 72)
    print(f"{'视频':<40}{'取料':>8}{'涂油':>8}   判定")
    print("-" * 72)
    for row in results:
        ref = row["baseline"]
        ref_text = f"{ref['picks']}/{ref['oilings']}" if ref else "—"
        mark = "✓" if row.get("ok") else ("✗" if ref else "·")
        print(
            f"{row['video']:<40}{row['pick_count']:>4}({ref_text:>5})"
            f"{row['oiling_count']:>6}   {mark}"
        )
    print("-" * 72)
    print(
        f"逐视频一致 {len(ok_rows)}/{len(paired)}；"
        f"合计 取料 {sum(r['pick_count'] for r in paired)}/"
        f"{sum(r['baseline']['picks'] for r in paired)}，"
        f"涂油 {sum(r['oiling_count'] for r in paired)}/"
        f"{sum(r['baseline']['oilings'] for r in paired)}"
    )

    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"逐视频结果: {out}")
    return 0 if len(ok_rows) == len(paired) else 2


if __name__ == "__main__":
    raise SystemExit(main())
