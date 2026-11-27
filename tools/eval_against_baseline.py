"""逐视频把线上判定与独立清点对账：列出漏检、误检、时刻偏差。

用法::

    uv run python tools/eval_against_baseline.py                       # 全部示例视频（原始帧率）
    uv run python tools/eval_against_baseline.py --dir data/sample_video_20fps
    uv run python tools/eval_against_baseline.py --video 122925 --verbose

口径：

- **基准**：``data/operations_preview/<视频名>.json``（``detect_operations.py``
  的独立清点，两个工序各判各的，不建作业周期）；
- **判定**：线上完全相同的 ``StationPipeline``（含 ``config/judge_*.json``
  二级判定）；
- **配对**：判定的动作区间与基准区间在 ``--tol``（默认 2.0s）内相接即配对。
  结算时刻天然晚于动作区间（取料要等杆出盘、涂油要攒够接触），按严格重叠
  比会把真动作标成「漏报 + 误报」。

与 ``eval_fps_baseline.py`` 的区别：那个只比次数，这个把每一处漏检/误检
的时刻都打出来，并区分取料来源（手部判据 / 丝杆判据）。
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

SUFFIXES = {".mp4", ".mkv", ".avi", ".mov"}


def _match(got: list[tuple[str, float, float]], ref: list[tuple[float, float]], tol: float):
    """贪心配对，返回 (配对数, 未配对 got, 未配对 ref)。

    配对只看**时间距离**：判定的结算区间与基准的动作区间是两种口径
    （判定要等杆出盘、攒够接触才结算，区间常常整体晚 0.2~1.5s、也可能
    被分段器截短），要求重叠会让同一件事被判成「一多一漏」。于是取
    「区间**重叠**，或区间间隙 ≤ tol」，再按间隙从小到大**全局**贪心配对。

    为什么不单看起点/中点：两边的区间是不同口径——判定事件的
    ``timestamp_end`` 是「判定成立那一刻」（涂油一攒够 ``brush_hold_s``
    就发，工人还接着刷，所以判定区间往往比脚本短一大截：105708 判定
    18.00~18.93、脚本 18.0~23.67）；而丝杆离盘判据的 ``start_ts`` 取的是
    「杆在盘里开始移动」的时刻，可能比脚本的动作区间早很多（091519 判定
    0.07~4.73、脚本 3.87~4.53）。区间重叠才是「同一件事」最稳的判据，
    tol 只用来兜住分段器把区间截短到碰不上的个例。

    必须全局贪心：按发生顺序逐个配时，先出现的那个会把后面那个本该配对的
    目标抢走，同样制造假的「一多一漏」。
    """

    def gap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(b0 - a1, a0 - b1, 0.0)

    pairs: list[tuple[float, int, int]] = []
    for gi, item in enumerate(got):
        for ri, (a, b) in enumerate(ref):
            g = gap(item[1], item[2], a, b)
            if g <= tol:
                pairs.append((g, gi, ri))
    pairs.sort(key=lambda p: p[0])
    used_got: set[int] = set()
    used_ref: set[int] = set()
    for _g, gi, ri in pairs:
        if gi in used_got or ri in used_ref:
            continue
        used_got.add(gi)
        used_ref.add(ri)
    unmatched_got = [item for i, item in enumerate(got) if i not in used_got]
    unmatched_ref = [r for i, r in enumerate(ref) if i not in used_ref]
    return len(used_ref), unmatched_got, unmatched_ref


def run_video(path: Path, *, pipeline: StationPipeline, detector, limit: int = 0):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    if fps <= 0 or fps > 240:
        fps = 15.0
    events: list = []
    pipeline._on_event = events.append  # noqa: SLF001
    confirmed: list = []
    original = pipeline._confirm_pick

    def confirm(*, ts, hand_track_id, screw, dwell_s, judge=None, **kw):
        confirmed.append(
            {
                "ts": round(float(ts), 2),
                "source": kw.get("source", "hand_exit"),
                "probability": None if judge is None else round(float(judge.probability), 3),
                "dwell": round(float(dwell_s), 2),
            }
        )
        return original(
            ts=ts, hand_track_id=hand_track_id, screw=screw,
            dwell_s=dwell_s, judge=judge, **kw,
        )

    pipeline._confirm_pick = confirm  # noqa: SLF001
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

    def pick_events(kind: str):
        return [
            (str(e.event_type.value), round(float(e.timestamp_start), 2), round(float(e.timestamp_end), 2))
            for e in events
            if str(getattr(e.event_type, "value", e.event_type)) == kind
        ]

    return {
        "fps": round(fps, 3),
        "frames": n,
        "duration_s": round(n / fps, 2),
        "picks": pick_events("PICK"),
        "oilings": pick_events("MANUAL_OILING"),
        "missing": pick_events("OILING_MISSING"),
        "confirmed": confirmed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="线上判定 vs 独立清点：逐条对账")
    parser.add_argument("--dir", default="data/sample_video", help="视频目录")
    parser.add_argument("--baseline", default="data/operations_preview", help="清点目录")
    parser.add_argument("--video", default="", help="只跑文件名包含该片段的视频")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--tol", type=float, default=2.0, help="配对容差（秒）")
    parser.add_argument("--verbose", action="store_true", help="打印事件逐步明细")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    service = OilCheckConfigService()
    cfg = service.load(create_if_missing=False)
    station = cfg.enabled_stations()[0] if cfg.stations else None
    if station is None:
        raise SystemExit(f"配置里没有启用的工位: {service.config_path}")
    pick_judge = _load_judge_file(service.config_dir / "judge_pick.json")
    oiling_judge = _load_judge_file(service.config_dir / "judge_oiling.json")
    detector = create_detector(station)

    video_dir = Path(args.dir)
    if not video_dir.is_absolute():
        video_dir = REPO_ROOT / video_dir
    baseline_dir = REPO_ROOT / args.baseline
    videos = [
        p for p in sorted(video_dir.iterdir(), key=lambda q: q.name)
        if p.suffix.lower() in SUFFIXES and p.is_file()
    ]
    if args.video:
        videos = [p for p in videos if args.video in p.name]

    totals = {"ok": 0, "bad": 0}
    for path in videos:
        truth_file = baseline_dir / f"{path.stem}.json"
        if not truth_file.is_file():
            print(f"{path.name}: 没有基准（{truth_file.name}），跳过")
            continue
        truth = json.loads(truth_file.read_text(encoding="utf-8"))
        ref_picks = [(float(x["start_ts"]), float(x["end_ts"])) for x in truth["picks"]]
        ref_oils = [(float(x["start_ts"]), float(x["end_ts"])) for x in truth["oilings"]]

        pipeline = StationPipeline(
            station=station, pipeline_cfg=cfg.pipeline, on_event=None,
            detector_name=getattr(detector, "name", "yolo"), cycle_id="REC",
        )
        pipeline.attach_judges(pick_judge, oiling_judge)
        result = run_video(path, pipeline=pipeline, detector=detector, limit=args.limit)

        hit_p, extra_p, miss_p = _match(result["picks"], ref_picks, args.tol)
        hit_o, extra_o, miss_o = _match(result["oilings"], ref_oils, args.tol)
        ok = not extra_p and not miss_p and not extra_o and not miss_o
        totals["ok" if ok else "bad"] += 1

        print(
            f"\n=== {path.name}  {result['fps']:g}fps {result['frames']} 帧 "
            f"({result['duration_s']:g}s)\n"
            f"  取料  基准 {len(ref_picks)}  判定 {len(result['picks'])}  "
            f"配对 {hit_p}  多判 {len(extra_p)}  漏判 {len(miss_p)}\n"
            f"  涂油  基准 {len(ref_oils)}  判定 {len(result['oilings'])}  "
            f"配对 {hit_o}  多判 {len(extra_o)}  漏判 {len(miss_o)}"
            + ("" if ok else "   ✗")
        )
        for kind, got, ref in (("取料", extra_p, miss_p), ("涂油", extra_o, miss_o)):
            for item in got:
                print(f"    {kind} 多判   {item[1]:7.2f}s → {item[2]:7.2f}s")
            for a, b in ref:
                print(f"    {kind} 漏判   基准 {a:7.2f}s → {b:7.2f}s")
        if args.verbose:
            print("    确认的取料（含被判重丢弃前）:")
            for item in result["confirmed"]:
                print(f"      {item}")
            print("    漏涂 NG:", result["missing"])

    print(f"\n逐视频全对 {totals['ok']}/{totals['ok'] + totals['bad']}")
    return 0 if totals["bad"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
