"""动作级评测：模型逐帧打分 → 分段 → 数动作次数 → 与脚本清点对比。

逐帧准确率不是现场关心的东西。现场关心的是「这段视频里判出了几次取料、
几次涂油」，以及**起止时刻**对不对。所以这里用 ``EpisodeSegmenter`` 把
逐帧概率切成动作，再和 ``data/operations_preview/*.json`` 的独立清点比：

- 次数是否一致（脚本 23 取 / 26 涂）；
- 每次动作的时间区间是否对得上（IoU / 中心偏差）。

用法::

    uv run python tools/eval_episodes.py                 # 全部算法
    uv run python tools/eval_episodes.py --model XGBoost
    uv run python tools/eval_episodes.py --kind pick
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.oilcheck.config import OilCheckConfigService  # noqa: E402
from app.oilcheck.detector import create_detector  # noqa: E402
from app.oilcheck.pipeline import StationPipeline  # noqa: E402
from app.paths import resolve_runtime_file, runtime_root  # noqa: E402

warnings.filterwarnings("ignore")

PICK_FEATURES = [
    "dwell", "hs_n", "hs_true", "carry_frames", "seen", "inventory_n",
    "carry_inhand_frames", "carry_inhand_max", "carry_rod_move",
    "carry_hand_move", "carry_align", "rods_in_tray_delta",
]
OILING_FEATURES = [
    "cum_s", "frames", "min_distance", "parts", "held_ratio", "rod_move",
    "contact_ratio",
]


@dataclass
class Frame:
    ts: float
    feats: dict[str, float]
    video: int
    visit: tuple[int, float] | None = None   # 取料：哪一趟进盘
    session: int | None = None               # 涂油：哪一段刷涂


def collect_frames(limit: int = 0) -> list[Frame]:
    """跑一遍示例视频，记下每帧的候选特征与归属。"""

    logging.basicConfig(level=logging.ERROR)
    cfg = OilCheckConfigService().load(create_if_missing=False)
    station = cfg.enabled_stations()[0]
    detector = create_detector(station)
    frames: list[Frame] = []
    for video_index, video in enumerate(cfg.demo.videos(), 1):
        path = Path(resolve_runtime_file(video.path, repo_root=runtime_root()))
        pipeline = StationPipeline(
            station=station, pipeline_cfg=cfg.pipeline,
            on_event=lambda e: None, detector_name="yolo", cycle_id="EV",
        )
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
        n = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            n += 1
            if limit and n > limit:
                break
            ts = n / fps
            pipeline.process(detections=detector.detect(bgr, ts))
            for hand_id, visit in pipeline._visits.items():
                if not visit.entered_confirmed or visit.holding_on_entry:
                    continue
                start = visit.first_inside_ts if visit.first_inside_ts is not None else ts
                frames.append(Frame(
                    ts=ts,
                    feats=pipeline.oiling_judge_features.__self__._pick_features(
                        visit, ts=ts, dwell=ts - start
                    ),
                    video=video_index,
                    visit=(hand_id, round(start, 1)),
                ))
            session = pipeline.active_oiling_session()
            if session is not None:
                frames.append(Frame(
                    ts=ts,
                    feats=pipeline.oiling_judge_features(session),
                    video=video_index,
                    session=id(session),
                ))
        capture.release()
    return frames


def segment(scores: list[tuple[float, float]], *, on=0.6, off=0.35,
            low=0.2, bridge=0.9) -> list[tuple[float, float]]:
    """把 (ts, prob) 序列切成动作区间。"""

    episodes: list[tuple[float, float]] = []
    start = None
    last_on = None
    for ts, p in scores:
        if start is None:
            if p >= low:
                start, last_on = ts, ts
            continue
        if p >= low:
            last_on = ts
        if p < off and ts - last_on > bridge:
            if last_on - start > 0.0:
                episodes.append((start, last_on))
            start = last_on = None
    if start is not None and last_on is not None and last_on > start:
        episodes.append((start, last_on))
    return episodes


def overlap(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def match(got: list[tuple[float, float]], ref: list[tuple[float, float]],
          min_iou: float = 0.1) -> tuple[int, int, int, list[float]]:
    """贪心匹配，返回 (命中, 多判, 漏判, 命中对的 IoU 列表)。"""

    used = set()
    hits: list[float] = []
    for g in got:
        best = -1
        best_iou = min_iou
        for index, r in enumerate(ref):
            if index in used:
                continue
            iou = overlap(g[0], g[1], r[0], r[1])
            if iou >= best_iou:
                best, best_iou = index, iou
        if best >= 0:
            used.add(best)
            hits.append(best_iou)
    return len(hits), len(got) - len(hits), len(ref) - len(hits), hits


def main() -> int:
    parser = argparse.ArgumentParser(description="动作级评测（次数 + 起止时刻）")
    parser.add_argument("--model", default="", help="只跑某个算法名")
    parser.add_argument("--kind", choices=("pick", "oiling", "both"), default="both")
    parser.add_argument("--min-iou", type=float, default=0.1, help="判定同一动作的 IoU 门限")
    parser.add_argument("--limit", type=int, default=0, help="每段视频只跑前 N 帧")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from compare_models import build_models  # noqa: PLC0415

    print("采集候选帧…")
    frames = collect_frames(args.limit)
    print(f"共 {len(frames)} 帧")

    cfg = OilCheckConfigService().load(create_if_missing=False)
    preview = runtime_root() / "data" / "operations_preview"
    videos = cfg.demo.videos()
    truth = {}
    for index, video in enumerate(videos, 1):
        path = Path(resolve_runtime_file(video.path, repo_root=runtime_root()))
        data = json.loads((preview / f"{path.stem}.json").read_text(encoding="utf-8"))
        truth[index] = data

    models = build_models()
    if args.model:
        models = {k: v for k, v in models.items() if k == args.model}
        if not models:
            print(f"没有这个算法: {args.model}")
            return 1

    kinds = ["pick", "oiling"] if args.kind == "both" else [args.kind]
    for kind in kinds:
        feats = PICK_FEATURES if kind == "pick" else OILING_FEATURES
        rows = [f for f in frames if (f.visit is not None) == (kind == "pick")]
        if not rows:
            continue
        print(f"\n=== {kind}: {len(rows)} 帧")
        for name, factory in models.items():
            # 训练：用按视频分折留出该视频的方式来预测每一帧
            tp = fp = fn = 0          # 动作级
            ious: list[float] = []
            counts: list[tuple[int, int]] = []
            single_video_mode = len(models) == 1
            for video in sorted({f.video for f in rows}):
                train = np.asarray(
                    [[f.feats.get(k, 0.0) for k in feats]
                     for f in rows if f.video != video], dtype=float
                )
                train_y = np.asarray([
                    int(_label(f, kind, truth[f.video])) for f in rows if f.video != video
                ])
                if train_y.sum() < 2 or len(train_y) - train_y.sum() < 2:
                    continue
                model = factory()
                try:
                    model.fit(train, train_y)
                except Exception:  # noqa: BLE001
                    continue
                proba = model.predict_proba(
                    np.asarray([[f.feats.get(k, 0.0) for k in feats]
                                for f in rows if f.video == video], dtype=float)
                )[:, 1]
                test = [f for f in rows if f.video == video]
                scores = [(f.ts, float(p)) for f, p in zip(test, proba)]
                # 一次只切一个动作序列：把整段视频串起来即可（同一时间只有一个候选）
                got = segment(scores)
                ref = [(item["start_ts"], item["end_ts"])
                       for item in truth[video][f"{kind}s"]]
                h, extra, missed, pair_ious = match(got, ref, args.min_iou)
                tp += h
                fp += extra
                fn += missed
                ious.extend(pair_ious)
                counts.append((len(got), len(ref)))
            total_ref = tp + fn
            count_ok = sum(1 for g, r in counts if g == r)
            mean_iou = sum(ious) / len(ious) if ious else 0.0
            print(
                f"  {name:<12} 命中 {tp:>3}/{total_ref:<3} 多判 {fp:>3} 漏判 {fn:>3} "
                f"次数一致的视频 {count_ok}/{len(counts)} 平均IoU {mean_iou:.2f}"
            )
    return 0


def _label(frame: Frame, kind: str, video_truth: dict) -> int:
    """这一帧是否落在脚本清点的动作区间内。"""

    key = "picks" if kind == "pick" else "oilings"
    pad = 0.3 if kind == "pick" else 0.6
    min_cum = 0.5
    if kind == "oiling" and frame.feats.get("cum_s", 0.0) < min_cum:
        return 0
    return int(any(
        item["start_ts"] - pad <= frame.ts <= item["end_ts"] + pad
        for item in video_truth[key]
    ))


if __name__ == "__main__":
    raise SystemExit(main())
