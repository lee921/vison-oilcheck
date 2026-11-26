"""事件级评测：每个候选动作过一次模型，数出次数，与脚本清点对比。

现场的动作计数是**事件级**的：一次「空手进盘→带杆离盘」是一个候选事件，
一次「刷头贴着手持杆累计」是一个候选事件。判对/判错就发生在这一个个
事件上，而不是逐帧。

所以这里为每个候选事件提取**结算时刻**的证据（正是线上做决定的那一刻），
交给模型判「是/否」，再数总次数与脚本（23 取 / 26 涂）比较。

用法::

    uv run python tools/eval_events.py                    # 全部算法对比
    uv run python tools/eval_events.py --model XGBoost
    uv run python tools/eval_events.py --dump data/events.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
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


def collect(kind: str, limit: int = 0) -> list[dict]:
    """采集候选事件（每个结算的进盘过程 / 每个结束的刷涂段）。"""

    logging.basicConfig(level=logging.ERROR)
    cfg = OilCheckConfigService().load(create_if_missing=False)
    station = cfg.enabled_stations()[0]
    detector = create_detector(station)
    preview = runtime_root() / "data" / "operations_preview"
    rows: list[dict] = []

    for video_index, video in enumerate(cfg.demo.videos(), 1):
        path = Path(resolve_runtime_file(video.path, repo_root=runtime_root()))
        truth = json.loads((preview / f"{path.stem}.json").read_text(encoding="utf-8"))
        items = truth["picks"] if kind == "pick" else truth["oilings"]
        pipeline = StationPipeline(
            station=station, pipeline_cfg=cfg.pipeline,
            on_event=lambda e: None, detector_name="yolo", cycle_id="EV",
        )
        captured: list[dict] = []

        if kind == "pick":
            original = pipeline._settle_pick

            def settle(*, visit, hand_track_id, hand_box, ts, holding, keep, _o=original):
                dwell = (
                    max(0.0, ts - visit.first_inside_ts)
                    if visit.first_inside_ts is not None
                    else visit.dwell_in_tray_s
                )
                values = pipeline.oiling_judge_features.__self__._pick_features(
                    visit, ts=ts, dwell=dwell
                )
                # 用「进盘—离盘」区间跟脚本比对，而不是结算时刻：结算要等
                # 杆出盘与采样窗口攒满，可能晚一两秒（示例⑤ 第 2 件晚 2.2s），
                # 按时刻配对会把真取料标成负例。
                start = float(values.get("_visit_start", ts))
                end = float(values.get("_visit_exit", ts))
                rows.append({
                    "kind": kind,
                    "video": video_index,
                    "ts": round(ts, 3),
                    "hand": hand_track_id,
                    "label": int(any(
                        max(start, p["start_ts"]) <= min(end, p["end_ts"])
                        for p in items
                    )),
                    **values,
                })
                return _o(visit=visit, hand_track_id=hand_track_id,
                          hand_box=hand_box, ts=ts, holding=holding, keep=keep)

            pipeline._settle_pick = settle
        else:
            original_close = pipeline._close_oiling

            def close(*, _o=original_close):
                session = pipeline.active_oiling_session()
                if session is not None:
                    values = pipeline.oiling_judge_features(session)
                    end = session.last_ts or 0.0
                    rows.append({
                        "kind": kind,
                        "video": video_index,
                        "ts": round(end, 3),
                        "label": int(
                            session.cum_s >= 0.5
                            and any(
                                o["start_ts"] - 0.6 <= end <= o["end_ts"] + 0.6
                                for o in items
                            )
                        ),
                        **values,
                    })
                return _o()

            pipeline._close_oiling = close

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
            pipeline.process(detections=detector.detect(bgr, n / fps))
        capture.release()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="事件级评测（次数对齐脚本）")
    parser.add_argument("--kind", choices=("pick", "oiling", "both"), default="both")
    parser.add_argument("--model", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dump", default="", help="把采集到的事件写成 JSONL")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from compare_models import build_models  # noqa: PLC0415

    kinds = ["pick", "oiling"] if args.kind == "both" else [args.kind]
    dumped: list[dict] = []
    for kind in kinds:
        rows = collect(kind, args.limit)
        dumped.extend(rows)
        feats = PICK_FEATURES if kind == "pick" else OILING_FEATURES
        x = np.asarray([[float(r.get(f, 0.0)) for f in feats] for r in rows], dtype=float)
        y = np.asarray([int(r["label"]) for r in rows], dtype=int)
        groups = np.asarray([int(r["video"]) for r in rows], dtype=int)
        print(f"\n=== {kind}: {len(rows)} 个候选事件（脚本正例 {int(y.sum())}）")
        models = build_models()
        if args.model:
            models = {k: v for k, v in models.items() if k == args.model}
        results = []
        for name, factory in models.items():
            exactly = 0
            total_tp = total_fp = total_fn = 0
            for video in sorted(set(groups.tolist())):
                train = groups != video
                test = groups == video
                if not test.any():
                    continue
                y_train = y[train]
                if y_train.sum() < 2 or len(y_train) - y_train.sum() < 2:
                    continue
                model = factory()
                try:
                    model.fit(x[train], y_train)
                    pred = model.predict(x[test])
                except Exception:  # noqa: BLE001
                    continue
                hits = int(((pred == 1) & (y[test] == 1)).sum())
                extras = int(((pred == 1) & (y[test] == 0)).sum())
                missed = int(((pred == 0) & (y[test] == 1)).sum())
                total_tp += hits
                total_fp += extras
                total_fn += missed
                if int(pred.sum()) == int(y[test].sum()):
                    exactly += 1
            results.append((exactly, total_tp, total_fp, total_fn, name))
        results.sort(key=lambda item: (-item[0], item[2] + item[3]))
        ref_total = int(y.sum())
        for exactly, tp, fp, fn, name in results:
            print(
                f"  {name:<12} 命中 {tp:>3}/{ref_total:<3} 多判 {fp:>3} 漏判 {fn:>3} "
                f"次数完全一致的视频 {exactly}/{len(set(groups.tolist()))}"
            )

    if args.dump:
        path = Path(args.dump)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in dumped),
            encoding="utf-8",
        )
        print(f"\n事件样本已写入 {path}（{len(dumped)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
