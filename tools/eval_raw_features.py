"""用**原始检测框特征**做事件级评测：多算法按视频分折对比。

与 ``eval_events.py`` 的区别：特征来自 ``app/oilcheck/raw_features.py``，
即直接吃 YOLO 检测框 + tray/work/oiling 区域 + 最近若干帧的丝杆位置，
**不读 pipeline 已经算好的规则结论**（``seen`` / ``hs_true`` / carry 累积
帧数等）。这样模型必须自己从几何里学判别，才有独立价值。

用法::

    uv run python tools/eval_raw_features.py                  # 全部算法
    uv run python tools/eval_raw_features.py --kind pick
    uv run python tools/eval_raw_features.py --dump data/raw_events.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.oilcheck.config import OilCheckConfigService  # noqa: E402
from app.oilcheck.detector import create_detector  # noqa: E402
from app.oilcheck.pipeline import StationPipeline  # noqa: E402
from app.oilcheck.raw_features import (  # noqa: E402
    OILING_FEATURE_NAMES,
    PICK_FEATURE_NAMES,
    OilingAggregator,
    PickAggregator,
    RawFrame,
    measure_oiling,
    measure_pick,
)
from app.paths import resolve_runtime_file, runtime_root  # noqa: E402

warnings.filterwarnings("ignore")


def _make_frame(pipeline: StationPipeline, ts: float) -> RawFrame:
    """从 pipeline 的**原始状态**组一帧素材（不读任何判定结论）。"""

    width, height = pipeline._frame_size
    station = pipeline.station
    screws = [
        det for det in pipeline._last_detections
        if det.label == station.detector.screw_label
    ]
    hands = [
        (track.track_id, track.observation_box())
        for track in pipeline._hand_tracks
        if track.matched
    ]
    brushes = [
        (track.track_id, track.observation_box())
        for track in pipeline._brush_tracks
        if track.matched
    ]
    return RawFrame(
        ts=ts,
        width=width,
        height=height,
        screws=screws,
        hands=hands,
        brushes=brushes,
        tray=pipeline._tray_region_box((width, height)),
        work=station.work_roi.pixel_bbox(width, height),
        oiling=station.oiling_roi.pixel_bbox(width, height),
        screw_history=list(pipeline._screw_history),
        detector=station.detector,
    )


def collect(kind: str, limit: int = 0) -> list[dict]:
    """采集候选事件；特征是**整个动作期间**原始测量的聚合。

    主流程不改：这里自己遍历 ``pipeline._visits`` / ``_active_oiling``，每帧
    往对应的聚合器里塞一条原始测量，事件结束时取出聚合特征。取料事件的
    边界用「首次进盘 → 结算」，涂油用「这段 session 存在期间」。
    """

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
            on_event=lambda e: None, detector_name="yolo", cycle_id="RAW",
        )
        # 手: (轨迹 ID) -> 聚合器；涂油: session id -> 聚合器
        pick_acc: dict[int, PickAggregator] = {}
        pick_start: dict[int, float] = {}
        pick_entry_rods: dict[int, int] = {}
        oil_acc: dict[int, OilingAggregator] = {}

        if kind == "pick":
            original = pipeline._settle_pick

            def settle(*, visit, hand_track_id, hand_box, ts, holding, keep, _o=original):
                dwell = (
                    max(0.0, ts - visit.first_inside_ts)
                    if visit.first_inside_ts is not None
                    else visit.dwell_in_tray_s
                )
                aggregator = pick_acc.pop(hand_track_id, None)
                start = pick_start.pop(hand_track_id, ts)
                entry_rods = pick_entry_rods.pop(hand_track_id, None)
                if aggregator is not None:
                    values = aggregator.features(
                        dwell=dwell, start_tray_rods=entry_rods
                    )
                    rows.append({
                        "kind": kind, "video": video_index, "ts": round(ts, 3),
                        # 与脚本比对用「进盘—结算」区间，不是单看结算时刻
                        "label": int(any(
                            max(start, p["start_ts"]) <= min(ts, p["end_ts"])
                            for p in items
                        )),
                        **values,
                    })
                return _o(visit=visit, hand_track_id=hand_track_id,
                          hand_box=hand_box, ts=ts, holding=holding, keep=keep)

            pipeline._settle_pick = settle
        else:
            # 在**事件触发那一刻**采样，不是在段收尾时。
            #
            # 线上第一次咨询模型就是 ``cum_s`` 刚够 ``brush_hold_s`` 的那一帧
            # （``_finish_oiling``），此刻 ``cum_s≈1.0 / frames≈15``；而按段
            # 收尾采样拿到的是 ``cum_s≈5.7 / frames≈84``。两者是同一段动作的
            # 不同截面，训练与推理的分布对不上——模型在线上看到「frames=15」
            # 会当成训练里的短接触负例，把真涂油否掉（示例①③④ 共 3 件）。
            #
            # 固定在触发截面后，``cum_s`` / ``frames`` / ``samples`` 在样本上
            # 基本恒定（模型自然用不上它们），模型必须去看接触质量本身。
            original_finish = pipeline._finish_oiling

            def finish(ts, *, session=None, episode=None, _o=original_finish):
                target = session if session is not None else pipeline.active_oiling_session()
                if target is not None and id(target) in oil_acc:
                    aggregator = oil_acc.pop(id(target))
                    values = aggregator.features(
                        cum_s=target.cum_s, frames=target.frames
                    )
                    rows.append({
                        "kind": kind, "video": video_index, "ts": round(ts, 3),
                        "label": int(any(
                            o["start_ts"] - 0.6 <= ts <= o["end_ts"] + 0.6
                            for o in items
                        )),
                        **values,
                    })
                return _o(ts, session=session, episode=episode)

            pipeline._finish_oiling = finish

            original_close = pipeline._close_oiling

            def close(*, _o=original_close):
                # 没攒够 brush_hold_s 就收掉的段（短接触）：记一条负例。
                # 这些在 ``_finish_oiling`` 里不会出现，但它们正是模型要
                # 学会拒绝的那一类，不能漏。
                #
                # **已经报过的段不能再记**：涂油达成后工人往往还接着刷几秒
                # （示例① 报完 1.0s 后一直刷到 6.6s），段收尾时 ``reported_ts``
                # 已经非空。不判这一条就会把同一段动作记成「一正一负」，模型
                # 学到的信号正好相反。
                session = pipeline.active_oiling_session()
                if (
                    session is not None
                    and session.reported_ts is None
                    and id(session) in oil_acc
                ):
                    aggregator = oil_acc.pop(id(session))
                    end = session.last_ts or 0.0
                    values = aggregator.features(
                        cum_s=session.cum_s, frames=session.frames
                    )
                    rows.append({
                        "kind": kind, "video": video_index, "ts": round(end, 3),
                        # 短接触一定是负例：独立清点里没有这个动作。
                        "label": 0,
                        **values,
                    })
                else:
                    oil_acc.pop(id(session), None) if session is not None else None
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
            ts = n / fps
            pipeline.process(detections=detector.detect(bgr, ts))
            frame = _make_frame(pipeline, ts)

            if kind == "pick":
                # 逐帧往「这趟进盘的聚合器」里累积原始测量
                for hand in pipeline._hand_tracks:
                    if not hand.matched:
                        continue
                    visit = pipeline._visits.get(hand.track_id)
                    if visit is None or not visit.entered_confirmed:
                        continue
                    if visit.holding_on_entry:
                        continue
                    aggregator = pick_acc.get(hand.track_id)
                    if aggregator is None:
                        aggregator = pick_acc[hand.track_id] = PickAggregator()
                        start = (
                            visit.first_inside_ts
                            if visit.first_inside_ts is not None else ts
                        )
                        pick_start[hand.track_id] = start
                        pick_entry_rods[hand.track_id] = sum(
                            1 for det in frame.screws if frame.in_tray(det.box)
                        )
                    aggregator.add(
                        measure_pick(frame, hand.track_id, hand.observation_box())
                    )
            else:
                session = pipeline.active_oiling_session()
                if session is not None:
                    aggregator = oil_acc.get(id(session))
                    if aggregator is None:
                        aggregator = oil_acc[id(session)] = OilingAggregator()
                    hand_box = None
                    if frame.hands:
                        anchor = session.anchor
                        if anchor is None:
                            hand_box = frame.hands[0][1]
                        else:
                            hand_box = min(
                                (b for _, b in frame.hands),
                                key=lambda b: (
                                    (b.center[0] - anchor[0]) ** 2
                                    + (b.center[1] - anchor[1]) ** 2
                                ),
                            )
                    aggregator.add(measure_oiling(frame, hand_box))
        capture.release()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="原始检测框特征 · 多算法事件级评测")
    parser.add_argument("--kind", choices=("pick", "oiling", "both"), default="both")
    parser.add_argument("--model", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dump", default="")
    args = parser.parse_args()

    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from compare_models import build_models  # noqa: PLC0415

    kinds = ["pick", "oiling"] if args.kind == "both" else [args.kind]
    dumped: list[dict] = []
    for kind in kinds:
        rows = collect(kind, args.limit)
        dumped.extend(rows)
        feats = PICK_FEATURE_NAMES if kind == "pick" else OILING_FEATURE_NAMES
        x = np.asarray([[float(r.get(f, 0.0)) for f in feats] for r in rows], dtype=float)
        y = np.asarray([int(r["label"]) for r in rows], dtype=int)
        groups = np.asarray([int(r["video"]) for r in rows], dtype=int)
        print(f"\n=== {kind}: {len(rows)} 个候选事件（脚本正例 {int(y.sum())}），"
              f"{len(feats)} 维原始特征")
        models = build_models()
        if args.model:
            models = {k: v for k, v in models.items() if k == args.model}
        results = []
        for name, factory in models.items():
            exactly = 0
            tp = fp = fn = 0
            for video in sorted(set(groups.tolist())):
                train, test = groups != video, groups == video
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
                tp += int(((pred == 1) & (y[test] == 1)).sum())
                fp += int(((pred == 1) & (y[test] == 0)).sum())
                fn += int(((pred == 0) & (y[test] == 1)).sum())
                if int(pred.sum()) == int(y[test].sum()):
                    exactly += 1
            results.append((exactly, tp, fp, fn, name))
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
        print(f"\n原始特征样本已写入 {path}（{len(dumped)} 条）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
