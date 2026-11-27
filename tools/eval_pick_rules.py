"""逐规则评测取料判定：每条规则单独跑，与脚本清点（data/operations_preview）比对。

用法::

    uv run python tools/eval_pick_rules.py                # 两条规则 + 合并
    uv run python tools/eval_pick_rules.py --rule hand    # 只看手部规则
    uv run python tools/eval_pick_rules.py --rule screw   # 只看丝杆规则

口径：

- **基准**：``data/operations_preview/<视频>.json`` 的 ``picks``（独立清点
  脚本产出，合计 23 次）；
- **命中**：判定出的取料与某条基准的 ``[start_ts, end_ts]`` 在
  ``MATCH_TOLERANCE_S``（1.5s）内相接。两条判据的**结算时刻**天然比基准的
  动作区间晚（手部判据要等手离盘后攒采样、丝杆判据要等杆出盘并确认），
  按严格重叠比会把同一件料误判成「漏报 + 误报」；
- **漏报**：基准里没有对应判定的那几次；
- **误报**：判定出的取料对不上任何基准。

丝杆规则的漏报里有一部分是**规则定义内判不到**的：现场有「把杆取出来、
在盘区里刷完又放回原处」的动作，杆心全程没有离开手工 ``tray_roi``。
那样的件落在基准里，却不可能被「杆移到盘区外」这条规则判出。终端会把
这类漏报标成 ``(杆未出盘)``——诊断方式见 ``tools/diag_screw_exit.py``。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass, replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from app.oilcheck.config import OilCheckConfigService  # noqa: E402
from app.oilcheck.detector import create_detector  # noqa: E402
from app.oilcheck.pipeline import StationPipeline, _exit_depth  # noqa: E402
from app.oilcheck.types import EventType  # noqa: E402
from app.paths import resolve_runtime_file, runtime_root  # noqa: E402

warnings.filterwarnings("ignore")

PREVIEW_DIR = REPO_ROOT / "data" / "operations_preview"

# 判定与基准配对的时间容差（秒）。见模块文档「命中」一条。
MATCH_TOLERANCE_S = 1.5

# 「杆确实出盘过」的判定余量，与 ``screw_exit_pick_margin`` 同量级。
_EXIT_MARGIN = 0.010


@dataclass
class VideoResult:
    name: str
    truth: list[tuple[float, float]]
    found: list[tuple[float, float, str]]
    # 基准每次取料「杆有没有真的出过盘」（按检测框判定，与规则无关）。
    leave_tray: list[bool]

    @staticmethod
    def _match(
        truth: tuple[float, float], found: tuple[float, float, str]
    ) -> bool:
        start, end = truth
        other, other_end, _src = found
        return other <= end + MATCH_TOLERANCE_S and start <= other_end + MATCH_TOLERANCE_S

    @property
    def hits(self) -> int:
        return sum(
            1 for start, end in self.truth
            if any(self._match((start, end), item) for item in self.found)
        )

    @property
    def misses(self) -> list[tuple[float, float]]:
        return [
            (start, end) for start, end in self.truth
            if not any(self._match((start, end), item) for item in self.found)
        ]

    def missed_inside_only(self, rule: str) -> int:
        """漏报里「杆从未出盘」的件数（丝杆规则定义内判不到的那些）。"""

        if rule != "screw":
            return 0
        count = 0
        for (start, end), left in zip(self.truth, self.leave_tray, strict=False):
            if left:
                continue
            if not any(self._match((start, end), item) for item in self.found):
                count += 1
        return count

    @property
    def false_positives(self) -> list[tuple[float, float, str]]:
        return [
            item for item in self.found
            if not any(self._match(truth, item) for truth in self.truth)
        ]


def _truth_for(path: Path) -> list[tuple[float, float]]:
    meta = PREVIEW_DIR / f"{path.stem}.json"
    if not meta.exists():
        return []
    data = json.loads(meta.read_text(encoding="utf-8"))
    return [(float(p["start_ts"]), float(p["end_ts"])) for p in data.get("picks", [])]


def _leave_tray_flags(
    path: Path, truth: list[tuple[float, float]], detector, station
) -> list[bool]:
    """基准每次取料期间，「动作前在盘区里的那几根杆」有没有真的出盘。

    这里自己跑一个极简的最近邻追踪（只跟细长杆），**不复用**管线里的判据
    逻辑——目的是独立回答「这次取料，杆到底有没有离开盘区」，从而把「丝杆
    规则定义内判不到」的漏报与真正的 bug 分开：

    - 现场有「把杆取出来、在盘区里刷完又放回」的动作，杆心全程没离开手工
      ``tray_roi``，按「移到盘区外才算取料」的定义永远判不到；
    - 其余漏报就属于规则本身要修的问题。
    """

    tray = station.tray_roi.pixel_bbox(2560, 1440)
    diag = (2560**2 + 1440**2) ** 0.5
    rod_boxes = max(1, int(station.work_roi.pixel_bbox(2560, 1440).width))

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    tracks: list[dict] = []
    history: list[tuple[float, list[tuple[int, tuple[float, float], bool]]]] = []
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = index / fps
            index += 1
            dets = [
                det
                for det in detector.detect(frame, ts).detections
                if det.label == "screw"
                and det.box.width / max(1e-6, det.box.height) >= 2.0
                and max(det.box.width, det.box.height) > 100
            ]
            pairs = sorted(
                (
                    (
                        math.hypot(
                            det.box.center[0] / 2560 - item["c"][0] / 2560,
                            det.box.center[1] / 1440 - item["c"][1] / 1440,
                        ),
                        item["id"],
                        det_index,
                    )
                    for det_index, det in enumerate(dets)
                    for item in tracks
                ),
                key=lambda pair: pair[0],
            )
            used_tracks: set[int] = set()
            used_dets: set[int] = set()
            for distance, track_id, det_index in pairs:
                if distance > 0.05 or track_id in used_tracks or det_index in used_dets:
                    continue
                used_tracks.add(track_id)
                used_dets.add(det_index)
                item = next(t for t in tracks if t["id"] == track_id)
                item["c"] = dets[det_index].box.center
                item["ts"] = ts
                inside = (
                    tray.x1 <= item["c"][0] <= tray.x2
                    and tray.y1 <= item["c"][1] <= tray.y2
                )
                item["ever_outside"] = item["ever_outside"] or not inside
                item["min_depth"] = min(
                    item["min_depth"], _exit_depth(dets[det_index].box, tray) / diag
                )
            for det_index, det in enumerate(dets):
                if det_index in used_dets:
                    continue
                inside = (
                    tray.x1 <= det.box.center[0] <= tray.x2
                    and tray.y1 <= det.box.center[1] <= tray.y2
                )
                tracks.append(
                    {
                        "id": len(tracks) + 1_000_000,
                        "c": det.box.center,
                        "ts": ts,
                        "ever_outside": not inside,
                        "min_depth": _exit_depth(det.box, tray) / diag,
                    }
                )
            history.append(
                (
                    ts,
                    [
                        (item["id"], item["c"], item["ever_outside"], item["min_depth"])
                        for item in tracks
                    ],
                )
            )
    finally:
        cap.release()
    del rod_boxes

    flags: list[bool] = []
    for start, end in truth:
        before = [
            item
            for ts, items in history
            if start - 1.2 <= ts < start
            for item in items
            if tray.x1 <= item[1][0] <= tray.x2 and tray.y1 <= item[1][1] <= tray.y2
        ]
        ids = {item[0] for item in before}
        left = False
        for ts, items in history:
            if not (start - 0.2 <= ts <= end + 1.5):
                continue
            for track_id, _center, ever_outside, min_depth in items:
                if track_id in ids and ever_outside and min_depth < -_EXIT_MARGIN:
                    left = True
                    break
            if left:
                break
        flags.append(left)
    return flags


def run_video(
    path: Path, cfg, station, detector, *, hand: bool, screw: bool
) -> VideoResult:
    events: list = []
    pipeline = StationPipeline(
        station=station,
        pipeline_cfg=replace(cfg.pipeline, screw_exit_pick_enabled=screw),
        on_event=events.append,
        detector_name="yolo",
        cycle_id="EVAL",
    )
    if not hand:
        # 手部规则没有独立开关（生产上它始终启用）。单独评测丝杆规则时必须
        # 真的把它关掉，不能只在结果里按来源过滤：手部规则先判出后会把
        # 丝杆判据去重掉，过滤出来的结果就是「丝杆规则 0 命中」的伪结论。
        pipeline._update_pick = lambda **_: None  # type: ignore[method-assign]

    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pipeline.process(detections=detector.detect(frame, index / fps))
        index += 1
    cap.release()

    found = [
        (
            float(event.timestamp_start),
            float(event.timestamp_end),
            str(event.evidence.get("pick_source", "hand_exit")),
        )
        for event in events
        if event.event_type == EventType.PICK
    ]
    if not hand:
        found = [item for item in found if item[2] != "hand_exit"]
    if not screw:
        found = [item for item in found if item[2] != "screw_exit"]

    truth = _truth_for(path)
    return VideoResult(
        name=path.stem,
        truth=truth,
        found=found,
        leave_tray=_leave_tray_flags(path, truth, detector, station),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="逐规则评测取料判定")
    parser.add_argument(
        "--rule",
        choices=["hand", "screw", "both", "all"],
        default="all",
        help="评测哪条规则（默认逐条评测并给出合并结果）",
    )
    args = parser.parse_args()

    cfg = OilCheckConfigService().load(create_if_missing=False)
    station = cfg.enabled_stations()[0]
    detector = create_detector(station)

    combos = [
        ("hand", True, False),
        ("screw", False, True),
        ("both", True, True),
    ]
    if args.rule != "all":
        combos = [combo for combo in combos if combo[0] == args.rule]

    ok = True
    for label, hand, screw in combos:
        print(f"\n=== 规则：{label} ===")
        total_hits = total_misses = total_fp = total_truth = 0
        total_inside_only = 0
        for video in cfg.demo.videos():
            path = Path(resolve_runtime_file(video.path, repo_root=runtime_root()))
            result = run_video(path, cfg, station, detector, hand=hand, screw=screw)
            total_truth += len(result.truth)
            total_hits += result.hits
            total_misses += len(result.misses)
            total_fp += len(result.false_positives)
            inside_only = result.missed_inside_only(label)
            total_inside_only += inside_only
            bad = result.misses or result.false_positives
            print(
                f"  {result.name}: 基准 {len(result.truth):2d} / 命中 "
                f"{result.hits:2d} / 漏 {len(result.misses)} / 误 "
                f"{len(result.false_positives)}{'  ✗' if bad else ''}"
            )
            for start, end in result.misses:
                idx = result.truth.index((start, end))
                tag = ""
                if label == "screw" and not result.leave_tray[idx]:
                    tag = "  (杆未出盘)"
                print(f"      漏报 {start:6.2f}~{end:6.2f}{tag}")
            for start, end, src in result.false_positives:
                print(f"      误报 {start:6.2f}~{end:6.2f} ({src})")

        rate = total_hits / max(1, total_truth)
        print(
            f"  合计：基准 {total_truth} / 命中 {total_hits} / 漏报 {total_misses}"
            f" / 误报 {total_fp}  → 准确率 {rate:.1%}"
        )
        if total_inside_only:
            print(
                f"    其中「杆从未离开盘区」的漏报 {total_inside_only} 次"
                f"（丝杆规则定义内判不到，hand 规则本该覆盖）"
            )
        # 只有合并规则要求 0 漏报 0 误报；单条规则只要求 0 误报。
        if label == "both":
            ok = ok and total_misses == 0 and total_fp == 0
        else:
            ok = ok and total_fp == 0
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
