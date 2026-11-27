"""丝杆出盘判据的逐步诊断。

按帧打印每条丝杆记录的状态（位置、深度、在盘内证据、出盘确认进度），
并列出本段视频里判出的取料。用来回答「这次真取料为什么没判出、为什么
误报」。

用法::

    uv run python tools/diag_screw_exit.py --video 20261125_170216 --from 2.5 --to 7
    uv run python tools/diag_screw_exit.py --video 20261125_170216 --track 4
    uv run python tools/diag_screw_exit.py --list        # 列出可用的示例视频

``--video`` 接受文件名里唯一的片段（例如 ``170216``）。
"""

from __future__ import annotations

import argparse
import sys
import warnings
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


def main() -> int:
    parser = argparse.ArgumentParser(description="丝杆出盘判据逐步诊断")
    parser.add_argument("--video", default="", help="文件名片段")
    parser.add_argument("--from", dest="start", type=float, default=0.0)
    parser.add_argument("--to", dest="end", type=float, default=1e9)
    parser.add_argument("--track", type=int, default=0, help="只看这条记录")
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--no-hand", action="store_true", help="关掉手部判据，只看丝杆判据"
    )
    args = parser.parse_args()

    cfg = OilCheckConfigService().load(create_if_missing=False)

    if args.list or not args.video:
        for video in cfg.demo.videos():
            print(Path(video.path).stem)
        return 0

    station = cfg.enabled_stations()[0]
    detector = create_detector(station)
    matches = [v for v in cfg.demo.videos() if args.video in v.path]
    if len(matches) != 1:
        print(f"--video {args.video!r} 匹配到 {len(matches)} 段，要求唯一")
        return 2
    path = Path(resolve_runtime_file(matches[0].path, repo_root=runtime_root()))

    events: list = []
    pipeline = StationPipeline(
        station=station,
        pipeline_cfg=cfg.pipeline,
        on_event=events.append,
        detector_name="yolo",
        cycle_id="DIAG",
    )
    if args.no_hand:
        pipeline._update_pick = lambda **_: None  # type: ignore[method-assign]

    advance = pipeline._advance_screw_exit
    printed: set[tuple[int, float]] = set()

    def spy(state, det, **kwargs):  # noqa: ANN001, ANN003
        ts = kwargs["ts"]
        if args.start <= ts <= args.end and (not args.track or state.track_id == args.track):
            tray = kwargs["tray_box"]
            diag = kwargs["diag"]
            depth = _exit_depth(det.box, tray) / diag
            key = (state.track_id, round(ts, 3))
            if key not in printed:
                printed.add(key)
                cx, cy = det.box.center
                print(
                    f"t={ts:6.2f} id={state.track_id:3d} ({cx:5.0f},{cy:5.0f}) "
                    f"depth={depth:+.3f} in={state.inside} "
                    f"static_s={state.static_s:5.2f} conf={state.confirmed} "
                    f"out={state.outside_frames} rep={state.reported}"
                )
        return advance(state, det, **kwargs)

    pipeline._advance_screw_exit = spy  # type: ignore[method-assign]

    print(f"tray={pipeline._tray_region_box((2560, 1440))}")
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

    print("\n判出的取料：")
    for event in events:
        if event.event_type == EventType.PICK:
            print(
                f"  {event.timestamp_start:6.2f}~{event.timestamp_end:6.2f} "
                f"src={event.evidence.get('pick_source')}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
