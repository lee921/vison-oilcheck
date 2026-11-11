"""现场诊断：用**当前生效配置**把示例视频离线跑一遍判定，打印事件与关键状态。

用途是回答「视频在放、模型在跑，为什么没有取料/涂油事件」：

    uv run python tools/diagnose_replay.py                 # 默认跑第一段示例视频
    uv run python tools/diagnose_replay.py --frames 300    # 只跑前 300 帧（更快）
    uv run python tools/diagnose_replay.py --video 2       # 跑第二段示例视频

脚本只读：不写配置、不写事件库、不启动 GUI。判定走与线上完全相同的
``StationPipeline``，因此输出的事件数应当与监控页一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.oilcheck.config import OilCheckConfigService  # noqa: E402
from app.oilcheck.detector import create_detector  # noqa: E402
from app.oilcheck.pipeline import StationPipeline  # noqa: E402
from app.oilcheck.roi import Roi  # noqa: E402
from app.paths import data_root, resolve_runtime_file, runtime_root  # noqa: E402


def _hr(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=0, help="只跑前 N 帧，0=全部")
    parser.add_argument("--video", type=int, default=1, help="示例视频序号（1 基）")
    parser.add_argument(
        "--debug", type=int, default=0,
        help="前 N 帧打印管线内部视角（原始检测 vs 显示检测、ROI、帧尺寸）",
    )
    args = parser.parse_args()

    service = OilCheckConfigService(data_root())
    # 只读：不走 ensure_defaults()，避免诊断动作触发配置迁移/写回。
    cfg = service.load(create_if_missing=False)
    info = service.source_info()

    _hr("0) 实际加载的模块")
    import app.oilcheck.config as _cfg_mod
    import app.oilcheck.pipeline as _pipe_mod
    import app.oilcheck.roi as _roi_mod

    for mod in (_cfg_mod, _pipe_mod, _roi_mod):
        print(f"  {mod.__name__}: {getattr(mod, '__file__', '?')}")

    _hr("1) 生效配置")
    print("配置文件:", info["path"])
    print("数据目录:", info["data_root"])
    print("仓库模板:", info["template"])
    print("配置版本:", cfg.version)
    print("工位数:", len(cfg.enabled_stations()))
    for station in cfg.enabled_stations():
        print(f"  {station.station_id} {station.name} 相机={station.camera_id}")
        print(f"    涂油盘区 tray_roi = {station.tray_roi.points}")
        print(f"    工作追踪区 work_roi = {station.work_roi.points}")
        print(f"    涂油区 oiling_roi = {station.oiling_roi.points}")
        det = station.detector
        print(f"    检测器 kind={det.kind} conf={det.confidence} raw_mode={det.raw_mode}")
        print(f"    model_path={det.model_path} device={det.device}")
        print(f"    类别名: screw={det.screw_label!r} hand={det.hand_label!r} "
              f"brush={det.brush_label!r} tray={det.tray_label!r}")

    _hr("2) 关键判定参数")
    pipe = cfg.pipeline
    for key in (
        "max_fps", "pick_hand_distance", "pick_confirm_s",
        "pick_sample_every_frames", "pick_history_samples",
        "hand_screw_max_distance", "held_infer_distance",
        "brush_contact_distance", "brush_hold_s",
        "hand_track_max_missing_s", "hand_track_max_active",
        "brush_track_max_missing_s", "brush_track_max_active",
        "hand_tracker_backend", "brush_tracker_backend",
        "bytetrack_buffer_s", "bytetrack_match_thresh",
        "screw_anchor_max_step", "screw_anchor_max_step_abs",
    ):
        print(f"  {key} = {getattr(pipe, key, '<无此参数>')}")

    _hr("3) 示例视频")
    for i, video in enumerate(cfg.demo.videos, 1):
        path = Path(resolve_runtime_file(video.path, repo_root=runtime_root()))
        size = f"{path.stat().st_size / 1048576:.1f} MB" if path.exists() else "-"
        print(f"  [{i}] {video.name}: {path}  存在={path.exists()} {size}")
    if not cfg.demo.videos:
        print("  （空：监控页「画面源」只有实时相机）")
        return 1

    index = max(1, min(args.video, len(cfg.demo.videos))) - 1
    video = cfg.demo.videos[index]
    video_path = Path(resolve_runtime_file(video.path, repo_root=runtime_root()))
    if not video_path.exists():
        print(f"\n视频文件不存在: {video_path}")
        return 1

    _hr(f"4) 回放判定：{video.name}")
    import cv2

    station = cfg.enabled_stations()[0]
    events: list = []
    detector = create_detector(station)
    class_names = getattr(detector, "class_names", None) or {}
    if class_names:
        print("模型输出的类别:", class_names)
        configured = {
            "screw": station.detector.screw_label,
            "hand": station.detector.hand_label,
            "brush": station.detector.brush_label,
            "tray": station.detector.tray_label,
        }
        known = set(class_names.values())
        missing = {k: v for k, v in configured.items() if v not in known}
        if missing:
            print("!! 配置的类别名与模型输出不一致 ->", missing)
            print("!! 这些类别的检测会被全部过滤掉，判定不可能产生事件。")
            print("!! 模型实际输出:", sorted(known))
    print("检测器:", getattr(detector, "name", "?"),
          "device=", getattr(detector, "device", "?"),
          "imgsz=", getattr(detector, "imgsz", "?"))
    pipeline = StationPipeline(
        station=station,
        pipeline_cfg=pipe,
        on_event=events.append,
        detector_name=getattr(detector, "name", "yolo"),
        cycle_id="DIAG",
    )
    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = args.frames if args.frames > 0 else total
    print(f"视频 {video_path.name}: {fps:.1f} fps, {total} 帧, 本次跑 {limit} 帧")
    # 指纹：两端对比时先确认跑的是同一份权重和同一段录像。GPU/CPU 的推理
    # 结果可能有细微差异，但同一份输入 + 同一套 ROI 不该出现「丝杆全在区外」。
    import hashlib

    def _digest(path: Path) -> str:
        h = hashlib.md5()
        with path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:12]

    model_file = Path(
        resolve_runtime_file(station.detector.model_path, repo_root=runtime_root())
    )
    print("权重 md5:", _digest(model_file))
    print("视频 md5:", _digest(video_path))

    frame_no = 0
    last_task = None
    in_work_counts: list[int] = []
    frames_with_hand_in_tray = 0
    hand_in_tray_total = 0
    frames_with_screw_detected = 0
    while frame_no < limit:
        ok, bgr = capture.read()
        if not ok:
            break
        frame_no += 1
        ts = frame_no / fps
        frame_dets = detector.detect(bgr, ts)
        snapshot = pipeline.process(detections=frame_dets)
        counts = snapshot.stats.get("label_counts") or {}
        raw = frame_dets.detections
        fw, fh = snapshot.frame_size
        screw_in = sum(
            1
            for d in raw
            if d.label == station.detector.screw_label
            and station.work_roi.contains_box_center(d.box, fw, fh)
        )
        in_work_counts.append(screw_in)

        # 取料判定的起点是「手空手进入涂油盘区」。盘区 = 手工 tray_roi ∪ 模型
        # 检出的 oil_station 框。这里统计有多少帧真的出现了「手在盘内」——
        # 若为 0，无论后面的参数怎么调都不可能有 PICK。
        tray_rois = [station.tray_roi]
        tray_label = station.detector.tray_label
        tray_rois.extend(
            Roi(kind="rect", points=(
                (d.box.x1 / snapshot.frame_size[0], d.box.y1 / snapshot.frame_size[1]),
                (d.box.x2 / snapshot.frame_size[0], d.box.y2 / snapshot.frame_size[1]),
            ))
            for d in raw if d.label == tray_label
        )
        hands = [d for d in raw if d.label == station.detector.hand_label]
        if any(d.label == station.detector.screw_label for d in raw):
            frames_with_screw_detected += 1
        inside = 0
        for hand in hands:
            cx, cy = hand.box.center
            if any(r.contains_point(cx, cy, *snapshot.frame_size) for r in tray_rois):
                inside += 1
        if inside:
            frames_with_hand_in_tray += 1
            hand_in_tray_total += inside

        if frame_no <= args.debug:
            shown = snapshot.detections
            shown_screw = [d for d in shown if d.label == station.detector.screw_label]
            raw_screw = [d for d in raw if d.label == station.detector.screw_label]
            print(f"  [debug] 帧{frame_no} pipeline.station is station={pipeline.station is station}"
                  f" work_roi={pipeline.station.work_roi.points}"
                  f" frame_size={snapshot.frame_size}"
                  f" 原始丝杆={len(raw_screw)} 显示丝杆={len(shown_screw)}"
                  f" 显示检测总数={len(shown)}")
            for d in raw_screw:
                cx, cy = d.box.center
                inside = station.work_roi.contains_box_center(d.box, *snapshot.frame_size)
                print(f"      screw 中心=({cx:.0f},{cy:.0f}) "
                      f"归一化=({cx / max(1, snapshot.frame_size[0]):.3f},"
                      f"{cy / max(1, snapshot.frame_size[1]):.3f}) 区内={inside}")

        task = snapshot.stats.get("active_task_id")
        if task != last_task:
            stage = snapshot.tasks[-1].stage.value if snapshot.tasks else "-"
            msg = (f"  第 {frame_no:4d} 帧（{ts:6.2f}s）"
                   f"任务={task or '无'} 阶段={stage} 全画面={counts} 区内丝杆={screw_in}")
            print(msg)
            last_task = task

    capture.release()
    try:
        detector.close()
    except Exception:
        pass

    _hr("5) 结果")
    picks = [e for e in events if e.event_type.value == "PICK"]
    oils = [e for e in events if e.event_type.value == "MANUAL_OILING"]
    for e in events:
        print(f"  {e.event_type.value:14s} {e.result.value:9s} {getattr(e, 'task_id', '') or ''}")
    print()
    print(f"共 {frame_no} 帧：PICK {len(picks)} 次，MANUAL_OILING {len(oils)} 次，"
          f"事件 {len(events)} 条")
    if in_work_counts:
        inside = sum(1 for n in in_work_counts if n > 0)
        print(f"工作追踪区内出现过丝杆的帧: {inside}/{len(in_work_counts)}"
              f"（中位数 {sorted(in_work_counts)[len(in_work_counts) // 2]} 个）")
    print(f"全画面检出过丝杆的帧: {frames_with_screw_detected}/{frame_no}")
    print(f"有手落在涂油盘区内的帧: {frames_with_hand_in_tray}/{frame_no}"
          f"（累计 {hand_in_tray_total} 次手-盘内命中）")
    if not events and frames_with_hand_in_tray == 0:
        print()
        print(">>> 手从未被判进涂油盘区，取料判定不可能启动。")
        print(">>> 说明涂油盘区 tray_roi 与这段视频的画面不匹配（多半是按另一台")
        print(">>> 相机/另一种机位标定的）。用设置页「打开 ROI 标定」→ 画面源选")
        print(">>> 这段示例视频 → 重新划涂油盘区与工作追踪区。")
    if not events:
        print()
        print("没有任何事件。按下面顺序核对：")
        print("  1) 工作追踪区 work_roi 是否盖住了取料动作发生的区域（上面第 1 节的值）；")
        print("  2) 涂油盘区 tray_roi 是否盖住了盘体（取料要靠进盘/离盘判断）；")
        print("  3) 若「区内丝杆」始终为 0，说明 work_roi 没对准，用设置页的 ROI 标定重划。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
