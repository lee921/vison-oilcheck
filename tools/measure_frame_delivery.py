"""端到端实测：相机 → 解码 → 服务取帧，真实交付帧率与帧间隔分布。

用法::

    uv run python tools/measure_frame_delivery.py --seconds 20 [--station ST01]

只跑相机链路（不加载模型、不判定），把两件事分开量：

1. **解码侧**：``FrameBuffer.last_ts()`` 每 50ms 采一次，看相机帧到底以
   多快进入缓冲；
2. **服务侧**：与服务同构的轮询（``_poll_interval`` + 一轮工作耗时），
   看判定侧实际能拿到多少帧、帧间隔分布如何。

输出帧间隔直方图，以及「超过 2×中位间隔」的比例——那个比例就是日志里
「视频时间戳异常」的条数来源。
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.camera_manager import CameraManager  # noqa: E402
from app.config_service import ConfigService  # noqa: E402
from app.logging_service import LogStore, setup_logging  # noqa: E402
from app.paths import data_root, runtime_root  # noqa: E402
from hik.sdk_context import HikSdkContext  # noqa: E402


@dataclass
class _Stats:
    sample_gaps: list[float] = field(default_factory=list)
    served_gaps: list[float] = field(default_factory=list)
    decoded_count: int = 0
    served: int = 0
    polls: int = 0
    dropped: int = 0


def _quantiles(values: list[float], points=(0.5, 0.9, 0.99)) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        f"p{int(p * 100)}": ordered[min(len(ordered) - 1, int(len(ordered) * p))]
        for p in points
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--camera", default="", help="相机 ID；留空取第一台启用的")
    parser.add_argument(
        "--poll-fps", type=float, default=0.0, help="覆盖轮询帧率；0 = 用 pipeline.max_fps"
    )
    parser.add_argument(
        "--work-ms", type=float, default=-1.0,
        help="模拟一轮工作耗时；负数 = 实测真实耗时（含 JPEG 编码）",
    )
    args = parser.parse_args()

    storage = data_root()
    logger = setup_logging(
        log_dir=(storage / "logs").resolve(), store=LogStore()
    )
    cfg_service = ConfigService(storage_root=storage)
    app_cfg = cfg_service.load_app_config()
    cameras = cfg_service.load_cameras()
    enabled = [cam for cam in cameras if cam.enabled]
    if not enabled:
        print("没有启用的相机（配置文件 config/cameras.json）")
        return 2
    camera = next((c for c in enabled if c.id == args.camera), enabled[0])

    # 判定帧率：默认按 oilcheck.json 的 pipeline.max_fps；测试用的覆盖值优先。
    poll_fps = args.poll_fps
    if poll_fps <= 0:
        try:
            from app.oilcheck.config import OilCheckConfigService

            poll_fps = float(OilCheckConfigService().load().pipeline.max_fps)
        except Exception:  # noqa: BLE001
            poll_fps = 15.0

    print(
        f"相机 {camera.id} ({camera.name}) {camera.ip}  码流={camera.stream_type}  "
        f"decode_max_fps={app_cfg.decode_max_fps}  轮询={poll_fps:g}fps"
    )

    sdk_ctx = HikSdkContext(
        repo_root=runtime_root(), log_dir=(storage / "logs" / "hik_sdk"), logger=logger
    )
    manager = CameraManager(
        sdk_ctx=sdk_ctx,
        cfg_service=cfg_service,
        app_cfg=app_cfg,
        logger=logger,
    )
    stats = _Stats()
    stop = threading.Event()

    try:
        manager.start_async(camera.id).result(timeout=20.0)
        deadline = time.monotonic() + 15.0
        session = None
        while time.monotonic() < deadline:
            session = manager.get_session(camera.id)
            if session is not None and session.state.value == "previewing":
                break
            time.sleep(0.05)
        if session is None or session.state.value != "previewing":
            state = "missing" if session is None else session.state.value
            print(f"相机没进入预览状态（当前={state}）")
            return 3
        print("预览已启动，开始采样……")
        time.sleep(1.0)  # 丢掉启动阶段的抖动

        # 1) 解码侧：直接计数 ``FrameBuffer.update_bgr`` 的调用。不能用
        #    「定时读 last_ts」——20Hz 的采样本身就把 25fps 的帧率测成 16fps。
        from hik import decode as decode_module

        decoded_ts: list[float] = []
        real_update = decode_module.FrameBuffer.update_bgr

        def counting_update(self, bgr, width, height):  # noqa: ANN001
            decoded_ts.append(time.monotonic())
            return real_update(self, bgr, width, height)

        decode_module.FrameBuffer.update_bgr = counting_update

        # 2) 服务侧：与服务同构的固定节拍轮询 + 每帧 JPEG 编码。
        interval = 1.0 / max(1.0, poll_fps)
        next_tick = time.monotonic()
        end = time.monotonic() + args.seconds
        last_served = 0.0
        while time.monotonic() < end:
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            next_tick += interval
            stats.polls += 1
            started = time.perf_counter()
            # 与服务侧同一套取帧方式：等「比上一帧更新」的帧，而不是取最新帧。
            frame = session.frame_buffer.wait_newer(last_served, timeout=interval)
            if frame is None:
                stats.dropped += 1
            else:
                if last_served:
                    stats.served_gaps.append(frame.ts - last_served)
                last_served = frame.ts
                stats.served += 1
                # 现场每帧都要做一次片段缓冲的 JPEG 编码（2560×1440 约 5ms）。
                if args.work_ms >= 0:
                    time.sleep(args.work_ms / 1000.0)
                else:
                    import cv2

                    cv2.imencode(".jpg", frame.bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            # 其余同步开销（PLC 快照、stat、flush_idle）用固定量近似。
            if args.work_ms < 0:
                spent = time.perf_counter() - started
                if spent < 0.003:
                    time.sleep(0.003 - spent)

        stop.set()
        decode_module.FrameBuffer.update_bgr = real_update
        stats.sample_gaps = [
            b - a for a, b in zip(decoded_ts, decoded_ts[1:], strict=False)
        ]
        stats.decoded_count = len(decoded_ts)
    finally:
        try:
            manager.shutdown()
        except Exception:  # noqa: BLE001
            pass
        sdk_ctx.close(timeout=3.0)

    _report(stats, args.seconds, poll_fps)
    return 0


def _report(stats: _Stats, seconds: float, poll_fps: float) -> None:
    print()
    if stats.sample_gaps:
        gaps = stats.sample_gaps
        q = _quantiles(gaps)
        fps = stats.decoded_count / max(1e-9, seconds)
        print(
            f"解码侧（FrameBuffer 收到新帧）: {fps:.1f} fps（共 {stats.decoded_count} 帧），"
            f"帧间隔 中位 {q.get('p50', 0):.3f}s / p90 {q.get('p90', 0):.3f}s / "
            f"p99 {q.get('p99', 0):.3f}s"
        )
    else:
        print("解码侧：没采到样本")

    if stats.served:
        gaps = stats.served_gaps
        median = _quantiles(gaps).get("p50", 0.0)
        # 与 ``StationPipeline._max_plausible_dt`` 同一口径：阈值取
        # ``2 × max(帧间隔中位数, 1/max_fps)``。只用中位数会低估低帧率配置下的
        # 容忍度，报出偏高的「异常」比例。
        base_dt = 1.0 / max(1.0, poll_fps)
        plausible = 2.0 * max(median, base_dt)
        abnormal = sum(1 for g in gaps if g > plausible) if median else 0
        hist = Counter(round(g, 2) for g in gaps)
        served_fps = stats.served / max(1e-9, seconds)
        print(
            f"服务侧（判定实际拿到）: {served_fps:.1f} fps，帧间隔 中位 {median:.3f}s / "
            f"p90 {_quantiles(gaps).get('p90', 0.0):.3f}s / max {max(gaps):.3f}s"
        )
        share = 100 * abnormal / max(1, len(gaps))
        print(
            f"  超过可信上限 {plausible:.3f}s（会记“时间戳异常”）: "
            f"{abnormal} / {len(gaps)} = {share:.1f}%"
        )
        print("  间隔分布:", hist.most_common(6))
    hit = 100 * stats.served / max(1, stats.polls)
    print(
        f"轮询 {stats.polls} 次，取到帧 {stats.served} 次（空轮 {stats.dropped}），"
        f"配置 {poll_fps:g} fps → 命中率 {hit:.0f}%"
    )


if __name__ == "__main__":
    raise SystemExit(main())
