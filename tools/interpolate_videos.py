"""把示例视频插帧到指定帧率（默认 15 → 20fps），供判定侧按现场帧率核对。

为什么需要这个脚本：示例录像都是 15fps，而现场相机主码流是 25fps、
判定侧取 ``pipeline.max_fps=20``。15fps 的录像上「判得对」不等于
20fps 节拍下也判得对——帧数变了，`frame_scale`、连续帧数门限、采样
窗口都在按帧走。这里生成一份 20fps 的对照集，用同一套判定跑一遍。

用法::

    uv run python tools/interpolate_videos.py                 # data/sample_video → data/sample_video_20fps
    uv run python tools/interpolate_videos.py --fps 25
    uv run python tools/interpolate_videos.py --jobs 4        # 同时跑几段（默认 min(4, CPU/3)）
    uv run python tools/interpolate_videos.py --dry-run

插帧用 ffmpeg 的 ``minterpolate``（运动补偿 + 双向光流），不是重复帧：
重复帧骗不过判定——同一画面被算两次会让「连续帧数」门限提前满足。
代价是慢（2560×1440 上约 30s/视频秒，单进程满核）。产物只用于离线
核对，不覆盖原始录像；文件名与原始一致，方便按文件名配对基准。
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _filter(fps: float) -> str:
    return (
        f"minterpolate=fps={fps:g}:mi_mode=mci:mc_mode=aobmc"
        f":me_mode=bidir:vsbmc=1"
    )


def convert(
    src: Path,
    dst: Path,
    *,
    fps: float,
    crf: int,
    preset: str,
    threads: int,
    overwrite: bool,
    dry_run: bool,
) -> tuple[Path, bool, float]:
    """转换一段视频；返回 (目标路径, 是否成功, 耗时秒)。已存在且不覆盖时跳过。"""

    if dst.exists() and not overwrite:
        return dst, True, 0.0
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src),
        "-vf", _filter(fps),
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-threads", str(threads), "-pix_fmt", "yuv420p",
        "-an", "-movflags", "+faststart",
        str(dst),
    ]
    if dry_run:
        print("  " + " ".join(cmd))
        return dst, True, 0.0
    started = time.perf_counter()
    # 先写临时文件再改名：中途失败不会留下「看着有、其实是半截」的录像。
    tmp = dst.with_suffix(dst.suffix + ".part")
    cmd[-1] = str(tmp)
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        print(f"  失败 {src.name}: {proc.stderr.strip()[:400]}", flush=True)
        return dst, False, elapsed
    os.replace(tmp, dst)
    return dst, True, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="示例视频插帧到目标帧率")
    parser.add_argument("--src", default="data/sample_video", help="源目录")
    parser.add_argument(
        "--dst", default="data/sample_video_20fps", help="输出目录"
    )
    parser.add_argument("--fps", type=float, default=20.0, help="目标帧率")
    parser.add_argument("--crf", type=int, default=16, help="x264 质量（越小越清）")
    parser.add_argument("--preset", default="veryfast", help="x264 preset")
    parser.add_argument(
        "--jobs", type=int, default=0, help="并行段数；0 = min(4, CPU/3)"
    )
    parser.add_argument(
        "--threads", type=int, default=0, help="每段 ffmpeg 线程数；0 = CPU/jobs"
    )
    parser.add_argument("--overwrite", action="store_true", help="重转已存在的")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("找不到 ffmpeg，请先安装（apt install ffmpeg）")

    src_dir = Path(args.src)
    if not src_dir.is_absolute():
        src_dir = REPO_ROOT / src_dir
    dst_dir = Path(args.dst)
    if not dst_dir.is_absolute():
        dst_dir = REPO_ROOT / dst_dir
    if not src_dir.is_dir():
        raise SystemExit(f"源目录不存在: {src_dir}")

    videos = sorted(p for p in src_dir.iterdir() if p.suffix.lower() == ".mp4")
    videos = [p for p in videos if p.is_file() and not p.name.endswith(".part")]
    if not videos:
        raise SystemExit(f"源目录里没有 mp4: {src_dir}")

    cpu = os.cpu_count() or 4
    jobs = args.jobs or max(1, min(4, cpu // 3))
    threads = args.threads or max(1, cpu // jobs)
    dst_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"{len(videos)} 段 → {dst_dir}（{args.fps:g}fps，并行 {jobs} 段 × "
        f"{threads} 线程）\n"
        f"minterpolate 是运动补偿插帧，1440p 上约 30s/视频秒——全部转完要 "
        f"十几分钟到半小时，属正常\n"
    )
    jobs_list = [(p, dst_dir / p.name) for p in videos]
    results = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [
            pool.submit(
                convert,
                src, dst,
                fps=args.fps, crf=args.crf, preset=args.preset,
                threads=threads, overwrite=args.overwrite, dry_run=args.dry_run,
            )
            for src, dst in jobs_list
        ]
        for (src, _dst), future in zip(jobs_list, futures, strict=True):
            path, ok, elapsed = future.result()
            if elapsed:
                print(f"  {src.name} → {path.name} {elapsed:.0f}s", flush=True)
            results.append((path, ok))

    failed = [path for path, ok in results if not ok]
    print(f"\n完成 {len(results) - len(failed)}/{len(results)}")
    if failed:
        for path in failed:
            print(f"  失败: {path.name}")
        return 1
    print(f"产物目录: {dst_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
