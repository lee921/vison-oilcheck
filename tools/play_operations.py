"""回放检查结果：逐条标出每次取料与人工二次涂油。

数据来自 ``tools/detect_operations.py`` 写出的 JSON（``picks`` / ``oilings``）。
播放时：

- 左上是录像，当前正在发生的动作会高亮（取料闪黄框、涂油显示接触进度）；
- 右侧是这段视频里**全部**动作的时间线，正在发生的那一条高亮、已发生的变暗；
- 底部是进度条，``[`` / ``]`` 跳到上/下一次动作，空格暂停。

用法：

    uv run python tools/play_operations.py                  # 全部已出结果的视频
    uv run python tools/play_operations.py --video 5         # 只看第 5 段（1 基）
    uv run python tools/play_operations.py --results x.json  # 指定结果文件
    uv run python tools/play_operations.py --snapshot 3      # 第 3 段第 8 秒存一张图
    uv run python tools/play_operations.py --speed 0.5

没有结果文件时会提示先跑 ``tools/detect_operations.py``；``--snapshot`` 与
``--video`` 配合可以在没有图形界面的机器上导出核对图。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402


# ----------------------------------------------------------------------
# 结果模型
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    kind: str  # "PICK" | "OIL"
    label: str
    index: int  # 该类型内从 1 开始的序号
    start_ts: float
    end_ts: float
    detail: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)


@dataclass
class Clip:
    path: Path
    fps: float
    duration_s: float
    actions: list[Action]
    source_json: Path


def _fmt_span(action: Action) -> str:
    return f"{action.start_ts:5.2f}s → {action.end_ts:5.2f}s"


def load_clip(json_path: Path) -> Clip:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    picks = data.get("picks", [])
    oils = data.get("oilings", [])
    actions: list[Action] = []
    for i, item in enumerate(picks, 1):
        actions.append(
            Action(
                kind="PICK",
                label="取料",
                index=i,
                start_ts=float(item["start_ts"]),
                end_ts=float(item["end_ts"]),
                detail=f"盘内停留 {item.get('dwell_in_tray_s', 0):.2f}s",
            )
        )
    for i, item in enumerate(oils, 1):
        parts = item.get("contact_parts", 1)
        extra = f"，{parts} 段" if parts > 1 else ""
        actions.append(
            Action(
                kind="OIL",
                label="人工二次涂油",
                index=i,
                start_ts=float(item["start_ts"]),
                end_ts=float(item["end_ts"]),
                detail=f"接触 {item.get('contact_s', 0):.2f}s{extra}",
            )
        )
    # 同一时刻先列取料再列涂油（取料在前更符合现场顺序）。
    actions.sort(key=lambda a: (a.start_ts, 0 if a.kind == "PICK" else 1))
    video = Path(data.get("path") or "")
    if not video.exists():
        video = REPO_ROOT / "data" / "sample_video" / data["video"]
    return Clip(
        path=video,
        fps=float(data.get("fps") or 15.0),
        duration_s=float(data.get("duration_s") or 0.0),
        actions=actions,
        source_json=json_path,
    )


# ----------------------------------------------------------------------
# 画面叠加（PIL 画中文，OpenCV 的 Hershey 字体不支持）
# ----------------------------------------------------------------------
# 必须是含中日韩字形的字体：DejaVu 之类西文字体没有汉字，PIL 会画成方框，
# 而检测框又不能换成 QPainter（这条路径要能在无界面机器上导出图）。
_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)
_font_cache: dict[int, ImageFont.FreeTypeFont] = {}
_ZH_PROBE = "取料涂油"


def _font_path() -> str | None:
    """挑一个真正带汉字字形的字体文件。"""

    for candidate in _FONT_CANDIDATES:
        if not Path(candidate).exists():
            continue
        try:
            font = ImageFont.truetype(candidate, 24)
        except OSError:
            continue
        # 仅存在不代表有汉字（某些字体是纯西文库）；用字形宽度粗判：
        # 缺字时 PIL 画的是方框，与汉字的宽度一致，所以改比对「能否渲染」
        # 之外的字段——直接查字体表是否有 cmap 覆盖。
        if _has_glyphs(font):
            return candidate
    return None


def _has_glyphs(font: ImageFont.FreeTypeFont) -> bool:
    try:
        mask = font.getmask(_ZH_PROBE)
        return mask.size[0] > 0 and mask.getbbox() is not None
    except Exception:  # noqa: BLE001
        return False


_FONT_FILE: str | None = None


def _font(size: int) -> ImageFont.FreeTypeFont:
    global _FONT_FILE
    if size in _font_cache:
        return _font_cache[size]
    if _FONT_FILE is None:
        _FONT_FILE = _font_path() or ""
    if _FONT_FILE:
        font = ImageFont.truetype(_FONT_FILE, size)
    else:
        print(
            "警告：没找到含中文字形的字体，画面上的中文会显示成方框。"
            "请安装 fonts-noto-cjk 或文泉驿正黑。",
            file=sys.stderr,
        )
        font = ImageFont.load_default(size)
    _font_cache[size] = font
    return font


_PICK_COLOR = (255, 196, 0)
_OIL_COLOR = (0, 132, 255)
_DIM = (150, 150, 150)


def _active_at(actions: list[Action], ts: float) -> list[Action]:
    return [a for a in actions if a.start_ts <= ts <= a.end_ts]


def draw_overlay(
    frame: np.ndarray,
    clip: Clip,
    *,
    ts: float,
    speed: float,
    paused: bool,
    flash: float,
    max_width: int = 0,
) -> np.ndarray:
    """把时间线与当前动作画到帧上，返回 BGR。

    ``max_width`` > 0 时先把画面缩到这个宽度再叠加：2560×1440 的原帧上跑
    PIL 画字每帧要几十毫秒，播放会卡；窗口本来也只有 1600 宽。
    """

    if max_width > 0 and frame.shape[1] > max_width:
        height = int(round(frame.shape[0] * max_width / frame.shape[1]))
        frame = cv2.resize(frame, (max_width, height), interpolation=cv2.INTER_AREA)

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    width, height = image.size
    scale = max(1.0, width / 1600.0)

    f_big = _font(int(30 * scale))
    f_mid = _font(int(24 * scale))
    f_small = _font(int(20 * scale))

    def text(xy, text, font, fill, shadow=True):
        if shadow:
            draw.text((xy[0] + 2, xy[1] + 2), text, font=font, fill=(0, 0, 0))
        draw.text(xy, text, font=font, fill=fill)

    # ---- 标题 ----
    header = f"{clip.path.name}   {ts:6.2f}s / {clip.duration_s:.1f}s"
    box = draw.textbbox((0, 0), header, font=f_big)
    draw.rectangle(
        [16, 14, 16 + box[2] + 20, 14 + box[3] - box[1] + 18], fill=(0, 0, 0)
    )
    text((26, 18), header, f_big, (255, 255, 255))

    ys = 70
    speed_txt = "暂停" if paused else f"{speed:g}×"
    text(
        (26, ys),
        f"播放 {speed_txt}   空格暂停   [ ] 上一次/下一次动作",
        f_small,
        (230, 230, 230),
    )
    ys += int(34 * scale)

    # ---- 当前正在发生的动作 ----
    active = _active_at(clip.actions, ts)
    for action in active:
        color = _PICK_COLOR if action.kind == "PICK" else _OIL_COLOR
        if action.kind == "PICK":
            head = f"取料 #{action.index}  进行中"
        else:
            progress = ts - action.start_ts
            head = f"人工二次涂油 #{action.index}  进行中  {progress:.2f}s"
        body = f"{_fmt_span(action)}   {action.detail}"
        block_w = max(
            draw.textbbox((0, 0), head, font=f_big)[2],
            draw.textbbox((0, 0), body, font=f_mid)[2],
        )
        pad = 14
        box_h = int(36 * scale) + int(30 * scale) + pad * 2
        draw.rectangle([16, ys, 16 + block_w + pad * 2, ys + box_h], fill=(20, 20, 20))
        draw.rectangle([16, ys, 22, ys + box_h], fill=color)
        text((16 + pad + 10, ys + pad - 4), head, f_big, color)
        text((16 + pad + 10, ys + pad + int(34 * scale)), body, f_mid, (240, 240, 240))
        ys += box_h + 8

    # 取料瞬间闪一下（0.9s）
    if flash > 0:
        alpha = int(120 * min(1.0, flash / 0.9))
        overlay = Image.new("RGB", image.size, _PICK_COLOR)
        image = Image.blend(image, overlay, alpha / 255 * 0.35)
        draw = ImageDraw.Draw(image)

    # ---- 右侧时间线：全部动作 ----
    panel_w = int(430 * scale)
    x0 = width - panel_w
    draw.rectangle([x0, 0, width, height], fill=(12, 12, 12))

    title = (
        f"本段共 取料 {sum(1 for a in clip.actions if a.kind == 'PICK')} 次 · "
        f"涂油 {sum(1 for a in clip.actions if a.kind == 'OIL')} 次"
    )
    draw.text((x0 + 18, 18), title, font=f_mid, fill=(255, 255, 255))
    # 行高按两行文字的实际行高算：f_mid + f_small + 间距。写死行高会让
    # 相邻两条动作叠在一起（实测 38px 不够，中文行高比西文大）。
    line1_h = int(28 * scale)
    line2_h = int(24 * scale)
    row_h = line1_h + line2_h + int(10 * scale)
    row_y = 66
    max_rows = max(1, (height - row_y - 16) // row_h)
    rows = clip.actions[:max_rows]

    # 时间轴竖条
    track_x = x0 + panel_w - 22
    total = max(0.001, clip.duration_s)
    draw.line([(track_x, 60), (track_x, height - 12)], fill=(60, 60, 60), width=3)
    play_y = 60 + (height - 72) * min(1.0, ts / total)
    draw.ellipse(
        [track_x - 7, play_y - 7, track_x + 7, play_y + 7], fill=(255, 255, 255)
    )

    for action in rows:
        color = _PICK_COLOR if action.kind == "PICK" else _OIL_COLOR
        is_active = action in active
        done = action.end_ts < ts
        label_color = color if (is_active or not done) else _DIM
        if is_active:
            draw.rectangle(
                [x0 + 8, row_y - 4, track_x - 20, row_y + row_h - 6],
                fill=(45, 45, 45),
            )
        mark = "▶" if is_active else ("✓" if done else "·")
        line1 = f"{mark} {action.label} #{action.index}"
        line2 = f"   {_fmt_span(action)}  {action.detail}"
        draw.text((x0 + 18, row_y), line1, font=f_mid, fill=label_color)
        draw.text(
            (x0 + 18, row_y + line1_h),
            line2,
            font=f_small,
            fill=label_color
            if is_active
            else (200, 200, 200)
            if not done
            else (120, 120, 120),
        )
        # 每条动作在时间轴上的区间
        y_a = 60 + (height - 72) * min(1.0, action.start_ts / total)
        y_b = 60 + (height - 72) * min(1.0, action.end_ts / total)
        draw.line([(track_x, y_a), (track_x, max(y_a + 3, y_b))], fill=color, width=6)
        row_y += row_h

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------
# 播放
# ----------------------------------------------------------------------
class _Player:
    """按媒体时间推进的播放器；空格暂停，[ ] 跳转动作。"""

    def __init__(self, clip: Clip, speed: float) -> None:
        self.clip = clip
        self.speed = max(0.05, float(speed))
        self.paused = False
        self.quit = False
        self.frame_no = 0
        self._flash_until = 0.0
        self._capture = cv2.VideoCapture(str(clip.path))
        if not self._capture.isOpened():
            raise SystemExit(f"无法打开视频: {clip.path}")
        self.fps = self._capture.get(cv2.CAP_PROP_FPS) or clip.fps or 15.0
        # 播放窗口宽度：原帧 2560 宽，先缩到这里再叠加标注。标注字号按
        # 这个宽度调过；缩得再小中文会挤在一起。
        self.display_width = 1600
        self._clock_origin = 0.0
        self._pending_seek = 0.0

    # -- 时间控制 -----------------------------------------------------
    @property
    def ts(self) -> float:
        return self.frame_no / self.fps

    def seek(self, ts: float) -> None:
        ts = max(0.0, min(self.clip.duration_s or ts, ts))
        self.frame_no = max(0, int(round(ts * self.fps)))
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no)

    # -- 主循环 -------------------------------------------------------
    def _render(self, frame: np.ndarray) -> np.ndarray:
        return draw_overlay(
            frame,
            self.clip,
            ts=self.ts,
            speed=self.speed,
            paused=self.paused,
            flash=max(0.0, self._flash_until - self.ts),
            max_width=self.display_width,
        )

    def _note_new_frame(self) -> None:
        """刚进入新的一帧：取料区间内就点亮取料提示（区间结束自动熄灭）。"""

        ts = self.ts
        self._flash_until = max(
            (
                a.end_ts
                for a in self.clip.actions
                if a.kind == "PICK" and a.start_ts <= ts <= a.end_ts
            ),
            default=0.0,
        )

    def _jump(self, ts: float) -> np.ndarray | None:
        """跳转并取跳转后的画面（跳转期间画面不推进时间）。"""

        self.seek(ts)
        ok, frame = self._capture.read()
        if not ok:
            return None
        self.frame_no += 1
        self._sync_clock()
        return frame

    def _sync_clock(self) -> None:
        """把播放节拍的参考点对齐到当前帧，跳转后不会一口气追帧。"""

        self._clock_origin = time.perf_counter() - self.frame_no / (
            self.fps * self.speed
        )

    def run(self) -> None:
        window = self.clip.path.stem
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        aspect = 9 / 16
        cap_h = self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1
        cap_w = self._capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1
        if cap_w > 0:
            aspect = cap_h / cap_w
        # 目标高度 = 播放画面高 + 面板；WINDOW_NORMAL 下 OpenCV 会按窗口
        # 尺寸缩放画面，留出这点余量才不会被裁掉。
        cv2.resizeWindow(
            window,
            self.display_width + 2,
            int(self.display_width * aspect) + 34,
        )
        self._sync_clock()
        frame: np.ndarray | None = None
        drawn = None
        ended = False
        while not self.quit:
            if ended:
                # 播完停在最后一帧：只有 r / Esc / q 有意义。
                key = cv2.waitKeyEx(30) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    ended = False
                    self.paused = False
                    frame = self._jump(0.0)
                continue

            if frame is None:
                ok, frame = self._capture.read()
                if not ok:
                    ended = True
                    if drawn is not None:
                        cv2.imshow(window, drawn)
                    continue
                self.frame_no += 1
                self._note_new_frame()
            drawn = self._render(frame)
            cv2.imshow(window, drawn)
            frame = None

            key = cv2.waitKeyEx(1) & 0xFF
            if self._handle_key(key):
                # 跳转：重新取帧后重画，不推进时间。
                self._note_new_frame()
                frame = self._jump(self._pending_seek)
                continue
            if self.paused:
                while self.paused and not self.quit:
                    key = cv2.waitKeyEx(30) & 0xFF
                    if self._handle_key(key):
                        self._note_new_frame()
                        frame = self._jump(self._pending_seek)
                        break
                continue

            # 按墙钟节流：目标是「这一帧本该显示的时刻」。
            target = self._clock_origin + self.frame_no / (self.fps * self.speed)
            delay = target - time.perf_counter()
            if delay > 0:
                key = cv2.waitKeyEx(int(delay * 1000)) & 0xFF
                if self._handle_key(key):
                    self._note_new_frame()
                    frame = self._jump(self._pending_seek)
        cv2.destroyAllWindows()

    def _handle_key(self, key: int) -> bool:
        """处理按键；返回 True 表示需要跳转（``_pending_seek`` 有效）。"""

        self._pending_seek = self.ts
        if key == 27 or key == ord("q"):  # Esc / q
            self.quit = True
        elif key == ord(" "):
            self.paused = not self.paused
        elif key in (ord("]"), ord(".")):
            self._pending_seek = self._next_action_ts()
            return True
        elif key in (ord("["), ord(",")):
            self._pending_seek = self._prev_action_ts()
            return True
        elif key == ord("r"):
            self._pending_seek = 0.0
            return True
        return False

    def _next_action_ts(self) -> float:
        now = self.ts
        later = sorted(
            (a for a in self.clip.actions if a.start_ts > now + 0.05),
            key=lambda a: a.start_ts,
        )
        return max(0.0, later[0].start_ts - 0.3) if later else self.ts

    def _prev_action_ts(self) -> float:
        now = self.ts
        earlier = sorted(
            (a for a in self.clip.actions if a.start_ts < now - 0.6),
            key=lambda a: a.start_ts,
            reverse=True,
        )
        return max(0.0, earlier[0].start_ts - 0.3) if earlier else 0.0


# ----------------------------------------------------------------------
# 无界面导出
# ----------------------------------------------------------------------
def snapshot(clip: Clip, ts: float, out: Path, *, speed: float = 1.0) -> Path:
    capture = cv2.VideoCapture(str(clip.path))
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频: {clip.path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(ts * clip.fps)))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"{clip.path.name} 取不到 {ts:.2f}s 的帧")
    shown = draw_overlay(
        frame,
        clip,
        ts=ts,
        speed=speed,
        paused=True,
        flash=0.0,
        max_width=1600,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), shown)
    return out


# ----------------------------------------------------------------------
def _collect(results_dir: Path) -> list[Path]:
    return sorted(
        path for path in results_dir.glob("*.json") if path.name != "summary.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="回放取料/涂油检测结果")
    parser.add_argument("--video", type=int, default=0, help="只看第 N 段（1 基）")
    parser.add_argument(
        "--results",
        action="append",
        default=[],
        help="结果 JSON 路径（可重复；默认读 data/operations_preview/）",
    )
    parser.add_argument("--results-dir", default="", help="结果目录")
    parser.add_argument("--speed", type=float, default=1.0, help="播放速度（默认 1.0）")
    parser.add_argument(
        "--snapshot",
        type=float,
        default=None,
        help="不播放，只把该时刻的画面存成图片（配合 --video 指定片段）",
    )
    parser.add_argument("--out", default="", help="--snapshot 的输出路径")
    args = parser.parse_args()

    results_dir = (
        Path(args.results_dir)
        if args.results_dir
        else REPO_ROOT / "data" / "operations_preview"
    )
    paths = [Path(p) for p in args.results]
    if not paths:
        paths = _collect(results_dir)
    if not paths:
        raise SystemExit(
            f"{results_dir} 里没有结果文件。先跑 "
            f"`uv run python tools/detect_operations.py`。"
        )
    if args.video:
        index = max(1, min(args.video, len(paths))) - 1
        paths = [paths[index]]

    clips = [load_clip(path) for path in paths]

    if args.snapshot is not None:
        clip = clips[0]
        out = (
            Path(args.out)
            if args.out
            else (
                REPO_ROOT
                / "data"
                / "operations_preview"
                / f"{clip.path.stem}_{args.snapshot:.2f}s.jpg"
            )
        )
        saved = snapshot(clip, args.snapshot, out, speed=args.speed)
        print(f"已保存: {saved}")
        for action in clip.actions:
            mark = "▶" if action.start_ts <= args.snapshot <= action.end_ts else " "
            print(
                f" {mark} {action.label} #{action.index} {_fmt_span(action)}  {action.detail}"
            )
        return 0

    for clip in clips:
        print(f"\n=== {clip.path.name} ===")
        print(f"  {clip.duration_s:.1f}s，{len(clip.actions)} 个动作")
        for action in clip.actions:
            print(
                f"    {action.label} #{action.index}  {_fmt_span(action)}  {action.detail}"
            )
        _Player(clip, args.speed).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
