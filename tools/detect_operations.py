"""独立检测「取料」与「人工二次涂油」两个工序。

与线上 ``StationPipeline`` 的区别：这里**不建作业周期**，两个工序各用一个
独立检测器，互不看对方的结论，也不要求先后顺序。因此：

- 视频开头就已经拿在手里、正在刷的丝杆（取料发生在录制之前）也能判出涂油；
- 只取料没涂油、只涂油没取料，都各自如实报出来，不会被「周期」串起来；
- 一遍视频只做一次检测 + 一次跟踪，两个检测器共享同一帧证据。

判定口径：

1. **取料**：一只手**空手**进入涂油盘区（手工 ``tray_roi``），随后同一只手
   **带着丝杆离开**，杆心已经出盘（不在 ``tray_roi`` 内），盘内停留
   ≥ ``pick_confirm_s``。等价于 ``docs/05`` 的 4.1/4.2 节。
2. **人工二次涂油**：刷头工作点到**手持丝杆**杆身中线的距离
   ≤ ``brush_contact_distance``，累计接触 ≥ ``brush_hold_s``。丝杆必须是
   被手握着的那根（排除盘里/零件盒里的静置杆），不要求它来自取料工序。

用法：

    uv run python tools/detect_operations.py                  # 配置里的全部示例视频
    uv run python tools/detect_operations.py --video a.mp4    # 指定视频（可重复）
    uv run python tools/detect_operations.py --no-annotate    # 只出结论，不写带框图
    uv run python tools/detect_operations.py --limit 200      # 每段只跑前 200 帧

产物在 ``data/operations_preview/``：``<名字>.json``（两个工序各自的动作清单）
与 ``<名字>.mp4``（带框图，可 ``--no-annotate`` 关闭）。终端打印每段视频的
取料/涂油次数与时间点。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.oilcheck.association import brush_work_point  # noqa: E402
from app.oilcheck.bytetrack_adapter import ByteTrackObjectTracker  # noqa: E402
from app.oilcheck.config import (  # noqa: E402
    OilCheckConfigService,
    PipelineConfig,
    StationConfig,
)
from app.oilcheck.detector import create_detector  # noqa: E402
from app.oilcheck.types import BBox, Detection, FrameDetections  # noqa: E402
from app.paths import resolve_runtime_file, runtime_root  # noqa: E402

# ----------------------------------------------------------------------
# 几何门限（与 app/oilcheck/pipeline.py 同一套实测口径）
# ----------------------------------------------------------------------
# 「位置没动过」的判定：与 8 帧前的框 IoU ≥ 0.5。盘里那排杆、零件盒里的整排
# 杆都不动；被手带走的杆 ≤0.3。见 pipeline._screw_static 的注释。
_STATIC_BACK = 8
_STATIC_IOU = 0.5
_STATIC_HISTORY_MAX = _STATIC_BACK + 2
# 握持判据里「手框右上角区域」：右边 50%、上边 50%。拳头攥住杆身时杆从
# 虎口伸出去，与这块区域明显相交。
_GRIP_RX = 0.5
_GRIP_RY = 0.5
# 静置杆要有这么多面积落在右上区才算握着（盘边蹭到 0.25~0.42，真攥住 ≥0.44）。
_GRIP_PARKED_MIN_FRAC = 0.50
# 「这根杆是进盘时那批存货」的 IoU 门限。
_INVENTORY_SAME_IOU = 0.5
# 手已出盘、杆心还没出盘时最多等这么久（按帧率缩放）。
_CARRY_OUT_OF_TRAY_MAX_S = 5.0


def _majority(samples: list[bool]) -> bool:
    if not samples:
        return False
    return sum(1 for value in samples if value) * 2 > len(samples)


def _overlaps(a: BBox, b: BBox) -> bool:
    return (
        min(a.x2, b.x2) - max(a.x1, b.x1) > 0 and min(a.y2, b.y2) - max(a.y1, b.y1) > 0
    )


def _center_in(box: BBox, region: BBox) -> bool:
    cx, cy = box.center
    return region.x1 <= cx <= region.x2 and region.y1 <= cy <= region.y2


def _grip_corner_fraction(hand_box: BBox, screw_box: BBox) -> float:
    """杆有多大比例的面积落在手框右上角区域里。"""

    x1 = hand_box.x2 - hand_box.width * _GRIP_RX
    y2 = hand_box.y1 + hand_box.height * _GRIP_RY
    inter = BBox(x1, hand_box.y1, hand_box.x2, y2).intersection_area(screw_box)
    area = screw_box.width * screw_box.height
    return inter / area if area > 0 else 0.0


def _distance_to_long_axis(box: BBox, point: tuple[float, float]) -> float:
    """点到杆身中线的垂直距离；投影落在杆身外时按端点计（像素）。

    接触的物理含义是「刷子碰到杆身」，不是「碰到杆的中点」。丝杆框细长，
    贴着手持杆的中下部刷时工作点离杆心天然就有 0.05~0.07 对角线，按杆心
    算会有一半的帧被判成「没靠近」（见 docs/04）。
    """

    cx, cy = box.center
    if box.width >= box.height:
        over = max(box.x1 - point[0], 0.0, point[0] - box.x2)
        perp = point[1] - cy
    else:
        over = max(box.y1 - point[1], 0.0, point[1] - box.y2)
        perp = point[0] - cx
    return math.hypot(over, perp)


# ----------------------------------------------------------------------
# 单帧证据：两个检测器共用
# ----------------------------------------------------------------------
class Observation:
    """一帧里检测器需要的全部信息（检测、跟踪轨迹、ROI、帧尺寸）。"""

    def __init__(
        self,
        *,
        ts: float,
        frame_size: tuple[int, int],
        detections: tuple[Detection, ...],
        hands: tuple[Any, ...],
        brushes: tuple[Any, ...],
        tray: BBox | None,
        screw_history: deque[tuple[BBox, ...]],
        frame_scale: float,
        cfg: PipelineConfig,
        station: StationConfig,
    ) -> None:
        self.ts = ts
        self.width, self.height = frame_size
        self.detections = detections
        self.hands = hands
        self.brushes = brushes
        self.tray = tray
        self.screw_history = screw_history
        self.frame_scale = max(1.0, float(frame_scale))
        self.cfg = cfg
        self.station = station
        self.diag = max(1.0, math.hypot(self.width, self.height))

    @property
    def label_screw(self) -> str:
        return self.station.detector.screw_label

    def screws(self) -> list[Detection]:
        return [det for det in self.detections if det.label == self.label_screw]

    def in_tray(self, box: BBox) -> bool:
        return self.tray is not None and _center_in(box, self.tray)

    def screw_static(self, box: BBox) -> bool:
        """这根杆的位置是否没动过（与 8 帧前同位）。

        历史不足时返回 False（开局没有对照，不能凭它否定握持）。
        参考帧取窗口内**最早的两帧**，跳过单帧漏检；不能拿全部帧比，
        否则移动中的杆总能跟最近一两帧对上。
        """

        if len(self.screw_history) < _STATIC_BACK:
            return False
        window = list(self.screw_history)[-_STATIC_BACK:]
        return any(
            box.iou(past) >= _STATIC_IOU for frame in window[:2] for past in frame
        )

    def holding_screw(self, hand_box: BBox) -> Detection | None:
        """手里拿着的那根丝杆（严格版：带杆**出**盘才算）。

        1. 手在盘外时，杆心仍在 ``tray_roi`` 内的一律不算——那是盘里的料；
        2. 杆心落在手框内，或两者 IoU ≥ 0.15；
        3. 否则按虎口几何：杆与手框右上区相交；位置没动过的杆要求更高的
           重叠比例，动着的杆相交即可。
        """

        outside = not self.in_tray(hand_box)
        hx, hy = hand_box.center
        best: tuple[float, Detection] | None = None
        for det in self.screws():
            cx, cy = det.box.center
            distance = math.hypot(cx - hx, cy - hy) / self.diag
            if distance > self.cfg.hand_screw_max_distance:
                continue
            if outside and self.in_tray(det.box):
                continue
            gripped = _center_in(det.box, hand_box) or hand_box.iou(det.box) >= 0.15
            if not gripped and outside:
                if self.screw_static(det.box):
                    gripped = (
                        _grip_corner_fraction(hand_box, det.box)
                        >= _GRIP_PARKED_MIN_FRAC
                    )
                else:
                    frac = _grip_corner_fraction(hand_box, det.box)
                    gripped = frac > 0
            if not gripped:
                continue
            if best is None or distance < best[0]:
                best = (distance, det)
        return None if best is None else best[1]

    def held_rod_boxes(self) -> list[BBox]:
        """本帧被手握着、且刷头可能够得着的丝杆框（涂油检测的候选）。

        与 :meth:`holding_screw` 的口径不同：**不要求杆心出盘**。``tray_roi``
        是给取料判定划的，它下沿往往盖住工人涂油的位置；涂油时杆就压在
        盘区里刷是常态，按盘区排除会把这一段的接触全丢掉。

        握着的事实用「杆框与手框交叠」判断——握着时丝杆一定被手挡住，
        ``docs/05`` §5.0.1 的重新捕获用的也是这一条。这里不看位置稳不稳：
        杆被攥住不动时静止判据会失效。
        """

        hand_boxes = [hand.observation_box() for hand in self.hands]
        boxes: list[BBox] = []
        for det in self.screws():
            cx, cy = det.box.center
            for hand_box in hand_boxes:
                if not _overlaps(det.box, hand_box):
                    continue
                hx, hy = hand_box.center
                # 手框很大（实测 350×400 px），手伸在盘上方时会与整排杆交叠；
                # 再要求杆心在手附近，把「盘里那排存货」排除掉。
                if (
                    math.hypot(cx - hx, cy - hy) / self.diag
                    > self.cfg.hand_screw_max_distance
                ):
                    continue
                boxes.append(det.box)
                break
        return boxes


# ----------------------------------------------------------------------
# 取料检测
# ----------------------------------------------------------------------
@dataclass
class _Visit:
    """一只手的一次「进盘 — 离盘」。"""

    hand_id: int
    first_inside_ts: float
    entered: bool
    inside: bool = True
    out_frames: int = 0
    last_ts: float = 0.0
    last_box: BBox | None = None
    # 进盘前最近的盘外空手观测（True = 空手）。
    entry_outside: list[bool] = field(default_factory=list)
    entry_empty: bool = True
    # 离盘后攒到的「手里有杆」采样。
    exit_ts: float | None = None
    hold_samples: list[bool] = field(default_factory=list)
    hold_seen: bool = False
    last_outside_holding: bool | None = None
    # 进盘那一刻盘区里那批丝杆（存货快照）。
    inventory: tuple[BBox, ...] = ()
    frames_since_sample: int = 0


@dataclass
class PickAction:
    start_ts: float
    end_ts: float
    hand_track_id: int
    dwell_s: float
    rod_center: tuple[float, float] | None = None

    def to_dict(self, fps: float) -> dict[str, Any]:
        item = {
            "type": "PICK",
            "label": "取料",
            "start_ts": round(self.start_ts, 3),
            "end_ts": round(self.end_ts, 3),
            "start_frame": int(round(self.start_ts * fps)),
            "end_frame": int(round(self.end_ts * fps)),
            "dwell_in_tray_s": round(self.dwell_s, 3),
            "hand_track_id": self.hand_track_id,
        }
        if self.rod_center is not None:
            item["rod_center_norm"] = [
                round(self.rod_center[0], 4),
                round(self.rod_center[1], 4),
            ]
        return item


class PickDetector:
    """空手进盘 → 带杆离盘 ⇒ 一次取料。完全不看涂油状态。"""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.every = max(1, int(cfg.pick_sample_every_frames))
        self.keep = max(1, int(cfg.pick_history_samples))
        self.visits: dict[int, _Visit] = {}
        # 盘外观测历史（按手 id），用于进盘定性。
        self.outside_history: dict[int, deque[tuple[bool, float]]] = {}
        self.actions: list[PickAction] = []
        self._live: set[int] = set()

    # -- 内部工具 -----------------------------------------------------
    def _entry_grace_s(self, scale: float) -> float:
        # 与 pipeline._entry_grace_s 相同：低帧率下不超过 2s。
        return min(self.cfg.holding_lost_timeout_s * scale, 2.0)

    def _settle_window_s(self, scale: float) -> float:
        return (self.every * self.keep + 1) / max(1, self.cfg.max_fps) * scale

    def _carry_wait_s(self, scale: float) -> float:
        return _CARRY_OUT_OF_TRAY_MAX_S * scale

    def _record_outside(self, obs: Observation, hand_id: int, holding: bool) -> None:
        """记一次「手在盘外、手里有没有杆」。空手观测会清掉之前手持杆的历史。"""

        window = self._entry_grace_s(obs.frame_scale)
        history = self.outside_history.setdefault(hand_id, deque())
        if not holding:
            history.clear()
        history.append((holding, obs.ts))
        while history and obs.ts - history[0][1] > window:
            history.popleft()

    def _entry_is_empty(self, obs: Observation, hand_id: int) -> bool:
        """进盘前最近几次盘外观测里，「空手」是否占多数。证据不足按空手。"""

        history = self.outside_history.get(hand_id)
        window = self._entry_grace_s(obs.frame_scale)
        recent = (
            [holding for holding, ts in history if obs.ts - ts <= window]
            if history
            else []
        )
        recent = recent[-self.keep :]
        if len(recent) < 2:
            return True
        return sum(1 for holding in recent if not holding) * 2 >= len(recent)

    def _carrying_out(
        self, obs: Observation, hand_box: BBox, inventory: tuple[BBox, ...]
    ) -> bool:
        """手已出盘、但手边那根正被带走的杆还没出盘。

        这种帧不能记成「手里没杆」：杆心比手晚几帧出盘是常态。判据是
        「手边有随手动、杆心仍在盘区内的杆，且它不在进盘时的存货快照里」。
        """

        if obs.tray is None:
            return False
        hx, hy = hand_box.center
        for det in obs.screws():
            cx, cy = det.box.center
            if (
                math.hypot(cx - hx, cy - hy) / obs.diag
                > self.cfg.hand_screw_max_distance
            ):
                continue
            if not _center_in(det.box, obs.tray):
                continue
            if inventory:
                # 有存货快照时只按「是不是存货」判：工人把杆停在盘区下沿
                # 歇一下（杆暂时不动）也仍是正在带出去的那根，不能按静置排除。
                if any(det.box.iou(box) >= _INVENTORY_SAME_IOU for box in inventory):
                    continue
                return True
            if obs.screw_static(det.box):
                continue
            return True
        return False

    def _settle(self, obs: Observation, visit: _Visit, hand_box: BBox) -> None:
        if not visit.entry_empty:
            return
        if visit.exit_ts is None:
            return
        dwell = max(0.0, visit.exit_ts - visit.first_inside_ts)
        if dwell < self.cfg.pick_confirm_s:
            return
        if not (visit.hold_seen or _majority(visit.hold_samples)):
            return
        rod = obs.holding_screw(hand_box)
        self.actions.append(
            PickAction(
                start_ts=visit.first_inside_ts,
                end_ts=visit.exit_ts,
                hand_track_id=visit.hand_id,
                dwell_s=dwell,
                rod_center=(
                    None
                    if rod is None
                    else (rod.box.center[0] / obs.width, rod.box.center[1] / obs.height)
                ),
            )
        )

    # -- 主流程 -------------------------------------------------------
    def update(self, obs: Observation) -> None:
        self._live = set()
        for hand in obs.hands:
            hand_box = hand.observation_box()
            inside = obs.in_tray(hand_box)
            screw = obs.holding_screw(hand_box)
            holding = screw is not None
            hand_id = int(hand.track_id)

            visit = self.visits.get(hand_id)
            if visit is None:
                visit = _Visit(
                    hand_id=hand_id,
                    first_inside_ts=obs.ts if inside else 0.0,
                    entered=inside,
                    inside=inside,
                    last_ts=obs.ts,
                    last_box=hand_box,
                )
                self.visits[hand_id] = visit
            self._live.add(hand_id)
            visit.last_ts = obs.ts
            visit.last_box = hand_box
            visit.last_outside_holding = None if inside else holding

            if not inside:
                self._record_outside(obs, hand_id, holding)
            if inside:
                if visit.first_inside_ts <= 0.0:
                    visit.first_inside_ts = obs.ts
                if not visit.entered:
                    visit.entered = True
                    visit.entry_empty = self._entry_is_empty(obs, hand_id)
                    visit.inventory = obs.screw_history[-1] if obs.screw_history else ()
                visit.inside = True
                visit.out_frames = 0
                visit.exit_ts = None
                visit.hold_samples.clear()
                visit.hold_seen = False
                continue

            # ---- 手在盘外 ----
            visit.out_frames += 1
            if visit.exit_ts is not None:
                if self._carrying_out(obs, hand_box, visit.inventory):
                    if obs.ts - visit.exit_ts <= self._carry_wait_s(obs.frame_scale):
                        continue
                visit.frames_since_sample += 1
                if holding:
                    visit.hold_seen = True
                if visit.frames_since_sample >= self.every:
                    visit.frames_since_sample = 0
                    visit.hold_samples.append(holding)
                    if len(visit.hold_samples) > self.keep:
                        visit.hold_samples.pop(0)
                ready = len(visit.hold_samples) >= self.keep
                timed_out = obs.ts - visit.exit_ts > self._settle_window_s(
                    obs.frame_scale
                )
                if ready or timed_out:
                    self._settle(obs, visit, hand_box)
                    self.visits.pop(hand_id, None)
                continue

            if not visit.entered:
                continue
            # 离盘只按采样窗口确认：连续 out_frames 达到窗口，避免盘边抖动劈段。
            if visit.out_frames >= self.keep:
                visit.inside = False
                visit.exit_ts = obs.ts
                visit.hold_samples.clear()
                visit.hold_seen = holding
                visit.frames_since_sample = 0

    def finish(self, obs: Observation | None = None) -> None:
        """视频结束时把还挂着的进盘过程按已有证据结算。"""

        if obs is None:
            return
        for hand_id, visit in list(self.visits.items()):
            if hand_id in self._live:
                continue
            if obs.ts - visit.last_ts < self.cfg.holding_lost_timeout_s:
                continue
            if (
                visit.exit_ts is None
                and visit.entered
                and visit.last_outside_holding is not None
            ):
                # 手在盘外直接消失：最后观测在盘外就按那一刻离盘结算。
                visit.exit_ts = visit.last_ts
                if visit.last_outside_holding:
                    visit.hold_seen = True
            if visit.exit_ts is not None and visit.last_box is not None:
                self._settle(obs, visit, visit.last_box)
            self.visits.pop(hand_id, None)


# ----------------------------------------------------------------------
# 人工二次涂油检测
# ----------------------------------------------------------------------
@dataclass
class _OilingSession:
    """一段连续接触（刷头工作点持续贴近同一根手持丝杆）。"""

    anchor: tuple[float, float]
    start_ts: float
    last_ts: float
    cum_s: float = 0.0
    frames: int = 1
    min_distance: float = 1.0
    rod_box: BBox | None = None
    brush_track_id: int | None = None
    # 这段接触由几个「接触片段」拼成：刷手把手抬开又放回来时，中间那几帧
    # 既没有贴合的杆也没有接触，但仍是同一件料上的同一次涂油。
    parts: int = 1


@dataclass
class OilingAction:
    start_ts: float
    end_ts: float
    hold_s: float
    frames: int
    min_distance: float
    rod_center: tuple[float, float]
    parts: int = 1

    def to_dict(self, fps: float) -> dict[str, Any]:
        return {
            "type": "MANUAL_OILING",
            "label": "人工二次涂油",
            "start_ts": round(self.start_ts, 3),
            "end_ts": round(self.end_ts, 3),
            "start_frame": int(round(self.start_ts * fps)),
            "end_frame": int(round(self.end_ts * fps)),
            "contact_s": round(self.hold_s, 3),
            "contact_frames": self.frames,
            "min_distance_norm": round(self.min_distance, 4),
            "rod_center_norm": [
                round(self.rod_center[0], 4),
                round(self.rod_center[1], 4),
            ],
            "contact_parts": self.parts,
        }


class OilingDetector:
    """刷头贴着手持丝杆累计接触 ≥ ``brush_hold_s`` ⇒ 一次人工二次涂油。

    不依赖取料：视频开始时手里已经有杆、正在刷的，照样能判。

    「一次涂油」的边界由**杆**决定，不由时间决定：

    - 刷头靠近手持杆的帧按 ``dt`` 累加；
    - 中途刷手把手抬开、杆短暂离开手框（现场每件都有 0.5~1.3s 的遮挡），
      只要下一段贴合的仍是同一根杆（位置跳变 ≤ ``stitch_step_norm``），
      就接着累计，记一个接触片段；
    - 换了一根杆（跳变很大），即另起一次涂油。用位置而不是时间做这门限，
      是因为现场的间隔很不稳定：同一件料上抬手的空档可能 0.67s，不同料
      之间最短只有 1.07s，两者在时间上分不开。
    """

    def __init__(
        self,
        cfg: PipelineConfig,
        *,
        stitch_gap_s: float,
        stitch_step_norm: float,
        diag: float,
    ) -> None:
        self.cfg = cfg
        self.stitch_gap_s = max(0.0, float(stitch_gap_s))
        self.stitch_step_norm = max(0.0, float(stitch_step_norm))
        self.diag = max(1.0, float(diag))
        self.sessions: list[_OilingSession] = []
        self._active: _OilingSession | None = None

    def _bridge_gap_s(self, obs: Observation) -> float:
        """刷头/杆整帧漏检时允许补回的空档（与 ``brush_hold_max_gap_s`` 同口径）。"""

        return self.cfg.brush_hold_max_gap_s * obs.frame_scale

    def _close(self) -> None:
        if self._active is not None:
            self.sessions.append(self._active)
            self._active = None

    def _contacts(self, obs: Observation) -> list[tuple[float, Any, BBox]]:
        """本帧「刷头工作点贴近**手持**丝杆」的候选：(距离, 刷头轨迹, 杆框)。"""

        held = obs.held_rod_boxes()
        if not held:
            return []
        contacts: list[tuple[float, Any, BBox]] = []
        for brush in obs.brushes:
            point = brush_work_point(brush, obs.station.detector)
            for box in held:
                distance = _distance_to_long_axis(box, point) / obs.diag
                if distance <= self.cfg.brush_contact_distance:
                    contacts.append((distance, brush, box))
        contacts.sort(key=lambda item: item[0])
        return contacts

    def _same_rod(
        self, obs: Observation, anchor: tuple[float, float], box: BBox
    ) -> bool:
        cx, cy = box.center
        return (
            math.hypot(cx - anchor[0], cy - anchor[1]) / obs.diag
            <= self.stitch_step_norm
        )

    def update(self, obs: Observation) -> None:
        contacts = self._contacts(obs)
        matched: tuple[float, Any, BBox] | None = None
        if self._active is not None:
            # 只要还在跟同一根杆，不论中间空了几帧，优先按杆的位置续上。
            for item in contacts:
                if self._same_rod(obs, self._active.anchor, item[2]):
                    matched = item
                    break
            if matched is None and contacts:
                # 换了一根杆：这一段的接触到此为止，下一段另起一次涂油。
                self._close()
        if matched is None and self._active is None and contacts:
            matched = contacts[0]

        if matched is not None:
            distance, brush, box = matched
            if self._active is None:
                self._active = _OilingSession(
                    anchor=box.center,
                    start_ts=obs.ts,
                    last_ts=obs.ts,
                    min_distance=distance,
                    rod_box=box,
                    brush_track_id=int(brush.track_id),
                )
                return
            gap = max(0.0, obs.ts - self._active.last_ts)
            if gap > self._bridge_gap_s(obs):
                # 中间隔得比漏检桥接窗口还久：算一个新接触片段。
                self._active.parts += 1
            self._active.cum_s += min(gap, self._bridge_gap_s(obs))
            self._active.last_ts = obs.ts
            self._active.frames += 1
            self._active.anchor = box.center
            self._active.min_distance = min(self._active.min_distance, distance)
            self._active.rod_box = box
            self._active.brush_track_id = int(brush.track_id)
            return

        if self._active is None:
            return
        # 没有接触。短空档先挂着（可能是漏检或短暂抬手），超过拼接容差
        # 才算这一段结束。
        if contacts:
            self._close()
            distance, brush, box = contacts[0]
            self._active = _OilingSession(
                anchor=box.center,
                start_ts=obs.ts,
                last_ts=obs.ts,
                min_distance=distance,
                rod_box=box,
                brush_track_id=int(brush.track_id),
            )
            return
        if obs.ts - self._active.last_ts > self.stitch_gap_s * obs.frame_scale:
            self._close()

    def finish(self) -> None:
        self._close()

    def actions(self) -> list[OilingAction]:
        actions: list[OilingAction] = []
        for session in sorted(self.sessions, key=lambda item: item.start_ts):
            if session.cum_s < self.cfg.brush_hold_s - 1e-9:
                continue
            actions.append(
                OilingAction(
                    start_ts=session.start_ts,
                    end_ts=session.last_ts,
                    hold_s=session.cum_s,
                    frames=session.frames,
                    min_distance=session.min_distance,
                    rod_center=session.anchor,
                    parts=session.parts,
                )
            )
        return actions


# ----------------------------------------------------------------------
# 单段视频
# ----------------------------------------------------------------------
class OperationDetector:
    """把检测、跟踪、两个工序检测串起来；两个检测器互不引用。"""

    def __init__(
        self,
        *,
        station: StationConfig,
        cfg: PipelineConfig,
        detector: Any,
        stitch_gap_s: float = 1.0,
        stitch_step_norm: float = 0.06,
    ) -> None:
        self.station = station
        self.cfg = cfg
        self.detector = detector
        self.pick = PickDetector(cfg)
        # 涂油接触片段的拼接容差，帧尺寸在 process() 里补齐。
        self.oiling_stitch_gap_s = max(0.0, float(stitch_gap_s))
        self.oiling_stitch_step_norm = max(0.0, float(stitch_step_norm))
        self.oiling: OilingDetector | None = None
        self.screw_history: deque[tuple[BBox, ...]] = deque(maxlen=_STATIC_HISTORY_MAX)
        self._gap_samples: deque[float] = deque(maxlen=15)
        self._last_ts: float | None = None
        self._frame_scale = 1.0
        self._last_obs: Observation | None = None
        hand_kwargs = dict(
            max_distance=cfg.hand_track_max_distance,
            max_missing_s=cfg.hand_track_max_missing_s,
            min_hits=cfg.hand_track_min_hits,
            max_active=cfg.hand_track_max_active,
            cost_alpha=cfg.bytetrack_cost_alpha,
            match_thresh=cfg.bytetrack_match_thresh,
            high_thresh=cfg.bytetrack_high_thresh,
            low_thresh=cfg.bytetrack_low_thresh,
            new_thresh=cfg.bytetrack_new_thresh,
            buffer_frames=max(1, int(round(cfg.max_fps * cfg.bytetrack_buffer_s))),
            backend=cfg.hand_tracker_backend,
        )
        brush_kwargs = dict(hand_kwargs)
        brush_kwargs.update(
            max_distance=cfg.brush_track_max_distance,
            max_missing_s=cfg.brush_track_max_missing_s,
            max_active=cfg.brush_track_max_active,
            backend=cfg.brush_tracker_backend,
        )
        self.hand_tracker = ByteTrackObjectTracker(id_base=100_000, **hand_kwargs)
        self.brush_tracker = ByteTrackObjectTracker(id_base=200_000, **brush_kwargs)
        self.picks: list[PickAction] = []
        self.oilings: list[OilingAction] = []

    # -- ROI 过滤与延续跟踪 -------------------------------------------
    def _tray_box(self, frame_size: tuple[int, int]) -> BBox | None:
        width, height = frame_size
        if width <= 0 or height <= 0:
            return None
        return self.station.tray_roi.pixel_bbox(width, height)

    def _in_roi(
        self, box: BBox, frame_size: tuple[int, int], tray: BBox | None
    ) -> bool:
        width, height = frame_size
        return self.station.work_roi.contains_box_center(box, width, height) or (
            tray is not None and _center_in(box, tray)
        )

    def _feed(
        self, label: str, dets: list[Detection], frame_size, tray, ts, tracker
    ) -> tuple[Detection, ...]:
        """区内检测正常喂；区外但贴着已有轨迹的也喂（延续跟踪，不断轨）。"""

        kept = [det for det in dets if self._in_roi(det.box, frame_size, tray)]
        tracks = tuple(tracker.active_tracks())
        if not tracks:
            return tuple(kept)
        limit = min(self.cfg.roi_track_margin * self._frame_scale, 0.10)
        width, height = frame_size
        diag = max(1.0, math.hypot(width, height))
        kept_ids = {id(det) for det in kept}
        for det in dets:
            if id(det) in kept_ids:
                continue
            cx, cy = det.box.center
            for track in tracks:
                px, py = track.predicted_box(ts).center
                if math.hypot(cx - px, cy - py) / diag <= limit:
                    kept.append(det)
                    break
        return tuple(kept)

    # -- 每帧 ---------------------------------------------------------
    def process(self, detections: FrameDetections) -> Observation | None:
        ts = float(detections.ts)
        width, height = detections.frame_size
        if width <= 0 or height <= 0:
            return None
        if self._last_ts is not None:
            gap = ts - self._last_ts
            if gap <= 0:
                return self._last_obs
            self._gap_samples.append(gap)
            typical = statistics.median(self._gap_samples)
            self._frame_scale = max(
                1.0, min(self.cfg.frame_scale_max, typical * max(1, self.cfg.max_fps))
            )
        self._last_ts = ts
        frame_size = (int(width), int(height))
        tray = self._tray_box(frame_size)
        cfg = self.station.detector

        hands = self._feed(
            cfg.hand_label,
            [d for d in detections.detections if d.label == cfg.hand_label],
            frame_size,
            tray,
            ts,
            self.hand_tracker,
        )
        brushes = self._feed(
            cfg.brush_label,
            [d for d in detections.detections if d.label == cfg.brush_label],
            frame_size,
            tray,
            ts,
            self.brush_tracker,
        )
        hand_out = self.hand_tracker.update(
            ts=ts,
            detections=hands,
            frame_size=frame_size,
            distance_scale=self._frame_scale,
        )
        brush_out = self.brush_tracker.update(
            ts=ts,
            detections=brushes,
            frame_size=frame_size,
            distance_scale=self._frame_scale,
        )
        obs = Observation(
            ts=ts,
            frame_size=frame_size,
            detections=detections.detections,
            hands=tuple(
                t
                for t in hand_out.active
                if t.matched and t.hits >= self.hand_tracker.min_hits
            ),
            brushes=tuple(
                t
                for t in brush_out.active
                if t.matched and t.hits >= self.brush_tracker.min_hits
            ),
            tray=tray,
            screw_history=self.screw_history,
            frame_scale=self._frame_scale,
            cfg=self.cfg,
            station=self.station,
        )
        if self.oiling is None:
            # 涂油合并门限按画面对角线归一化，首次拿到帧尺寸时建实例。
            self.oiling = OilingDetector(
                self.cfg,
                stitch_gap_s=self.oiling_stitch_gap_s,
                stitch_step_norm=self.oiling_stitch_step_norm,
                diag=math.hypot(*frame_size),
            )
        self.pick.update(obs)
        self.oiling.update(obs)
        self.pick.finish(obs)
        # 丝杆历史在判定之后入库：_screw_history 表示「此前若干帧」。
        self.screw_history.append(
            tuple(d.box for d in detections.detections if d.label == cfg.screw_label)
        )
        self._last_obs = obs
        return obs

    def finish(self) -> None:
        if self.oiling is not None:
            self.oiling.finish()
            self.oilings = sorted(self.oiling.actions(), key=lambda item: item.start_ts)
        self.picks = sorted(self.pick.actions, key=lambda item: item.end_ts)

    def contact_segments(self, fps: float) -> list[dict[str, Any]]:
        """全部涂油接触片段（含未达 ``brush_hold_s`` 的），用于调阈值与复核。

        输出格式与 ``picks``/``oilings`` 的 JSON 一致（带帧号与杆心像素坐标），
        另外标出是否达到完成阈值。
        """

        if self.oiling is None:
            return []
        items: list[dict[str, Any]] = []
        for session in sorted(self.oiling.sessions, key=lambda item: item.start_ts):
            items.append(
                {
                    "start_ts": round(session.start_ts, 3),
                    "end_ts": round(session.last_ts, 3),
                    "start_frame": int(round(session.start_ts * fps)),
                    "end_frame": int(round(session.last_ts * fps)),
                    "contact_s": round(session.cum_s, 3),
                    "contact_frames": session.frames,
                    "contact_parts": session.parts,
                    "min_distance_norm": round(session.min_distance, 4),
                    "rod_center_px": [
                        round(session.anchor[0], 1),
                        round(session.anchor[1], 1),
                    ],
                    "confirmed": session.cum_s >= self.cfg.brush_hold_s - 1e-9,
                }
            )
        return items


# ----------------------------------------------------------------------
# 画框
# ----------------------------------------------------------------------
_COLORS = {
    "hand": (255, 159, 10),
    "brush": (255, 55, 95),
    "screw": (50, 215, 75),
    "oil_station": (255, 214, 10),
}
_PICK_COLOR = (0, 215, 255)
_OIL_COLOR = (255, 120, 0)


def _draw_frame(
    frame: np.ndarray,
    obs: Observation,
    *,
    flash_pick: bool,
    oil_progress: tuple[float, float] | None,
    title: str,
) -> np.ndarray:
    for det in obs.detections:
        x1, y1, x2, y2 = det.box.as_int_tuple()
        color = _COLORS.get(det.label, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame, title, (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6, cv2.LINE_AA
    )
    cv2.putText(
        frame,
        title,
        (24, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    y = 90
    if flash_pick:
        cv2.putText(
            frame,
            "PICK  取料",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "PICK  取料",
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            _PICK_COLOR,
            2,
            cv2.LINE_AA,
        )
        y += 44
    if oil_progress is not None:
        cum, need = oil_progress
        text = f"OIL  涂油 {cum:.2f}/{need:.2f}s"
        cv2.putText(
            frame,
            text,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            6,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            _OIL_COLOR,
            2,
            cv2.LINE_AA,
        )
    return frame


# ----------------------------------------------------------------------
# 单段视频跑一遍
# ----------------------------------------------------------------------
def run_video(
    path: Path,
    detector: Any,
    station: StationConfig,
    cfg: PipelineConfig,
    out_dir: Path,
    *,
    limit: int = 0,
    annotate: bool = True,
    stitch_gap_s: float = 1.0,
    stitch_step_norm: float = 0.06,
    save_contacts: bool = False,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频: {path}")
    native_fps = capture.get(cv2.CAP_PROP_FPS) or 15.0
    if native_fps <= 0 or native_fps > 240:
        native_fps = 15.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    stem = path.stem

    writer = None
    if annotate:
        writer = cv2.VideoWriter(
            str(out_dir / f"{stem}.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(native_fps),
            (width, height),
        )

    ops = OperationDetector(
        station=station,
        cfg=cfg,
        detector=detector,
        stitch_gap_s=stitch_gap_s,
        stitch_step_norm=stitch_step_norm,
    )

    # 预先不知道事件时刻，画框时按时间窗口闪一下即可（只影响可视化）。
    pick_flash: deque[float] = deque()
    frame_no = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        frame_no += 1
        ts = frame_no / native_fps
        detections = detector.detect(bgr, ts)
        obs = ops.process(detections)
        if obs is None:
            continue
        # 取料事件在 process 里产生；取最新一条用于闪烁提示。
        for action in ops.pick.actions:
            if action.end_ts > ts - 1e-6 and action.hand_track_id is not None:
                if not pick_flash or abs(pick_flash[-1] - action.end_ts) > 1e-6:
                    pick_flash.append(action.end_ts)
        while pick_flash and ts - pick_flash[0] > 1.2:
            pick_flash.popleft()
        if writer is not None:
            oil_progress = None
            active = ops.oiling._active if ops.oiling is not None else None
            if active is not None and ts - active.last_ts <= 0.5:
                oil_progress = (active.cum_s, float(cfg.brush_hold_s))
            frame = _draw_frame(
                bgr,
                obs,
                flash_pick=bool(pick_flash),
                oil_progress=oil_progress,
                title=f"{path.name}  f{frame_no}  {ts:6.2f}s",
            )
            writer.write(frame)
        if limit and frame_no >= limit:
            break
        if frame_no % 100 == 0:
            print(f"  ... {frame_no}/{total or '?'} 帧", flush=True)

    capture.release()
    if writer is not None:
        writer.release()
    ops.finish()

    result = {
        "video": path.name,
        "path": str(path),
        "fps": native_fps,
        "frames": frame_no,
        "duration_s": round(frame_no / native_fps, 3),
        "picks": [a.to_dict(native_fps) for a in ops.picks],
        "oilings": [a.to_dict(native_fps) for a in ops.oilings],
        "pick_count": len(ops.picks),
        "oiling_count": len(ops.oilings),
    }
    if save_contacts:
        result["oiling_contact_segments"] = ops.contact_segments(native_fps)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"\n=== {path.name} ===")
    print(
        f"  {width}x{height} @ {native_fps:.1f}fps, {frame_no} 帧, {frame_no / native_fps:.1f}s"
    )
    print(f"  取料 {len(ops.picks)} 次:")
    for action in ops.picks:
        print(
            f"    {action.start_ts:6.2f}s → {action.end_ts:6.2f}s"
            f"（盘内停留 {action.dwell_s:.2f}s，手 #{action.hand_track_id}）"
        )
    print(f"  人工二次涂油 {len(ops.oilings)} 次:")
    for action in ops.oilings:
        print(
            f"    {action.start_ts:6.2f}s → {action.end_ts:6.2f}s"
            f"（接触 {action.hold_s:.2f}s，最近距离 {action.min_distance:.3f}"
            f"，{action.parts} 个接触片段）"
        )
    print(f"  结果文件: {out_dir / f'{stem}.json'}")
    return result


# ----------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="独立检测取料与人工二次涂油")
    parser.add_argument(
        "--video", action="append", default=[], help="指定视频路径（可重复）"
    )
    parser.add_argument("--limit", type=int, default=0, help="每段视频最多处理帧数")
    parser.add_argument(
        "--out", default="", help="输出目录（默认 data/operations_preview/）"
    )
    parser.add_argument("--config", default="", help="配置目录（默认仓库 config/）")
    parser.add_argument("--no-annotate", action="store_true", help="不写带框视频")
    parser.add_argument(
        "--save-contacts",
        action="store_true",
        help="JSON 里额外写入全部涂油接触片段（含未达阈值的），便于调阈值",
    )
    parser.add_argument(
        "--stitch-gap",
        type=float,
        default=1.0,
        help="同一件料上刷手抬手、刷头离开的最大空档（秒，默认 1.0）",
    )
    parser.add_argument(
        "--stitch-step",
        type=float,
        default=0.06,
        help="拼接接触片段时允许的杆位置跳变（按画面对角线归一化，默认 0.06）",
    )
    args = parser.parse_args()

    service = OilCheckConfigService(Path(args.config) if args.config else None)
    cfg = service.load(create_if_missing=False)
    station = cfg.enabled_stations()[0] if cfg.stations else None
    if station is None:
        raise SystemExit(f"配置里没有启用的工位: {service.config_path}")
    if station.detector.kind != "yolo":
        raise SystemExit(
            f"工位 {station.station_id} 的 detector.kind={station.detector.kind}，"
            f"需要设为 yolo 才能离线检测。配置: {service.config_path}"
        )

    detector = create_detector(station)
    out_dir = Path(args.out) if args.out else REPO_ROOT / "data" / "operations_preview"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos: list[Path] = [Path(item) for item in args.video]
    if not videos:
        videos = [
            Path(resolve_runtime_file(video.path, repo_root=runtime_root()))
            for video in cfg.demo.videos
        ]
    if not videos:
        raise SystemExit(
            "没有可用视频：请用 --video 指定，或在 config/oilcheck.json 的 demo 里登记"
        )

    results: list[dict[str, Any]] = []
    for path in videos:
        if not path.exists():
            print(f"跳过不存在的视频: {path}")
            continue
        results.append(
            run_video(
                path,
                detector,
                station,
                cfg.pipeline,
                out_dir,
                limit=args.limit,
                annotate=not args.no_annotate,
                stitch_gap_s=args.stitch_gap,
                stitch_step_norm=args.stitch_step,
                save_contacts=args.save_contacts,
            )
        )
    detector.close()

    print("\n" + "=" * 66)
    print(f"{'视频':<34}{'取料':>6}{'涂油':>6}")
    print("-" * 66)
    for item in results:
        name = item["video"]
        print(f"{name:<34}{item['pick_count']:>6}{item['oiling_count']:>6}")
    print("-" * 66)
    print(
        f"{'合计':<34}"
        f"{sum(i['pick_count'] for i in results):>6}"
        f"{sum(i['oiling_count'] for i in results):>6}"
    )
    summary = out_dir / "summary.json"
    summary.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n汇总已写入: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
