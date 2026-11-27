"""涂油：正在刷的那根杆整帧漏检时，按锚点补接触时长。

现场（示例 122925）一次涂油的接触被切成 0.2~0.3 秒的碎段：拳头压住杆、
那根杆整帧没检出，但**盘里别的杆还在**，所以走不到「一根杆都检不到」
的整帧遮挡桥接，累计停在 0.93 s 判不出来。补帧只应在「那根杆在锚点附近
确实不可见、刷头仍贴着锚点」时发生，且总补时长不超过
``brush_hold_max_gap_s × frame_scale``。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.oilcheck.pipeline import _OILING_ROD_OCCLUDED_RADIUS
from app.oilcheck.types import EventType
from tests.test_pipeline import (
    HAND_TARGET,
    brush_at,
    do_pick,
    feed,
    hand_at,
    held_screw_at,
    make_pipeline,
    screw_at,
    timestamps,
)

# 盘里另外那根「存货」：刷油时它一直在，正是让整帧遮挡桥接失效的原因。
OTHER_ROD = (HAND_TARGET[0] + 260.0, HAND_TARGET[1] + 260.0)


def _brush_frames(
    pipeline, ts_iter, *, frames: int, occlude_rod: bool
) -> None:
    """手持杆 + 刷头贴着刷；``occlude_rod`` 时手里那根杆整帧不出现。"""

    for _ in range(frames):
        ts = next(ts_iter)
        items = [
            hand_at(*HAND_TARGET),
            screw_at(*OTHER_ROD),
            brush_at(HAND_TARGET[0] + 10, HAND_TARGET[1]),
        ]
        if not occlude_rod:
            items.append(held_screw_at(*HAND_TARGET))
        feed(pipeline, ts, *items)


def _start_oiling(pipeline, ts_iter, *, seconds: float = 0.5, fps: float = 15.0):
    """先取一件料，再刷够一小段，让涂油段建立起来。"""

    do_pick(pipeline, ts_iter)
    # 重新对准 15fps 节拍：do_pick 用的是 0.2s 步长，而这里要按帧算。
    frames = int(round(seconds * fps))
    _brush_frames(pipeline, ts_iter, frames=frames, occlude_rod=False)


def _make(fps: float = 15.0):
    """15fps 的判定参数：``frame_scale`` 必须为 1，补帧上限才是可预期的。"""

    pipe, events = make_pipeline(max_fps=int(fps), brush_hold_s=1.0)
    return pipe, events


def _oils(events) -> list:
    return [e for e in events if e.event_type == EventType.MANUAL_OILING]


def test_occluded_rod_is_bridged_when_brush_stays_on_anchor():
    """刷头仍贴着、手里那根杆整帧漏检 ⇒ 补时间，接触攒够判出涂油。"""

    pipe, events = _make()
    ts_iter = timestamps(step=1 / 15.0)
    _start_oiling(pipe, ts_iter, seconds=0.5)
    # 杆被拳头完全压住 0.6s（9 帧）。盘里那根存货一直在。
    _brush_frames(pipe, ts_iter, frames=9, occlude_rod=True)
    # 杆重新可见再刷一会儿，确保写进同一次接触。
    _brush_frames(pipe, ts_iter, frames=6, occlude_rod=False)

    assert len(_oils(events)) == 1, "杆短暂漏检不应把一次涂油判丢"


def test_occlusion_bridging_is_capped():
    """杆一直不可见时补帧有上限，不能凭空把短接触凑成完成。"""

    pipe, events = _make()
    pipe.cfg = replace(pipe.cfg, brush_hold_s=3.0)
    ts_iter = timestamps(step=1 / 15.0)
    _start_oiling(pipe, ts_iter, seconds=0.5)
    # 之后杆一直不出现、刷头也早就离开了：补满上限就该收段。
    for _ in range(120):
        feed(pipe, next(ts_iter), hand_at(*HAND_TARGET), screw_at(*OTHER_ROD))
    assert _oils(events) == []


def test_rod_visible_at_anchor_is_not_bridged():
    """手里那根杆在锚点处仍被检出 ⇒ 这不是遮挡，不补时间。"""

    pipe, events = _make()
    ts_iter = timestamps(step=1 / 15.0)
    _start_oiling(pipe, ts_iter, seconds=0.5)
    # 杆一直在（只是不在手里），刷头也离开了：不做补帧。
    for _ in range(60):
        feed(pipe, next(ts_iter), hand_at(*HAND_TARGET), screw_at(*HAND_TARGET))
    assert _oils(events) == []


def test_occluded_radius_is_smaller_than_rod_track_step():
    """补帧的「不可见」半径要比「换了一根杆」的门限小，避免误并。"""

    from app.oilcheck.pipeline import _OILING_SAME_ROD_STEP

    assert _OILING_ROD_OCCLUDED_RADIUS <= _OILING_SAME_ROD_STEP


@pytest.mark.parametrize("fps", [15, 20])
def test_bridging_works_at_both_frame_rates(fps):
    """补帧按秒计，15/20fps 下都应判出同一次涂油。"""

    pipe, events = _make(fps)
    ts_iter = timestamps(step=1 / fps)
    _start_oiling(pipe, ts_iter, seconds=0.5, fps=fps)
    _brush_frames(pipe, ts_iter, frames=int(round(0.6 * fps)), occlude_rod=True)
    _brush_frames(pipe, ts_iter, frames=int(round(0.4 * fps)), occlude_rod=False)
    assert len(_oils(events)) == 1
