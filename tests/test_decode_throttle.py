"""解码节流（hik/session_actor.py）。

现场症状：相机主码流 25fps，``app.json`` 的 ``decode_max_fps`` 也是 25，
服务侧实际只看到 8~12fps，日志里一小时上千条「视频时间戳异常：0.16s」。

根因是节流的判定方式：``now - last < 1/decode_max_fps`` 要求两帧间隔
**严格不小于** 40ms，而 25fps 的帧间隔正是 40ms 减去网络/PlayM4 抖动，
于是每第二帧被判「早到」丢掉、下一帧隔了 80ms 又能过 —— 恒定丢一半。
这里用纯逻辑用例把「按节拍推进」的行为钉住，不需要相机与 SDK。
"""

from __future__ import annotations

import random

from hik.session_actor import _DECODE_GATE_TOLERANCE, SessionActor


class _Throttle:
    """借用 SessionActor 的方法，绕开它的构造（要 SDK 上下文）。"""

    _decode_on_schedule = SessionActor._decode_on_schedule
    _decode_interval_s = lambda self: 1.0 / 25  # noqa: E731


def _run(frames: list[float], interval_last: float = 0.0) -> tuple[int, list[float]]:
    throttle = _Throttle()
    passed = 0
    last = interval_last
    accepted: list[float] = []
    for now in frames:
        allowed, last = throttle._decode_on_schedule(last, now)
        if allowed:
            passed += 1
            accepted.append(now)
    return passed, accepted


def test_25fps_stream_at_25fps_cap_keeps_every_frame():
    """相机 25fps、上限 25fps：抖动下也必须全部放行，不能恒定丢一半。

    用固定种子的抖动复现现场：帧间隔在 40ms 上下 ±6ms 波动。旧实现
    （``now - last < 40ms`` 就丢）在这里放行率约 50%。
    """

    rnd = random.Random(20261127)
    now = 0.0
    frames = []
    for _ in range(500):
        now += 0.040 + rnd.uniform(-0.006, 0.006)
        frames.append(now)

    passed, accepted = _run(frames)
    ratio = passed / len(frames)
    assert ratio > 0.95, f"25fps 流在 25fps 上限下只放行 {ratio:.0%}"
    # 放行帧的时间轴仍然单调，且平均节拍接近 40ms。
    assert accepted == sorted(accepted)
    span = accepted[-1] - accepted[0]
    assert abs(span / (len(accepted) - 1) - 0.040) < 0.005


def test_faster_than_cap_is_thinned_to_the_cap():
    """相机 50fps、上限 25fps：应压到约 25fps，而不是全放或全丢。"""

    frames = [index * 0.020 for index in range(500)]
    passed, _ = _run(frames)
    ratio = passed / len(frames)
    assert 0.45 <= ratio <= 0.55, f"50fps→25fps 放行率 {ratio:.0%}"


def test_early_frame_does_not_push_the_schedule_back():
    """早到的帧只被跳过，不能把下一帧的可放行时刻也往后挪。

    旧实现每次都把 ``last = now``，即使这帧被丢掉；现场抖动因此会一路
    累积。这里喂「半拍一个」的帧：正确的节流每拍放行一个（9 帧 ≈ 4~5 个），
    旧实现则会因为每次都把基准推后而全丢。
    """

    throttle = _Throttle()
    interval = 1.0 / 25
    # 第一帧放行，建立一个节拍点。
    allowed, last = throttle._decode_on_schedule(0.0, 100.0)
    assert allowed
    accepted = 0
    for i in range(1, 10):
        allowed, last = throttle._decode_on_schedule(
            last, 100.0 + i * interval * 0.5
        )
        accepted += int(allowed)
    # 4.5 拍 ⇒ 4~5 帧放行；关键是「不是 0」，且不会把节拍越推越后。
    assert 4 <= accepted <= 5, f"半拍喂帧只放行 {accepted} 帧"


def test_gap_tolerance_allows_a_slightly_early_frame():
    """留 10% 容差：只差一点点（>0.9 拍）视为到点，避免整拍被推后。"""

    throttle = _Throttle()
    interval = 1.0 / 25
    _, last = throttle._decode_on_schedule(0.0, 0.0)
    early = last + interval * (_DECODE_GATE_TOLERANCE - 0.01)
    allowed, _ = throttle._decode_on_schedule(last, early)
    assert allowed


def test_first_frame_always_passes():
    throttle = _Throttle()
    allowed, last = throttle._decode_on_schedule(0.0, 123.456)
    assert allowed and last == 123.456
