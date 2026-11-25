"""模拟 3 次完整工序，打印每一拍写出的字节与灯色；可选用 --write 直接写 PLC 看真灯。

**用途**

人肉验收灯色映射：桌面推演 → 看日志 → 上真机。这个工具把整条「等待取料 →
取料完成 → 涂油中 → 涂油完成/异常」的时间线按 200 ms 一拍跑一遍，每一拍都
经过真实的 ``PlcStepMapper``（配置里的 ``plc.steps`` / ``plc.hold_s`` 照常
生效），所以打印出来的就是主程序会写的东西。

默认**不连 PLC**，只打印。加 ``--write`` 才会真的写寄存器，让你盯着灯看。

**用法**

    # 只打印（推荐先跑这个，核对时间线）
    uv run python tools/plc_simulate.py

    # 真的写 PLC（必须先停主程序，该 PLC 只允许一条 Modbus 会话）
    uv run python tools/plc_simulate.py --write

    # 换保持窗 / 换节奏
    uv run python tools/plc_simulate.py --hold 3 --speed 2
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.oilcheck.config import OilCheckConfigService  # noqa: E402
from app.oilcheck.plc_io import (  # noqa: E402
    PlcEvidence,
    PlcFrame,
    PlcStepMapper,
)
from app.oilcheck.plc_link import (  # noqa: E402
    ModbusError,
    ModbusLinkError,
    PlcLink,
)
from app.oilcheck.types import TaskResult, TaskStage  # noqa: E402

_LAMP_CHAR = {"green": "绿", "yellow": "黄", "red": "红"}


@dataclass(frozen=True)
class _Segment:
    """时间线的一段：持续多久、这期间给 mapper 什么证据。"""

    label: str
    seconds: float
    evidence: PlcEvidence


def _cycle_segments(
    cycle: int, result: TaskResult, hold_s: float
) -> list[_Segment]:
    """构造一次完整工序的时间线。"""

    task_id = f"TASK-{cycle:03d}"
    picked = PlcEvidence(
        online=True,
        task_id=task_id,
        stage=TaskStage.PICKED,
        final_result=TaskResult.PROCESSING,
        brush_hold_s=0.0,
    )
    oiling = PlcEvidence(
        online=True,
        task_id=task_id,
        stage=TaskStage.PICKED,
        final_result=TaskResult.PROCESSING,
        brush_hold_s=1.5,
    )
    finished = PlcEvidence(
        online=True,
        task_id=task_id,
        stage=TaskStage.OILED,
        final_result=result,
        brush_hold_s=1.5,
    )
    return [
        _Segment("等待取料", 1.0, PlcEvidence(online=True)),
        _Segment("取料完成", 1.0, picked),
        _Segment("涂油中", 1.5, oiling),
        _Segment(
            "良品完成" if result == TaskResult.COMPLETED else "漏涂 NG（超时判定）",
            hold_s + 0.4,
            finished,
        ),
    ]


def _lamp_columns(frame: PlcFrame, lamps: int = 4) -> str:
    """把前几盏灯拼成一行，例如 ``  灭  绿  绿  黄``。"""

    parts = []
    for color in frame.lamps[:lamps]:
        parts.append(f"{_LAMP_CHAR.get(color, '灭'):>2}")
    return " ".join(parts)


def _write_with_reconnect(link: PlcLink, base: int, values: list[int]) -> None:
    """写一次，链路断了就重连再写（该 PLC 空闲 10~18s 会断连）。"""

    try:
        link.write_registers(base, values)
        return
    except ModbusLinkError:
        pass
    link.close()
    time.sleep(0.3)
    link.open()
    link.write_registers(base, values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="模拟 3 次完整工序并打印/写入 PLC 字节",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default=None,
        help="oilcheck.json 路径（默认仓库 config/oilcheck.json）",
    )
    parser.add_argument(
        "--hold", type=float, default=None,
        help="覆盖完成保持窗秒数；默认取配置的 plc.hold_s",
    )
    parser.add_argument(
        "--tick", type=float, default=0.2,
        help="模拟节拍秒数，默认 0.2（与现场 poll_interval_ms 一致）",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="只影响 --write 时的实际等待；1.0 = 实时，2.0 = 两倍速（默认 2.0）",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="真的写 PLC（会改变灯的状态；必须先停主程序）",
    )
    args = parser.parse_args()

    cfg_service = OilCheckConfigService(
        Path(args.config).parent if args.config else None
    )
    cfg = cfg_service.load()
    hold_s = cfg.plc.hold_s if args.hold is None else args.hold
    speed = max(0.1, float(args.speed))

    print(f"配置: {cfg_service.config_path}")
    print(
        f"PLC 目标: {cfg.plc.host}:{cfg.plc.port} unit={cfg.plc.unit_id} "
        f"基址={cfg.plc.register_base}"
    )
    print(
        f"灯位配置: {cfg.plc.steps}  保持窗: {hold_s:.1f}s  "
        f"结果灯: 常亮到下一个结论  节拍: {args.tick:.2f}s"
    )
    print(
        "编码: VB0.bitN=灯N+1 绿 / VB1.bitN=灯N+1 黄 / 同位置双置=红；"
        "HR1 与灯无关"
    )

    mapper = PlcStepMapper(hold_s=hold_s, steps=cfg.plc.steps)

    # 三次工序：良品 → 漏涂 NG → 良品
    plan = [
        (1, TaskResult.COMPLETED),
        (2, TaskResult.NG),
        (3, TaskResult.COMPLETED),
    ]

    link: PlcLink | None = None
    if args.write:
        print("\n⚠ --write：会真实写入 PLC，请确认主程序已停止（该 PLC 只允许一条会话）")
        link = PlcLink(
            cfg.plc.host,
            cfg.plc.port,
            unit=cfg.plc.unit_id,
            connect_timeout_s=3.0,
            request_timeout_s=2.0,
        )
        try:
            link.open()
            original = link.read_holding(cfg.plc.register_base, 2)
            print(
                f"  当前 HR0=0x{original[0]:04X} HR1=0x{original[1]:04X}"
                f"（结束后写回）"
            )
        except ModbusLinkError as exc:
            print(f"  ✗ 连接失败：{exc}")
            print("    最常见原因：主程序还在跑，占着唯一的那条 Modbus 会话。")
            return 1
    else:
        original = None
        print("\n（只打印，不连 PLC；加 --write 可写真机看灯）")

    header = (
        f"\n{'时间':>6}  {'工序/状态':<22}"
        f"{'灯1':>3}{'灯2':>4}{'灯3':>4}{'灯4':>4}"
        f"   {'VB0':<6}{'VB1':<6}{'HR0':<8}{'HR1':<8}说明"
    )
    print(header)
    print("-" * len(header.expandtabs()))

    clock = 0.0
    last_registers: tuple[int, int] | None = None
    changes = 0
    tick = max(0.05, float(args.tick))

    try:
        for cycle, result in plan:
            segments = _cycle_segments(cycle, result, hold_s)
            print(f"\n=== 第 {cycle} 次工序（{result.value}）===")
            for seg in segments:
                ticks = max(1, round(seg.seconds / tick))
                for _ in range(ticks):
                    frame = mapper.update(seg.evidence, clock)
                    if frame.registers != last_registers:
                        last_registers = frame.registers
                        changes += 1
                        print(
                            f"{clock:6.1f}  {seg.label:<22}"
                            f"{_lamp_columns(frame):>15}"
                            f"   0x{frame.vb0:02X}  0x{frame.vb1:02X}"
                            f"  0x{frame.hr0:04X}  0x{frame.hr1:04X}"
                            f"  {frame.reason}"
                        )
                        if link is not None:
                            _write_with_reconnect(
                                link, cfg.plc.register_base, list(frame.registers)
                            )
                    elif link is not None:
                        # 真实写入每拍都写（与主程序的 1s 重写一致），
                        # 保证连接不因空闲断掉。
                        _write_with_reconnect(
                            link, cfg.plc.register_base, list(frame.registers)
                        )
                    if link is not None:
                        time.sleep(tick / speed)
                    clock += tick

        # 收尾：让 mapper 收到「没有任务」，回到等待取料
        for _ in range(30):
            frame = mapper.update(PlcEvidence(online=True), clock)
            if frame.registers != last_registers:
                last_registers = frame.registers
                changes += 1
                print(
                    f"{clock:6.1f}  {'收尾·等待取料':<22}"
                    f"{_lamp_columns(frame):>15}"
                    f"   0x{frame.vb0:02X}  0x{frame.vb1:02X}"
                    f"  0x{frame.hr0:04X}  0x{frame.hr1:04X}"
                    f"  {frame.reason}"
                )
                if link is not None:
                    _write_with_reconnect(
                        link, cfg.plc.register_base, list(frame.registers)
                    )
            if link is not None:
                time.sleep(tick / speed)
            clock += tick
    finally:
        if link is not None:
            if original is not None:
                try:
                    _write_with_reconnect(link, cfg.plc.register_base, original)
                    print(
                        f"\n已写回原值 HR0=0x{original[0]:04X} "
                        f"HR1=0x{original[1]:04X}"
                    )
                except (ModbusLinkError, ModbusError) as exc:
                    print(f"\n⚠ 写回原值失败（{exc}），请手动在 PLC 在线监视里恢复")
            link.close()

    print("\n" + "=" * 76)
    print("预期灯色序列（每盏灯只有四种状态：灭 / 绿 / 黄 / 红）")
    print("=" * 76)
    print("  等待取料    灯2 黄")
    print("  取料完成    灯2、灯3 绿")
    print("  涂油中      灯2、灯3 绿，灯4 黄")
    print("  良品完成    灯2、灯3、灯4 绿   ← 保持窗后回「灯2 黄」")
    print("  漏涂 NG     灯2、灯3 绿，灯4 红 ← 保持窗后回「灯2 黄」")
    print(f"\n本次一共发生 {changes} 次字节变化。")
    print(
        "如果实际灯色与上面不符，跑 tools/plc_lamp_probe.py 复测逐位编码，"
        "再把结果发我。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
