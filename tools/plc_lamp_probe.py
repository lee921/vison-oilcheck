"""PLC 灯色探针：逐位点亮 HR0/HR1 的每一位，由现场人员记录灯的实际颜色。

**为什么需要这个工具**

协议表把 4 个字节解释成「VB0=绿、VB1=黄、VB2=红」，但现场实测与之不符：
写 `HR0=0x0100`（按 lsb 读即 `VB1=0x01`）时灯亮**红**，而写 `VB0`/`VB2` 的
位时灯没反应。这说明「字节 = 颜色」的读法不成立，或者字节序相反，或者灯
由 PLC 内部逻辑按别的规则驱动。

光看代码分不出来。这个工具把 **HR0 和 HR1 的每一位单独点亮一次**，停留
几秒，让你直接看「哪一位 → 哪盏灯什么颜色」。拿到这张对照表，映射才有
依据，代码才谈得上改。

**前提：主程序不能占用 PLC 会话**

该 PLC 只允许一条 Modbus TCP 会话。主程序里把 `plc.enabled` 设为 false
（默认就是 false），PLC 线程根本不会启动、也不会连 PLC；此时你可以把主
程序开着用相机看灯，同时运行本工具。

若仍报「连接失败」，说明还有别的客户端占着会话（主程序 PLC 未关、博途在
线监控、组态软件等），先用下面的命令确认没有残留连接::

    ss -tn | grep 192.168.1.88:502

**用法**

交互模式（推荐，按回车推进，不赶时间）::

    uv run python tools/plc_lamp_probe.py

定时模式（每条停留 3 秒）::

    uv run python tools/plc_lamp_probe.py --hold 3

只看 HR0 的 8 位::

    uv run python tools/plc_lamp_probe.py --bits 8

结束时会把开始时读到的原值写回。**归类结果请记下来填进工具末尾的表**。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.oilcheck.plc_link import (  # noqa: E402
    ModbusError,
    ModbusLinkError,
    PlcLink,
)

# 协议表里 4 个字节对应两个寄存器。逐位走查只碰这两个。
_VB_NAMES = {0: "VB0", 1: "VB1", 2: "VB2", 3: "VB3"}


def _bit_name(reg: int, bit: int) -> str:
    """``HR0 bit8`` → 同时给出按 lsb / msb 的两种字节位读法。"""

    vb_lsb = reg * 2 + (0 if bit < 8 else 1)
    vb_msb = reg * 2 + (1 if bit < 8 else 0)
    bit_lsb = bit % 8
    return (
        f"HR{reg} bit{bit:<2d}"
        f"  lsb→{_VB_NAMES.get(vb_lsb, '?')}.bit{bit_lsb}"
        f"  msb→{_VB_NAMES.get(vb_msb, '?')}.bit{bit_lsb}"
    )


def _plan_combos() -> list[tuple[str, int, int]]:
    """组合走查：**同一盏灯同时给绿+黄**，检验「多信号=红」。

    逐位走查已经证明：VB0=绿、VB1=黄、HR1 完全无反应。而现场约定里有
    「同时给同一个位置多个信号为红」，所以红灯极可能是**同一位在 VB0 与
    VB1 上都置 1** 时触发的。这一组把每盏灯依次这样点一遍。
    """

    steps: list[tuple[str, int, int]] = [("全零基线", 0x0000, 0x0000)]
    for bit in range(8):
        mask = (1 << bit) | (1 << (bit + 8))  # VB0.bitN + VB1.bitN 同时置位
        steps.append(
            (
                f"灯{bit + 1} 同置 VB0.bit{bit}+VB1.bit{bit}（期望红）",
                mask,
                0x0000,
            )
        )
    # 三色灯常见的另一种编码：同一盏灯的三种颜色占三个连续 VD 的同一位置。
    # 既然 HR1 无反应，这里再确认「HR0 全绿」「HR0 全黄」「HR0 全绿+全黄」。
    steps += [
        ("VB0 全置（8 盏绿）", 0x00FF, 0x0000),
        ("VB1 全置（8 盏黄）", 0xFF00, 0x0000),
        ("VB0+VB1 全置（8 盏红？）", 0xFFFF, 0x0000),
        ("全零基线（收尾）", 0x0000, 0x0000),
    ]
    return steps


def _plan(bits: int, regs: int, from_bit: int = 0) -> list[tuple[str, int, int]]:
    """生成走查表：(说明, HR0, HR1)。"""

    steps: list[tuple[str, int, int]] = [("全零基线（三盏应全灭）", 0x0000, 0x0000)]
    for reg in range(regs):
        for bit in range(from_bit, bits):
            values = [0x0000, 0x0000]
            values[reg] = 1 << bit
            steps.append((_bit_name(reg, bit), values[0], values[1]))
    steps.append(("全零基线（收尾）", 0x0000, 0x0000))
    return steps


def _ask_enter(prompt: str = "  看完按回车继续（q 退出并写回原值）... ") -> bool:
    try:
        answer = input(prompt).strip().lower()
    except EOFError:
        return False
    return answer not in ("q", "quit", "exit")


def _write_with_reconnect(
    link: PlcLink, base: int, values: list[int]
) -> None:
    """写一次，链路断了就重连再写一次。

    该 PLC **空闲 10~18 秒会断连，而且断得不干净**（socket 收不到 FIN），
    交互模式下每步都要等人看灯，必然超过这个窗口。所以每一步写入前都容忍
    一次断链：关掉旧 socket、重新建连、再写。

    重连本身有坑：实测「同一时刻连发第二条建连要等约 1 s」，所以断链后
    稍等一下再连，别贴着上一个 socket 的时间戳建连。
    """

    try:
        link.write_registers(base, values)
        return
    except ModbusLinkError:
        pass
    link.close()
    time.sleep(0.3)  # 避开「同一时刻第二次建连要等 1s」那段时间
    link.open()
    link.write_registers(base, values)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PLC 灯色逐位探针（会真实写入 HR0/HR1）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="192.168.1.88", help="PLC IP")
    parser.add_argument("--port", type=int, default=502, help="Modbus TCP 端口")
    parser.add_argument("--unit", type=int, default=1, help="从站 unit id")
    parser.add_argument("--base", type=int, default=0, help="保持寄存器首地址")
    parser.add_argument(
        "--bits", type=int, default=16, choices=(8, 16),
        help="每个寄存器走查到第几位（16 = 全走，默认；8 = 只走低字节）",
    )
    parser.add_argument(
        "--from-bit", type=int, default=0, choices=range(0, 16),
        help="从第几位开始走（8 = 只走高字节 VB1/VB3），默认 0",
    )
    parser.add_argument(
        "--regs", type=int, default=2, choices=(1, 2),
        help="走查几个寄存器（1 = 只 HR0，2 = HR0+HR1），默认 2",
    )
    parser.add_argument(
        "--mode",
        choices=("bits", "combos"),
        default="bits",
        help="bits = 逐位点亮（默认）；combos = 同位置绿+黄组合，检验「多信号=红」",
    )
    parser.add_argument(
        "--hold", type=float, default=0.0,
        help="定时模式：每条停留秒数。0 = 交互模式（按回车推进），默认 0",
    )
    parser.add_argument(
        "--no-restore", action="store_true",
        help="结束后不写回原值（默认写回）",
    )
    args = parser.parse_args()

    interactive = args.hold <= 0
    steps = _plan_combos() if args.mode == "combos" else _plan(
        args.bits, args.regs, args.from_bit
    )
    total = len(steps)

    # ---- 连接 ----
    link = PlcLink(
        args.host, args.port, unit=args.unit,
        connect_timeout_s=3.0, request_timeout_s=2.0,
    )
    try:
        link.open()
    except ModbusLinkError as exc:
        print(f"✗ 连接失败：{exc}\n")
        print("  该 PLC 只允许一条 Modbus TCP 会话。请确认：")
        print("    1. 主程序里 plc.enabled 是 false（或主程序已退出）")
        print("    2. 没有博途在线监控 / 组态软件 / 其它脚本连着它")
        print("    3. 用 `ss -tn | grep 192.168.1.88:502` 看有没有残留连接")
        return 1

    original: list[int] | None = None
    results: list[tuple[str, int, int, str]] = []
    try:
        original = link.read_holding(args.base, 2)
        print(
            f"当前 HR{args.base}=0x{original[0]:04X} "
            f"HR{args.base + 1}=0x{original[1]:04X}（结束时写回）"
        )
        print(
            "按 lsb 读："
            f"VB0=0x{original[0] & 0xFF:02X} VB1=0x{original[0] >> 8:02X} "
            f"VB2=0x{original[1] & 0xFF:02X} VB3=0x{original[1] >> 8:02X}"
        )
        print(
            "\n接下来逐位点亮，共 "
            f"{total} 步。每步请看三盏灯，记下实际颜色。"
        )
        if interactive:
            print("（按回车推进；输入 q 提前结束并写回原值）\n")
        else:
            print(f"（每步停留 {args.hold:.1f}s）\n")

        for index, (name, hr0, hr1) in enumerate(steps, start=1):
            values = [hr0 & 0xFFFF, hr1 & 0xFFFF]
            # 交互模式下人会停留很久，该 PLC 空闲 10~18s 就断连，所以每步
            # 都容忍一次断链重连（见 _write_with_reconnect）。
            _write_with_reconnect(link, args.base, values)
            back = link.read_holding(args.base, 2)
            echo = "✓" if back == values else f"✗ 回读 {[hex(v) for v in back]}"
            print(
                f"[{index:2d}/{total}] {name}\n"
                f"         写入 HR0=0x{values[0]:04X} HR1=0x{values[1]:04X}  {echo}"
            )
            if interactive:
                if not _ask_enter():
                    print("  提前结束。\n")
                    break
                # 人看完灯后把实际颜色填进来，最后汇总
                seen = input("  实际灯色（直接回车 = 跳过）：").strip()
            else:
                time.sleep(args.hold)
                seen = ""
            results.append((name, hr0, hr1, seen))

    except (ModbusLinkError, ModbusError) as exc:
        print(f"✗ 通信失败：{exc}")
        print("  已写过的步骤仍在走查表里，可以把结果整理给我。")
        return 1
    finally:
        if original is not None and not args.no_restore:
            try:
                _write_with_reconnect(link, args.base, original)
                back = link.read_holding(args.base, 2)
                print(f"已写回原值 HR0=0x{back[0]:04X} HR1=0x{back[1]:04X}")
            except (ModbusLinkError, ModbusError) as exc:
                print(f"⚠ 写回原值失败（{exc}），请手动在 PLC 在线监视里恢复")
        link.close()

    # ---- 结果表：留给现场填 ----
    print("\n" + "=" * 74)
    print("走查记录表（把「实际看到的灯色」填到最后一列）")
    print("=" * 74)
    print(f"{'步':<4}{'写入 HR0':<10}{'写入 HR1':<10}{'按 lsb 的字节位':<26}{'实际灯色'}")
    print("-" * 74)
    for index, (name, hr0, hr1, seen) in enumerate(results, start=1):
        short = name.split("  ")[0]
        print(f"{index:<4}0x{hr0:04X}    0x{hr1:04X}    {short:<26}{seen}")
    print("-" * 74)

    print(
        """
怎么读这张表（把观察结果对号入座）：

  · 只有某一位亮红，其它位全灭
        → 红灯由这一位驱动。看它是 VB? 的哪一位，重排红/黄/绿的字节映射。
  · HR0=0x0001（lsb 的 VB0.bit0）与 HR0=0x0100（lsb 的 VB1.bit0）表现不同
        → 字节序可以定死：哪个是 VB0、哪个是 VB1 一目了然。
  · 单独一位点亮时灯「不亮」，但某个组合亮
        → 灯由组合逻辑驱动，需要找组合（工具已覆盖 5 个真实状态，见下）。
  · 全零基线也亮红
        → 红色是 PLC 侧的默认/锁存态，与写入值无关，需要 PLC 侧清零或给真值表。
  · 逐位走查全都不亮
        → 灯不由 HR0/HR1 驱动，请 PLC 侧给出灯的真实输出地址。
"""
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
