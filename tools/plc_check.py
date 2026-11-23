"""PLC（警示灯）Modbus TCP 只读体检工具。

用途：现场确认「PLC 地址通不通、寄存器怎么映射、工步位对不对、链路稳不稳」。
**只读，不写任何寄存器**——写入联调请用主程序（``plc.enabled`` + ``dry_run``）。

用法：

    uv run python tools/plc_check.py
    uv run python tools/plc_check.py --host 192.168.1.88 --port 502
    uv run python tools/plc_check.py --watch 30     # 持续观察 30 秒
    uv run python tools/plc_check.py --byte-order msb   # 按相反字节序解读

输出：网络可达性、功能码支持情况、地址空间边界、VB0~VB3 的字节解读与置位
工步、请求时延、持续观察中的任何变化。传输层复用
:class:`app.oilcheck.plc_link.PlcLink`，与主程序是同一份协议实现。
"""

from __future__ import annotations

import argparse
import struct
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

# 协议表里定义的 4 个字节（VB0~VB3）。本方案给每盏三色灯驱动一种颜色，
# 位 = 灯号（bit(n-1) → 第 n 盏灯）。VB3 恒 0（PLC 只显示，不输入）。
VB_NAMES = {
    0: "绿（本灯亮的绿信号）",
    1: "黄（本灯亮的黄信号；与绿同位置 → 红）",
    2: "与灯无关（实测写任何位灯都不动，本项目恒 0）",
    3: "与灯无关（本项目恒 0）",
}


def _lamp_view(vbs: dict[int, int]) -> list[str]:
    """把 VB0/VB1 按现场实测的编码解成 8 盏灯的颜色。

    实测（docs/12 §3.4）：VB0.bitN → 灯 N+1 绿；VB1.bitN → 灯 N+1 黄；
    同一位两者都为 1 → 灯 N+1 红；VB2/VB3 与灯无关。
    """

    out: list[str] = []
    for index in range(8):
        bit = 1 << index
        green = bool(vbs.get(0, 0) & bit)
        yellow = bool(vbs.get(1, 0) & bit)
        if green and yellow:
            color = "红"
        elif yellow:
            color = "黄"
        elif green:
            color = "绿"
        else:
            color = "灭"
        if color != "灭":
            out.append(f"灯{index + 1}={color}")
    return out or ["全灭"]


def _byte_view(words: list[int], order: str) -> dict[int, int]:
    """把保持寄存器按指定字节序拆成 VB 字节。

    ``order``：
    - ``"lsb"``：寄存器低字节是编号较小的 VB 号。现场协议表的四个例子
      （0x0001 / 0x0500 / 0x0005 / 0x0900）全部只在这种读法下成立，
      见 ``docs/12-PLC警示灯通信与控制.md`` §3.2；
    - ``"msb"``：反过来。仅在现场确认字节序相反时使用。
    """

    out: dict[int, int] = {}
    for index, word in enumerate(words):
        hi, lo = (word >> 8) & 0xFF, word & 0xFF
        first, second = (hi, lo) if order == "msb" else (lo, hi)
        out[index * 2] = first
        out[index * 2 + 1] = second
    return out


def _bits(value: int) -> str:
    return " ".join(f"{(value >> shift) & 1}" for shift in range(7, -1, -1))


def _describe(words: list[int], order: str) -> list[str]:
    vbs = _byte_view(words, order)
    lines = []
    for vb in range(4):
        value = vbs.get(vb, 0)
        name = VB_NAMES.get(vb, "")
        steps = [str(step + 1) for step in range(8) if (value >> step) & 1]
        lines.append(
            f"    VB{vb} = 0x{value:02X} [{_bits(value)}] {name}"
            f"   置位的灯: {', '.join(steps) if steps else '无'}"
        )
    lines.append(f"    → 解读：{', '.join(_lamp_view(vbs))}")
    return lines


def _print_expected_frames() -> None:
    """打印每一拍应该写出的字节与灯色，供与 PLC 在线监视逐位对照。"""

    print("\n== 期望值（本项目写入的字节与灯色）==")
    print("    状态             VB0  VB1   HR0     HR1    点亮的灯      说明")
    rows = [
        ("等待取料", 0x00, 0x02, "灯2 黄：取料进行中"),
        ("取料完成", 0x06, 0x00, "灯2、灯3 绿"),
        ("涂油中", 0x06, 0x08, "灯2、灯3 绿 + 灯4 黄"),
        ("涂油完成(良)", 0x0E, 0x00, "灯2、灯3、灯4 全绿"),
        ("涂油异常(NG)", 0x0E, 0x08, "灯4 同时绿+黄 → 红"),
        ("未检测", 0x00, "保持", "只灭纯黄，绿灯保留"),
    ]
    for label, vb0, vb1, note in rows:
        if isinstance(vb1, int):
            hr0 = (vb0 & 0xFF) | (vb1 << 8)
            lamps = " ".join(_lamp_view({0: vb0, 1: vb1}))
            print(
                f"    {label:<14} 0x{vb0:02X} 0x{vb1:02X}  0x{hr0:04X}  0x0000"
                f"  {lamps:<14} {note}"
            )
        else:
            print(f"    {label:<14} 0x{vb0:02X} 保持    -       -      "
                  f"{'':<14} {note}")
    print(
        "    位 = 灯号（bit(n-1) → 第 n 盏灯）："
        "VB0.bitN = 灯 N+1 绿，VB1.bitN = 灯 N+1 黄"
    )
    print(
        "    同一位 VB0 与 VB1 同时为 1 → 灯 N+1 红（红不在 HR1！）。"
        "灯1 只能红，工步从灯2 起用。"
    )


def check(
    host: str,
    port: int,
    unit: int,
    watch_s: float,
    *,
    byte_order: str = "lsb",
    allow_write_probe: bool = False,
) -> int:
    print(f"== 1. TCP 可达性 {host}:{port} ==")
    t0 = time.perf_counter()
    link = PlcLink(host, port, unit=unit, connect_timeout_s=2.0, request_timeout_s=2.0)
    try:
        link.open()
    except ModbusLinkError as exc:
        print(f"  ✗ 连接失败：{exc}")
        print("  排查：网线/网段、本机是否有 192.168.1.x 地址、是否被其它主站独占")
        print("  （该 PLC 只允许一条 Modbus TCP 会话，实测第二条连接会 SYN 超时）")
        return 1
    print(f"  ✓ 建连成功（{(time.perf_counter() - t0) * 1000:.1f} ms）")

    try:
        print("\n== 2. 功能码支持 ==")
        # 读类功能码可以安全试探；写类功能码会真的改 addr=0，
        # 默认跳过，避免体检把 PLC 状态改坏。
        probes: list[tuple[str, bytes, bool]] = [
            ("FC1  读线圈        ", struct.pack(">BHH", 1, 0, 8), False),
            ("FC2  读离散输入    ", struct.pack(">BHH", 2, 0, 8), False),
            ("FC4  读输入寄存器  ", struct.pack(">BHH", 4, 0, 4), False),
            ("FC43 读设备标识    ", struct.pack(">BBBBB", 43, 14, 1, 0, 0), False),
            ("FC6  写单寄存器    ", struct.pack(">BHH", 6, 0, 0), True),
            (
                "FC16 写多寄存器    ",
                struct.pack(">BHHB", 16, 0, 1, 2) + b"\x00\x00",
                True,
            ),
        ]
        for label, pdu, is_write in probes:
            if is_write and not allow_write_probe:
                print(f"  {label} -> 跳过（写操作，需 --allow-write-probe）")
                continue
            print(f"  {label} -> {link.probe(pdu)}")

        print("\n== 3. 保持寄存器地址边界（FC3）==")
        valid = []
        for start in (0, 4, 8, 16, 20, 24, 100, 1000, 8192):
            try:
                regs = link.read_holding(start, 1)
                valid.append(start)
                print(f"  addr={start:5d} -> 0x{regs[0]:04X}  ✓")
            except ModbusError as exc:
                print(f"  addr={start:5d} -> {exc}")
            except ModbusLinkError as exc:
                print(f"  addr={start:5d} -> 链路错误 {exc}")
                return 1
        if not valid:
            print(
                f"  ✗ 没有可读寄存器，检查 unit id（当前 {unit}）"
                "与地址基址（从 0 开始）"
            )
            return 1

        print(f"\n== 4. VB0~VB3 当前值（按 {byte_order} 解读）==")
        words = link.read_holding(0, 2)
        print(f"  原始保持寄存器：HR0=0x{words[0]:04X}  HR1=0x{words[1]:04X}")
        for line in _describe(words, byte_order):
            print(line)
        if byte_order == "lsb":
            print("  另一种解读（msb，仅供对照）：")
            for line in _describe(words, "msb"):
                print(line)
        if words[1] & 0xFF == 0xFF:
            print("  ⚠ HR1 低字节为 0xFF（= VB2 八位全 1，三盏灯全红）这类残留值，")
            print("    说明这段数据区当前没有 PLC 侧生产者（上位机还没接管），")
            print("    请先看现场警示灯的实际状态。")
        print("  ⚠ 仍需一次现场确认字节序：上位机空闲时写 VB1=0x01（灯1 黄），")
        print("    期望 HR0 == 0x0100；若读到 0x0001 则字节序要改成 msb。")

        print("\n== 5. 全量寄存器（VB0~VB19）==")
        try:
            allv = link.read_holding(0, 20)
            for index, word in enumerate(allv):
                if word:
                    print(f"  HR{index:<2d} = 0x{word:04X}")
            if not any(allv):
                print("  全部为 0")
        except ModbusError as exc:
            print(f"  读 0..19 失败：{exc}")

        print("\n== 6. 请求时延（同一连接 20 次）==")
        lat = []
        for _ in range(20):
            t0 = time.perf_counter()
            link.read_holding(0, 2)
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        print(
            f"  min={lat[0]:.1f} p50={lat[len(lat) // 2]:.1f} "
            f"p95={lat[int(len(lat) * 0.95)]:.1f} max={lat[-1]:.1f} ms"
        )

        if watch_s > 0:
            print(f"\n== 7. 持续观察 {watch_s:.0f}s（每 0.5s 读一次）==")
            prev = None
            deadline = time.time() + watch_s
            failures = 0
            while time.time() < deadline:
                try:
                    current = tuple(link.read_holding(0, 2))
                except (ModbusError, ModbusLinkError) as exc:
                    failures += 1
                    print(f"  读取失败：{exc}")
                    time.sleep(1.0)
                    continue
                if current != prev:
                    print(
                        f"  HR0=0x{current[0]:04X} HR1=0x{current[1]:04X}"
                        f"   （变化于 {time.strftime('%H:%M:%S')}）"
                    )
                    prev = current
                time.sleep(0.5)
            if failures == 0:
                print("  观察期内无读取失败")
    finally:
        link.close()

    _print_expected_frames()
    print("\n== 结论 ==")
    print("  链路可用。写入联调前请先按 docs/12 §8 的第 2 步确认字节序，")
    print(
        "  再在 config/oilcheck.json 里开 plc.enabled 并保持 dry_run=true 观察一个班次。"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="PLC Modbus TCP 只读体检")
    parser.add_argument("--host", default="192.168.1.88", help="PLC IP")
    parser.add_argument("--port", type=int, default=502, help="PLC Modbus 端口")
    parser.add_argument("--unit", type=int, default=1, help="从站 unit id")
    parser.add_argument(
        "--byte-order",
        choices=("lsb", "msb"),
        default="lsb",
        help="字节序解读（默认 lsb，与 plc.byte_order 一致）",
    )
    parser.add_argument("--watch", type=float, default=0.0, help="额外持续观察的秒数")
    parser.add_argument(
        "--allow-write-probe",
        action="store_true",
        help="允许 FC6/FC16 写 0 到 addr=0（会改变 PLC 状态，默认关闭）",
    )
    args = parser.parse_args()
    if args.allow_write_probe:
        print("⚠ --allow-write-probe 会真实写入 addr=0，请确认现场允许")
    return check(
        args.host,
        args.port,
        args.unit,
        args.watch,
        byte_order=args.byte_order,
        allow_write_probe=args.allow_write_probe,
    )


if __name__ == "__main__":
    raise SystemExit(main())
