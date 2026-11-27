#!/usr/bin/env bash
# 一键启动丝杆取料与二次涂油视觉检测系统（桌面快捷方式调用这里）。
#
# 与 `uv run python main.py` 等价，但按「双击启动」的现场场景补了三件事：
#   1. 直接用 .venv 里的 Python，不经过 uv 解析依赖（不联网、启动更快）；
#   2. 单实例保护：已有窗口在跑时再双击不会开出第二个进程
#      —— 海康相机与 PLC 都只允许单连接，第二个实例只会抢流/抢连接；
#   3. 启动失败时弹窗说明原因，并把本次启动的输出写进 launcher.log。
#      双击启动看不到终端，没有这层兜底就只是「窗口闪一下」。
#
# 退出应用后脚本也随之退出，终端/桌面都不会残留进程。

set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO_ROOT/.venv/bin/python"

DATA_ROOT="${OILCHECK_DATA_ROOT:-$HOME/OilCheckData}"
LOG_DIR="$DATA_ROOT/logs"
LAUNCH_LOG="$LOG_DIR/launcher.log"
LOCK_FILE="$LOG_DIR/launcher.lock"

mkdir -p "$LOG_DIR" 2>/dev/null || true

# 双击启动没有终端可看，错误一律落到日志 + 弹窗。
notify() {
    local title="$1" text="$2"
    if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]] && command -v zenity >/dev/null 2>&1; then
        zenity --error --title="$title" --no-wrap --width=520 --text="$text" >/dev/null 2>&1
    elif command -v notify-send >/dev/null 2>&1; then
        notify-send "$title" "$text"
    fi
    printf '%s\n' "$text" >&2
}

fail() {
    local title="$1" text="$2"
    {
        echo "==== $(date '+%F %T') 启动失败 ===="
        echo "$title: $text"
    } >>"$LAUNCH_LOG" 2>/dev/null
    notify "$title" "$text"
    exit 1
}

cd "$REPO_ROOT" || fail "启动失败" "无法进入项目目录：$REPO_ROOT"

# 日志只保留最近一次启动的输出，避免反复双击把文件撑大。
if [[ -f "$LAUNCH_LOG" ]] && [[ $(stat -c %s "$LAUNCH_LOG" 2>/dev/null || echo 0) -gt 5242880 ]]; then
    mv -f "$LAUNCH_LOG" "$LAUNCH_LOG.1" 2>/dev/null || true
fi

if [[ ! -x "$PY" ]]; then
    fail "启动失败" "未找到 Python 环境：$PY

请先在项目目录执行一次依赖安装：
    cd $REPO_ROOT
    uv sync

安装完成后再双击本快捷方式。"
fi

# 单实例：拿不到锁说明已经有窗口在跑，直接提示并退出。
if command -v flock >/dev/null 2>&1; then
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
        notify "程序已在运行" "丝杆涂油检测已经在运行，请勿重复启动。

如需重启，先关闭已打开的窗口。"
        exit 0
    fi
fi

{
    echo "==== $(date '+%F %T') 启动 ===="
    echo "项目目录: $REPO_ROOT"
    echo "运行数据: $DATA_ROOT"
} >>"$LAUNCH_LOG" 2>/dev/null

# 用 exec 替换进程，应用自身的退出码直接成为本脚本的退出码，
# 打包成快捷方式后也不会留下一个空壳 bash 进程。
exec "$PY" "$REPO_ROOT/main.py" >>"$LAUNCH_LOG" 2>&1
