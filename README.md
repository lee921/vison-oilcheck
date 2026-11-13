# 丝杆取料与二次涂油视觉检测系统

基于固定摄像头的事件识别系统：确认工人从涂油盘取出一根丝杆，跟踪该丝杆，
并识别右手刷具对它的二次涂油过程。实现依据
《涂油工位视觉检测项目实现文档》（`涂油工位视觉检测项目实现文档.docx`）。

- 桌面端：PySide6 + QFluentWidgets，复用海康 HCNetSDK/PlayCtrl 相机接入
- 业务核心：区域过滤（涂油盘区 + 工作追踪区）→ 丝杆/手/刷头检测 →
  手/刷头 **ByteTrack** 跟踪 → **手-丝杆关联** → 取料判定 → 涂油判定
- **只有两条判定**：
  1. **取料完成**：一只手**空手**进入涂油盘（`oil_station` 检测框），
     随后同一只手**带着丝杆离开** → `PICK/CONFIRMED`；
  2. **人工二次涂油完成**：取料之后手持丝杆，刷头工作点持续在丝杆接触邻域内
     累计 ≥ `brush_hold_s` → `MANUAL_OILING/CONFIRMED`，周期结束并复位。
- 唯一的异常判定是**漏涂 NG**，两个触发点：取料后二次涂油没完成就来了下一次
  取料，或取料后超过 `oiling_timeout_s`（默认 10 秒）仍未完成；不做槽位识别，
  不做放料判定
- 视觉 ID 只做画面标注：手 `#1xxxxx`、刷头 `#2xxxxx`，丢失中的轨迹带 `?`
- 手/刷头跟踪的已知边界与实测数据见 [docs/04](docs/04-视觉检测管线.md) §5.1.1~5.1.2；
  对比参数用 `uv run python tools/tracker_ab.py`
- 无模型可运行：`null` 检测器保证不伪造事件
- 示例回放：`画面源` 下拉直接回放 `data/sample_video` 下的真实录像，
  支持 0.5×/1×/2× 变速（判定按视频媒体时间计时，变速不改变结果），
  画面左上角显示当前帧号；实时相机不显示帧号

## 快速开始

```bash
uv sync
uv run python main.py
```

**配置只有一份，在仓库的 `config/` 下**（`app.json`、`cameras.json`、
`oilcheck.json`）。UI 里保存的值、判定实际读取的值、仓库里的文件是同一个文件，
不会出现「改了没生效」或两份配置不一致。

运行数据（日志、截图、事件、训练录制）在用户主目录：

- 所有平台：`~/OilCheckData/`（Windows 为 `%USERPROFILE%\OilCheckData\`）
- 可用 `OILCHECK_DATA_ROOT` 覆盖；配置目录可用 `OILCHECK_CONFIG_ROOT` 覆盖

如需把运行数据也放进仓库目录：

```bash
OILCHECK_DATA_ROOT=$(pwd) uv run python main.py
```

## 目录

```text
app/oilcheck/          业务核心：配置、ROI、检测器、跟踪、关联、判定、事件、录制
app/oilcheck/bytetrack_adapter.py  手/刷头的 ByteTrack 适配器（带回退）
app/oilcheck/association.py        手-丝杆空间关联、刷头工作点
app/oilcheck/video_source.py  示例视频帧源（回放 mp4，按视频节拍投放帧）
app/camera_manager.py  相机会话与重连
hik/                   海康 SDK 封装
ui/pages/              监控、复核、训练录制、设备、设置、日志页面
config/                配置文件（唯一一份，UI 与判定都读写这里）
docs/                  各模块设计/使用文档（必读）
tests/                 取料/涂油判定、关联、跟踪、事件、配置单元测试
tools/detect_video.py  离线跑示例视频，输出带框视频与逐帧检测 JSONL
tools/tracker_ab.py    离线对比跟踪参数/策略下的手与刷头轨迹质量
lib/                   海康 SDK 运行库与 Python 封装
```

## 检测模型

模型训练在另一个项目中完成，本项目只消费导出权重。检测类别：

| id | 类别 | 作用 |
| --- | --- | --- |
| 0 | `hand` | 左右手统一类别 |
| 1 | `brush` | 刷具（含工作端） |
| 2 | `oil_station` | 涂油工位/盘体定位参考 |
| 3 | `screw` | 每根独立丝杆 |

仓库自带权重：`model/detect/exp/weights/best.pt`（YOLO11n，`imgsz=1024`，
mAP50 0.976）。`detector.imgsz=0` 会自适应训练尺寸；`detector.device=auto`
在 GPU 服务器上自动使用 CUDA，无显卡的开发机自动回退 CPU。

部署目标为 GPU 服务器，`uv.lock` 锁定 CUDA 12.8 版 torch；无 GPU 的开发机
同样可 `uv sync`（推理走 CPU，仅慢），详见 [docs/09](docs/09-开发环境与运行.md)。

没有模型时使用 `null`。示例视频放在 `data/sample_video/`，在监控页
“画面源”下拉里直接回放，检测仍由模型完成，并正常走取料/刷涂事件判定。
ROI 标定同样支持示例视频：设置页的标定窗口里选视频、拖位置条挑一帧，
就能划取涂油盘区/工作追踪区/涂油区，现场无相机时也能完成标定。
`detector.raw_mode=true` 是旁路排查模式：只输出全量检测结果、不产生事件。

## 事件与周期阶段

正常生产只产生两类确认事件（漏涂时另有 `OILING_MISSING`）：

| 事件 | 条件 |
| --- | --- |
| `PICK` | 手空手进盘、带料离盘；确认事件为 `CONFIRMED` |
| `MANUAL_OILING` | 取料后手持丝杆，刷子靠近累计 ≥ `brush_hold_s` |
| `OILING_MISSING` | 只取料、没涂完就被新的取料打断，或取料后超过 `oiling_timeout_s` 仍未涂完 ⇒ 该周期判 `NG`，结果 `NEEDS_REVIEW` |

作业周期阶段只有三档：

| 阶段 | 含义 |
| --- | --- |
| `WAITING_PICK` | 等待取料 |
| `PICKED` | 已取料，等待人工二次涂油 |
| `OILED` | 本周期的涂油已完成，周期结束并复位 |

`TaskResult` 除 `PROCESSING` / `COMPLETED` 外还有 `NG`（漏涂）。结算有两个
触发点：下一次取料到来，或取料后超过 `oiling_timeout_s`（默认 10 秒，按媒体
时间算、与帧率无关；设 0 关闭）。超时把「取走后视频就结束」这类悬空周期也能
及时判出来；剩余时长不足 `oiling_timeout_s` 的仍不会判出。

## 文档索引

从 [docs/README.md](docs/README.md) 开始。每个模块都有独立文档：配置、相机采集、
检测管线、取料与刷涂规则、事件复核、训练录制、UI、开发环境、测试与实施路线。

## 当前边界

- 不做模型训练与真实推理；`detector.kind=yolo` 仅为适配器与接入契约。
- 不识别取自涂油盘哪个槽位，不绑定槽位。
- `oiling_roi`（涂油区）本轮只做标定与显示，不参与判定，预留给后续涂油范围约束。
- 不判断机器自动涂油是否成功，不检测最终存储盘。
- 只有落在涂油盘区或工作追踪区内的目标才会被跟踪和判定，训练和推理保持一致范围。
- 第一版未启用事件前后视频片段；复核页保留入口与字段。

## 测试

```bash
uv run pytest
uv run ruff check app ui tests
```
