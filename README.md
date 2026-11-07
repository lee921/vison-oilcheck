# 丝杆取料与二次涂油视觉检测系统

基于固定摄像头的事件识别系统：确认工人从涂油盘取出一根丝杆，跟踪该丝杆，
并识别右手刷具对它的二次涂油过程。实现依据
《涂油工位视觉检测项目实现文档》（`涂油工位视觉检测项目实现文档.docx`）。

- 桌面端：PySide6 + QFluentWidgets，复用海康 HCNetSDK/PlayCtrl 相机接入
- 业务核心：ROI 屏蔽 → 丝杆/手/刷头检测 → 短时跟踪 → 取料事件引擎 →
  手持丝杆任务 → 刷涂行为引擎 → 任务状态机与事件
- **不做槽位识别**：涂油盘只作为取料参考区域，不建立四个槽位与丝杆的绑定
- **视觉 ID 与业务 ID 分离**：跟踪器给出短时 Track ID，业务层维护 `TASK-xxxxxx`
- **证据不足不硬判**：遮挡、关联不确定、刷头不可见时输出
  `PENDING_CONFIRMATION`
- 无模型可运行：`null` 检测器保证不伪造事件
- 示例回放：`画面源` 下拉直接回放 `data/sample_video` 下的真实录像

## 快速开始

```bash
uv sync
uv run python main.py
```

首次运行会在用户主目录下创建统一的数据目录：

- 所有平台：`~/OilCheckData/`（Windows 为 `%USERPROFILE%\OilCheckData\`）
- 配置、日志、截图、事件、训练录制都放在这里，便于备份和迁移
- 可用 `OILCHECK_DATA_ROOT` 覆盖

如需把配置/日志/事件/录制写回仓库目录：

```bash
OILCHECK_DATA_ROOT=$(pwd) uv run python main.py
```

## 目录

```text
app/oilcheck/          业务核心：配置、ROI、检测器、跟踪、取料/刷涂引擎、任务、事件、录制
app/oilcheck/video_source.py  示例视频帧源（回放 mp4，按视频节拍投放帧）
app/camera_manager.py  相机会话与重连
hik/                   海康 SDK 封装
ui/pages/              监控、复核、训练录制、设备、设置、日志页面
config/                配置模板（复制到数据目录后生效）
docs/                  各模块设计/使用文档（必读）
tests/                 取料证据、刷涂覆盖、跟踪、事件、配置单元测试
tools/detect_video.py  离线跑示例视频，输出带框视频与逐帧检测 JSONL
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
mAP50 0.976）。`detector.imgsz=0` 会自适应训练尺寸。

没有模型时使用 `null`。示例视频放在 `data/sample_video/`，在监控页
“画面源”下拉里直接回放，检测仍由模型完成（`detector.raw_mode=true`
时只输出全量检测结果，不做 SOP 判定）。

## 结果判定

| 结果 | 条件 |
| --- | --- |
| `OK` | 取料确认 + 刷头接触 + 有效刷涂运动 + 覆盖区段达到阈值 |
| `NG` | 可观测条件下确认未执行必要动作（如手持期间始终无刷涂证据） |
| `PENDING_CONFIRMATION` | 遮挡、身份关联不确定、刷头不可见、部分证据不足 |
| `PROCESSING` | 任务仍在进行 |

## 文档索引

从 [docs/README.md](docs/README.md) 开始。每个模块都有独立文档：配置、相机采集、
检测管线、取料与刷涂规则、事件复核、训练录制、UI、开发环境、测试与实施路线。

## 当前边界

- 不做模型训练与真实推理；`detector.kind=yolo` 仅为适配器与接入契约。
- 不识别取自涂油盘哪个槽位，不绑定槽位；不要求槽位持续可见。
- 不判断机器自动涂油是否成功，不检测最终存储盘。
- 物料盒区域应配置为排除 ROI，训练和推理保持一致屏蔽。
- 第一版未启用事件前后视频片段；复核页保留入口与字段。

## 测试

```bash
uv run pytest
uv run ruff check app ui tests
```
