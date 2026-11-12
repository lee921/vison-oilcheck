"""取料/二次涂油业务配置。

配置结构对应《涂油工位视觉检测项目实现文档》第 3、10 节：

- 每个工位配置工作追踪区（``work_roi``）与
  涂油区（``oiling_roi``）；跟踪与判定只处理前两个区域内的目标，
  涂油区本轮只做标定与显示；
- 不配置槽位，也不做槽位绑定；
- 判定参数按“取料事件 / 跟踪 / 刷涂 / 任务”分组。

配置文件默认位于数据目录的 ``config/oilcheck.json``。仓库
``config/oilcheck.json`` 只作为首次运行的模板，实际运行配置由
:class:`OilCheckConfigService` 复制并维护。启动时会把代码里有默认值、
文件里却缺的键补写回文件（只补不删、不改已有值），让文件里的键与实际
生效的参数始终一致。旧版配置中的 ``pick_roi``、
``oiling_roi``、``hand_roi``、``brush_roi``、``slot_layout`` 等旧字段在读取时
会自动迁移为 ``work_roi`` / ``oiling_roi``。涂油盘区 = 手工标定的
``tray_roi`` 与模型检出的 ``oil_station`` 框的并集：手工区覆盖盘体下方的
手臂活动范围，检测框跟随盘体上下移动。
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.paths import config_root, runtime_root

from .roi import Roi

logger = logging.getLogger("oilcheck.config")

# 配置结构版本。升级时 ensure_defaults 会把新增字段合并进已有配置文件。
_CONFIG_VERSION = "2.6"

# 手/刷头跟踪后端：auto 优先用 ByteTrack，失败回退自研 ObjectTracker。
_TRACKER_BACKENDS = frozenset({"auto", "bytetrack", "object"})

# 旧配置既没有 work_roi 也没有 hand_roi/brush_roi 时的兜底工作追踪区。
_DEFAULT_WORK_ROI = Roi(kind="rect", points=((0.0, 0.0), (1.0, 1.0)))

# 仓库自带的权重（相对仓库根解析）。
_DEFAULT_MODEL_PATH = "model/detect/exp/weights/best.pt"

# 2.3 新增、且在当前版本仍在使用的管线字段及默认值，迁移时只补缺失键。
_PIPELINE_DEFAULTS_2_3: dict[str, Any] = {
    "hand_screw_max_distance": 0.10,
    "hand_interaction_margin": 0.02,
    "held_infer_distance": 0.10,
    "held_predict_hold_s": 0.50,
    "hand_tracker_backend": "auto",
    "brush_tracker_backend": "auto",
    "bytetrack_high_thresh": 0.30,
    "bytetrack_low_thresh": 0.10,
    "bytetrack_new_thresh": 0.30,
    "bytetrack_match_thresh": 1.00,
    "bytetrack_buffer_s": 2.00,
    "bytetrack_cost_alpha": 0.40,
}

# 2.4 删除的管线字段：简化判定后不再有任何消费方，迁移时清掉，
# 避免现场配置里留一堆永不生效的键。
_PIPELINE_REMOVED_2_4: tuple[str, ...] = (
    "pick_motion_min",
    "pick_relative_max",
    "pick_comotion_min_s",
    "pick_abort_s",
    "pick_occlusion_timeout_s",
    "pick_ambiguity_margin",
    "pick_window_s",
    "screw_track_max_distance",
    "screw_track_max_missing_s",
    "screw_track_min_hits",
    "screw_track_max_active",
    "brush_min_duration_s",
    "brush_max_gap_s",
    "brush_min_motion",
    "brush_require_hand",
    "coverage_segments",
    "coverage_required",
    "identity_timeout_s",
    "identity_ambiguity_margin",
    "oiling_wait_timeout_s",
    "association_ambiguity_margin",
    "association_sustain_s",
    "association_lock_confidence",
    "association_unmatched_cost",
    "slot_occupied_confirm_s",
    "slot_empty_confirm_s",
    "slot_restock_confirm_s",
    "slot_cooldown_s",
    "slot_empty_stable_s",
    "slot_assign_margin",
    "slot_occlusion_hold_s",
    "slot_min_visible_ratio",
    "slot_layout_anchor_ratio",
    "slot_layout_shrink_ratio",
    "brush_hand_distance",
    "brush_hand_confirm_s",
    "brush_hand_link_grace_s",
    "brush_require_hand_link",
    "brush_hand_cross_hand_ok",
)

# 2.4 删除的工位字段（槽位与遮挡区）。
_STATION_REMOVED_2_4: tuple[str, ...] = (
    "slot_rois",
    "occlusion_rois",
    "slot_auto_count",
    "slot_follow_tray",
)

# 2.1 新增的检测器字段及其默认值，迁移时补齐。
_DETECTOR_DEFAULTS_2_1: dict[str, Any] = {
    "iou": 0.7,
    "max_det": 100,
    "imgsz": 0,
    "device": "auto",
    "half": False,
    # 迁移补默认值时应直接进入完整判定；旁路模式只作为现场临时排查手段，
    # 由用户显式打开，不能作为新旧配置的默认行为。
    "raw_mode": False,
    "brush_label": "brush",
    "tray_label": "oil_station",
    # 旧配置里的 rod_label / screw_rod 会迁移成这个值。
    "screw_label": "screw",
}


@dataclass(frozen=True)
class DetectorConfig:
    """检测器配置。

    ``kind`` 支持：

    - ``null``：不产生检测结果，用于无模型时的框架自检。
    - ``yolo``：Ultralytics YOLO 权重（普通 Detect 输出）。

    类别名与训练导出模型保持一致（``best.pt`` 的 4 个类别）：

    - ``0=hand``、``1=brush``、``2=oil_station``、``3=screw``。

    ``imgsz=0`` 表示自适应：优先取权重 checkpoint 中训练时的 ``imgsz``，
    取不到再回退 640，避免用 640 推理 1024 训练的模型而丢精度。

    ``raw_mode=True`` 时流水线只做“把这一帧的全部检测结果交出来”，跳过
    ROI 过滤、跟踪、取料/刷涂状态机与事件产出，用于先确认模型标签是否齐全。
    默认关闭，按需在配置里打开；关闭时走完整的 SOP 判定流程。
    """

    kind: str = "null"
    model_path: str = ""
    confidence: float = 0.5
    iou: float = 0.7
    max_det: int = 100
    imgsz: int = 0
    # auto/空 = 有 CUDA 用 cuda:0，否则 cpu；可显式写 cpu / cuda:0 / 0。
    # 默认写 auto 而不是空串：迁移与模板都用 auto，补键写回时才一致。
    device: str = "auto"
    half: bool = False
    raw_mode: bool = False
    screw_label: str = "screw"
    hand_label: str = "hand"
    brush_label: str = "brush"
    tray_label: str = "oil_station"
    # 模型只输出刷具框、没有关键点时，用框内相对坐标估计工作端。
    brush_point_offset_x: float = 0.5
    brush_point_offset_y: float = 0.9

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DetectorConfig:
        data = dict(data or {})
        d = cls()
        kind = str(data.get("kind", "null")).strip().lower()
        if kind not in {"null", "yolo"}:
            kind = "null"

        def _f(key: str, default: float, lo: float, hi: float) -> float:
            try:
                return max(lo, min(hi, float(data.get(key, default))))
            except (TypeError, ValueError):
                return default

        def _i(key: str, default: int, lo: int, hi: int) -> int:
            try:
                return max(lo, min(hi, int(data.get(key, default))))
            except (TypeError, ValueError):
                return default

        def _b(key: str, default: bool) -> bool:
            value = data.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        return cls(
            kind=kind,
            model_path=str(data.get("model_path", "")),
            confidence=_f("confidence", d.confidence, 0.0, 1.0),
            iou=_f("iou", d.iou, 0.0, 1.0),
            max_det=_i("max_det", d.max_det, 1, 1000),
            # 0 表示自适应训练尺寸；非 0 时会被向上取整到 stride 的整数倍。
            imgsz=_i("imgsz", d.imgsz, 0, 8192),
            device=str(data.get("device") or d.device).strip(),
            half=_b("half", d.half),
            raw_mode=_b("raw_mode", d.raw_mode),
            # 旧配置：rod_label → screw_label，brush_head_label → brush_label。
            screw_label=str(
                data.get("screw_label") or data.get("rod_label") or d.screw_label
            ),
            hand_label=str(data.get("hand_label") or d.hand_label),
            brush_label=str(
                data.get("brush_label")
                or data.get("brush_head_label")
                or d.brush_label
            ),
            tray_label=str(data.get("tray_label") or d.tray_label),
            brush_point_offset_x=_f(
                "brush_point_offset_x", d.brush_point_offset_x, 0.0, 1.0
            ),
            brush_point_offset_y=_f(
                "brush_point_offset_y", d.brush_point_offset_y, 0.0, 1.0
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "model_path": self.model_path,
            "confidence": self.confidence,
            "iou": self.iou,
            "max_det": self.max_det,
            "imgsz": self.imgsz,
            "device": self.device,
            "half": self.half,
            "raw_mode": self.raw_mode,
            "screw_label": self.screw_label,
            "hand_label": self.hand_label,
            "brush_label": self.brush_label,
            "tray_label": self.tray_label,
            "brush_point_offset_x": self.brush_point_offset_x,
            "brush_point_offset_y": self.brush_point_offset_y,
        }


@dataclass(frozen=True)
class PipelineConfig:
    """时序事件判定参数。所有时间单位为秒，距离为按画面对角线归一化。"""

    max_fps: int = 15

    # 取料事件：空手进盘 → 带料离盘
    pick_hand_distance: float = 0.05
    # 手必须在盘内停留这么久才算一次取料（滤掉从盘边划过的手）。
    pick_confirm_s: float = 0.30
    # 取料判定用「间隔采样 + 多帧投票」，不拿单帧当结论：
    # - ``pick_sample_every_frames``（k）：每隔 k 帧采样一次状态；
    # - ``pick_history_samples``（n）：保留最近 n 个采样点。
    #
    # 涉及两个状态——「手是否在盘内」「离盘时手里有没有杆」。
    #
    # 现场实测：盘区边线附近手框中心会来回抖动一帧；离盘那一帧的丝杆
    # 置信度也常掉到 0.5 左右（杆心还没进手框），只看当帧会判成
    # 「空手离盘」而丢掉整件。
    #
    # 取 k=2、n=3：每 2 帧采一个样、看最近 3 个采样点。
    #
    # 两段示例视频按「取料数 / 涂油数 / 误判 NG」在 15fps 与 7.5fps 下横扫
    # k∈{1,2,3} × n∈{1,2,3,5}：k=1,n=5（早先默认）在两组场景合计 4 次取料
    # 只判出 3 次；n=5 的观察窗太长，低帧率下盘边抖动会把一件拆成两段。
    # k=2,n=3 与 k=3,n=2 在四组场景下全部 4 取、0 MISS、0 假阳性。
    #
    # k×n 决定「离盘后要攒多少个盘外采样才按多数定论」：k=2、n=3 在
    # 15fps 下是 0.4s、7.5fps 下 0.8s。手出画面、视频结束、或等到超时仍
    # 不足这么多点时不再等，见过一帧手握杆就按握持结算——点数不足一律按
    # 空手处理会让真实取料连周期都建不出来。
    pick_sample_every_frames: int = 2
    pick_history_samples: int = 3

    # 目标跟踪（手/刷头使用 ByteTrack）
    hand_track_max_distance: float = 0.15
    hand_track_max_missing_s: float = 0.80
    hand_track_min_hits: int = 1
    # 同时保留的轨迹上限。漏检轨迹在 max_missing_s 内仍占名额，设小了会让
    # 新的手检测框建不了轨迹（画面里常有手臂/盘边等被误检成 hand 的框）。
    hand_track_max_active: int = 8
    brush_track_max_distance: float = 0.15
    brush_track_max_missing_s: float = 0.80
    brush_track_min_hits: int = 1
    brush_track_max_active: int = 4

    # 人工二次涂油：刷头工作点持续在丝杆接触邻域内累计到该时长即判完成
    brush_contact_distance: float = 0.060
    brush_hold_s: float = 1.00
    # 取料后超过该时长仍未完成二次涂油 ⇒ 判漏涂 NG，不等下一次取料。
    # 0 = 不限时（只靠下一次取料结算，旧行为）。
    #
    # 计时起点是取料确认时刻，按媒体时间累计，掉帧不暂停。与
    # ``screw_reacquire_timeout_s`` 有取舍关系：后者默认 0（工人拿着杆走出
    # 画面、过一会儿再回来涂油仍算这一件），开了本项之后这段时间会被计入
    # 超时，超出即判 NG。
    oiling_timeout_s: float = 10.0

    # 周期与保留
    holding_lost_timeout_s: float = 2.0
    single_active_task: bool = True
    task_max_active: int = 4
    # 已结束（完成 / NG）的周期在内存里保留最近多少条。
    #
    # 监控页的任务卡片要一直显示最近几件的结果，所以不能按时间清：原来
    # ``completed_retain_s`` 到点就删，卡片栏做完 10 秒就空掉，现场看不到
    # 刚才那几件的结论。现在改为「按条数保留」，只留最近
    # ``completed_keep_count`` 条，更早的才回收。
    completed_keep_count: int = 4
    # 兜底的时间上限：已结束周期在内存里最多再留这么久，超过即回收。
    #
    # 与 ``completed_keep_count`` 是「与」的关系——两个条件任意一个满足就回收。
    # 默认 3600 秒（1 小时）：卡片栏要常驻最近几件的结果，但内存不能随运行
    # 时长无限增长。一小时足够覆盖换班之间的查看需求，长时间停机后也不会有
    # 一堆早就不关心的事件占着内存。设 0 = 不限时，只按条数保留。
    completed_retain_s: float = 3600.0

    # 涂油盘定位（只作为取料参考区域）
    tray_max_distance: float = 0.20
    tray_max_missing_s: float = 1.0
    tray_min_confidence: float = 0.30

    # 手-丝杆关联（空间距离 + 交互关系）
    hand_screw_max_distance: float = 0.10
    hand_interaction_margin: float = 0.02
    held_infer_distance: float = 0.10
    held_predict_hold_s: float = 0.50

    # 丝杆锚点跟踪：取料那一刻把手里那根丝杆记下来，之后按最近邻连续性
    # 跟这根杆。丝杆是静置物、位移小，比手部视觉 ID 稳得多。
    # 单帧最大位移（按画面对角线归一化，按实际帧率缩放）。
    # 单帧最大位移。实测手持杆单帧位移中位 0.003、p90 0.081，取 0.12
    # 覆盖约 p95：既跟得住快速移动，又不会跳到相邻的静置杆上
    # （盘里丝杆间距只有 0.02~0.03）。
    screw_anchor_max_step: float = 0.12
    # 锚点单帧位移的**绝对**上限（不随帧率缩放）。盘里丝杆间距只有
    # 0.02~0.03，低帧率时 frame_scale 会把上面那个门限放大到 0.5，一跳就是
    # 相邻那根杆。
    #
    # 必须 ≥ ``screw_anchor_max_step``：取料瞬间手带着杆快速划出，实测单帧
    # 位移到 0.09~0.11。上限比这个大，锚点会一步跨不过去而永久卡在盘内那根
    # 杆上（之后每帧都找不到杆，整件涂油累计为 0）。
    screw_anchor_max_step_abs: float = 0.15
    # 锚点丢失后靠预测位置继续跟的时长（按实际帧率缩放）。
    screw_anchor_hold_s: float = 0.50

    # 刷子靠近累计允许跨过的最大空档：低帧率下一个漏检帧会吞掉整帧时间，
    # 用空档桥接把这段补回来（按实际帧率缩放）。
    brush_hold_max_gap_s: float = 0.50

    # 手持丝杆离开画面后，允许重新捕获的时长。工人取完料可能拿着杆走出
    # 画面、过一会儿再回来涂油，这段时间内周期保持 PICKED 并继续等；
    # 超过该时长仍未重新出现，才认为这一件不再回来。
    # 0 表示不限制（只要周期没结束就一直等）。
    screw_reacquire_timeout_s: float = 0.0

    # 帧率自适应：以配置 max_fps 为基准，按实测帧间隔放大上面所有
    # 距离门限与容差窗口。倍率上限避免低帧率时门限大到失去区分度。
    frame_scale_max: float = 4.0

    # 目标移出工作追踪区后，继续跟踪的额外门限余量（按画面对角线归一化）。
    # 区外目标不会新建轨迹，但如果它离已有轨迹足够近，说明是同一个目标
    # 走出了区，继续喂给跟踪器以保持轨迹不断。取 0.20：示例①上区外
    # 跟踪帧 429（占 51%），再放大只增加轨迹碎片，判定结果不变。
    roi_track_margin: float = 0.20

    # 跟踪后端（手/刷头使用 ByteTrack）
    hand_tracker_backend: str = "auto"
    brush_tracker_backend: str = "auto"
    bytetrack_high_thresh: float = 0.30
    bytetrack_low_thresh: float = 0.10
    bytetrack_new_thresh: float = 0.30
    # 关联代价阈值。门控未命中的代价固定为 1.5（高于任何阈值），因此 1.0 表示
    # 「只看距离/IoU 门限」，与 *_track_max_distance 的语义一致；调小可加严。
    bytetrack_match_thresh: float = 1.00
    bytetrack_buffer_s: float = 2.00
    bytetrack_cost_alpha: float = 0.40

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PipelineConfig:
        data = dict(data or {})
        d = cls()

        def _f(key: str, default: float, lo: float = 0.0) -> float:
            try:
                value = float(data.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(lo, value)

        def _b(key: str, default: bool) -> bool:
            value = data.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        def _i(key: str, default: int, lo: int, hi: int) -> int:
            try:
                value = int(data.get(key, default))
            except (TypeError, ValueError):
                value = default
            return max(lo, min(hi, value))

        def _backend(key: str, default: str) -> str:
            value = str(data.get(key, default)).strip().lower()
            return value if value in _TRACKER_BACKENDS else "auto"

        def _legacy(new_key: str, *old_keys: str, default: float) -> float:
            """新字段缺失时回退到旧配置字段，便于现场配置平滑迁移。"""

            if new_key in data:
                return _f(new_key, default)
            for old_key in old_keys:
                if old_key in data:
                    return _f(old_key, default)
            return default

        return cls(
            max_fps=_i("max_fps", d.max_fps, 1, 60),
            pick_hand_distance=_legacy(
                "pick_hand_distance",
                "slot_hand_margin",
                "hand_proximity_margin",
                default=d.pick_hand_distance,
            ),
            pick_confirm_s=_f("pick_confirm_s", d.pick_confirm_s, 0.0),
            hand_track_max_distance=_f(
                "hand_track_max_distance", d.hand_track_max_distance, 0.005
            ),
            hand_track_max_missing_s=_f(
                "hand_track_max_missing_s", d.hand_track_max_missing_s, 0.0
            ),
            hand_track_min_hits=_i(
                "hand_track_min_hits", d.hand_track_min_hits, 1, 20
            ),
            hand_track_max_active=_i(
                "hand_track_max_active", d.hand_track_max_active, 1, 32
            ),
            brush_track_max_distance=_f(
                "brush_track_max_distance", d.brush_track_max_distance, 0.005
            ),
            brush_track_max_missing_s=_f(
                "brush_track_max_missing_s", d.brush_track_max_missing_s, 0.0
            ),
            brush_track_min_hits=_i(
                "brush_track_min_hits", d.brush_track_min_hits, 1, 20
            ),
            brush_track_max_active=_i(
                "brush_track_max_active", d.brush_track_max_active, 1, 32
            ),
            brush_contact_distance=_legacy(
                "brush_contact_distance",
                "brush_part_distance",
                default=d.brush_contact_distance,
            ),
            brush_hold_s=_legacy(
                "brush_hold_s",
                "brush_min_duration_s",
                "oiling_min_duration_s",
                default=d.brush_hold_s,
            ),
            oiling_timeout_s=_f("oiling_timeout_s", d.oiling_timeout_s, 0.0),
            holding_lost_timeout_s=_f(
                "holding_lost_timeout_s", d.holding_lost_timeout_s, 0.0
            ),
            single_active_task=_b("single_active_task", d.single_active_task),
            task_max_active=_i(
                "task_max_active",
                int(data.get("operation_max_active", d.task_max_active))
                if "task_max_active" not in data
                else d.task_max_active,
                1,
                64,
            ),
            completed_keep_count=_i(
                "completed_keep_count", d.completed_keep_count, 1, 64
            ),
            completed_retain_s=_f(
                "completed_retain_s", d.completed_retain_s, 0.0
            ),
            tray_max_distance=_f(
                "tray_max_distance", d.tray_max_distance, 0.01
            ),
            tray_max_missing_s=_f(
                "tray_max_missing_s", d.tray_max_missing_s, 0.0
            ),
            tray_min_confidence=max(
                0.0,
                min(1.0, _f("tray_min_confidence", d.tray_min_confidence, 0.0)),
            ),
            hand_screw_max_distance=_f(
                "hand_screw_max_distance", d.hand_screw_max_distance, 0.005
            ),
            hand_interaction_margin=_f(
                "hand_interaction_margin", d.hand_interaction_margin, 0.0
            ),
            held_infer_distance=_f(
                "held_infer_distance", d.held_infer_distance, 0.005
            ),
            held_predict_hold_s=_f(
                "held_predict_hold_s", d.held_predict_hold_s, 0.0
            ),
            screw_anchor_max_step=_f(
                "screw_anchor_max_step", d.screw_anchor_max_step, 0.005
            ),
            screw_anchor_max_step_abs=_f(
                "screw_anchor_max_step_abs", d.screw_anchor_max_step_abs, 0.005
            ),
            screw_anchor_hold_s=_f(
                "screw_anchor_hold_s", d.screw_anchor_hold_s, 0.0
            ),
            brush_hold_max_gap_s=_f(
                "brush_hold_max_gap_s", d.brush_hold_max_gap_s, 0.0
            ),
            screw_reacquire_timeout_s=_f(
                "screw_reacquire_timeout_s", d.screw_reacquire_timeout_s, 0.0
            ),
            frame_scale_max=_f("frame_scale_max", d.frame_scale_max, 1.0),
            roi_track_margin=_f(
                "roi_track_margin", d.roi_track_margin, 0.0
            ),

            hand_tracker_backend=_backend(
                "hand_tracker_backend", d.hand_tracker_backend
            ),
            brush_tracker_backend=_backend(
                "brush_tracker_backend", d.brush_tracker_backend
            ),
            bytetrack_high_thresh=_f(
                "bytetrack_high_thresh", d.bytetrack_high_thresh, 0.0
            ),
            bytetrack_low_thresh=_f(
                "bytetrack_low_thresh", d.bytetrack_low_thresh, 0.0
            ),
            bytetrack_new_thresh=_f(
                "bytetrack_new_thresh", d.bytetrack_new_thresh, 0.0
            ),
            bytetrack_match_thresh=_f(
                "bytetrack_match_thresh", d.bytetrack_match_thresh, 0.01
            ),
            bytetrack_buffer_s=_f(
                "bytetrack_buffer_s", d.bytetrack_buffer_s, 0.0
            ),
            bytetrack_cost_alpha=max(
                0.0,
                min(1.0, _f("bytetrack_cost_alpha", d.bytetrack_cost_alpha, 0.0)),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_fps": self.max_fps,
            "pick_hand_distance": self.pick_hand_distance,
            "pick_confirm_s": self.pick_confirm_s,
            "pick_sample_every_frames": self.pick_sample_every_frames,
            "pick_history_samples": self.pick_history_samples,
            # 键顺序与 dataclass 声明一致：写回文件时按它重排，模板也照此排。
            "hand_track_max_distance": self.hand_track_max_distance,
            "hand_track_max_missing_s": self.hand_track_max_missing_s,
            "hand_track_min_hits": self.hand_track_min_hits,
            "hand_track_max_active": self.hand_track_max_active,
            "brush_track_max_distance": self.brush_track_max_distance,
            "brush_track_max_missing_s": self.brush_track_max_missing_s,
            "brush_track_min_hits": self.brush_track_min_hits,
            "brush_track_max_active": self.brush_track_max_active,
            "brush_contact_distance": self.brush_contact_distance,
            "brush_hold_s": self.brush_hold_s,
            "oiling_timeout_s": self.oiling_timeout_s,
            "holding_lost_timeout_s": self.holding_lost_timeout_s,
            "single_active_task": self.single_active_task,
            "task_max_active": self.task_max_active,
            "completed_keep_count": self.completed_keep_count,
            "completed_retain_s": self.completed_retain_s,
            "tray_max_distance": self.tray_max_distance,
            "tray_max_missing_s": self.tray_max_missing_s,
            "tray_min_confidence": self.tray_min_confidence,
            "hand_screw_max_distance": self.hand_screw_max_distance,
            "hand_interaction_margin": self.hand_interaction_margin,
            "held_infer_distance": self.held_infer_distance,
            "held_predict_hold_s": self.held_predict_hold_s,
            "screw_anchor_max_step": self.screw_anchor_max_step,
            "screw_anchor_max_step_abs": self.screw_anchor_max_step_abs,
            "screw_anchor_hold_s": self.screw_anchor_hold_s,
            "brush_hold_max_gap_s": self.brush_hold_max_gap_s,
            "screw_reacquire_timeout_s": self.screw_reacquire_timeout_s,
            "frame_scale_max": self.frame_scale_max,
            "roi_track_margin": self.roi_track_margin,
            "hand_tracker_backend": self.hand_tracker_backend,
            "brush_tracker_backend": self.brush_tracker_backend,
            "bytetrack_high_thresh": self.bytetrack_high_thresh,
            "bytetrack_low_thresh": self.bytetrack_low_thresh,
            "bytetrack_new_thresh": self.bytetrack_new_thresh,
            "bytetrack_match_thresh": self.bytetrack_match_thresh,
            "bytetrack_buffer_s": self.bytetrack_buffer_s,
            "bytetrack_cost_alpha": self.bytetrack_cost_alpha,
        }


@dataclass(frozen=True)
class RecordingConfig:
    """训练视频录制参数。"""

    training_dir: str = "recordings/training"
    fps: int = 15
    segment_seconds: float = 600.0
    codec: str = "mp4v"
    auto_start: bool = False
    event_clips_enabled: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RecordingConfig:
        data = dict(data or {})
        d = cls()

        def _f(key: str, default: float) -> float:
            try:
                return float(data.get(key, default))
            except (TypeError, ValueError):
                return default

        def _i(key: str, default: int) -> int:
            try:
                return int(data.get(key, default))
            except (TypeError, ValueError):
                return default

        def _b(key: str, default: bool) -> bool:
            value = data.get(key, default)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)

        return cls(
            training_dir=str(data.get("training_dir") or d.training_dir),
            fps=max(1, min(60, _i("fps", d.fps))),
            segment_seconds=max(10.0, _f("segment_seconds", d.segment_seconds)),
            codec=str(data.get("codec") or d.codec),
            auto_start=_b("auto_start", d.auto_start),
            event_clips_enabled=_b("event_clips_enabled", d.event_clips_enabled),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_dir": self.training_dir,
            "fps": self.fps,
            "segment_seconds": self.segment_seconds,
            "codec": self.codec,
            "auto_start": self.auto_start,
            "event_clips_enabled": self.event_clips_enabled,
        }


@dataclass(frozen=True)
class DemoVideo:
    """一段示例视频：显示名 + 相对仓库根或绝对路径。"""

    name: str
    path: str

    @classmethod
    def from_dict(cls, data: Any) -> DemoVideo | None:
        if isinstance(data, str):
            name, path = "", data
        elif isinstance(data, dict):
            path = str(data.get("path", "")).strip()
            name = str(data.get("name", "")).strip()
        else:
            return None
        if not path:
            return None
        if not name:
            name = Path(path).stem
        return cls(name=name, path=path)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "path": self.path}


@dataclass(frozen=True)
class DemoConfig:
    """演示视频源配置。

    演示模式不再使用脚本伪造检测，而是直接回放 ``data/sample_video`` 下的
    真实录像：画面来自视频文件，检测结果来自训练好的模型。
    """

    loop: bool = True
    fps_limit: int = 15
    # 示例回放默认速率倍数（0.5 为慢放）。判定计时用视频媒体时间，与速率无关。
    speed: float = 1.0
    videos: tuple[DemoVideo, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DemoConfig:
        data = dict(data or {})
        d = cls()
        loop = data.get("loop", d.loop)
        if isinstance(loop, str):
            loop = loop.strip().lower() in {"1", "true", "yes", "on"}
        try:
            fps_limit = int(data.get("fps_limit", d.fps_limit))
        except (TypeError, ValueError):
            fps_limit = d.fps_limit
        try:
            speed = float(data.get("speed", d.speed))
        except (TypeError, ValueError):
            speed = d.speed
        if speed <= 0:
            speed = d.speed

        videos: list[DemoVideo] = []
        for item in data.get("videos", ()) or ():
            video = DemoVideo.from_dict(item)
            if video is not None:
                videos.append(video)
        return cls(
            loop=bool(loop),
            fps_limit=max(1, min(60, fps_limit)),
            speed=max(0.1, min(4.0, speed)),
            videos=tuple(videos),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "loop": self.loop,
            "fps_limit": self.fps_limit,
            "speed": self.speed,
            "videos": [video.to_dict() for video in self.videos],
        }

    def video(self, name: str) -> DemoVideo | None:
        for video in self.videos:
            if video.name == name:
                return video
        return None


@dataclass(frozen=True)
class StationConfig:
    """一个工位的 ROI 与检测配置。

    工位有三个区域：

    - ``tray_roi``：涂油盘区的**手工底区**，用于判断“手进盘/离盘”。
      它覆盖盘体下方的手臂活动范围——模型只框得出盘体本身，手伸下去取料时
      会在盘体下方 100~280px，只靠检测框会把取料全部漏掉。运行时与模型检出
      的 ``oil_station`` 框取并集，跟随盘体上下移动。
    - ``work_roi``：工作追踪区，跟踪与判定只处理中心落在该区域
      （或涂油盘区内）的目标。
    - ``oiling_roi``：涂油区，标定与显示用，预留给后续涂油范围约束。

    ROI 使用归一化坐标，现场调整后无需改代码。
    """

    station_id: str
    name: str
    camera_id: str
    enabled: bool
    tray_roi: Roi
    work_roi: Roi
    oiling_roi: Roi = _DEFAULT_WORK_ROI
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StationConfig | None:
        # 涂油盘区（手工底区）：兼容旧配置的 pick_roi。
        tray_roi = Roi.from_dict(data.get("tray_roi") or data.get("pick_roi"))
        if tray_roi is None:
            return None

        # 工作追踪区：新字段优先；旧版手持区/刷涂区取并集迁移。
        # 2.2 之前的 ``oiling_roi`` 就是手持操作区，同样按旧键处理。
        work_roi = Roi.from_dict(data.get("work_roi"))
        if work_roi is None:
            legacy = [
                roi
                for roi in (
                    Roi.from_dict(data.get("hand_roi")),
                    Roi.from_dict(data.get("oiling_roi")),
                    Roi.from_dict(data.get("brush_roi")),
                )
                if roi is not None
            ]
            work_roi = (
                _union_roi(legacy[0], legacy[1:]) if legacy else _DEFAULT_WORK_ROI
            )

        # 涂油区：新字段优先，缺省沿用工作追踪区（旧配置没有这个概念）。
        # 2.2 之前的 ``oiling_roi`` 是“手持操作区”的历史名字，含义完全不同；
        # 那个年代的配置没有 ``work_roi``，因此只在已有 work_roi 时才认这个键。
        oiling_roi = (
            Roi.from_dict(data.get("oiling_roi")) if "work_roi" in data else None
        ) or work_roi

        station_id = str(data.get("station_id", "")).strip()
        camera_id = str(data.get("camera_id", "")).strip()
        if not station_id or not camera_id:
            return None

        return cls(
            station_id=station_id,
            name=str(data.get("name") or station_id),
            camera_id=camera_id,
            enabled=bool(data.get("enabled", True)),
            tray_roi=tray_roi,
            work_roi=work_roi,
            oiling_roi=oiling_roi,
            detector=DetectorConfig.from_dict(data.get("detector")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "station_id": self.station_id,
            "name": self.name,
            "camera_id": self.camera_id,
            "enabled": self.enabled,
            "tray_roi": self.tray_roi.to_dict(),
            "work_roi": self.work_roi.to_dict(),
            "oiling_roi": self.oiling_roi.to_dict(),
            "detector": self.detector.to_dict(),
        }


def _legacy_work_roi(raw_rois: list[Any]) -> dict[str, Any] | None:
    """旧配置的手持区/刷涂区取包围矩形，作为工作追踪区。"""

    rois = [roi for roi in (Roi.from_dict(item) for item in raw_rois) if roi is not None]
    if not rois:
        return None
    return _union_roi(rois[0], rois[1:]).to_dict()


def _union_roi(first: Roi, others: Iterable[Roi]) -> Roi:
    """把若干 ROI 合成一个包围矩形，用于旧配置迁移。"""

    boxes = [first.pixel_bbox(1, 1), *(roi.pixel_bbox(1, 1) for roi in others)]
    return Roi(
        kind="rect",
        points=(
            (min(b.x1 for b in boxes), min(b.y1 for b in boxes)),
            (max(b.x2 for b in boxes), max(b.y2 for b in boxes)),
        ),
    )


def _paired(raw: Any, effective: Any) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """把 ``raw`` 与 ``effective`` 里的对象数组按元素配对。

    优先按 ``station_id`` 配对，其次按下标：解析时会跳过不合法工位
    （缺 ``tray_roi`` 或 ID），下标会错位，补错工位比不补更糟。
    """

    if not isinstance(raw, list) or not isinstance(effective, list):
        return []
    raw_dicts = [item for item in raw if isinstance(item, dict)]
    eff_dicts = [item for item in effective if isinstance(item, dict)]
    by_id: dict[str, dict[str, Any]] = {}
    for item in raw_dicts:
        station_id = str(item.get("station_id", "")).strip()
        if station_id:
            by_id.setdefault(station_id, item)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for index, item in enumerate(eff_dicts):
        station_id = str(item.get("station_id", "")).strip()
        match = by_id.get(station_id) if station_id else None
        if match is None and index < len(raw_dicts):
            candidate = raw_dicts[index]
            if not station_id and not str(candidate.get("station_id", "")).strip():
                match = candidate
        if match is not None:
            pairs.append((match, item))
    return pairs


def _sync_keys(
    raw: dict[str, Any], effective: dict[str, Any], prefix: str
) -> list[str]:
    """把 ``effective`` 里有、``raw`` 里缺的键补进 ``raw``（就地修改）。

    只补不删、不改已有值：现场调过的参数必须原样保留，废弃键由各自的
    迁移分支删除。返回被补上的键路径，供日志与去重使用。
    """

    changed: list[str] = []
    for key, value in effective.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in raw:
            raw[key] = value
            changed.append(path)
            continue
        existing = raw[key]
        if isinstance(value, dict) and isinstance(existing, dict):
            changed.extend(_sync_keys(existing, value, path))
        elif isinstance(value, list) and isinstance(existing, list):
            for child_raw, child_eff in _paired(existing, value):
                changed.extend(_sync_keys(child_raw, child_eff, f"{path}[]"))
    return changed


def _ordered_like(raw: Any, effective: Any) -> Any:
    """按 ``effective`` 的键顺序重排 ``raw``，未知键排到末尾。

    写回的文件键顺序与仓库模板一致，`diff` 运行配置和模板时只会剩下
    「值不同」的部分，不会因为键序或缺失键产生噪声。
    """

    if isinstance(raw, dict) and isinstance(effective, dict):
        ordered: dict[str, Any] = {}
        for key, value in effective.items():
            ordered[key] = (
                _ordered_like(raw[key], value) if key in raw else value
            )
        for key, value in raw.items():
            if key not in ordered:
                ordered[key] = value
        return ordered
    if isinstance(raw, list) and isinstance(effective, list):
        # 保持文件里的元素顺序，只重排每个对象内部的键。
        synced = {
            id(child_raw): _ordered_like(child_raw, child_eff)
            for child_raw, child_eff in _paired(raw, effective)
        }
        return [synced.get(id(item), item) for item in raw]
    return raw


def _unknown_keys(
    raw: Any, effective: Any, prefix: str = ""
) -> list[str]:
    """列出 ``raw`` 里当前代码不认识的键（不删除，只提示）。

    删掉未知键风险更大：同版本的旧/新程序交替运行时，对方写的键会被当成
    垃圾清掉。这里只报出来，由人决定。
    """

    found: list[str] = []
    if isinstance(raw, dict) and isinstance(effective, dict):
        for key, value in raw.items():
            path = f"{prefix}.{key}" if prefix else key
            if key not in effective:
                found.append(path)
            else:
                found.extend(_unknown_keys(value, effective[key], path))
    elif isinstance(raw, list) and isinstance(effective, list):
        for child_raw, child_eff in _paired(raw, effective):
            found.extend(_unknown_keys(child_raw, child_eff, f"{prefix}[]"))
    return found


def _template_demo() -> dict[str, Any] | None:
    """仓库模板里的 ``demo`` 段；模板缺失或不可解析时返回 None。"""

    template = runtime_root() / "config" / "oilcheck.json"
    try:
        # 模板带注释（JSONC），不能用 json.loads 直接读。
        tpl = load_json_with_comments(template)
    except (OSError, ValueError):
        return None
    demo = tpl.get("demo") if isinstance(tpl, dict) else None
    return demo if isinstance(demo, dict) and demo else None


def _demo_videos_missing(demo: Any) -> bool:
    """demo 段或其中的 videos 键是否「缺失」。

    ``"videos": []`` 不算缺失：用户可以把清单清空，让下拉里只剩实时相机。
    只有整段不在、或 videos 不是数组时才需要按模板补回来。
    """

    if not isinstance(demo, dict):
        return True
    return not isinstance(demo.get("videos"), list)


@dataclass(frozen=True)
class OilCheckConfig:
    version: str = _CONFIG_VERSION
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    demo: DemoConfig = field(default_factory=DemoConfig)
    stations: tuple[StationConfig, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OilCheckConfig:
        stations: list[StationConfig] = []
        for item in data.get("stations", ()):
            if not isinstance(item, dict):
                continue
            station = StationConfig.from_dict(item)
            if station is not None:
                stations.append(station)
        return cls(
            version=str(data.get("version", _CONFIG_VERSION)),
            pipeline=PipelineConfig.from_dict(data.get("pipeline")),
            recording=RecordingConfig.from_dict(data.get("recording")),
            demo=DemoConfig.from_dict(data.get("demo")),
            stations=tuple(stations),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "pipeline": self.pipeline.to_dict(),
            "recording": self.recording.to_dict(),
            "demo": self.demo.to_dict(),
            "stations": [station.to_dict() for station in self.stations],
        }

    def station(self, station_id: str) -> StationConfig | None:
        for station in self.stations:
            if station.station_id == station_id:
                return station
        return None

    def enabled_stations(self) -> tuple[StationConfig, ...]:
        return tuple(s for s in self.stations if s.enabled)


class OilCheckConfigService:
    """读写 ``<仓库>/config/oilcheck.json``。

    配置固定在仓库的 ``config/`` 里，不再复制到数据目录：UI 保存的值、
    判定实际读的值、仓库里的文件是同一份，不会再出现「改了仓库配置不
    生效」或「数据目录那份和仓库那份不一致」。
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        # 生产路径不传参：读取仓库的 config/（见 app.paths.config_root）。
        # 测试与离线工具显式传入临时目录。
        self._config_dir = Path(config_dir).resolve() if config_dir else config_root()

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    @property
    def config_path(self) -> Path:
        return self.config_dir / "oilcheck.json"

    def ensure_defaults(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            self._migrate_existing()
            return
        template = runtime_root() / "config" / "oilcheck.json"
        if template.exists() and template.resolve() != self.config_path.resolve():
            try:
                shutil.copy2(template, self.config_path)
                logger.info("已从模板复制检测配置: %s", self.config_path)
                return
            except OSError as exc:
                logger.warning("复制检测配置模板失败: %s", exc)
        self.save(OilCheckConfig())

    def _migrate_existing(self) -> None:
        """把新增配置项合并进已存在的配置文件。

        已有配置不会被模板覆盖（现场 ROI 等不能丢），但旧版本缺少
        ``demo`` 段、``detector`` 仍是 ``null`` 时，服务器拉取新代码后
        会出现“接入不了模型 / 没有示例视频”的假故障。这里按版本号
        做一次幂等合并，并在改动前备份原文件。
        """

        try:
            data = load_json_with_comments(self.config_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("检测配置无法解析，跳过迁移 (%s): %s", self.config_path, exc)
            return
        if not isinstance(data, dict):
            return

        old_version = str(data.get("version", "2.0"))
        try:
            old_parts = tuple(int(x) for x in old_version.split("."))
        except ValueError:
            old_parts = (2, 0)
        outdated = old_parts < (2, 1)
        # 2.2 之前每个工位配置手持操作区/刷涂区/排除区，改为涂油盘区 + 工作追踪区。
        rois_outdated = old_parts < (2, 2)
        # 2.3 引入槽位身份、手-丝杆关联、刷-手关联与 ByteTrack 后端。
        slots_outdated = old_parts < (2, 3)
        # 2.5 跟踪稳定性：锚点加绝对步长上限；关联代价门控语义对齐
        # （未进门限的代价改为 1.5，阈值 0.80 会把有效距离截到 0.667×）。
        stability_outdated = old_parts < (2, 5)
        # 2.6 修复空示例视频清单：2.0 及更早的版本保存配置时，没有 demo 段
        # 会被写成 "videos": []，之后按版本号补 demo 的分支不再触发，服务器
        # 上从此一条示例视频都没有（文件里段在、视频也在，下拉里却是空的）。
        demo_videos_outdated = old_parts < (2, 6)

        changed: list[str] = []

        # 1) demo 段：缺「整段」或「videos 键」时按模板补上示例视频清单。
        #
        #    显式写成 "videos": [] 的保持不动——那是用户主动清空，下拉里只剩
        #    「实时相机」是预期行为。但整段/整键缺失不能补成空清单：下面第 6
        #    步按「生效参数」补键，缺段会被补出 videos: []，现场看起来 demo
        #    段在、视频文件也在，下拉里却一条示例都没有，日志也不报错。
        #    这类文件多半来自早期版本的保存或手工删段，与版本号无关，所以
        #    不再用 outdated 门控。
        demo = data.get("demo")
        videos = demo.get("videos") if isinstance(demo, dict) else None
        # 升级到 2.6 时，把旧版本遗留的空清单也补回模板：那份空清单不是用户
        # 主动清空的，是当时保存配置时凭空写出来的。
        legacy_empty = demo_videos_outdated and videos == []
        if _demo_videos_missing(demo) or legacy_empty:
            template_demo = _template_demo()
            template_videos = (
                template_demo.get("videos") if template_demo is not None else None
            )
            if isinstance(template_videos, list) and template_videos:
                if isinstance(demo, dict):
                    demo["videos"] = template_videos
                    changed.append("demo.videos")
                else:
                    data["demo"] = template_demo
                    changed.append("demo")

        # 2) 工位 ROI：只保留涂油盘区，新增工作追踪区，删除旧的
        #    手持操作区/刷涂区/排除区。
        if rois_outdated:
            for station in data.get("stations", ()) or ():
                if not isinstance(station, dict):
                    continue
                if not isinstance(station.get("work_roi"), dict):
                    legacy = [
                        station.get(key)
                        for key in ("hand_roi", "oiling_roi", "brush_roi")
                        if isinstance(station.get(key), dict)
                    ]
                    work = _legacy_work_roi(legacy)
                    if work is not None:
                        station["work_roi"] = work
                        changed.append("stations.work_roi")
                for legacy_key in (
                    "hand_roi",
                    "oiling_roi",
                    "brush_roi",
                    "exclude_rois",
                    "material_box_roi",
                    "material_box_rois",
                    "pick_roi",
                ):
                    if legacy_key in station:
                        station.pop(legacy_key)
                        changed.append(f"-stations.{legacy_key}")

        # 3) 槽位身份与关联参数：只补缺失键，不改现场已调过的值。
        if slots_outdated:
            pipeline = data.get("pipeline")
            if not isinstance(pipeline, dict):
                pipeline = data["pipeline"] = {}
                changed.append("pipeline")
            for key, value in _PIPELINE_DEFAULTS_2_3.items():
                if key not in pipeline:
                    pipeline[key] = value
                    changed.append(f"pipeline.{key}")

        # 4) 2.4 简化判定：删除槽位/遮挡区与一大批不再生效的管线参数，
        #    给工位补上「涂油区」（缺省沿用工作追踪区）。
        simplified_outdated = old_parts < (2, 4)
        if simplified_outdated:
            pipeline = data.get("pipeline")
            if isinstance(pipeline, dict):
                # 旧字段名 → 新的「刷子靠近判完成时长」。
                # 必须在删除废弃键之前取值，否则 legacy 键已被清掉。
                if "brush_hold_s" not in pipeline:
                    for legacy_key in ("brush_min_duration_s", "oiling_min_duration_s"):
                        legacy_value = pipeline.get(legacy_key)
                        if isinstance(legacy_value, (int, float)):
                            pipeline["brush_hold_s"] = float(legacy_value)
                            changed.append("pipeline.brush_hold_s")
                            break
                for key in _PIPELINE_REMOVED_2_4:
                    if key in pipeline:
                        pipeline.pop(key)
                        changed.append(f"-pipeline.{key}")
            for station in data.get("stations", ()) or ():
                if not isinstance(station, dict):
                    continue
                if not isinstance(station.get("oiling_roi"), dict):
                    # 没有涂油区时沿用工位的工作追踪区作为初值，现场可再标定。
                    work = station.get("work_roi")
                    if isinstance(work, dict):
                        station["oiling_roi"] = work
                        changed.append("stations.oiling_roi")
                for key in _STATION_REMOVED_2_4:
                    if key in station:
                        station.pop(key)
                        changed.append(f"-stations.{key}")

        # 4.4) 2.5 跟踪稳定性：补锚点绝对步长上限；把仍是旧默认值的
        #      关联代价阈值提到 1.0（现场已手工调过的值保留，不覆盖）。
        #
        # 本节**不看版本号**：2.5 早期版本把 screw_anchor_max_step_abs 写成
        # 0.08（小于基准步长 0.12），会让锚点在取料瞬间跨不过去而永久卡住；
        # 那份配置的 version 已经是 2.5，按版本号判断会漏掉。这里每次都做
        # 幂等纠正——只在值确实不合法时才改，不会反复改写文件。
        pipeline = data.get("pipeline")
        if not isinstance(pipeline, dict):
            pipeline = {}
            if stability_outdated:
                data["pipeline"] = pipeline
                changed.append("pipeline")
        abs_step = pipeline.get("screw_anchor_max_step_abs")
        base_step = pipeline.get("screw_anchor_max_step")
        if isinstance(abs_step, (int, float)) and (
            isinstance(base_step, (int, float))
            and float(abs_step) < float(base_step)
        ):
            # 必须不小于基准步长：取料瞬间单帧位移实测可达 0.11。
            pipeline["screw_anchor_max_step_abs"] = float(base_step)
            changed.append("pipeline.screw_anchor_max_step_abs")
        elif stability_outdated and not isinstance(abs_step, (int, float)):
            pipeline["screw_anchor_max_step_abs"] = 0.15
            changed.append("pipeline.screw_anchor_max_step_abs")
        if stability_outdated:
            match_thresh = pipeline.get("bytetrack_match_thresh")
            if isinstance(match_thresh, (int, float)) and float(match_thresh) <= 0.80:
                # 旧语义下 0.80 等价于「距离门限打 0.667 折」；新语义下
                # 1.0 才是「只用距离门限」，等价于把配置写的行为还回来。
                pipeline["bytetrack_match_thresh"] = 1.0
                changed.append("pipeline.bytetrack_match_thresh")

        # 4.45) 检测置信度必须低于 ByteTrack 高置信度阈值，否则低分框在
        #       检测层就被丢掉，第二阶段补救关联永远收不到输入。旧配置里
        #       常见的 0.5 > 0.3 就属于这种情况，等价于关掉了第二阶段。
        high_thresh = pipeline.get("bytetrack_high_thresh")
        for station in data.get("stations", ()) or ():
            if not isinstance(station, dict):
                continue
            detector = station.get("detector")
            if not isinstance(detector, dict):
                continue
            conf = detector.get("confidence")
            if not isinstance(conf, (int, float)):
                continue
            if isinstance(high_thresh, (int, float)) and float(conf) > float(high_thresh):
                detector["confidence"] = float(high_thresh)
                changed.append("detector.confidence")

        # 4.5) demo：旧配置没有示例回放速率时补默认值，不改现场已设的值。
        demo = data.get("demo")
        if isinstance(demo, dict):
            if "speed" not in demo:
                demo["speed"] = _default_demo_speed()
                changed.append("demo.speed")

        # 5) detector：旧配置的 null / 空路径不会自动用上仓库权重
        for station in data.get("stations", ()) or ():
            if not isinstance(station, dict):
                continue
            detector = station.get("detector")
            if not isinstance(detector, dict):
                station["detector"] = detector = {}
            kind = str(detector.get("kind", "null")).strip().lower()
            if outdated and kind in {"", "null"}:
                detector["kind"] = "yolo"
                changed.append("detector.kind")
            if outdated and not str(detector.get("model_path", "")).strip():
                detector["model_path"] = _DEFAULT_MODEL_PATH
                changed.append("detector.model_path")
            for key, value in _DETECTOR_DEFAULTS_2_1.items():
                if key not in detector:
                    detector[key] = value
                    changed.append(f"detector.{key}")

            # 旧类别名与训练模型（0=hand 1=brush 2=oil_station 3=screw）
            # 不一致：模型只输出 oil_station / screw，配置成 oil_tray / rod
            # 会整整一类目标都匹配不到。不看版本号——这类旧值可能已经跟着
            # 配置一路升到了 2.6（老字段改过名，但值从没被纠正过），按版本号
            # 判断会永远漏掉；只在值确实还是旧名时才改，不会反复改写文件。
            #
            # 症状很隐蔽：画面上的检测框照常显示（那用的是模型输出的标签），
            # 但判定侧按配置里的类别名过滤，这类目标全部被当成"未配置类别"
            # 丢掉，取料/涂油一个事件都产生不了。
            legacy_labels = {
                "tray_label": ("oil_tray", "tray"),
                "brush_label": ("brush_head",),
                "screw_label": ("rod", "screw_rod", "screw_rod_head"),
            }
            for key, legacy_values in legacy_labels.items():
                current = str(detector.get(key, "")).strip()
                if current in legacy_values:
                    detector[key] = _DETECTOR_DEFAULTS_2_1[key]
                    changed.append(f"detector.{key}")

        # 6) 键集合同步：代码里有默认值、文件里缺的键补写回去，让「配置文件
        #    里看到的键」和「实际生效的参数」一致。只补不删、不改已有值——
        #    现场调过的 ROI 与阈值一律保留；废弃键由上面各版本分支删除。
        #
        #    不对齐这一步的话，缺键的参数走代码默认值却不在文件里，现场对着
        #    配置排查时会以为「还是老参数」，例如 pick_sample_every_frames /
        #    pick_history_samples 和 screw_reacquire_timeout_s。
        effective = OilCheckConfig.from_dict(data).to_dict()
        changed.extend(_sync_keys(data, effective, ""))
        unknown = _unknown_keys(data, effective)
        if unknown:
            logger.warning(
                "检测配置里有当前版本不认识的键（原样保留）: %s",
                ", ".join(unknown),
            )

        if not changed:
            return

        backup = self.config_path.with_suffix(".json.bak")
        try:
            shutil.copy2(self.config_path, backup)
        except OSError as exc:
            logger.warning("备份检测配置失败: %s", exc)

        data["version"] = _CONFIG_VERSION
        # 写回前按生效结构重排：键顺序与模板一致，diff 时只剩值差异。
        data = _ordered_like(data, effective)
        try:
            _write_json_atomic(self.config_path, data, comments=CONFIG_COMMENTS)
        except OSError as exc:
            logger.error("写回迁移后的检测配置失败: %s", exc)
            return
        fields = ", ".join(dict.fromkeys(changed))
        if old_version == _CONFIG_VERSION:
            # 版本没变、只是把文件里缺的键补齐（例如换了一台机器或回滚过）。
            logger.warning(
                "检测配置已按当前默认值补齐缺失的键: %s（原文件备份为 %s）",
                fields,
                backup.name,
            )
        else:
            logger.warning(
                "检测配置已升级到 %s（%s → %s），字段变更: %s"
                "（-前缀为删除，原文件备份为 %s）",
                _CONFIG_VERSION,
                old_version,
                _CONFIG_VERSION,
                fields,
                backup.name,
            )

    def source_info(self) -> dict[str, Any]:
        """当前生效配置文件的位置与状态，供 UI/日志核对。

        配置就在仓库的 ``config/`` 下，UI 保存与代码读取都是这一份。
        路径仍亮出来，便于现场确认程序确实读的是这个检出里的文件。
        """

        template = runtime_root() / "config" / "oilcheck.json"
        info: dict[str, Any] = {
            "path": str(self.config_path),
            "exists": self.config_path.exists(),
            "template": str(template),
            "config_dir": str(self.config_dir),
            "mtime": None,
            "size": None,
            "same_as_template": False,
        }
        try:
            stat = self.config_path.stat()
            info["mtime"] = stat.st_mtime
            info["size"] = stat.st_size
        except OSError:
            pass
        try:
            info["same_as_template"] = (
                template.exists()
                and self.config_path.exists()
                and template.resolve() == self.config_path.resolve()
            )
        except OSError:
            pass
        return info

    def load(self, *, create_if_missing: bool = True) -> OilCheckConfig:
        if create_if_missing:
            self.ensure_defaults()
        if not self.config_path.exists():
            return OilCheckConfig()
        try:
            data = load_json_with_comments(self.config_path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("加载检测配置失败 (%s): %s", self.config_path, exc)
            return OilCheckConfig()
        if not isinstance(data, dict):
            return OilCheckConfig()
        cfg = OilCheckConfig.from_dict(data)
        if not cfg.demo.videos:
            # 空清单本身是合法配置（下拉只剩实时相机），但现场最常见的困惑
            # 就在这里：文件里有 demo 段、data/sample_video 里也有视频，却
            # 没有一条可选。启动日志里写明原因和文件路径，便于对照排查。
            logger.warning(
                "示例视频清单为空（demo.videos=[]）：监控页「画面源」只有实时相机。"
                "配置文件：%s",
                self.config_path,
            )
        return cfg

    def save(self, cfg: OilCheckConfig) -> None:
        _write_json_atomic(
            self.config_path, cfg.to_dict(), comments=CONFIG_COMMENTS
        )


# 配置文件里的注释表。保存时自动写回，UI 改参数不会把注释弄丢。
# 键是用点分隔的路径，对象数组用下标（例如 stations[0].tray_roi）。
CONFIG_COMMENTS: dict[str, str] = {
    "version": "配置结构版本，供自动迁移判断，不要手工改",
    # ---- pipeline：取料判定 ----
    "pipeline": "时序判定参数。单位：时间=秒，距离=按画面对角线归一化",
    "pipeline.max_fps": (
        "检测最大帧率。也是帧率自适应的基准：实际帧间隔超过 1/max_fps 时，"
        "距离门限按倍率放大（见 frame_scale_max）"
    ),
    "pipeline.pick_hand_distance": (
        "手-丝杆关联时视为「够得着」的最大归一化中心距离"
    ),
    "pipeline.pick_sample_every_frames": (
        "取料判定的采样间隔（帧）：每隔这么多帧记一次「是否在盘内 / "
        "手里是否有杆」的状态"
    ),
    "pipeline.pick_history_samples": (
        "取料判定保留的采样点数：用最近这么多个采样点投票，避免单帧抖动导致漏判或误判。\n"
        "与上一项相乘决定离盘后攒够多少个盘外采样才定论（= 发取料事件的延迟）"
    ),
    "pipeline.pick_confirm_s": (
        "手在涂油盘内的最短停留时长。低于该值视为从盘边划过，不判取料。"
        "纯时间阈值，不随帧率缩放"
    ),
    # ---- pipeline：手部跟踪 ----
    "pipeline.hand_track_max_distance": (
        "手部轨迹匹配的最大归一化中心距离。低帧率时按帧间隔自动放大"
    ),
    "pipeline.hand_track_max_missing_s": "手部轨迹漏检后保留时长，超过即删除轨迹",
    "pipeline.hand_track_max_active": (
        "同时保留的手部轨迹上限。漏检轨迹在保留期内仍占名额，设小了会让新的"
        "手检测框建不了轨迹（画面里常有手臂、盘边被误检成 hand）"
    ),
    "pipeline.hand_track_min_hits": "手部轨迹需命中多少帧才参与刷头靠近判定",
    # ---- pipeline：刷头跟踪 ----
    "pipeline.brush_track_max_distance": "刷头轨迹匹配的最大归一化中心距离",
    "pipeline.brush_track_max_missing_s": "刷头轨迹漏检后保留时长",
    "pipeline.brush_track_max_active": "同时保留的刷头轨迹上限",
    "pipeline.brush_track_min_hits": "刷头轨迹需命中多少帧才参与靠近判定",
    # ---- pipeline：二次涂油判定 ----
    "pipeline.brush_contact_distance": (
        "刷头工作点到手持丝杆中心的归一化距离小于该值，本帧计为「靠近」"
    ),
    "pipeline.brush_hold_s": (
        "刷头靠近累计到该时长即判「人工二次涂油完成」。纯时间阈值，"
        "不随帧率缩放"
    ),
    "pipeline.oiling_timeout_s": (
        "取料后超过该时长仍未完成二次涂油即判漏涂 NG，不等下一次取料；"
        "0=不限时。按媒体时间累计，掉帧不暂停"
    ),
    "pipeline.brush_hold_max_gap_s": (
        "刷头整帧漏检时允许补回的最大空档。用于低帧率下不丢时间；"
        "刷子检测到了但离得远不补"
    ),
    # ---- pipeline：周期 ----
    "pipeline.holding_lost_timeout_s": (
        "手部轨迹消失后，进盘过程作废的时长（取料判定用）"
    ),
    "pipeline.single_active_task": (
        "预留项：当前判定逻辑本身就只维护一个活动周期，改这里不会变成多任务"
    ),
    "pipeline.task_max_active": "内存中最多保留多少个作业周期",
    "pipeline.completed_keep_count": (
        "已结束（完成/NG）的周期保留最近多少条，供监控页任务卡片显示"
    ),
    "pipeline.completed_retain_s": (
        "已结束周期最多再保留这么久即回收（与上一项取交集）；0=不限时"
    ),
    # ---- pipeline：涂油盘定位 ----
    "pipeline.tray_max_distance": "涂油盘跟踪的匹配距离（盘体上下移动时用）",
    "pipeline.tray_max_missing_s": "涂油盘检测漏检后保留预测框的时长",
    "pipeline.tray_min_confidence": "低于该置信度的盘体检测不参与盘区更新",
    # ---- pipeline：手-丝杆与丝杆锚点 ----
    "pipeline.hand_screw_max_distance": (
        "手框与丝杆框的关联距离上限；还要求两者交叠才认作「握着」"
    ),
    "pipeline.hand_interaction_margin": (
        "预留项：当前判定未使用，手-丝杆关联直接按交叠与距离判定"
    ),
    "pipeline.held_infer_distance": (
        "预留项：丝杆锚点跟踪上线后不再使用（锚点按位置连续性跟随）"
    ),
    "pipeline.held_predict_hold_s": (
        "手/丝杆短时遮挡时按预测位置继续的时长。按帧率自动放大"
    ),
    "pipeline.screw_anchor_max_step": (
        "丝杆锚点单帧最大位移。取料后按最近邻连续性跟这根杆，"
        "既跟得住快速移动，又不会跳到相邻丝杆（盘里间距很小）"
    ),
    "pipeline.screw_anchor_max_step_abs": (
        "锚点单帧位移的绝对上限，不随帧率放大。用于低帧率下防止锚点跳到相邻丝杆；\n"
        "必须不小于 screw_anchor_max_step（取料瞬间手带杆打出画面，单帧位移可达 0.11）"
    ),
    "pipeline.screw_anchor_hold_s": (
        "锚点当帧找不到时，按锚点位置继续跟踪的时长；超出则在下面时限内"
        "尝试重新捕获"
    ),
    "pipeline.screw_reacquire_timeout_s": (
        "手持丝杆离开画面后允许重新捕获的时长。工人取完料可能拿着杆走出画面、\n"
        "过一会儿再回来涂油，这段时间内周期保持已取料并继续等；0=不限时"
    ),
    "pipeline.frame_scale_max": (
        "帧率自适应倍率上限。门限按「实际帧间隔 / (1/max_fps)」放大，"
        "但不超过该值，避免低帧率时门限大到失去区分度"
    ),
    "pipeline.roi_track_margin": (
        "已跟踪目标移出工作追踪区后继续跟踪的距离余量。区外目标不会新建轨迹"
    ),
    # ---- pipeline：跟踪后端 ----
    "pipeline.hand_tracker_backend": "auto=优先 ByteTrack，失败回退自研；可选 bytetrack / object",
    "pipeline.brush_tracker_backend": "同上，刷头专用",
    "pipeline.bytetrack_high_thresh": "ByteTrack 高置信度检测阈值（第一阶段关联）",
    "pipeline.bytetrack_low_thresh": "ByteTrack 低置信度检测阈值（第二阶段补救关联）",
    "pipeline.bytetrack_new_thresh": "达到该置信度才允许新建轨迹",
    "pipeline.bytetrack_match_thresh": (
        "关联代价阈值。未进门限的代价固定为 1.5，所以 1.0 表示「只看距离/IoU 门限」；\n"
        "调小可加严（现场标定用）"
    ),
    "pipeline.bytetrack_buffer_s": (
        "轨迹丢失后保留的时长（按 max_fps 换算成帧数交给 ByteTrack）"
    ),
    "pipeline.bytetrack_cost_alpha": (
        "关联代价权重：α·(1-IoU) + (1-α)·(中心距离/上限)。"
        "调大更看重框重叠，调小更看重中心距离"
    ),
    # ---- recording ----
    "recording": "训练视频录制参数",
    "recording.training_dir": "录制输出目录，相对数据目录解析",
    "recording.fps": "录制帧率",
    "recording.segment_seconds": "单个分段视频的时长，超过即切新文件",
    "recording.codec": "OpenCV VideoWriter 编码，例如 mp4v / XVID",
    "recording.auto_start": "启动后是否自动开始录制",
    "recording.event_clips_enabled": "预留项：事件前后片段的开关，当前未实现",
    # ---- demo ----
    "demo": "示例视频回放（画面来自 mp4，检测仍由模型完成）",
    "demo.loop": "播放结束后是否从头循环",
    "demo.fps_limit": "投放帧率上限，实际取 min(视频原帧率, 该值)",
    "demo.speed": (
        "播放速率倍数（0.1~4）。只改播放快慢，判定按视频媒体时间计时，"
        "结果不受影响"
    ),
    "demo.videos": "示例视频清单，监控页「画面源」下拉按这里的 name 生成",
    "demo.videos[].name": "下拉里显示的名字",
    "demo.videos[].path": "视频路径，相对仓库根或绝对路径",
    # ---- stations ----
    "stations": "工位列表，每个工位一台相机",
    "stations[].station_id": "工位 ID，事件与日志里用它区分",
    "stations[].name": "工位显示名",
    "stations[].camera_id": "对应 cameras.json 里的相机 ID",
    "stations[].enabled": "是否启用该工位",
    "stations[].tray_roi": (
        "涂油盘区的手工底区，用于判断「手进盘/离盘」。必须把盘体连同"
        "下方手臂活动范围一起框进去：模型只框得出盘体本身，运行时两者取并集"
    ),
    "stations[].work_roi": (
        "工作追踪区，跟踪与判定的有效范围。只有中心落在该区（或涂油盘区）内"
        "的目标才新建轨迹"
    ),
    "stations[].oiling_roi": (
        "涂油区，标定与显示用，本轮不参与判定，预留给后续涂油范围约束"
    ),
    "stations[].tray_roi.kind": "区域类型，目前只支持 rect",
    "stations[].tray_roi.points": "归一化坐标 0~1：左上角、右下角两个点",
    "stations[].work_roi.kind": "区域类型，目前只支持 rect",
    "stations[].work_roi.points": "归一化坐标 0~1：左上角、右下角两个点",
    "stations[].oiling_roi.kind": "区域类型，目前只支持 rect",
    "stations[].oiling_roi.points": "归一化坐标 0~1：左上角、右下角两个点",
    # ---- detector ----
    "stations[].detector": "检测模型与类别名",
    "stations[].detector.kind": "null=不检出（框架自检）；yolo=用 Ultralytics 权重",
    "stations[].detector.model_path": "权重路径，相对仓库根或绝对路径",
    "stations[].detector.confidence": (
        "检测置信度阈值。要低于 bytetrack_high_thresh（0.3），否则低于该值的框\n"
        "在检测层就被丢掉，ByteTrack 的低分补救关联（第二阶段）永远收不到输入"
    ),
    "stations[].detector.iou": "NMS 的 IoU 阈值",
    "stations[].detector.max_det": "单帧最多输出多少个检测框",
    "stations[].detector.imgsz": (
        "推理输入尺寸。0=自适应：优先用权重里训练时的尺寸，取不到回退 640"
    ),
    "stations[].detector.device": "auto/空=有 CUDA 用 cuda:0；也可显式 cpu / cuda:0 / 0",
    "stations[].detector.half": "是否用 FP16 推理（只在 CUDA 上生效）",
    "stations[].detector.raw_mode": (
        "旁路排查模式：只交出全量检测结果，不建轨迹、不判定、不产生事件"
    ),
    "stations[].detector.screw_label": "模型输出的丝杆类别名",
    "stations[].detector.hand_label": "模型输出的手部类别名",
    "stations[].detector.brush_label": "模型输出的刷具类别名",
    "stations[].detector.tray_label": "模型输出的涂油盘类别名",
    "stations[].detector.brush_point_offset_x": (
        "刷子没有关键点时，用框内相对坐标（0~1）估计工作端：横向位置"
    ),
    "stations[].detector.brush_point_offset_y": (
        "刷子没有关键点时，用框内相对坐标（0~1）估计工作端：纵向位置"
    ),
}


def _default_demo_speed() -> float:
    """示例回放默认速率（与 :class:`DemoConfig` 保持一致）。"""

    return DemoConfig().speed


def strip_json_comments(text: str) -> str:
    """去掉 JSONC 里的 ``//`` 与 ``/* */`` 注释，供 ``json.loads`` 使用。

    配置文件带中文注释，标准 JSON 解析器不认。这里只处理**字符串之外**的
    注释，避免把 ``"http://..."`` 这类值里的双斜杠当成注释。
    """

    out: list[str] = []
    i = 0
    length = len(text)
    in_string = False
    escaped = False
    while i < length:
        ch = text[i]
        if in_string:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < length:
            nxt = text[i + 1]
            if nxt == "/":
                # 行注释：跳到行尾，换行符保留（行号不偏移）。
                while i < length and text[i] != "\n":
                    i += 1
                continue
            if nxt == "*":
                i += 2
                while i + 1 < length and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def add_json_comments(payload: object, comments: dict[str, str]) -> str:
    """按 ``{"a.b": "说明"}`` 的注释表序列化 JSON，把注释写在对应键的上一行。

    自己实现序列化而不是先在文本里找行，是因为配置里有嵌套对象和数组
    （``stations`` 是对象数组），靠行扫描判断「当前处在哪个键下」很容易错位。
    这里递归下降，路径天然正确。
    """

    def lookup(path: str) -> str | None:
        """先查精确路径，再查 ``stations[].xxx`` 这种通配写法。"""

        note = comments.get(path)
        if note is not None:
            return note
        wildcard = re.sub(r"\[\d+\]", "[]", path)
        return comments.get(wildcard)

    def dump(value: object, path: str, indent: int) -> list[str]:
        pad = " " * indent
        if isinstance(value, dict):
            if not value:
                return ["{}"]
            lines = ["{"]
            items = list(value.items())
            for pos, (key, item) in enumerate(items):
                child = f"{path}.{key}" if path else key
                note = lookup(child)
                if note:
                    # 说明里可以带换行：按行各写一条 // 注释，避免出现超长行，
                    # 也让注释表与仓库模板逐字一致（模板由注释表渲染生成）。
                    for note_line in note.splitlines() or [""]:
                        lines.append(f"{pad}  // {note_line}".rstrip())
                rendered = dump(item, child, indent + 2)
                body = rendered[0] if len(rendered) == 1 else None
                if body is not None:
                    suffix = "," if pos < len(items) - 1 else ""
                    lines.append(f'{pad}  "{key}": {body}{suffix}')
                else:
                    rendered[-1] += "," if pos < len(items) - 1 else ""
                    lines.append(f'{pad}  "{key}": {rendered[0]}')
                    lines.extend(rendered[1:])
            lines.append(f"{pad}}}")
            return lines
        if isinstance(value, list):
            if not value:
                return ["[]"]
            # 简单标量数组写成一行，对象数组逐项展开。
            if all(not isinstance(v, (dict, list)) for v in value):
                return ["[" + ", ".join(json.dumps(v, ensure_ascii=False) for v in value) + "]"]
            lines = ["["]
            items = list(value)
            for pos, item in enumerate(items):
                child = f"{path}[{pos}]"
                rendered = dump(item, child, indent + 2)
                rendered[-1] += "," if pos < len(items) - 1 else ""
                lines.extend(f"{pad}  {line}" if i == 0 else line for i, line in enumerate(rendered))
            lines.append(f"{pad}]")
            return lines
        return [json.dumps(value, ensure_ascii=False)]

    return "\n".join(dump(payload, "", 0))


def load_json_with_comments(path: Path) -> Any:
    """读取带注释的 JSON 文件。"""

    text = path.read_text(encoding="utf-8")
    return json.loads(strip_json_comments(text))


def _write_json_atomic(
    path: Path, payload: object, *, comments: dict[str, str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            text = (
                add_json_comments(payload, comments)
                if comments
                else json.dumps(payload, ensure_ascii=False, indent=2)
            )
            f.write(text)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        Path(tmp_name).replace(path)
    finally:
        try:
            os.remove(tmp_name)
        except FileNotFoundError:
            pass
        except OSError as exc:
            logger.debug("删除临时文件失败 (%s): %s", tmp_name, exc)
