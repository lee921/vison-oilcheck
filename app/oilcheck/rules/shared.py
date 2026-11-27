"""跨判定环节共用的小工具（mixin）。

这些方法被取料、丝杆离盘、涂油三处调用，单独放在这里，三个判定 mixin 都
继承它。它们只依赖 ``self`` 上的配置与状态字段（由
:class:`~app.oilcheck.pipeline.StationPipeline` 提供），不引入新的状态。

分组：

- **时间门限的帧率换算**：``_max_plausible_dt`` / ``_at_least_frame``
  / ``_carry_wait_s``。判定按秒计时，只给「比一帧还短」的门限兜一个
  帧间隔下限，不随帧率放大（见 docs/05 §5.0.3）。
- **进盘定性**：``_record_outside_holding`` / ``_mostly_empty_outside``
  / ``_holding_screw`` / ``_same_rod_as_entry``。
- **静置杆与 carry 证据**：``_screw_static`` / ``_screw_parked_in_tray``
  / ``_carrying_out_of_tray`` / ``_hand_over_split_visits``。
- **取料确认与二级判定**：``_confirm_pick`` / ``_pick_features`` /
  ``_judge_pick``。
- **涂油与周期的挂接**：``_episode_bounds`` / ``_drain_pending_oilings``
  / ``_close_oiling_if_settled`` / ``_same_anchor``。
"""

from __future__ import annotations

import math
from collections import deque
from statistics import median

from ..constants import (
    _CARRY_LOCK_STEP,
    _CARRY_OUT_OF_TRAY_MAX_S,
    _GRIP_PARKED_MIN_FRAC,
    _INVENTORY_SAME_IOU,
    _OILING_SAME_ROD_STEP,
    _PICK_SOURCE_HAND,
    _RULE_TRUE_VETO_P,
    _SCREW_STATIC_IOU,
    _SCREW_STATIC_LOOKBACK_S,
)
from ..geometry import (
    _box_center_in,
    _grip_corner_fraction,
    _gripped_by_corner,
)
from ..judge import Episode, JudgeResult
from ..review import build_pick_summary
from ..state import _CarryEvidence, _Cycle, _HandVisit
from ..types import (
    BBox,
    Detection,
    EventType,
    TaskStage,
)


class SharedRules:
    """三处判定共用的辅助方法。"""

    def _pick_features(
        self, visit: _HandVisit, *, ts: float, dwell: float
    ) -> dict[str, float]:
        """取料判定的特征向量：**规则字段 + 原始检测框聚合特征**。

        两套字段并存，各有分工：

        - ``hs_n`` / ``hs_true`` / ``seen`` / ``carry_*``：规则自己算出来的
          结论，用于 ``_judge_pick`` 的门限（规则这一路投没投票）；
        - 原始几何聚合（``rod_moved_max`` / ``rod_iou_max`` / ``found_ratio``
          …）：模型的输入，全部能从 YOLO 检测框复算（见 ``raw_features``）。

        模型只读原始那套——它必须能独立判断，而不是复述规则结论。
        """

        carry = visit.carry
        samples = list(visit.holding_samples)
        rods_in_tray = sum(
            1
            for det in self._last_detections
            if det.label == self.station.detector.screw_label
            and self._tray_region_box(self._frame_size) is not None
            and _box_center_in(
                det.box, self._tray_region_box(self._frame_size)  # type: ignore[arg-type]
            )
        )
        values: dict[str, float] = {
            "dwell": float(dwell),
            "hs_n": float(len(samples)),
            "hs_true": float(sum(1 for value in samples if value)),
            "carry_frames": float(carry.frames),
            "seen": float(visit.holding_seen),
            "inventory_n": float(len(visit.entry_inventory)),
            "carry_inhand_frames": float(carry.inhand_frames),
            "carry_inhand_max": float(carry.max_inhand),
            "carry_rod_move": float(carry.rod_move),
            "carry_hand_move": float(carry.hand_move),
            "carry_align": self._carry_alignment(carry),
            "rods_in_tray_delta": float(
                len(visit.entry_inventory) - rods_in_tray
            ),
            # 以下两项只给训练脚本打标签用（进盘/离盘区间），不是判定特征。
            "_visit_start": float(visit.first_inside_ts or ts - dwell),
            "_visit_exit": float(visit.exited_at_ts or ts),
        }
        values.update(
            visit.raw.features(dwell=dwell, start_tray_rods=visit.raw_entry_rods)
        )
        return values

    def _judge_pick(
        self, values: dict[str, float], *, rule: bool
    ) -> JudgeResult | None:
        """取料二级判定：规则与逻辑回归的融合。

        融合的关键是**分工边界**。现场实测（10 段示例视频）说明两层证据
        互补：

        - 规则擅长「持握采样多数为真」这一路——盘外手持杆检出率 96~99%；
        - 模型擅长「采样不足、但有零散线索」这一路——杆跟着手动、杆持续
          落在手框里、或只见过一两帧握持。这类帧恰恰是规则投不出票的地方。

        但规则也会**误投**：工人整理料盘时，手伸进盘里把杆挪一挪、停留
        六七秒，规则凭「杆长时间落在手框里 + 杆确实动过」投「是」，而
        那根杆从没跟着手走。模型对这种形态给的概率非常低（实测 ≤0.196），
        所以规则投「是」时也要过一道模型否决门（``_RULE_TRUE_VETO_P``）。

        门限按「规则这一路有没有投出票」来分：

        - 规则多数投「是」（``rule=True``）：模型概率 ≥ ``_RULE_TRUE_VETO_P``
          才维持「是」，否则改判「否」。
        - 规则投「否」但见过握持（``hs_true > 0``）：**模型说了算**。
          这正是示例⑦ 第 1 件的形态（``hs=1/3``、carry 一帧都没有），
          规则多数票判否会把真取料丢掉，模型 p=0.65 能救回来。
        - 一帧握持都没有（``hs_true == 0``）：空手离开，规则与模型都判否。
        """

        if self._pick_judge is None:
            return None
        if values["hs_true"] <= 0.0 and not rule:
            # 一帧握持都没有、规则也投否：空手离开，模型不必参与。
            return self._pick_judge.decide(values, rule=False)
        if rule:
            # 规则投「是」，但模型**强烈反对**时以模型为准。
            #
            # 原先这里直接 ``rule=rule``——规则投真就完全不让模型参与。
            # 代价是整理料盘被判成取料：工人手伸进盘里把杆挪一挪、停留
            # 6.75 秒，规则凭「杆长时间落在手框里 + 杆确实动过」投真，而
            # 模型给 p=0.146 明确判否（现场 2026-11-27 10:40，片段
            # TASK-000002_oiling_missing_timeout_20261127_104002.mp4）。
            # 这次误判还连带报了一次漏涂 NG。
            #
            # 门限取 0.4 而不是 0.5：实测 25 次真取料（``rule=True``）的
            # 概率全部 ≥ 0.653，4 次误判全部 ≤ 0.196，中间空档很大；取
            # 0.4 离两侧都留有余量。模型判否只否决**明确的假阳性**，
            # 概率落在 [0.4, 0.5) 的中间带仍按规则处理。
            probability = self._pick_judge.probability(values)
            return JudgeResult(
                probability, probability >= _RULE_TRUE_VETO_P, True, dict(values)
            )
        # 规则投否、但有过握持：中间带，模型定。
        return self._pick_judge.decide(values, rule=None)

    def _record_outside_holding(
        self, hand_id: int, holding: Detection | None, ts: float
    ) -> None:
        """记录一次「手在盘外、手里有没有杆」的观测。

        这条历史只用于**进盘定性**（这一趟是空手进盘还是带料进盘），
        记录前要剔除盘内的杆：

        手要经过盘边时，手框（实测 348×399 px，接近盘宽）必然把盘里那排
        静置杆框进去，这些杆的**中心**就落在手框内，命中「杆心在手框内」
        那条基础判据。一条这样的观测进了历史，本次进盘就被判成「带料进盘」，
        整件取料丢掉——示例② 第 95 帧附近的取料即如此（f94 手在盘外，
        盘内 (1544,319) 那根静置杆被判成握持）。

        只在这里剔除、不动 ``_holding_screw`` 本身：离盘后的握持采样仍要
        认盘内的杆——取料瞬间手里那根杆往往还压在盘区上方，那正是这一件
        料被带走的证据（视频② f463 那次取料的前两个采样点就落在盘区内，
        提前剔除会把整件取料判丢）。
        """

        if holding is not None and self._screw_parked_in_tray(holding):
            holding = None
        # 同一帧只记一次。三处调用点（新建轨迹、采样点、未确认进盘）
        # 看似互斥，实测同一 ``ts`` 会被调用 2~3 次：``pick_history_samples``
        # 默认 3，一帧就能把投票窗口灌满，多数票退化成「单帧随机数」。
        last = self._outside_holding_last_ts.get(hand_id)
        if last is not None and last == ts:
            return
        self._outside_holding_last_ts[hand_id] = ts
        window = max(0.0, self._entry_grace_s())
        history = self._outside_holding.setdefault(hand_id, deque())
        # 「手里没杆」是一次**状态复位**：从这一刻起的观测才是这一趟进盘前
        # 的有效依据。不清掉之前的「手持杆」观测，上一次取料后拿着杆走动的
        # 那串记录会一直留在窗口里，下一次空手进盘被投成「带料进盘」，整件
        # 取料判丢（示例③在 7.5fps 下第 2 件即如此）。
        if holding is None:
            history.clear()
        history.append((holding is None, ts))
        while history and ts - history[0][1] > window:
            history.popleft()

    def _screw_static(self, box: BBox) -> bool:
        """这根杆的位置是否**没动过**（与 ``_SCREW_STATIC_LOOKBACK_S`` 秒前同位）。

        只按位置稳定性判断，与杆在不在盘区里无关：盘里那排杆、零件盒里的
        整排杆都不动；被手带走的杆即使中心还没离开盘区，框也已经在动。
        实测四段示例视频里不动的杆对该窗口内位置的 IoU ≥0.5（中位 0.9+），
        被带走的杆 ≤0.3。

        历史不足时不作判断（返回 False）：开局/复位后的前几帧没有对照，
        此时不能凭它否定握持，否则真取料会被吞掉。
        """

        if self._last_frame_ts is None:
            return False
        # 对照取「回看窗口内**最早的两帧**」，前提是窗口里真有两帧以上。
        #
        # 单帧比对时，只要那一帧恰好漏检了这根杆，位置从未变动的杆就会被
        # 判成「在动」；一旦锚点借此捕获了静置杆，`_static_anchor_ok` 的
        # 「锚点 ±0.02 同根」例外会让它一直合法，之后刷真杆的接触时长不再
        # 累计 → 漏涂 NG。
        #
        # 但低帧率下（每帧间隔接近或超过回看窗口）窗口里可能只有一两帧，
        # 「最早两帧」和「最近两帧」是同一份，这时只能拿它当唯一对照。
        reference = self._screw_history_at(
            self._last_frame_ts - _SCREW_STATIC_LOOKBACK_S, take=2
        )
        if not reference:
            return False
        return any(
            box.iou(past_box) >= _SCREW_STATIC_IOU
            for frame in reference
            for past_box in frame
        )

    def _hand_over_split_visits(self, tray_box: BBox | None, ts: float) -> None:
        """手部 ID 在进盘途中被重建时，把没走完的进盘过程交给真正在盘里的那只手。

        跟踪器在掉帧/遮挡后会保留旧轨迹（``matched=False``，位置停在最后一次
        观测），同时给回来的手**新建**一条轨迹。于是出现这一幕：旧 ID 停在
        盘外不动、新 ID 在盘里干活，随后出盘又回落到旧 ID——进盘记在新 ID
        名下、离盘发生在旧 ID 名下，两边永远凑不齐，整件取料丢掉。示例④
        第 3 次取料即如此（f339 盘内是 100004，f347 出盘时已是 100003）。

        做法：找出「已确认进盘、但主人轨迹这一帧已失配」的过程，把它转给
        这一帧**确实在盘内**、且离它最后一次位置最近的那只手。宁可漏接也
        不要张冠李戴，所以三个条件都要满足：

        - 过程是进行中的（``entered_confirmed`` 且未离盘）；
        - 主人这一帧没有匹配上（``matched=False``）；
        - 接手的这只手这一帧在盘内，且距离在 ``hand_track_max_distance`` 内。
        """

        if not self._visits:
            return
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        matched = {
            track.track_id: track
            for track in self._hand_tracks
            if track.matched and track.hits >= max(1, self._hand_tracker.min_hits)
        }
        limit = self.cfg.hand_track_max_distance * self._frame_scale
        for owner_id, visit in list(self._visits.items()):
            if not visit.entered_confirmed or visit.exited_at_ts is not None:
                continue
            if owner_id in matched or visit.last_hand_box is None:
                continue
            ox, oy = visit.last_hand_box.center
            best: tuple[float, int] | None = None
            for track_id, track in matched.items():
                if track_id == owner_id:
                    continue
                # 候选轨迹自己也在走一段**已确认进盘**的过程时不动它，否则
                # 会把两只同时干活的手搅在一起。只是挂着个还没进盘的空壳
                # visit 的（掉帧期间新建、还没确认）可以直接顶掉：那几帧盘外
                # 握持历史本来就属于同一只手。
                other = self._visits.get(track_id)
                if other is not None and other.entered_confirmed:
                    continue
                hb = track.observation_box()
                hx, hy = hb.center
                if tray_box is not None and not (
                    tray_box.x1 <= hx <= tray_box.x2
                    and tray_box.y1 <= hy <= tray_box.y2
                ):
                    continue
                distance = math.hypot(ox - hx, oy - hy) / diag
                if distance > limit:
                    continue
                if best is None or distance < best[0]:
                    best = (distance, track_id)
            if best is None:
                continue
            _, new_id = best
            self._visits.pop(owner_id, None)
            self._visits.pop(new_id, None)
            visit.hand_track_id = new_id
            visit.last_seen_ts = ts
            visit.rebinds += 1
            visit.restart_holding()
            visit.holding_seen = False
            self._visits[new_id] = visit
            self._logger.debug(
                "手部轨迹 %s → %s 接管进盘过程（盘内已停留 %.2fs，接管 %d 次）",
                owner_id,
                new_id,
                visit.dwell_in_tray_s,
                visit.rebinds,
            )

    def _carrying_out_of_tray(
        self,
        hand_box: BBox,
        inventory: tuple[BBox, ...] = (),
        evidence: _CarryEvidence | None = None,
        *,
        ts: float | None = None,
    ) -> bool:
        """手已在盘外、但手边那根正随着动的杆还没出盘。

        这种帧不能当成「手里没杆」的采样点：杆心晚几帧出盘是常态（示例④
        第 3 次取料手 f348 出盘、杆 f352 才出盘），把它记成 False 会让
        采样窗口多数票投出「空手」。它是这一件正在出门的证据，只是还没
        跨过盘区边线。等待上限见 ``_CARRY_OUT_OF_TRAY_MAX_S``。

        **盘里的存货不算**。判据是「这根杆不在进盘时的存货快照里」——
        存货在这趟进盘前后一直在盘里；手上那根是这趟新进入盘区的。只看
        位置稳不稳不够：工人把杆停在盘区下沿歇一下（示例⑤ 第 2 次取料
        停了 2s 多），``_screw_static`` 就不再认为它在动，整件取料被当成
        「空手离盘」丢掉。

        ``inventory`` 为空（历史不足）时退回位置稳定性判据。

        传了 ``evidence`` 时，顺带把这根候选杆的握持证据累积进去（见
        :class:`_CarryEvidence`）——「手边有盘内杆」只是必要条件，够不够
        算握持由调用方按 :attr:`_CarryEvidence.carried` 决定。
        """

        tray_box = self._tray_region_box(self._frame_size)
        if tray_box is None:
            return False
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        hx, hy = hand_box.center
        width, height = self._frame_size
        candidates: list[tuple[float, BBox]] = []
        for det in self._last_detections:
            if det.label != self.station.detector.screw_label:
                continue
            cx, cy = det.box.center
            distance = math.hypot(cx - hx, cy - hy) / diag
            if distance > self.cfg.hand_screw_max_distance:
                continue
            if not _box_center_in(det.box, tray_box):
                continue
            if inventory:
                # 与存货快照里任一根重合 ⇒ 它是盘里的料，与本件无关。
                if any(det.box.iou(box) >= _INVENTORY_SAME_IOU for box in inventory):
                    continue
            elif self._screw_static(det.box):
                continue
            candidates.append((distance, det.box))
        if not candidates:
            if evidence is not None:
                evidence.gaps += 1
            return False
        rod = None
        if evidence is not None and evidence.last_rod is not None:
            # 已锁定：这一帧只认锁定位置附近的那根。附近找不到就记一次
            # 空缺、跳过这一帧，**不**退回到「离手最近的那根」——那正是换
            # 目标的地方，两根杆的位移拼在一起会拼出虚假的「随手动」
            # （示例⑨ 夹具上相邻杆间距 0.03，跳一次 rod_move 就到 0.015）。
            near = [
                (math.hypot(box.center[0] / width - evidence.last_rod[0],
                            box.center[1] / height - evidence.last_rod[1]), box)
                for _, box in candidates
            ]
            near = [item for item in near if item[0] <= _CARRY_LOCK_STEP]
            if not near:
                evidence.gaps += 1
                return True
            rod = min(near, key=lambda item: item[0])[1]
        else:
            rod = min(candidates, key=lambda item: item[0])[1]
        if evidence is not None:
            self._update_carry_evidence(evidence, hand_box, rod, ts=ts)
        return True

    def _screw_parked_in_tray(self, screw: Detection) -> bool:
        """这根杆是不是盘里那根**没被动过**的静置杆。

        只看「中心在不在盘区」不够：手工 ``tray_roi`` 把盘体下方的手臂活动
        范围也框进去了，取料时手里那根杆的中心常常还落在盘区内（示例④第 2、
        3 次取料的离盘初期整段如此），按中心判定就会把正被带走的杆当成静置
        杆排除掉，整件取料判不出来。

        真正的区别在**位置稳定性**（见 :meth:`_screw_static`）。

        中心不在盘区的杆一律不是静置杆；历史不足时按静置处理（保守：
        盘区外的手在这一点上不做宽松推断）。
        """

        tray_box = self._tray_region_box(self._frame_size)
        if tray_box is None or not _box_center_in(screw.box, tray_box):
            return False
        if not self._screw_history_at(
            (self._last_frame_ts or 0.0) - _SCREW_STATIC_LOOKBACK_S
        ):
            return True
        return self._screw_static(screw.box)

    def _mostly_empty_outside(
        self, hand_id: int, ts: float, *, default_empty: bool = False
    ) -> bool:
        """进盘前**最近几次**盘外观测里，「空手」是否占多数。

        只看最近 ``pick_confirm_frames + 1`` 次观测，不用时间窗：时间窗在
        低帧率下会被放大到几秒，把上一件涂油时「手持杆」的观测也卷进来，
        于是每次进盘都被当成「带料进盘」。

        观测不足 2 次时直接采用那一次的结果——低帧率下盘外可能只采到一帧，
        此时没有更好的依据，退回单帧判断总好过一律拒绝取料。
        """

        history = self._outside_holding.get(hand_id)
        window = max(0.0, self._entry_grace_s())
        recent = (
            [(empty, t) for empty, t in history if ts - t <= window]
            if history
            else []
        )
        if not recent:
            # 这只手从未在盘外被观测到（例如第一次建轨迹就在盘内，或者
            # 太久没见已滚出时间窗）：没有依据，按调用方的保守值处理。
            # 之前这里直接返回 False（=“不是空手”），会让“空手进盘”一律
            # 被当成“带料进盘”，取料整件丢掉。
            return default_empty
        limit = max(2, int(self.cfg.pick_history_samples))
        recent = recent[-limit:]
        if len(recent) < 2:
            # 观测不足（低帧率下盘外只采到一帧）：按调用方给的保守值处理。
            # 进盘判定宁可放过（``default_empty=True``），因为漏判一件的
            # 代价高于多判一次——多出来的那次取料会因涂油未完成而进复核。
            return default_empty
        empty_count = sum(1 for empty, _ in recent if empty)
        # 平局按「空手」处理：进盘判定宁可放过一件（后续涂油判定会拦），
        # 也不要因为一两帧的杆交叠把整件取料吞掉。
        return empty_count * 2 >= len(recent)

    def _max_plausible_dt(self, current_gap: float | None = None) -> float:
        """单帧可信的最大间隔：按**已建立的节拍**的两倍计。

        掉帧时 ``dt`` 越大，累计时长越容易被凭空放大：一帧 0.95s 的间隔
        能让刷头只看过 3 帧就判完成（实测）。上限取「最近帧间隔中位数」的
        两倍，而不是配置里的 ``max_fps``——现场可能本来就只有 2~5fps，
        用配置值会把正常帧全判成不可信。还没建立节拍时，用本帧间隔自身
        作为节拍（第一帧无法区分「低帧率」和「掉帧」，按前者处理）。
        """

        base_dt = 1.0 / max(1, self.cfg.max_fps)
        if self._gap_samples:
            typical = median(self._gap_samples)
        elif current_gap is not None and current_gap > 0:
            typical = float(current_gap)
        else:
            typical = base_dt
        return 2.0 * max(typical, base_dt)

    def _at_least_frame(self, seconds: float) -> float:
        """时间门限的下限：至少覆盖一个实际帧间隔。

        以秒计的门限**不需要**按帧率缩放（秒本来就是帧率无关的量）。但低帧率
        下如果门限比一帧还短，判定会退化成「逐帧随机」——比如 2fps 时
        ``brush_hold_max_gap_s`` 只有 0.5s、还不到一帧，任何两帧之间都会被
        算成「超出补回上限」。所以下限取一帧，上限不动。

        这也顺带修掉原来「乘以 ``frame_scale``」带来的反向漂移：15fps 下
        scale=1.33、20fps 下 scale=1.0，同一个配置值在高帧率下反而更严格，
        判定随帧率朝反方向变。
        """

        return max(float(seconds), self._last_frame_gap_s or 0.0)

    def _carry_wait_s(self) -> float:
        """手出盘后等「杆也出盘」的最长时长。

        见 ``_CARRY_OUT_OF_TRAY_MAX_S``：这段时间不能用离盘后的固定结算
        时长，杆心比手晚出门一秒上下是常态。
        """

        return self._at_least_frame(_CARRY_OUT_OF_TRAY_MAX_S)

    def _holding_screw(self, hand_box: BBox) -> Detection | None:
        """手里拿着的那根丝杆。严格按「空手进盘、带丝杆出盘」的口径：

        1. **杆必须已经出盘**。手在盘外时，杆心仍落在手工 ``tray_roi`` 里
           的一律不算手里拿着——那是盘里的料。取料的定义就是「带丝杆**出**
           盘」，杆心还在盘区里就还没出盘。这条不加，「离盘」这一帧手上
           蹭到的盘内杆会被当成取料证据：示例① 第 88~103 帧工人只是把手
           伸进盘里点按夹具，离盘瞬间杆心还在盘区内的那根杆和手框交叠，
           于是凭空多判一件取料；
        2. **杆心落在手框内**，或两者 IoU ≥ 0.15 —— 杆被手完全握住/遮住；
        3. **手框右上角区域与杆框相交** —— 实际握姿是拳头攥住杆身、杆从虎口
           （手框右上角）伸出去，此时杆心已经在手框外、IoU 也上不去。丝杆
           又细又长（面积只有手框的 17~26%），IoU 的结构上限就是面积比，
           0.15 这条门限在握持状态下几乎不可达（示例③实测握持段 IoU 只有
           0.03~0.13，而「杆心在手框内」会随手的朝向时真时假）。

        第 3 条只对**已经在动**的杆放开到「相交即可」；位置没动过的杆
        要求杆有 ``_GRIP_PARKED_MIN_FRAC`` 的面积落进右上区（盘边蹭到的
        实测 0.25~0.42，真攥住的 ≥0.44）。杆被带走时手常常是「握着杆身、
        杆从虎口伸出去」，与右上区的重叠刚好落在两档之间，用同一个 0.30
        会把整段离盘初期判成空手（示例④ 第 3 次取料 f353~f358 即如此）。

        盘内不启用第 2、3 条：手伸在盘上方必然蹭到盘里的杆，那条读数与
        实际有没有握着无关（盘内握持读数本来就不参与投票，见 ``_HandVisit``）。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        label = self.station.detector.screw_label
        hx, hy = hand_box.center
        tray_box = self._tray_region_box((width, height))
        hand_outside = tray_box is None or not _box_center_in(hand_box, tray_box)
        best: tuple[float, Detection] | None = None
        for det in self._last_detections:
            if det.label != label:
                continue
            cx, cy = det.box.center
            distance = math.hypot(cx - hx, cy - hy) / diag
            if distance > self.cfg.hand_screw_max_distance:
                continue
            if (
                hand_outside
                and tray_box is not None
                and _box_center_in(det.box, tray_box)
            ):
                # 杆心还在手工盘区内：手在盘外，说明这根还没被带出盘。
                continue
            gripped = _box_center_in(det.box, hand_box) or hand_box.iou(det.box) >= 0.15
            if not gripped and hand_outside:
                if self._screw_static(det.box):
                    # 位置没动过的杆不可能在手里：盘边/夹具边蹭到的不算。
                    gripped = (
                        _grip_corner_fraction(hand_box, det.box)
                        >= _GRIP_PARKED_MIN_FRAC
                    )
                else:
                    gripped = _gripped_by_corner(hand_box, det.box, min_frac=0.0)
            if not gripped:
                continue
            if best is None or distance < best[0]:
                best = (distance, det)
        return None if best is None else best[1]

    def _confirm_pick(
        self,
        *,
        ts: float,
        hand_track_id: int | None,
        screw: Detection,
        dwell_s: float,
        judge: JudgeResult | None = None,
        source: str = _PICK_SOURCE_HAND,
    ) -> None:
        # 两条取料判据在同一件料上几乎同时确认时（手部判据在手离盘后攒采样、
        # 丝杆判据在杆出盘后确认，相差几百毫秒到一两秒），只报一次。
        # 放在这一层而不是各判据内部：哪一条先判出都有可能（实测两种顺序
        # 都出现过），单边去重挡不住。
        if self._duplicate_pick(ts, screw.box.center):
            self._logger.debug(
                "取料判据 %s 与 %.1fs 前同一位置的取料重复，不重复上报",
                source,
                ts - (self._last_pick_ts or ts),
            )
            return
        self._last_pick_ts = ts
        self._last_pick_center = screw.box.center
        task = self._ensure_cycle(ts, hand_track_id=hand_track_id)
        task.stage = TaskStage.PICKED
        task.stage_since = ts
        task.picked_at = ts
        task.hand_track_id = hand_track_id
        # 取料与涂油并行判定：工人一边取料一边刷时，涂油段起于取料结算
        # 之前，**不能**因为它还没挂上周期就清掉（示例⑤ 6.47~10.20 就是
        # 一整段连续涂抹，切成两段会多报一次）。
        #
        # 但「已经在某一件上报过的段」要收：那是上一件的涂油，新一件的
        # 刷动不该并进去（两件的杆可能落在同一位置，位置判才分不开）。
        self._close_oiling_if_settled()
        task.screw_observed = True
        # 记下取料那一刻手里那根杆的位置：之后这次涂油要么就发生在它上面，
        # 要么在它旁边（工人会走动）。没有它就无从判断「这次涂油属于哪一件」，
        # 另一只手握着别的杆刷够了会被错挂到本周期上（实测）。
        task.screw_anchor = screw.box.center
        task.screw_anchor_ts = ts
        task.confidence = screw.confidence
        task.reason_code = "PICK_CONFIRMED"
        task.last_seen_ts = ts
        task.evidence.append(
            "SCREW_LEFT_TRAY_WITH_HAND"
            if source == _PICK_SOURCE_HAND
            else "SCREW_MOVED_OUT_OF_TRAY"
        )
        task.timeline.append((TaskStage.PICKED.value, ts))
        # 事件的区间优先用逻辑回归分段出来的「这件事从什么时候到什么时候」；
        # 没有模型时退回「结算帧 - 盘内停留」到结算帧（旧口径）。丝杆判据
        # 没有分段器（它不依赖手部轨迹），直接走 fallback。
        start_ts, end_ts, episode = self._episode_bounds(
            "pick",
            owner=hand_track_id,
            fallback_start=ts - dwell_s,
            fallback_end=ts,
        )
        extra = {
            "hand_track_id": hand_track_id,
            "dwell_in_tray_s": round(dwell_s, 3),
            "screw_confidence": round(float(screw.confidence), 3),
            "pick_source": source,
        }
        if judge is not None:
            extra["judge"] = judge.to_dict()
        if episode is not None:
            extra["judge_episode"] = episode.to_dict()
        self._emit_event(
            task,
            event_type=EventType.PICK,
            reason="pick_confirmed",
            extra=extra,
            start_ts=start_ts,
            end_ts=end_ts,
            review=build_pick_summary(task, dwell_s=dwell_s, source=source),
        )
        self._logger.info(
            "取料完成 %s（来源=%s，手部轨迹=%s，盘内停留 %.2fs）",
            task.task_id,
            source,
            hand_track_id if hand_track_id is not None else "-",
            dwell_s,
        )
        # 「一边取料一边刷」：这次涂油在取料结算之前就判完了，补挂上来。
        self._drain_pending_oilings(task, ts)

    def _episode_bounds(
        self,
        kind: str,
        *,
        owner: int | None,
        fallback_start: float,
        fallback_end: float,
    ) -> tuple[float, float, Episode | None]:
        """取这次判定对应的时间区间。

        优先用逻辑回归分段器的结果——它给的是**动作本身的起止**（手进盘 →
        杆出盘、刷头贴上 → 离开），而不是「证据攒够」的那一帧。

        判定可能比动作结束更早（涂油攒够 ``brush_hold_s`` 时人还在继续刷），
        所以先把该动作的分段器**就地封口**：此时已经攒到的 ``start_ts``
        才是动作真正的起点。封口之后这一段不再参与后续帧，否则人继续刷
        会把它一直往后拖。

        ``owner`` 是动作的归属键（手部轨迹 ID / 涂油段 ``id()``），只认
        自己那一段，不会把相邻一件的区间拿过来。
        """

        if owner is not None:
            segmenter = self._segmenters.pop((kind, owner), None)
            if segmenter is not None:
                pending = segmenter.take_active()
                if pending is not None:
                    self._judge_episodes.append((kind, owner, pending))
        stale_s = self.cfg.screw_exit_pick_dedup_s
        for index, (name, key, episode) in enumerate(self._judge_episodes):
            if name != kind:
                continue
            if owner is not None and key != owner:
                continue
            if owner is None and abs(episode.end_ts - fallback_end) > 2.0:
                # 没有归属键（丝杆离盘判据不挂分段器）时**不能**拿队列里
                # 最近一个动作的区间顶替：那条区间很可能是别的手留下、
                # 或者上一次被去重压掉的取料的残留。实测 091519 就因此把
                # 4.33s 的区间挂到了 13.13s 的事件上，看起来像多了一件。
                continue
            if owner is not None and episode.end_ts < fallback_end - stale_s:
                # 有归属键也要看区间是不是**这一趟**的。手部轨迹 ID 会被
                # 复用（ByteTrack 回收后新建的轨迹拿到同一个 ID），上一次
                # 进盘的区间就留在队列里，下一次进盘结算时会把它取走，
                # 事件起点被拖到几十秒前（实测 122925：96.5s 那次取料取到
                # 89.2s 的区间，99.5s 又取到 96.5s 的，整串错位一格）。
                # 区间终点明显早于本次结算时刻的一律跳过，让它留在队列里
                # 等真正属于它的那次结算（等不到就自然过期）。
                continue
            del self._judge_episodes[index]
            return episode.start_ts, episode.end_ts, episode
        return fallback_start, fallback_end, None

    def _close_oiling_if_settled(self) -> None:
        """当前涂油段已经完整报过（且不是暂存待挂的），就收段。"""

        session = self._active_oiling
        if session is None or session.reported_ts is None:
            return
        # 还在 pending 里等这一件结算的，不能收——它马上就要挂到新周期上。
        if any(item[0] is session for item in self._pending_oilings):
            return
        self._close_oiling()

    def _drain_pending_oilings(self, task: _Cycle, ts: float) -> bool:
        """把暂存的涂油补挂到刚建立的周期上。

        只认「同一根杆」：暂存的是工人的涂油，取料确认的是从盘里拿出来的
        那一根，两者位置对得上才是同一件。
        """

        if not self._pending_oilings or task.screw_anchor is None:
            return False
        for session, oil_ts, episode in list(self._pending_oilings):
            if session.anchor is None:
                continue
            if not self._same_anchor(task.screw_anchor, session.anchor):
                continue
            self._pending_oilings.remove((session, oil_ts, episode))
            self._attach_oiling(task, session, oil_ts, episode=episode)
            return True
        return False

    def _same_rod_as_entry(
        self,
        visit: _HandVisit,
        holding: Detection | None,
        hand_box: BBox,
    ) -> bool:
        """离盘手里那根，是不是就是进盘时带进来的那根。

        判据：位置对得上（``_OILING_SAME_ROD_STEP`` 量级），**或者**杆框
        与「进盘时手里那根」的落点重合。对得上 ⇒ 只是带进来又带出去
        （放回去、换个位置再拿走），不是取料。

        进盘时没记下落点（``entry_held`` 为空）时按「是同一根」处理，
        保持原行为——那条路径本来就是保守的。
        """

        if visit.entry_held is None:
            return True
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        candidates: list[tuple[float, float]] = []
        if holding is not None:
            candidates.append(holding.box.center)
        if visit.last_outside_holding is not None or holding is None:
            candidates.append(hand_box.center)
        return any(
            math.hypot(cx - visit.entry_held[0], cy - visit.entry_held[1]) / diag
            <= _OILING_SAME_ROD_STEP
            for cx, cy in candidates
        )

    def _same_anchor(
        self, a: tuple[float, float], b: tuple[float, float]
    ) -> bool:
        """两个位置是不是同一根杆（按涂油判定的同杆容差，放宽两倍）。

        工人取完料会走动，同一根杆后来的位置和取料那一刻可以差一截；
        这里只用来**否定**「明显是另一根杆」，宁可放宽也不要错挂。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        return (
            math.hypot(a[0] - b[0], a[1] - b[1]) / diag
            <= _OILING_SAME_ROD_STEP * 2
        )
