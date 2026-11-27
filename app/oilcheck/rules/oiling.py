"""人工二次涂油判定（mixin）。

刷头工作点贴着「手工握着的那根丝杆」的杆身中线累计接触，达到
``brush_hold_s`` 即判涂油完成。与取料并行判定、不依赖作业周期——现场
工人常常一边取料一边就刷上了。

要点：

- 「一次涂油」的边界由**杆**决定：换一根杆另起一次，同一根杆上刷手抬一下
  再放回来仍算同一次（``_same_oiling_rod`` / ``_oiling_stitch_gap_s``）；
- 杆被拳头压住整帧漏检时，按锚点补接触时长（``_oiling_rod_occluded_contact``，
  上限 ``brush_hold_max_gap_s``），否则现场每次刷 1.0~1.4s 会被切成碎段；
- 涂油段结算时若还没有对应的取料周期，先暂存，取料确认后按同一根杆补挂
  （``_flush_stale_pending_oilings`` / ``_drain_pending_oilings``）；
- 二级判定在证据偏弱时补一票（``_judge_oiling``）。

判定状态（``_active_oiling`` / ``_cycles`` 等）仍由
:class:`~app.oilcheck.pipeline.StationPipeline` 持有，本 mixin 只放流程。
"""

from __future__ import annotations

import math

from ..association import brush_work_point
from ..constants import (
    _ANCHOR_SAME_SCREW_DISTANCE,
    _CONTINUITY_STEP_ABS,
    _OILING_ROD_OCCLUDED_RADIUS,
    _OILING_SAME_ROD_STEP,
    _OILING_STITCH_GAP_S,
)
from ..geometry import (
    _box_center_in,
    _distance_to,
    _distance_to_long_axis,
    _overlaps,
)
from ..judge import Episode, EpisodeSegmenter, JudgeResult, LogisticJudge
from ..raw_features import RawFrame, measure_oiling
from ..review import build_oiling_summary
from ..state import _Cycle, _OilingSession
from ..tracker import Track
from ..types import (
    BBox,
    EventRecord,
    EventResult,
    EventType,
    TaskResult,
    TaskStage,
)
from .shared import SharedRules


class OilingRules(SharedRules):
    """人工二次涂油判定。"""


    # ------------------------------------------------------------------
    # 人工二次涂油判定
    # ------------------------------------------------------------------
    def _update_oiling(self, *, ts: float, dt: float) -> None:
        """人工二次涂油：刷头贴着手持丝杆累计接触 ⇒ 完成。

        **与取料并行判定，不依赖作业周期**。现场就是这样：工人常常一边
        取料一边就刷上了（示例7 第 2 件、示例6 第 3 件），要等「手离开盘区」
        才建立取料周期的话，刷的那两秒正好落在周期建立之前，整段丢掉、
        随后还被下一次取料判成漏涂。

        「一次涂油」的边界由**杆**决定，和 ``tools/detect_operations.py``
        同一口径：换一根杆就另起一次；同一根杆上刷手抬一下再放回来
        （现场每件都有 0.5~1.3 s 遮挡）仍算同一次。
        """

        self._flush_stale_pending_oilings(ts)
        session = self._active_oiling
        contacts = self._oiling_contacts()
        if session is not None:
            # 只要还在跟同一根杆，不论中间空了几帧都优先续上。
            matched = next(
                (
                    item
                    for item in contacts
                    if self._same_oiling_rod(item[1], session)
                ),
                None,
            )
            if matched is None and contacts:
                # 场上还有别的杆在被刷：这一根换了，本段到此为止。
                self._close_oiling()
                session = None
        else:
            matched = None
        if matched is None and session is None and contacts:
            matched = contacts[0]

        if matched is not None:
            distance, box, brush = matched
            held = any(
                hand_box.iou(box) > 0.0
                for hand_box in (hand.observation_box() for hand in self._hand_tracks)
            )
            if session is None:
                # 首帧也把这一帧的 dt 记进去：刷头此刻确实贴着，和后续帧
                # 一样是有效接触。脚本没有 dt 概念、只能从第二帧起算，
                # 这里按判定的实际口径算，低帧率下才不会少算一帧。
                self._active_oiling = _OilingSession(
                    anchor=box.center,
                    start_ts=ts,
                    last_ts=ts,
                    last_seen_ts=ts,
                    cum_s=max(0.0, dt),
                    frames=1,
                    min_distance=distance,
                    brush_track_id=brush.track_id,
                    held_frames=1 if held else 0,
                    rod_start=box.center,
                )
                self._publish_oiling_progress(self._active_oiling)
                return
            gap = max(0.0, ts - session.last_ts)
            # 漏检/抬手的空档按帧率放大的桥接窗口补回来，超出就不再计入
            # ——把整段空档都算成接触会把「刷两下就走」凑成完成。
            bridge = min(
                self._at_least_frame(self.cfg.brush_hold_max_gap_s),
                self._max_plausible_dt(),
            )
            if gap > bridge:
                session.parts += 1
            session.cum_s += min(gap, bridge)
            session.last_ts = ts
            session.last_seen_ts = ts
            session.frames += 1
            # 重新贴上实地接触：之前「杆被压住」补的时长作废计数重新开始。
            session.occluded_s = 0.0
            if held:
                session.held_frames += 1
            if session.rod_start is not None:
                width, height = self._frame_size
                diag = max(1.0, math.hypot(width, height))
                moved = math.hypot(
                    box.center[0] - session.rod_start[0],
                    box.center[1] - session.rod_start[1],
                ) / diag
                session.rod_move = max(session.rod_move, moved)
            session.anchor = box.center
            session.min_distance = min(session.min_distance, distance)
            session.brush_track_id = brush.track_id
            # 二级判定的输入：本帧刷头与手持杆的**原始几何**。只喂模型。
            frame = self._raw_frame(ts)
            session.raw.add(measure_oiling(frame, self._oiling_hand_box(frame, box)))
            self._publish_oiling_progress(session)
            if (
                session.reported_ts is None
                and session.cum_s >= self.cfg.brush_hold_s - 1e-9
            ):
                self._finish_oiling(ts)
            return

        if session is None:
            return
        # 正在刷的那根杆这一帧没检出、但刷头仍贴在它身上：
        #
        # 现场实测（122925）：一次涂油的接触被切成 0.2~0.3 秒的碎段，
        # 每碎一次就少算一段，累计停在 0.93s，整件涂油判不出来。碎掉的原因
        # 是那根杆被拳头压住、整帧没有检出——但**盘里别的杆还在**，所以
        # 走不到下面「一根杆都检不到」的整帧遮挡桥接。
        #
        # 这里的判据比整帧遮挡更严：只补「这根杆在锚点附近确实不可见、
        # 而刷头工作点仍贴着锚点」的帧；盘里有别的杆、或这根杆就在锚点附近
        # 被检出，都说明手已经松开了，照常往下走。
        #
        # 连续补的总时长受 ``brush_hold_max_gap_s × frame_scale`` 限制
        # （与下面的桥接同一上限），不会把「刷两下就走」凑成完成。
        if self._oiling_rod_occluded_contact(session, ts):
            return

        # 本帧没贴着刷。三种情形要分开：
        # - 场上还有手持杆、只是刷头没贴上去 → 真的停手了，靠拼接窗口收段；
        # - 一根手持杆都没看见（被手完全挡住 / 整帧漏检）→ 接着往下判，
        #   把这段时间按桥接窗口补上：现场的杆本来就常被拳头压住，
        #   不补会让「刷了 2 秒、中间遮了 0.4 秒」凑不够 brush_hold_s；
        # - 超时 → 收段。
        if self._oiling_held_rods() or self._screw_visible_anywhere():
            # 场上有丝杆被检出：手确实松开了 / 刷头确实离开了，不补时间。
            # 只有「连一根杆都检不到」才算整帧遮挡，才往下桥接——
            # 用「有没有手持杆」区分不了这两种，杆画得出来但不在手里
            # 和杆被手完全压住是两回事。
            pass
        elif ts - session.last_seen_ts <= self._oiling_occluded_hold_s():
            bridge = min(
                self._at_least_frame(self.cfg.brush_hold_max_gap_s),
                self._max_plausible_dt(),
            )
            gap = max(0.0, ts - session.last_ts)
            if gap > bridge:
                session.parts += 1
            session.cum_s += min(gap, bridge)
            session.last_ts = ts
            session.frames += 1
            self._publish_oiling_progress(session)
            if (
                session.reported_ts is None
                and session.cum_s >= self.cfg.brush_hold_s - 1e-9
            ):
                self._finish_oiling(ts)
            return
        # 本帧没贴着刷：短暂抬手/漏检先挂着，超过拼接窗口再收段。
        if ts - session.last_seen_ts > self._oiling_stitch_gap_s():
            self._close_oiling()

    def _publish_oiling_progress(self, session: _OilingSession) -> None:
        """把当前接触段的进度同步到正在等涂油的那个周期上。

        ``brush_hold_s`` 等字段供监控卡片、复核页与 PLC 灯效实时显示，
        不能只在判完的那一刻才写一次。
        """

        task = self._oiling_target_cycle(session)
        if task is None:
            return
        task.brush_hold_s = session.cum_s
        task.brush_track_id = session.brush_track_id
        task.brush_min_distance = session.min_distance
        task.brush_near_frames = session.frames
        task.last_seen_ts = session.last_seen_ts

    def _oiling_held_rods(self) -> tuple[BBox, ...]:
        """本帧「被手握着」的丝杆框。

        不要求杆心出盘：``tray_roi`` 是为取料划的，下沿常盖住工人刷油的
        位置，刷的时候杆压在盘区里是常态（示例6 第 1、3 件）。握着的事实
        用「杆框与手框交叠 + 杆心在手附近」判断。
        """

        label = self.station.detector.screw_label
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        limit = self.cfg.hand_screw_max_distance * self._frame_scale
        anchor = None if self._active_oiling is None else self._active_oiling.anchor
        boxes: list[BBox] = []
        for det in self._last_detections:
            if det.label != label:
                continue
            cx, cy = det.box.center
            for hand in self._hand_tracks:
                if not hand.matched:
                    continue
                hand_box = hand.observation_box()
                if not _overlaps(det.box, hand_box):
                    continue
                hx, hy = hand_box.center
                if math.hypot(cx - hx, cy - hy) / diag > limit:
                    continue
                # 盘里那根没被动过的存货不算「被握着」：手框很大，伸进盘里
                # 抓料时会与整排存货交叠，不排除就会把盘里的料当成正在刷的
                # 那根，让没刷的那一件也累计够了（实测会吞掉漏涂 NG）。
                #
                # 只排除**在盘区内**的静置杆：盘区外台面上的待刷杆同样不动，
                # 但它就是工人要刷的那根。
                # 当前这一段自己那根除外——拳头压着杆时它可能连续几帧不动。
                if (
                    self._screw_parked_in_tray(det)
                    and not self._same_rod_pos(det.box, anchor, diag)
                ):
                    continue
                boxes.append(det.box)
                break
        return tuple(boxes)

    @staticmethod
    def _same_rod_pos(
        box: BBox, anchor: tuple[float, float] | None, diag: float
    ) -> bool:
        if anchor is None:
            return False
        cx, cy = box.center
        return (
            math.hypot(cx - anchor[0], cy - anchor[1]) / diag
            <= _OILING_SAME_ROD_STEP
        )

    def _oiling_contacts(self) -> list[tuple[float, BBox, Track]]:
        """本帧「刷头工作点贴着某只手里的杆」的全部候选，按距离升序。

        与 ``tools/detect_operations.py`` 的 ``OilingDetector._contacts``
        同一口径：遍历本帧全部刷头轨迹与全部手持杆，取刷头工作点到杆身
        中线的垂距 ≤ ``brush_contact_distance`` 的组合。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        held = self._oiling_held_rods()
        if not held:
            return []
        contacts: list[tuple[float, BBox, Track]] = []
        for brush in self._brush_tracks:
            if not brush.matched or brush.hits < self._brush_tracker.min_hits:
                continue
            point = brush_work_point(brush, self.station.detector)
            for box in held:
                distance = _distance_to_long_axis(box, point) / diag
                if distance <= self.cfg.brush_contact_distance:
                    contacts.append((distance, box, brush))
        contacts.sort(key=lambda item: item[0])
        return contacts

    def _same_oiling_rod(self, box: BBox, session: _OilingSession) -> bool:
        if session.anchor is None:
            return True
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        cx, cy = box.center
        return (
            math.hypot(cx - session.anchor[0], cy - session.anchor[1]) / diag
            <= _OILING_SAME_ROD_STEP
        )

    def _screw_visible_anywhere(self) -> bool:
        """本帧是否检出了任意一根丝杆（不要求在手边）。"""

        label = self.station.detector.screw_label
        return any(det.label == label for det in self._last_detections)

    def _oiling_occluded_hold_s(self) -> float:
        """杆整帧看不见时，还能按「还在刷」往下补多久。

        与「手持丝杆被预测保持多久」同一口径（``held_predict_hold_s``）：
        拳头压住杆是常态，补一两帧能救回真实接触；补得更久就等于凭空
        造时间，会把「刷两下就走」凑成完成。
        """

        # 不乘 frame_scale：那是给「判定门槛」用的放大，用在这里会让低帧率
        # 下把「手早就松开了」也补进来（实测 3 倍放大后能凭空补 1.5s）。
        # 只补得了「一两帧漏检」，最多一个可信帧间隔。
        return min(
            self.cfg.held_predict_hold_s,
            self._max_plausible_dt(),
        )

    def _oiling_stitch_gap_s(self) -> float:
        return self._at_least_frame(
            max(
                _OILING_STITCH_GAP_S,
                self.cfg.brush_hold_max_gap_s,
            )
        )

    def _close_oiling(self) -> None:
        """收掉当前接触段；还没报过且够了 ``brush_hold_s`` 才发一次涂油事件。"""

        session = self._active_oiling
        if session is None:
            return
        self._active_oiling = None
        # 先把这一段的分段器收掉——区间要退回到「刷头第一次贴上」。
        # 已经报过（攒够时长那帧就发了）的段，区间由事后补写落在事件上
        # （见 `_finish_oiling` 的 `_pending_episode`）。
        episode = self._take_oiling_episode(session)
        if session.reported_ts is None and (
            session.cum_s >= self.cfg.brush_hold_s - 1e-9
        ):
            self._finish_oiling(
                session.last_ts, session=session, episode=episode
            )
            return
        if episode is not None:
            # 事件在攒够 `brush_hold_s` 的那一帧就发了（那时人还在继续刷），
            # 这里记下整段动作的起止给复核统计——事件本身是 frozen 的，
            # 不能再写回；`_last_full_episodes` 供监控/复核展示。
            self._last_full_episodes.append(("oiling", episode))
            self._logger.debug(
                "涂油整段区间 %.2fs → %.2fs（%.2fs）",
                episode.start_ts,
                episode.end_ts,
                episode.duration,
            )
        # 没够 brush_hold_s 的短接触也交给训练钩子：涂油的负样本只能从这里
        # 来（规则没命中就没有事件），没有负样本就无法训练二级判定。
        if self._training_oiling_hook is not None and session.frames > 0:
            self._training_oiling_hook(
                self._oiling_features(session), session.last_ts or 0.0
            )

    def _oiling_rod_occluded_contact(
        self, session: _OilingSession, ts: float
    ) -> bool:
        """正在刷的杆被手压住、整帧没检出时，按锚点继续补接触时长。

        返回 True 表示这一帧已经按「仍在刷」处理，调用方不必再往下走。

        为什么需要它：现场（122925）一次涂油的接触被切成 0.2~0.3 秒的碎段，
        每碎一次少算一段，累计停在 0.93s 判不出来。碎的原因是拳头压住杆、
        整帧漏检；但**盘里别的杆还在**，所以走不到「一根杆都检不到」的
        整帧遮挡桥接（那条要求场上完全没有丝杆）。

        判据三条同时成立才补：

        1. 锚点附近（``_OILING_ROD_OCCLUDED_RADIUS``）没有丝杆框——
           那根杆确实不可见。有框就是手松开了，照常走正常路径；
        2. 刷头工作点仍贴着锚点（``brush_contact_distance``）——人还在刷；
        3. 这一段的补时长还没超过 ``brush_hold_max_gap_s × frame_scale``
           ——与整帧遮挡桥接同一上限，防止把「刷两下就走」凑成完成。
        """

        cap_s = self._at_least_frame(self.cfg.brush_hold_max_gap_s)
        if session.occluded_s >= cap_s:
            return False
        anchor = session.anchor
        if anchor is None:
            return False
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        label = self.station.detector.screw_label
        # 1) 那根杆在锚点附近确实不可见。
        if any(
            det.label == label
            and math.hypot(
                det.box.center[0] - anchor[0], det.box.center[1] - anchor[1]
            )
            / diag
            <= _OILING_ROD_OCCLUDED_RADIUS
            for det in self._last_detections
        ):
            return False
        # 2) 刷头仍贴着锚点。
        point_box = BBox(
            anchor[0] - 10.0, anchor[1] - 10.0, anchor[0] + 10.0, anchor[1] + 10.0
        )
        touching = False
        for brush in self._brush_tracks:
            if not brush.matched or brush.hits < self._brush_tracker.min_hits:
                continue
            point = brush_work_point(brush, self.station.detector)
            if (
                _distance_to_long_axis(point_box, point) / diag
                <= self.cfg.brush_contact_distance
            ):
                touching = True
                break
        if not touching:
            return False
        gap = max(0.0, ts - (session.last_ts or ts))
        add = min(gap, cap_s - session.occluded_s)
        if add <= 0.0:
            return False
        session.cum_s += add
        session.occluded_s += add
        session.last_ts = ts
        session.last_seen_ts = ts
        session.frames += 1
        self._logger.debug(
            "涂油：正在刷的杆整帧未检出，按锚点补 %.3fs（本段累计补 %.3fs/%.3fs）",
            add,
            session.occluded_s,
            cap_s,
        )
        self._publish_oiling_progress(session)
        if session.reported_ts is None and session.cum_s >= self.cfg.brush_hold_s - 1e-9:
            self._finish_oiling(ts)
        return True

    def _take_oiling_episode(self, session: _OilingSession) -> Episode | None:
        """收掉这个涂油段的分段器，返回它的区间。"""
        segmenter = self._segmenters.pop(("oiling", id(session)), None)
        if segmenter is None:
            return None
        segmenter.close()
        episodes = segmenter.drain()
        return episodes[-1] if episodes else None

    def _finish_oiling(
        self,
        ts: float,
        *,
        session: _OilingSession | None = None,
        episode: Episode | None = None,
    ) -> None:
        """一次涂油达成：并入最近的取料周期，或单独报一条。

        ``ts`` 是这一段的结束时刻（收段时传 ``session.last_ts``），
        ``episode`` 是逻辑回归分段出来的整段区间（刷头贴上 → 离开）。
        """

        session = session or self._active_oiling
        if session is None:
            return
        session.reported_ts = ts
        values = self._oiling_features(session)
        if self._training_oiling_hook is not None:
            self._training_oiling_hook(values, ts)
        judge_result = self._judge_oiling(values)
        if judge_result is not None and judge_result.used and not judge_result.decision:
            # 二级判定否掉了这一段的「人工二次涂油」结论。规则已经攒够
            # brush_hold_s，但证据（杆不在手里、杆没跟着刷手动、接触零散）
            # 说明刷的不是手上这根——判成完成会让漏刷件直接过关。
            self._logger.info(
                "涂油完成被二级判定否掉：累计 %.2fs，逻辑回归 p=%.3f",
                session.cum_s,
                judge_result.probability,
            )
            return
        task = self._oiling_target_cycle(session)
        if task is not None:
            self._attach_oiling(task, session, ts, judge=judge_result, episode=episode)
            return
        # 还没有取料周期可挂。分两种情形：
        #
        # - 有手正在盘区里（取料进行中）：工人常常一边取料一边就刷上了，
        #   取料要等手离开盘区才结算，那会儿这次涂油已经判完。先存着，
        #   取料确认时按杆补挂——立刻报成无主涂油会让这一件没有归属。
        # - 没有手在盘区里：这是纯涂油（录像开头就在刷、取完又补刷），
        #   现场按规则不算异常，直接报出去。
        if session.anchor is not None and self._pick_in_progress():
            self._pending_oilings.append((session, ts, episode))
            return
        self._emit_free_oiling(session, ts, episode=episode)

    def _oiling_hand_box(
        self, frame: RawFrame, rod_box: BBox
    ) -> BBox | None:
        """刷的这根杆，此刻最可能是哪只手在拿。

        没有关键点也没有直接关联，只能按位置取最近的那只手；比刷头本身
        更可靠——刷头是工具，杆在谁手里由手决定。
        """

        if not frame.hands:
            return None
        cx, cy = rod_box.center
        return min(
            (box for _, box in frame.hands),
            key=lambda box: (
                (box.center[0] - cx) ** 2 + (box.center[1] - cy) ** 2
            ),
        )

    def _oiling_features(self, session: _OilingSession) -> dict[str, float]:
        """涂油判定的特征向量：**规则字段 + 原始检测框聚合特征**。

        同取料：规则字段供 ``_judge_oiling`` 的弱证据门限用，原始聚合
        （``contact_ratio`` / ``held_ratio`` / ``brush_rod_dist_min`` …）
        是模型真正的输入，全部能从检测框复算。
        """

        frames = max(1, session.frames)
        values: dict[str, float] = {
            "cum_s": float(session.cum_s),
            "frames": float(session.frames),
            "min_distance": float(session.min_distance),
            "parts": float(session.parts),
            "held_ratio": session.held_frames / frames,
            "rod_move": float(session.rod_move),
            "contact_ratio": float(session.cum_s) / max(
                1e-6, float(self.cfg.brush_hold_s)
            ),
        }
        values.update(
            session.raw.features(cum_s=session.cum_s, frames=session.frames)
        )
        return values

    def _judge_oiling(self, values: dict[str, float]) -> JudgeResult | None:
        """涂油二级判定。

        规则已经很保守（贴着杆身中线累计够 ``brush_hold_s``），所以这里
        **只在证据偏弱时**（杆在手里的比例不高、或杆几乎没跟着刷手动）
        让它参与；证据充分时直接用规则结论，避免模型把稳定的判定搅乱。
        """

        if self._oiling_judge is None:
            return None
        weak = values["held_ratio"] < 0.5 or values["rod_move"] < 0.005
        return self._oiling_judge.decide(values, rule=None if weak else True)

    def _update_judge_episodes(self, *, ts: float) -> None:
        """逐帧打分并切段，记录「取料/涂油从什么时候到什么时候」。

        规则判定回答的是「这一帧够不够格算一次」，分段器回答的是「这件事
        真正发生在哪个时间段」：取料是手进盘到带杆出盘，涂油是刷头贴上到
        真正离开。事件里的 ``start_ts`` / ``end_ts`` 直接用分段结果，不再
        由结算帧往回推。

        **一段动作一个分段器**（每趟进盘、每次刷涂各一个）：整段视频共用
        一个分段器时，前一件的尾巴和后一件的开头会被 ``bridge_s`` 的空档
        并成一段，区间横跨两件（实测示例④ 一次涂油的区间被拉成 8.0s，
        把取料也框进去了）。分到各个动作上，边界就落在这件自己的动作上。
        """

        if self._pick_judge is not None:
            live: set[tuple[str, int]] = set()
            for hand_id, visit in self._visits.items():
                if not visit.entered_confirmed or visit.holding_on_entry:
                    continue
                live.add(("pick", hand_id))
                start = (
                    visit.first_inside_ts if visit.first_inside_ts is not None else ts
                )
                values = self._pick_features(visit, ts=ts, dwell=ts - start)
                segmenter = self._segmenters.get(("pick", hand_id))
                if segmenter is None:
                    segmenter = self._make_segmenter(self._pick_judge, kind="pick")
                    self._segmenters[("pick", hand_id)] = segmenter
                segmenter.update(ts, self._pick_judge.probability(values))
            self._retire_segmenters("pick", live)
        if self._oiling_judge is not None:
            live = set()
            session = self._active_oiling
            if session is not None:
                key = ("oiling", id(session))
                live.add(key)
                values = self._oiling_features(session)
                segmenter = self._segmenters.get(key)
                if segmenter is None:
                    segmenter = self._make_segmenter(self._oiling_judge, kind="oiling")
                    self._segmenters[key] = segmenter
                segmenter.update(ts, self._oiling_judge.probability(values))
            self._retire_segmenters("oiling", live)

    def _make_segmenter(
        self, judge: LogisticJudge, *, kind: str
    ) -> EpisodeSegmenter:
        bridge = (
            self._at_least_frame(0.6)
            if kind == "pick"
            else max(0.8, self._oiling_stitch_gap_s() * 0.5)
        )
        return EpisodeSegmenter(
            on=judge.high,
            off=judge.low,
            low=judge.low * 0.6,
            bridge_s=bridge,
        )

    def _retire_segmenters(self, kind: str, live: set[tuple[str, int]]) -> None:
        """动作结束的段收掉；同一动作再次出现时沿用旧段，不重复收。"""

        for key in list(self._segmenters):
            if key[0] != kind or key in live:
                continue
            segmenter = self._segmenters.pop(key)
            segmenter.close()
            for episode in segmenter.drain():
                self._judge_episodes.append((kind, key[1], episode))

    def live_pick_features(self, ts: float) -> list[dict[str, float]]:
        """本帧所有「进行中的取料」的特征（训练脚本 / 排查用）。

        取料在盘内那一段还没攒到离盘后的握持采样，除 ``dwell`` 以外的特征
        基本都是空的；分段器靠 ``dwell`` 与 carry 证据的逐帧增长定位起点，
        所以这几帧照样要打分。
        """

        sets: list[dict[str, float]] = []
        for visit in self._visits.values():
            if not visit.entered_confirmed or visit.holding_on_entry:
                continue
            start = visit.first_inside_ts if visit.first_inside_ts is not None else ts
            sets.append(self._pick_features(visit, ts=ts, dwell=ts - start))
        return sets

    def active_oiling_session(self) -> _OilingSession | None:
        """当前正在累计的涂油段（没有则 None）。"""

        return self._active_oiling

    def oiling_judge_features(self, session: _OilingSession) -> dict[str, float]:
        """某段涂油此刻的特征（训练脚本 / 排查用）。"""

        return self._oiling_features(session)

    def judge_episodes(self, kind: str | None = None) -> list[Episode]:
        """逻辑回归分段出来的动作（按结束时刻排序）。"""

        items = [
            episode
            for name, _owner, episode in self._judge_episodes
            if kind is None or name == kind
        ]
        return sorted(items, key=lambda item: item.end_ts)



    def _pick_in_progress(self) -> bool:
        """取料是不是正在进行中（进盘过程还挂着、尚未结算）。

        判据用「有没挂着未结算的进盘过程」而不是「盘区里有没有手」：
        工人常常站在盘边刷，手刚好在盘区下沿外一点点；按盘区判会在
        涂油判完的那一刻返回 False，这次涂油就被报成无主涂油
        （示例⑤ 6.47s、示例⑦ 11.93s 实测）。进盘过程挂着，说明这一趟
        还没定论，涂油应该等它。
        """

        # 「已确认进盘、还没结算」才算取料进行中。手在旁边走过（还没进盘）
        # 或只是被跟踪着，不算。
        #
        # 注意离盘之后、结算之前这段也算：那时正在攒盘外握持采样，
        # 取料马上就要确认了，这一件的涂油该等它挂上去（示例⑤ 6.47s、
        # 示例⑦ 11.93s 就落在这一段里）。
        if any(visit.entered_confirmed for visit in self._visits.values()):
            return True
        # 手已在盘区里但离盘还没按采样窗口确认：这也是取料进行中。
        tray_box = self._tray_region_box(self._frame_size)
        if tray_box is None:
            return False
        return any(
            visit.inside_tray
            and visit.last_hand_box is not None
            and _box_center_in(visit.last_hand_box, tray_box)
            for visit in self._visits.values()
        )

    def _flush_stale_pending_oilings(self, ts: float) -> None:
        """暂存太久、始终没等到对应取料的涂油，按无主涂油报出去。

        现场确实有「录前就在刷、整段都没有取料」的情形（示例③④⑦ 开头）。
        等 `oiling_timeout_s` 这么久还没有取料，就不再等；设 0 时用一个
        保守的固定值，保证视频结束前也能报出来。
        """

        if not self._pending_oilings:
            return
        wait = self.cfg.oiling_timeout_s if self.cfg.oiling_timeout_s > 0 else 10.0
        for session, oil_ts, episode in list(self._pending_oilings):
            if ts - oil_ts < wait:
                continue
            self._pending_oilings.remove((session, oil_ts, episode))
            self._emit_free_oiling(session, oil_ts, episode=episode)

    def _attach_oiling(
        self,
        task: _Cycle,
        session: _OilingSession,
        ts: float,
        *,
        judge: JudgeResult | None = None,
        episode: Episode | None = None,
    ) -> None:
        """把一次涂油记到某个周期上并判完成。"""

        if not task.active or task.stage != TaskStage.PICKED:
            # 这一件已经判过（多次涂油的情形），不重复上报。
            return
        self._oiling_start_ts = session.start_ts
        task.brush_hold_s = session.cum_s
        task.brush_track_id = session.brush_track_id
        task.brush_min_distance = session.min_distance
        task.brush_near_frames = session.frames
        if task.screw_anchor is None and session.anchor is not None:
            task.screw_anchor = session.anchor
        self._complete_oiling(task, ts, judge=judge, session=session, episode=episode)



    def _oiling_target_cycle(self, session: _OilingSession) -> _Cycle | None:
        # 第一步就排除「这根杆在被别的手握着」：取料时记下了那是哪只手，
        # 画面里另一只手握着另一根杆刷，不该算到这一件上。

        """这次涂油该记到哪个取料周期上。

        只有「同一根杆」才认：周期在取料那一刻记下了手里那根杆的位置，
        这次刷的杆必须就是它。不校验的话，画面里另一只手握着别的杆刷够了，
        会被错挂到本周期上、把这一件提前判成已涂油（实测会吞掉真实漏涂）。
        """

        candidates = [
            task
            for task in self._cycles.values()
            if task.active and task.stage == TaskStage.PICKED
        ]
        if not candidates:
            return None
        session_anchor = session.anchor
        if session_anchor is not None:
            by_hand = [
                task
                for task in candidates
                if self._hand_holds_point(task.hand_track_id, session_anchor)
            ]
            if by_hand:
                candidates = by_hand
            else:
                # 取料那几只手里没有一只握着这根杆。先看**位置**能不能定：
                # 这次刷的杆与某个周期取料时记下的那根对得上，就是这一件，
                # 与「手还看不看得见」无关——工人取完料把杆换到另只手上、
                # 或原来那只手松开闲着继续刷时，手部那一路必然判不出来，
                # 位置这一路仍然可靠（实测 122925 第 2 件：14.27s 刷的正是
                # 13.00s 取出的那根，只因取料那只手已松开而被拒，整件连带
                # 下一次取料一起判成漏涂 NG）。
                matched = [
                    task
                    for task in candidates
                    if task.screw_anchor is not None
                    and self._same_anchor(task.screw_anchor, session_anchor)
                ]
                if matched:
                    candidates = matched
                elif any(task.hand_track_id is not None for task in candidates):
                    # 位置也对不上，才按手区分两种情形：
                    #
                    # - 取料那只手**还在画面里**（只是空着手）：这根杆是别的
                    #   手里的，不归这一件。现场实测就有一只空手站在原处、
                    #   另一只手刷另一根的情形，认了会把这一件提前判成已涂油。
                    # - 取料那只手**已经不在画面里**（拿杆走开、或视觉 ID 重建）：
                    #   无歧义，就是这一件，认。
                    if self._any_cycle_hand_still_visible(candidates):
                        return None
                # 位置对不上时：只有**确实存在另一个候选周期**才据此拒绝。
                # 工人取完料走动一段再刷是常有的事，只有一个周期在等涂油
                # 时无歧义，认它；有两个以上在等，才按位置挑。
                if len(candidates) > 1 and not matched:
                    return None
        # 多次涂油时挂到最近一次取料的那一件上。
        return max(candidates, key=lambda item: item.picked_at or item.created_at)

    def _any_cycle_hand_still_visible(self, tasks: list[_Cycle]) -> bool:
        """这些周期记下的取料那只手，还有没有一只在画面里。"""

        return any(
            task.hand_track_id is not None
            and self._live_hand(task.hand_track_id) is not None
            for task in tasks
        )

    def _other_hand_holds_other_rod(
        self, anchor: tuple[float, float]
    ) -> bool:
        """画面的其他手里是不是也握着杆（即「这根杆归谁」存在歧义）。"""

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        label = self.station.detector.screw_label
        for det in self._last_detections:
            if det.label != label:
                continue
            if (
                math.hypot(det.box.center[0] - anchor[0], det.box.center[1] - anchor[1])
                / diag
                <= _OILING_SAME_ROD_STEP
            ):
                continue
            # 这里按「任意手与任意杆交叠」判，不套 ``_oiling_held_rods`` 的
            # 盘内存货过滤——那只负责选当前要跟的杆，这里要回答的是
            # 「场上有几只手里有杆」。
            for hand in self._hand_tracks:
                if not hand.matched:
                    continue
                if _overlaps(det.box, hand.observation_box()):
                    return True
        return False

    def _hand_holds_point(
        self, hand_track_id: int | None, point: tuple[float, float]
    ) -> bool:
        """那只手此刻是不是正握着这个位置上的杆。"""

        if hand_track_id is None:
            return False
        hand = self._live_hand(hand_track_id)
        if hand is None:
            return False
        hand_box = hand.observation_box()
        if not _box_center_in(BBox(point[0] - 8, point[1] - 8, point[0] + 8, point[1] + 8), hand_box):
            # 手框不一定「包含」杆心（握在手里时杆心常在手框边缘外），
            # 退一步按中心距离判。
            hx, hy = hand_box.center
            width, height = self._frame_size
            diag = max(1.0, math.hypot(width, height))
            if math.hypot(point[0] - hx, point[1] - hy) / diag > (
                self.cfg.hand_screw_max_distance
            ):
                return False
        return True



    def _emit_free_oiling(
        self, session: _OilingSession, ts: float, *, episode: Episode | None = None
    ) -> None:
        """没有对应取料周期的涂油（录像开头就已在刷、或取完又补刷）。"""

        self._free_oiling_seq += 1
        event = EventRecord.create(
            timestamp_start=float(
                episode.start_ts if episode is not None else (session.start_ts or ts)
            ),
            timestamp_end=float(episode.end_ts if episode is not None else ts),
            station_id=self.station.station_id,
            camera_id=self.station.camera_id,
            cycle_id=self.cycle_id,
            task_id="",
            event_type=EventType.MANUAL_OILING,
            result=EventResult.CONFIRMED,
            evidence={
                "task_id": "",
                "reason": "manual_oiling_completed",
                "cycle_attached": False,
                "brush_track_id": session.brush_track_id,
                "brush_hold_s": round(session.cum_s, 3),
                "required_s": self.cfg.brush_hold_s,
                "brush_min_distance": round(session.min_distance, 4),
                "brush_near_frames": session.frames,
                "contact_parts": session.parts,
                "detector": self.detector_name,
                "free_oiling_seq": self._free_oiling_seq,
                **({} if episode is None else {"judge_episode": episode.to_dict()}),
            },
        )
        self._events_emitted += 1
        self._logger.info(
            "人工二次涂油完成（未挂作业周期，刷头靠近 %.2fs）", session.cum_s
        )
        if self._on_event is not None:
            try:
                self._on_event(event)
            except Exception:  # noqa: BLE001
                self._logger.exception("事件回调失败")

    def _static_anchor_ok(self, box: BBox, anchor: tuple[float, float] | None) -> bool:
        """这根杆能不能当「手里那根」的候选。

        位置没动过的杆（盘里的料、零件盒里的整排、夹具上的）一律排除——
        手框很大，随手一放就与整排杆交叠，那些杆不可能是正被带走/正被刷的
        这一根。示例① 第一次涂油时，锚点从工件漂到零件盒里的静置杆上，
        刷子在真杆上刷了 5 秒、接触累计却只有 0.13s，整件被判漏涂。

        例外：**就是锚点当前那根**。拳头压着杆、杆贴着夹具刷油时这一刻
        确实可能不动；允许它继续，否则锚点每抖一下就要丢一次。
        """

        if not self._screw_static(box):
            return True
        if anchor is None:
            return False
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        return (
            math.hypot(box.center[0] - anchor[0], box.center[1] - anchor[1]) / diag
            <= _ANCHOR_SAME_SCREW_DISTANCE
        )

    def _current_screw_box(self, task: _Cycle, ts: float) -> BBox | None:
        """当前手持丝杆的位置：**丝杆锚点跟踪**，不依赖手部视觉 ID。

        取料那一刻把手里那根杆记成锚点，之后每帧更新：

        1. 优先取「与**本周期那只手**的交叠框交叠」的杆——握着的时候丝杆一定
           被手框遮住，这条约束把候选从满盘丝杆缩到手上那一根；只看本周期
           这只手，别的工人/另一只手握的杆不能把锚点抢走（实测会直接导致
           刷别处的杆也判完成）；
        2. 交叠候选里取离锚点最近的（``screw_anchor_max_step`` 按实际帧率
           缩放，但不超过 ``screw_anchor_max_step_abs``，防跳杆）；
        3. 手上没有交叠杆（手框偏了、杆被遮，或杆其实在另一只手里）时退化
           为按位置连续性取离锚点最近的杆，但**仅在锚点仍贴近最后一次确认
           握持的位置时**（``hand_screw_max_distance``）——否则锚点会在密集
           的静置丝杆之间逐帧漂走；
        4. 全都找不到时，在 ``screw_anchor_hold_s`` 内按锚点位置继续，
           让刷子靠近的累计不因一两帧漏检中断。

        为什么不跟手：手部视觉 ID 在遮挡/快速移动后会重建——示例①一个 12 秒
        周期里手部 ID 换了 6 次，其中某个周期 80% 的帧绑定的 ID 都未匹配，
        涂油累计因此停在 0 并误判漏涂（实测该周期 19.5~22.5s 刷子距杆
        0.011~0.031，远低于阈值 0.06，实际是刷了的）。丝杆是静置物、位移小。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        label = self.station.detector.screw_label
        # 绝对上限（不随帧率放大）：盘里丝杆间距只有 0.02~0.03，
        # frame_scale 顶到 4 时单帧步长会到 0.48，一跳就是相邻那根杆。
        #
        # 上限只允许**收紧**帧率放大的结果、不允许低于配置值本身：取料
        # 瞬间手带着杆快速划出，单帧位移实测 0.09~0.11，上限比这还小时
        # 锚点一步跨不过去就永久卡在盘内那根杆上，整件涂油累计为 0
        # （示例视频第一件即如此）。
        step = min(
            self.cfg.screw_anchor_max_step * self._frame_scale,
            max(
                self.cfg.screw_anchor_max_step_abs,
                self.cfg.screw_anchor_max_step,
            ),
        )
        anchor = task.screw_anchor
        hand_center = self._hand_center(task.hand_track_id)

        # 1) 握在手里的杆：与**本周期那只手**的框交叠的那些。
        #
        # 加一条「位置没动过的不算」：手框很大（实测 348×399 px），手在盘中
        # 或零件盒边随手一放就与整排杆交叠；那些杆纹丝不动，不可能是手里
        # 正在被带走的这一根。锚点已经锁定的那根除外——拳头压着杆时它可能
        # 连续几帧不动。
        cycle_hand = self._live_hand(task.hand_track_id)
        gripped: list[tuple[float, BBox]] = []
        for det in self._last_detections:
            if det.label != label:
                continue
            if cycle_hand is None or not _overlaps(
                det.box, cycle_hand.observation_box()
            ):
                continue
            if not self._static_anchor_ok(det.box, anchor):
                continue
            cx, cy = det.box.center
            distance = (
                0.0
                if anchor is None
                else math.hypot(cx - anchor[0], cy - anchor[1]) / diag
            )
            if anchor is not None and distance > step:
                continue
            gripped.append((distance, det.box))

        grabbed = bool(gripped)
        if gripped and anchor is None:
            # 初始化：取交叠杆中离手中心最近的那根（握着的一定贴着这只手）。
            best = min(
                gripped, key=lambda item: _distance_to(item[1], hand_center, diag)
            )[1] if hand_center is not None else gripped[0][1]
        else:
            best = min(gripped, key=lambda item: item[0])[1] if gripped else None

        # 位置连续性只在「离最后一次确认握持的位置不远」时成立。锚点每帧
        # 都会被刷新，不加这道约束它会在密集的静置丝杆之间逐帧漂走
        # （实测漂进零件盒里的整排静置杆，手里那根反而累计不上）。
        continuity_ok = (
            task.screw_gripped_at is not None
            and anchor is not None
            and math.hypot(
                anchor[0] - task.screw_gripped_at[0],
                anchor[1] - task.screw_gripped_at[1],
            )
            / diag
            <= self.cfg.hand_screw_max_distance * self._frame_scale
        )

        # 2) 手上没有可见的杆（手框偏了/杆被遮/刷的是另一只手拿的杆）：
        #    按位置连续性找离锚点最近的杆。
        #
        # 安全性由两道约束共同保证：
        # - ``continuity_ok``：锚点仍贴「最后一次确认握持」的位置，不会在
        #   密集的静置丝杆之间逐帧漂走；
        # - ``_CONTINUITY_STEP_ABS``：单帧跨度比「握持证据」时小得多。没有
        #   握持证据却允许跨 0.15，会一步跳到另一只手握的杆上。
        #
        # 不能要求「本周期那只手不可见」才走这条分支：现场常常一只手拿杆、
        # 另一只手拿刷子，被跟踪的那只（拿刷子的那只）一直在画面里，此时
        # 杆本身才是唯一的定位依据（实测会让整件涂油累计为 0）。
        continuity_step = min(step, _CONTINUITY_STEP_ABS)
        if best is None and anchor is not None and continuity_ok:
            nearest: tuple[float, BBox] | None = None
            for det in self._last_detections:
                if det.label != label:
                    continue
                if not self._static_anchor_ok(det.box, anchor):
                    continue
                cx, cy = det.box.center
                distance = math.hypot(cx - anchor[0], cy - anchor[1]) / diag
                if distance > continuity_step:
                    continue
                if nearest is None or distance < nearest[0]:
                    nearest = (distance, det.box)
            best = None if nearest is None else nearest[1]

        if best is not None:
            task.screw_anchor = best.center
            task.screw_anchor_ts = ts
            task.screw_observed = True
            task.last_screw_ts = ts
            if grabbed:
                task.screw_gripped_ts = ts
                task.screw_gripped_at = best.center
            return best

        # 3) hold 窗口内：按锚点位置继续（跨过一两帧漏检，不中断累计）。
        #
        # 窗口从「最后一次锚点更新」算起。加位置约束（``continuity_ok``）：
        # 锚点一旦漂离最后确认握持的位置就不再往下接，避免它在密集的静置
        # 丝杆之间逐帧漂走（实测漂进零件盒里的整排静置杆）。
        #
        # 注：杆被手完全遮住、超过窗口时按既定规则暂停累计、周期保持
        # PICKED（见 docs/05 已知限制）——现场确实会这样，属于已知取舍。
        if (
            anchor is not None
            and continuity_ok
            and task.screw_anchor_ts is not None
            and ts - task.screw_anchor_ts
            <= self._at_least_frame(self.cfg.screw_anchor_hold_s)
        ):
            task.screw_observed = False
            cx, cy = anchor
            return BBox(cx - 8.0, cy - 8.0, cx + 8.0, cy + 8.0)

        # 4) 锚点已失效：尝试**重新捕获**。
        #
        # 现场会这样：取完料拿着杆走出画面（去别处、被遮挡、或杆暂时检测
        # 不到），过一会儿再回来涂油。原实现在 hold 窗口过期后每帧直接返回
        # None，周期永久停在 PICKED，直到下一次取料把它判成 NG——中间回来
        # 涂了也不算。这里只要还在重新捕获时限内、且本帧又出现了「被手握着」
        # 的丝杆，就把它接成新锚点，继续累计涂油。
        reacquire = self._reacquire_anchor(task, ts)
        if reacquire is not None:
            reacquired, hand = reacquire
            gap = 0.0 if task.last_screw_ts is None else ts - task.last_screw_ts
            if hand.track_id != task.hand_track_id:
                # 换绑：本周期那只手的视觉 ID 已重建，跟着新 ID 继续判。
                self._logger.info(
                    "周期 %s 手部轨迹 %s → %s（视觉 ID 重建）",
                    task.task_id,
                    task.hand_track_id,
                    hand.track_id,
                )
                task.hand_track_id = hand.track_id
            task.screw_anchor = reacquired.center
            task.screw_anchor_ts = ts
            task.last_screw_ts = ts
            task.screw_observed = True
            if "SCREW_REACQUIRED" not in task.evidence:
                task.evidence.append("SCREW_REACQUIRED")
                self._logger.info(
                    "周期 %s 重新捕获手持丝杆（距上次观测 %.1fs）",
                    task.task_id,
                    gap,
                )
            return reacquired

        task.screw_observed = False
        return None

    def _reacquire_anchor(
        self, task: _Cycle, ts: float
    ) -> tuple[BBox, Track] | None:
        """锚点失效后重新捕获手持丝杆。

        条件是「本帧有丝杆检测与本周期那只手的框交叠」——握着的时候丝杆
        必然被手框遮住，这条约束足以把候选从满盘丝杆缩到手上那一根，不需要
        位置先验。手部轨迹已被回收时改为接最近的那只有效手。

        返回 ``(框, 手)``，调用方在换手时更新周期的 ``hand_track_id``。

        ``screw_reacquire_timeout_s=0`` 表示不限时：只要周期还活着就一直等，
        因为「多久算这一件不回来了」是工艺问题，交给下一次取料去判 NG 更准。
        """

        timeout = self.cfg.screw_reacquire_timeout_s
        if timeout > 0 and task.last_screw_ts is not None:
            if ts - task.last_screw_ts > timeout:
                return None

        # 只认**本周期那只手**握着的杆：别的工人/另一只手握着的是另一件料，
        # 接成新锚点会让刷别处的杆也算作这一件涂完了（已复现误判）。
        #
        # 例外是「本周期那只手的轨迹已经被跟踪器彻底回收」：手部视觉 ID 在
        # 遮挡/快速移动后必然重建，这时回来涂油的是同一个人，必须允许换绑。
        # 判据是「轨迹还在不在跟踪器里」而不是「本帧匹不匹配」——后者分不开
        # 「同一只手暂时漏检」和「别的工人伸手进来」。
        if self._hand_tracked(task.hand_track_id):
            hand = self._live_hand(task.hand_track_id)
            if hand is not None and task.screw_anchor is not None:
                hx, hy = hand.observation_box().center
                width, height = self._frame_size
                diag = max(1.0, math.hypot(width, height))
                if (
                    math.hypot(hx - task.screw_anchor[0], hy - task.screw_anchor[1])
                    / diag
                    > self.cfg.hand_screw_max_distance * self._frame_scale
                ):
                    # 本周期那只手还在、却已经跑到别处：这是另一只手/另一件料，
                    # 不能接成本周期的锚点（实测会直接导致刷别处的杆也判完成）。
                    return None
        else:
            # 换绑：本周期那只手的轨迹已经不在跟踪器里。这一件被拿走了很久
            # （手和杆都出过画面），回来时在哪都合理，不限制位置；但杆仍要
            # 被这只手握着（下面按「杆心落在手框内」筛）。
            hand = self._nearest_live_hand_to(task.screw_anchor)
        if hand is None:
            return None
        label = self.station.detector.screw_label
        hand_box = hand.observation_box()
        best: tuple[float, BBox] | None = None
        for det in self._last_detections:
            if det.label != label:
                continue
            # 与「持有判定」同一口径：杆心落在手框内，或两者明显交叠。
            # 只要求「有一点交叠」不够——手臂从零件盒上方经过时会蹭到盒里
            # 整排静置杆，锚点会被接过去，之后刷手里那根永远累计不上。
            gripped = (
                _box_center_in(det.box, hand_box)
                or hand_box.iou(det.box) >= 0.15
            )
            if not gripped:
                continue
            # 位置没动过的杆不算：手臂从零件盒/盘上方经过时整排杆都落在
            # 手框内，接成锚点之后刷手里那根永远累计不上（同
            # ``_static_anchor_ok``）。
            if not self._static_anchor_ok(det.box, task.screw_anchor):
                continue
            # 多根都满足时取离旧锚点最近的（离得远说明是别人手里的）。
            cx, cy = det.box.center
            score = (
                0.0
                if task.screw_anchor is None
                else math.hypot(cx - task.screw_anchor[0], cy - task.screw_anchor[1])
            )
            if best is None or score < best[0]:
                best = (score, det.box)
        if best is None:
            return None
        return best[1], hand

    def _hand_tracked(self, hand_track_id: int | None) -> bool:
        """该 id 的轨迹是否仍在跟踪器里（无论本帧是否匹配）。"""

        return (
            hand_track_id is not None
            and self._hand_tracker.get(hand_track_id) is not None
        )

    def _nearest_live_hand_to(
        self,
        anchor: tuple[float, float] | None,
        *,
        limit: float | None = None,
    ) -> Track | None:
        """离给定位置最近的本帧有效手；给了 ``limit`` 就只接受够近的。"""

        live = [
            hand
            for hand in self._hand_tracks
            if hand.matched and hand.hits >= max(1, self._hand_tracker.min_hits)
        ]
        if not live:
            return None
        if anchor is None:
            return live[0]
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))

        def distance(hand: Track) -> float:
            cx, cy = hand.observation_box().center
            return math.hypot(cx - anchor[0], cy - anchor[1]) / diag

        best = min(live, key=distance)
        if limit is not None and distance(best) > limit:
            return None
        return best

    def _live_hand(self, hand_track_id: int | None) -> Track | None:
        """本帧真正观测到、且达到 ``min_hits`` 的那只手（否则 None）。

        ``min_hits`` 之前对手部完全不生效：单帧误检就能建轨迹并写进盘状态。
        """

        if hand_track_id is None:
            return None
        hand = self._hand_tracker.get(hand_track_id)
        if hand is None or not hand.matched:
            return None
        if hand.hits < max(1, self._hand_tracker.min_hits):
            return None
        return hand

    def _hand_center(self, hand_track_id: int | None) -> tuple[float, float] | None:
        """本周期那只手的中心，用于丝杆锚点初始化。

        与其余判定（握持、进盘/离盘、锚点交叠）统一用 ``observation_box()``
        ——原始检测框。``box`` 是卡尔曼滤波后的状态，快速移动时明显滞后
        （实测 200px 的跳变只走到 155px）；锚点初始化时用滤波框会把杆锁到
        手「上一帧所在」的位置，取料瞬间尤其容易锁错。
        """

        hand = self._live_hand(hand_track_id)
        return None if hand is None else hand.observation_box().center

    def _has_live_brush(self) -> bool:
        """本帧是否有任一已匹配的刷头轨迹（用于区分漏检和远离）。"""

        return any(
            brush.matched and brush.hits >= self._brush_tracker.min_hits
            for brush in self._brush_tracks
        )

    def _brush_work_distance(self, screw_box: BBox, brush: Track) -> float:
        """刷头工作点到杆身中线的归一化距离（复核统计用）。

        与 :meth:`_nearest_brush` 同一算法（都用 ``brush_work_point``），
        不另立口径——否则复核时看到的距离与判定用的距离会对不上。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        point = brush_work_point(brush, self.station.detector)
        return _distance_to_long_axis(screw_box, point) / diag

    def _close_brush_run(self, task: _Cycle) -> None:
        """结束当前连续接触段，把它的时长并入「最长连续接触」。

        段长取「段内最后一次刷头靠近的时刻 - 段起点」，不用收尾那帧的 ts：
        杆不可见时 ``_update_oiling`` 会提前 return，那段空档并没有累计进
        ``brush_hold_s``，也不该算进连续接触时长。
        """

        if task.brush_run_start_ts is None:
            return
        end = task.brush_last_near_ts or task.brush_run_start_ts
        task.brush_longest_run_s = max(
            task.brush_longest_run_s, end - task.brush_run_start_ts
        )
        task.brush_run_start_ts = None
        task.brush_last_near_ts = None

    def _nearest_brush(self, screw_box: BBox) -> Track | None:
        """离杆身接触邻域最近、且已被本帧检测到的刷头。

        距离量的是刷头工作点到**杆身中线**的垂距，不是到杆心的距离：杆框
        细长，贴着杆身中下部刷时工作点离杆心天然就在门限附近，按杆心算
        会有一半的帧被判成「没靠近」（见 ``_distance_to_long_axis``）。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        best: tuple[float, Track] | None = None
        for brush in self._brush_tracks:
            if not brush.matched or brush.hits < self._brush_tracker.min_hits:
                continue
            point = brush_work_point(brush, self.station.detector)
            distance = _distance_to_long_axis(screw_box, point) / diag
            if distance > self.cfg.brush_contact_distance:
                continue
            if best is None or distance < best[0]:
                best = (distance, brush)
        return None if best is None else best[1]

    def _complete_oiling(
        self,
        task: _Cycle,
        ts: float,
        *,
        judge: JudgeResult | None = None,
        session: _OilingSession | None = None,
        episode: Episode | None = None,
    ) -> None:
        self._close_brush_run(task)
        task.stage = TaskStage.OILED
        task.stage_since = ts
        task.completed_at = ts
        task.final_result = TaskResult.COMPLETED
        task.reason_code = "MANUAL_OILING_COMPLETED"
        task.evidence.append("BRUSH_NEAR_SCREW_HELD")
        task.timeline.append((TaskStage.OILED.value, ts))
        if episode is not None:
            start_ts, end_ts = episode.start_ts, episode.end_ts
        else:
            # 起点是**这次接触**的开始，不是取料时刻。
            #
            # 取料与涂油并行判定，工人常常取完料过几秒才开刷（示例① 13.0s
            # 取料、18.0s 才开刷）。按「取料时刻」当起点，事件区间会从取料
            # 一直横跨到刷完（实测 13.93→18.93，看着像刷了 5 秒），复核页
            # 核对时间点时对不上。``_attach_oiling`` 记下的
            # ``_oiling_start_ts`` 才是这一段接触的起点。
            #
            # 终点取「接触结束」与「取料时刻」中较晚的那个：补挂的涂油是在
            # 取料结算**之前**就刷上的（工人一边取料一边刷），接触结束可能
            # 早于取料时刻（实测 8.73→7.47 那种倒挂），取较晚的才不出现
            # end < start。
            picked_at = task.picked_at or task.created_at
            start_ts = (
                picked_at if self._oiling_start_ts is None else self._oiling_start_ts
            )
            end_ts = max(ts, picked_at)
        extra = {
            "hand_track_id": task.hand_track_id,
            "brush_track_id": task.brush_track_id,
            "brush_hold_s": round(task.brush_hold_s, 3),
            "required_s": self.cfg.brush_hold_s,
        }
        if judge is not None:
            extra["judge"] = judge.to_dict()
        if episode is not None:
            extra["judge_episode"] = episode.to_dict()
        self._emit_event(
            task,
            event_type=EventType.MANUAL_OILING,
            reason="manual_oiling_completed",
            extra=extra,
            # 没有模型时：起点取「取料时刻」和「这次接触起点」中较早的那个，
            # 终点取较晚的那个——补挂的涂油是在取料结算之前就刷上的，按取料
            # 时刻起点会得到倒挂区间（实测 8.73→7.47）。
            start_ts=start_ts,
            end_ts=end_ts,
            review=build_oiling_summary(task, required_s=self.cfg.brush_hold_s),
        )

        self._logger.info(
            "人工二次涂油完成 %s（手持丝杆，刷头靠近 %.2fs）",
            task.task_id,
            task.brush_hold_s,
        )
