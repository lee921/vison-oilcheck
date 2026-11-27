"""取料判据（二）：丝杆自己从盘内移到盘外（mixin）。

第二条取料判据，**完全不看手**，只看杆自己在哪。手被遮挡、漏检、或整段
没被检出时，手部判据会把一次真实取料整件丢掉；丝杆是刚体、检测稳定得多。

判定要求三件事，缺一不可：

1. **杆确实在油盘里**：杆心在 ``tray_roi`` 内，且杆本身没动过的累计时长
   够 ``screw_exit_pick_inside_s``（``_screw_static``）；
2. **移出盘区**：杆心越过盘区边线至少 ``screw_exit_pick_margin``，并连续
   ``_SCREW_EXIT_MIN_FRAMES`` 帧、至少 ``_SCREW_EXIT_MIN_OUTSIDE_S`` 这么久
   都在外；
3. **落点在工作台一侧**（盘区中线以下）——把杆搬到盘区上方是整理料盘。

杆的身份靠「位置 + 尺寸」双门限的贪心最近邻维护，记录按时释放
（``_SCREW_EXIT_IDLE_RELEASE_S``），否则盘里那排静置杆会长期占坑。

判出后与手部判据走同一条 ``_confirm_pick``（去重也在那里），事件里
``pick_source = "screw_exit"``。
"""

from __future__ import annotations

import math

from ..association import brush_work_point
from ..constants import (
    _CARRY_INHAND_MIN,
    _OILING_SAME_ROD_STEP,
    _PICK_SOURCE_SCREW_EXIT,
    _ROD_MIN_ASPECT,
    _SCREW_EXIT_IDLE_RELEASE_S,
    _SCREW_EXIT_MIN_FRAMES,
    _SCREW_EXIT_MIN_OUTSIDE_S,
    _SCREW_TRACK_MISS,
)
from ..geometry import (
    _box_center_in,
    _distance_to_long_axis,
    _exit_depth,
    _grip_corner_fraction,
)
from ..state import _CarryEvidence, _ScrewExit
from ..types import BBox, Detection
from .shared import SharedRules


class ScrewExitRules(SharedRules):
    """丝杆离盘判据。"""

    # 取料判定（二）：丝杆自己从盘内移到盘外
    # ------------------------------------------------------------------
    def _update_screw_exit_pick(
        self, *, ts: float, dt: float, tray_box: BBox | None
    ) -> None:
        """丝杆从盘区内移到盘区外 ⇒ 取料完成。

        **与手部判据并行、互不依赖**：这条只看杆的位置，不看手。现场手被
        遮挡、漏检、或者整段没被检出时，手部判据会把一次真实取料整件丢掉；
        丝杆是刚体、检测稳定得多。

        判定要求三件事，缺一不可：

        1. **杆确实在油盘里**：杆心在 ``tray_roi`` 内，且杆**本身没动过**
           （``_screw_static``）的累计时长够 ``screw_exit_pick_inside_s``。
           位置一直在变的杆（被手举着在盘区上方刷、或只是从盘区上空划过）
           累计不上。这一条正是「确保丝杆本身确实在油盘中」——没有它，工人
           拿着杆从盘前走过、把杆放回盘里再拿走，都会被判成取料；
        2. **移出盘区**：杆心越过盘区边线至少 ``screw_exit_pick_margin``
           （按画面对角线归一化），并连续 ``_SCREW_EXIT_MIN_FRAMES`` 帧、
           至少 ``_SCREW_EXIT_MIN_OUTSIDE_S`` 这么久都在外面。贴着边线抖一下、
           或杆被举起来又放回去，都不算；
        3. **不重复上报**：与手部判据在同一件料上几乎同时确认时按
           ``screw_exit_pick_dedup_s`` 去重，同一次出盘只报一次。

        杆的身份由一条专用的 ByteTrack 轨迹维护（``_screw_tracker``）：盘里
        那排杆各走各的轨迹，被带走的那根一路跟出盘区；杆被手完全遮住几帧时
        轨迹还在，不会把一次取料切成两段。

        判出后与手部判据走同一条 ``_confirm_pick``：建作业周期、记锚点，
        后续涂油判定完全一致，只是事件里标注来源为 ``screw_exit``。

        ``screw_exit_pick_enabled = false`` 时整条关闭，行为与旧版本一致。
        """

        if not self.cfg.screw_exit_pick_enabled or tray_box is None:
            return
        width, height = self._frame_size
        if width <= 0 or height <= 0:
            return
        diag = max(1.0, math.hypot(width, height))
        margin = self.cfg.screw_exit_pick_margin

        # 候选杆：盘区内、盘区外紧邻一圈，以及**已经建了记录**的杆（它正
        # 在走出去，可能已经离盘区很远）。
        region = BBox(
            tray_box.x1 - margin * diag,
            tray_box.y1 - margin * diag,
            tray_box.x2 + margin * diag,
            tray_box.y2 + margin * diag,
        )
        candidates = [
            det
            for det in self._all_screw_detections()
            if _box_center_in(det.box, region)
        ]
        candidates.extend(
            det
            for det in self._all_screw_detections()
            if not _box_center_in(det.box, region)
            and any(
                self._same_track_position(state, det.box, width, height)
                for state in self._screw_exits.values()
            )
        )

        # 与记录配对：**位置 + 尺寸双门限的贪心最近邻**。
        #
        # 为什么不用速度预测：丝杆是长条形刚体，手把它举起来带走时朝向会变，
        # 框心并不沿速度方向移动（实测 170216 第 1 件：y 从 323 走到 222，
        # 而 x 在 1533→1599→1522 之间来回），匀速预测反而会把它配到旁边
        # 那根杆上。
        #
        # 为什么必须带尺寸：环境里「丝杆」的框尺寸差得很远——盘里那根
        # 275×30，盘区下方摆着的是 150×150 的方块，零件盒里的又是另一种。
        # 只按位置最近配对，手一抬杆框就会和下方那个大方块抢同一个检测。
        # 同一根杆的框宽在几帧内会变（遮挡多少），但不会跨量级，所以用
        # 「更新前后尺寸比」做门限比绝对值稳。
        #
        # 门限两步：先全局贪心取「位置最近且尺寸相容」的配对；位置上限用
        # 单帧容许位移乘以漏检帧数，长时间跟丢的杆才允许一次跨很远配回去。
        strict = min(
            self.cfg.screw_exit_track_distance * self._frame_scale,
            self.cfg.screw_exit_track_distance * 2.0,
        )
        scored: list[tuple[float, int, int]] = []
        for index, det in enumerate(candidates):
            cx, cy = det.box.center
            for track_id, state in self._screw_exits.items():
                if state.last_center is None:
                    continue
                distance = math.hypot(
                    cx / max(1, width) - state.last_center[0],
                    cy / max(1, height) - state.last_center[1],
                )
                # 位移上限：静止的杆每帧只动 ~0.001，用严格门限就够；已经
                # 在动的杆（正被带走）单帧能跳 0.1 以上，必须放宽到
                # ``screw_exit_pick_margin`` 的量级，否则会跟丢、过几帧以
                # 新记录的身份出现在盘外——新记录没有「在盘里放稳过」的
                # 证据，那次取料就整件丢掉。
                #
                # 放宽只在「记录本来就在动」时生效：盘里那排静止杆之间
                # 间距 0.02~0.03，给静止记录放宽就会互相串。
                base = strict if state.static else max(
                    strict, self.cfg.screw_exit_pick_margin * 12.0
                )
                limit = min((1 + state.misses) * base, base * 3.0)
                if distance > limit:
                    continue
                if not state.size_compatible(det.box):
                    continue
                scored.append((distance, track_id, index))
        scored.sort(key=lambda item: item[0])
        assignments: dict[int, int] = {}
        taken: set[int] = set()
        for _distance, track_id, index in scored:
            if track_id in assignments or index in taken:
                continue
            assignments[track_id] = index
            taken.add(index)

        # 已有记录：续上，或连续漏检后丢弃。
        for track_id, state in list(self._screw_exits.items()):
            index = assignments.get(track_id)
            if index is None:
                state.misses += 1
                if state.misses > _SCREW_TRACK_MISS:
                    self._screw_exits.pop(track_id, None)
                continue
            # 一条从没动过、也没报过的旧记录活过 ``_SCREW_EXIT_IDLE_RELEASE_S``
            # 就释放：它是盘里的存货，一直占着位置会让「这一根被取走」没法
            # 从新记录开始累计（见 ``_ScrewExit.age_s`` 的说明）。
            if (
                not state.active
                and not state.reported
                and state.age_s > _SCREW_EXIT_IDLE_RELEASE_S
            ):
                self._screw_exits.pop(track_id, None)
                continue
            self._advance_screw_exit(
                state,
                candidates[index],
                ts=ts,
                dt=dt,
                tray_box=tray_box,
                diag=diag,
                width=width,
                height=height,
            )

        # 新出现的杆建记录：杆心**严格**落在盘区内、且框是**细长杆**才建。
        #
        # 前一条是「这根杆确实来自油盘」的地域条件。后一条是形状条件：盘区
        # 下沿/手边常被误检出一坨近方形的「screw」（实测 146×154、170×170、
        # 136×130），它们随手动、静止一会儿再被带走，会完整走完「在盘里放稳
        # → 出盘」的流程、凭空报一次取料（示例 092817 8.6~9.3s）。丝杆框的
        # 长宽比实测 p50≈4，取 2.0 作下限足以把这类方块挡在建记录这一步。
        for index, det in enumerate(candidates):
            if index in taken or _exit_depth(det.box, tray_box) < 0.0:
                continue
            long_side, short_side = (
                max(det.box.width, det.box.height),
                min(det.box.width, det.box.height),
            )
            if short_side <= 0 or long_side / short_side < _ROD_MIN_ASPECT:
                continue
            self._screw_exit_seq += 1
            state = _ScrewExit(track_id=self._screw_exit_seq, inside=True)
            self._screw_exits[state.track_id] = state
            self._advance_screw_exit(
                state,
                det,
                ts=ts,
                dt=dt,
                tray_box=tray_box,
                diag=diag,
                width=width,
                height=height,
                fresh=True,
            )

    @staticmethod
    def _same_track_position(
        state: _ScrewExit, box: BBox, width: int, height: int
    ) -> bool:
        """这根检测框是不是该记录那根杆（粗判，只用于扩大候选范围）。"""

        if state.last_center is None:
            return False
        cx, cy = box.center
        return (
            math.hypot(
                cx / max(1, width) - state.last_center[0],
                cy / max(1, height) - state.last_center[1],
            )
            <= 0.15
        )

    def _all_screw_detections(self) -> list[Detection]:
        """本帧全部丝杆检测（不做区域过滤）。"""

        label = self.station.detector.screw_label
        return [det for det in self._last_detections if det.label == label]

    def _advance_screw_exit(
        self,
        state: _ScrewExit,
        det: Detection,
        *,
        ts: float,
        dt: float,
        tray_box: BBox,
        diag: float,
        width: int,
        height: int,
        fresh: bool = False,
    ) -> None:
        """把这一帧的观测折进一根杆的出盘记录，必要时判出取料。"""

        box = det.box
        state.misses = 0
        state.age_s += max(0.0, dt)
        state.static = self._screw_static(box)
        if not state.static:
            # 动过：这根杆有可能就是被带走的那根，记录要留住。
            state.active = True
        state.last_size = (box.width, box.height)
        state.max_short = max(state.max_short, min(box.width, box.height))
        cx, cy = box.center
        center = (cx / max(1, width), cy / max(1, height))
        if state.last_center is not None and state.last_ts is not None:
            gap = max(1e-3, ts - state.last_ts)
            state.velocity = (
                (center[0] - state.last_center[0]) / gap,
                (center[1] - state.last_center[1]) / gap,
            )
        state.last_center = center
        state.last_ts = ts
        margin = self.cfg.screw_exit_pick_margin
        depth = _exit_depth(box, tray_box) / diag
        # 三个区间，不是两个：
        # - ``depth >= 0``：确实在盘区**inside**。只有这里才累计「放在盘里」
        #   的证据、才建记录；
        # - ``-margin <= depth < 0``：贴着边线的容差带。不算在盘里（否则
        #   盘沿外那排摆着的杆会被当成「盘里的料」），也不算出盘——杆框细长，
        #   倾斜时框心本来就贴着边线摆动；
        # - ``depth < -margin``：真的出盘了。
        inside = depth >= 0.0

        if inside:
            if not state.inside or state.reported:
                # 进盘，或报过之后又被放回盘里：这一次出盘作废，从头累计。
                state.restart_excursion()
            if state.entered_ts is None:
                state.entered_ts = ts
            if not self._screw_static(box):
                # 杆在盘区里动了：取料动作从这一刻开始，事件区间用它当起点。
                # 已经确认过「确实在盘里」的锁存不变——它正是取料的前提。
                #
                # 起点要跟着**最近一次**开始移动走，不能在第一次动过之后就锁死：
                # 盘里的杆在整段视频里会因为检测框抖动被偶尔判成「在动」
                # （实测 091519 的静止杆在 3.9s 就动了一下、之后一直不动到
                # 4.5s 才真被取走），锁死会把事件起点拖到几分钟前。
                state.moving_since_ts = ts
                state.static_s = 0.0
            else:
                state.static_s += max(0.0, dt)
                if state.static_s >= self.cfg.screw_exit_pick_inside_s:
                    state.confirmed = True
            return

        if depth >= -margin:
            # 容差带：既不算在盘里，也不算出去。杆被举起来又放回盘沿时，
            # 这一帧要把出盘计时清掉，否则「抬一下又放下」会被当成出盘。
            state.inside = False
            state.static_s = 0.0
            state.outside_since_ts = None
            state.outside_frames = 0
            return

        # 在盘外。
        state.inside = False
        state.static_s = 0.0
        if state.reported or not state.confirmed:
            # 没在盘里放稳过，或者这一趟已经报过：不是从盘里取料。
            state.entered_ts = None
            state.moving_since_ts = None
            state.outside_since_ts = None
            state.outside_frames = 0
            state.confirmed = False
            return
        if state.outside_since_ts is None:
            state.outside_since_ts = ts
            state.outside_frames = 0
        state.outside_frames += 1
        if (
            state.outside_frames < _SCREW_EXIT_MIN_FRAMES
            or ts - state.outside_since_ts < _SCREW_EXIT_MIN_OUTSIDE_S
        ):
            return
        # 出盘的落点必须在盘区**下方**（工作台一侧）。取料是把杆从盘里取出、
        # 拿到盘体下方的工作台上二次涂油；而工人整理料盘时会把杆搬到盘区
        # 上方的高处（示例 091519 15.7s：杆从盘内 (1537,227) 被搬到画面顶部
        # (1574,40) 并放稳），那种搬动不是取料，只按「杆心出了盘区」会误报。
        cy = box.center[1]
        if cy <= (tray_box.y1 + tray_box.y2) * 0.5:
            self._logger.debug(
                "丝杆离盘判据：杆出盘后落在盘区上方（y=%.0f 不高于盘区中线 "
                "%.0f），不是取料",
                cy,
                (tray_box.y1 + tray_box.y2) * 0.5,
            )
            state.reported = True
            state.confirmed = False
            state.entered_ts = None
            state.moving_since_ts = None
            state.outside_since_ts = None
            state.outside_frames = 0
            return
        start_ts = state.moving_since_ts or state.entered_ts or ts
        state.reported = True
        state.confirmed = False
        state.entered_ts = None
        state.moving_since_ts = None
        state.outside_since_ts = None
        state.outside_frames = 0
        self._confirm_screw_exit_pick(ts=ts, det=det, start_ts=start_ts)

    def _duplicate_pick(self, ts: float, center: tuple[float, float]) -> bool:
        """这次取料是不是刚判过的同一件料。

        两条判据都会在「杆离开盘区」之后几百毫秒到一两秒内确认，先后顺序
        不固定（实测两种都出现过）：手部判据要等手离盘后攒满采样窗口，
        丝杆判据要等杆出盘并确认。去重必须放在共用的确认点上，单边去重
        挡不住反过来的顺序。
        """

        if self._last_pick_ts is None or self._last_pick_center is None:
            return False
        if ts - self._last_pick_ts > self.cfg.screw_exit_pick_dedup_s:
            return False
        return self._same_anchor(self._last_pick_center, center)

    def _brush_touches_rod(self, rod: BBox) -> bool:
        """本帧有没有刷头工作点贴着这根杆的杆身中线。

        与涂油判定（``_oiling_contacts``）同一口径：到杆身中线的垂距小于
        ``brush_contact_distance``。用来区分「杆从盘里被取出来」与
        「取出后已经在刷」——后者不该再算一次取料。
        """

        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        for brush in self._brush_tracks:
            if not brush.matched or brush.hits < self._brush_tracker.min_hits:
                continue
            point = brush_work_point(brush, self.station.detector)
            if (
                _distance_to_long_axis(rod, point) / diag
                <= self.cfg.brush_contact_distance
            ):
                return True
        return False

    def _oiling_session_covers(self, rod: BBox) -> bool:
        """这根杆是不是**已经有一段涂油在跟进**（本帧或刚才几帧贴过）。

        取料和「取出后在同一个位置刷」这两件事只差一个时间先后：杆出盘
        那一两帧刷头往往已经贴上去了（现场是取出就着盘沿刷）。只按
        「此刻刷头贴着」判，会把真空取料也一起否掉。
        这里改按「这一段涂油已经在这根杆上累计过」判：只有涂油会话
        已经建立、且锚点就在这根杆上，才认为「这不是新取的一件」。
        """

        session = self._active_oiling
        if session is None or session.anchor is None:
            return False
        width, height = self._frame_size
        diag = max(1.0, math.hypot(width, height))
        cx, cy = rod.center
        if (
            math.hypot(cx - session.anchor[0], cy - session.anchor[1]) / diag
            > _OILING_SAME_ROD_STEP
        ):
            return False
        # 至少累计过一点接触：刚建立、纯靠一帧 dt 的不算。
        return session.cum_s >= self._at_least_frame(0.1)

    def _confirm_screw_exit_pick(
        self, *, ts: float, det: Detection, start_ts: float
    ) -> None:
        """丝杆离盘判据确认取料：交给 ``_confirm_pick``（那里统一去重）。"""

        self._confirm_pick(
            ts=ts,
            hand_track_id=None,
            screw=det,
            dwell_s=max(0.0, ts - start_ts),
            judge=None,
            source=_PICK_SOURCE_SCREW_EXIT,
        )


    def _carry_alignment(self, carry: _CarryEvidence) -> float:
        """carry 那根杆与手的位移方向余弦（无证据时为 0）。"""

        if carry.rod_vec is None or carry.hand_vec is None:
            return 0.0
        rn = math.hypot(*carry.rod_vec)
        hn = math.hypot(*carry.hand_vec)
        if rn <= 1e-9 or hn <= 1e-9:
            return 0.0
        cosine = (
            carry.rod_vec[0] * carry.hand_vec[0]
            + carry.rod_vec[1] * carry.hand_vec[1]
        ) / (rn * hn)
        return max(-1.0, min(1.0, cosine))




    def _screw_history_at(
        self, target_ts: float, *, take: int = 1
    ) -> list[tuple[BBox, ...]]:
        """取 ``target_ts`` 之前最近的 ``take`` 份丝杆框。

        历史按时间索引，所以「多久之前」这个对照窗口与帧率无关：同一段动作
        在 15fps 与 20fps 下看的是同一个时间距离。取不到（历史还没攒够那么长）
        时返回空列表，调用方按「无对照」处理。

        回看窗口内不足 ``take`` 帧时（帧间隔比窗口还长，例如 2fps），退一步
        用已有的最早那几帧——真有一两帧对照就够跳出单帧漏检，总比判成
        「无对照」而把静止判据整个关掉好。仍然一帧都没有才返回空。
        """

        frames = [entry for entry in self._screw_history if entry[0] <= target_ts]
        if not frames:
            # 窗口比整段历史还早（开局）：用最早的几帧作对照。
            frames = list(self._screw_history)
        if not frames:
            return []
        # 与原先「窗口内最早的两帧」一致：这两帧互为漏检的备份。低帧率下
        # 窗口里只有一两帧时，它们同时也是唯一可用的对照。
        return [entry[1] for entry in frames[:take]]



    def _update_carry_evidence(
        self,
        evidence: _CarryEvidence,
        hand_box: BBox,
        rod_box: BBox,
        *,
        ts: float | None = None,
    ) -> None:
        """把这一帧的候选杆折进 carry 证据（见 :class:`_CarryEvidence`）。

        杆是**锁定**跟踪的：第一次锁定后，后续帧只在离锁定位置
        ``_CARRY_LOCK_STEP`` 以内找同一根杆，找不到就丢这一帧，不换目标。
        不锁的话，手一划过去最近的那根就变成旁边另一根，两根杆的位移混在
        一起拼出虚假的「随手动」（实测夹具上那排杆两两间距 0.02~0.03，
        换目标一次就够把 memory 里的 rod_move 抬到门限以上）。
        """

        width, height = self._frame_size
        hand = (hand_box.center[0] / width, hand_box.center[1] / height)
        rod = (rod_box.center[0] / width, rod_box.center[1] / height)
        if evidence.start_rod is None:
            evidence.start_hand = hand
            evidence.start_rod = rod
        evidence.last_rod = rod
        evidence.frames += 1
        area = rod_box.width * rod_box.height
        if area > 0:
            inhand = hand_box.intersection_area(rod_box) / area
            evidence.max_inhand = max(evidence.max_inhand, inhand)
            # 持续落在手里：手从杆旁边划过只会零星一两帧达标，真握着刷/带走
            # 时是连续的一大片。判定按**时长**（``inhand_s``），帧数只留作
            # 诊断——按帧数数会让门限随帧率漂移。
            if inhand >= _CARRY_INHAND_MIN:
                evidence.inhand_frames += 1
                if ts is not None:
                    evidence.inhand_s += max(0.0, ts - (evidence.last_ts or ts))
            if ts is not None:
                evidence.last_ts = ts
        evidence.max_iou = max(evidence.max_iou, hand_box.iou(rod_box))
        evidence.max_grip = max(
            evidence.max_grip, _grip_corner_fraction(hand_box, rod_box)
        )
        rod_vec = (
            rod[0] - evidence.start_rod[0],
            rod[1] - evidence.start_rod[1],
        )
        hand_vec = (
            hand[0] - evidence.start_hand[0],
            hand[1] - evidence.start_hand[1],
        )
        rod_len = math.hypot(*rod_vec)
        if rod_len > evidence.rod_move:
            evidence.rod_move = rod_len
            evidence.rod_vec = rod_vec
        hand_len = math.hypot(*hand_vec)
        if hand_len > evidence.hand_move:
            evidence.hand_move = hand_len
            evidence.hand_vec = hand_vec







    def _entry_grace_s(self) -> float:
        """进盘前多久之内的「盘外空手」观测仍算数。

        窗口取 ``holding_lost_timeout_s``，但设 2s 的绝对上限：低帧率下
        乘上帧率倍率会长到 8s，那时「八秒前见过这只手空手」已经不足以
        说明本次是空手进盘。

        也不能取得太窄：工人从盘外走到盘内要 1~2s，窗口只有两三帧时，
        进盘那一刻永远查不到「最近空手」，真空手取料会被判成带料进盘
        而整件漏掉（示例视频第一件的取料即如此）。
        """

        return min(
            self._at_least_frame(self.cfg.holding_lost_timeout_s),
            2.0,
        )



    # ------------------------------------------------------------------
