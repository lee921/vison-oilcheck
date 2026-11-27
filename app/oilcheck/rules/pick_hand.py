"""取料判据（一）：空手进盘 → 带杆离盘（mixin）。

跟踪到一只手在**未持有丝杆**的状态下进入涂油盘（手工标定的 ``tray_roi``），
随后同一只手**带着丝杆离开**涂油盘，即判取料完成。

几个关键口径：

- **盘区只用手工 tray_roi**：``oil_station`` 检测框只框得住盘体本身，而
  取料时手在盘体下方 100~280px，按检测框判会全部漏掉；
- **离盘后的握持按采样窗口投票**（每 ``pick_sample_every_frames / 15`` 秒
  一个点，最近 ``pick_history_samples`` 个点），不拿单帧当结论；
- **杆心必须已经出盘**：手在盘外时，杆心还在盘区里的那根是盘里的料，
  不算「手里拿着」。杆心比手晚出盘的那几帧既不采样也不定论
  （``_carrying_out_of_tray``）；
- **带料进盘**时，只有带出去的仍是带进来的那根才不判取料
  （``_same_rod_as_entry``）——带进来一根、又取走盘里另一根仍然是取料。
"""

from __future__ import annotations

from ..constants import (
    _PICK_SAMPLE_CALIBRATION_FPS,
    _RULE_TRUE_VETO_P,
)
from ..raw_features import measure_pick
from ..state import _HandVisit
from ..types import BBox, Detection
from .shared import SharedRules


class HandPickRules(SharedRules):
    """取料判据（一）：手部进盘 / 离盘。"""

    # 取料判定
    # ------------------------------------------------------------------
    def _update_pick(self, *, ts: float, dt: float, tray_box: BBox | None) -> None:
        """空手进盘 → 带料离盘 ⇒ 取料完成。

        所有子判定都用最近 n 个采样点投票（``pick_sample_every_frames`` /
        ``pick_history_samples``），不拿单帧当结论：

        - **是否在盘内**：窗口内出现过盘内采样即可确认「进过盘」（事件性，
          用 any——短时取料不会被后来的盘外采样淹没）；窗口多数在盘外才
          确认离盘（用多数——避免盘边抖动劈段）；
        - **离盘后手里有没有杆**：只采**盘外**的读数。攒满 n 个采样点就
          按多数投票；点数不足就得定论（手出画面、视频结束、等到超时）
          时，见过一帧手握杆就算握持。离盘那一帧的杆检测往往还没稳（杆心
          刚离开盘区、置信度 0.5 上下），盘内读数又被静置杆污染，两者都
          不能混进窗口；
        - **进盘时手里有没有杆**：同样按盘外观测历史投票。

        带料进盘又带料离开（例如工人把丝杆放回去再拿走）不算取料，空手进、
        空手出也不算。点数不足时**不能**一律按空手处理：那样一次真实取料
        会连周期都建不出来，后面的涂油判定无从谈起（现场「取料没跑完就
        停了」）。只有一帧正面证据都没有时才按空手处理。
        """

        every = max(1, int(self.cfg.pick_sample_every_frames))
        keep = max(1, int(self.cfg.pick_history_samples))
        # 采样节拍**按秒算，不按帧数**：同一段动作在任何帧率下都应采到同样
        # 多的点。``pick_sample_every_frames`` 保留原名（现场配置与文档都用
        # 它），但它是「在标定帧率 15fps 下每 k 帧」的意思——换算成秒即
        # ``k / 15``，与运行时帧率无关。按帧数数点会让高帧率采到更多样本、
        # 窗口更早攒满，判定随帧率漂移（实测 20fps 比 15fps 少判 4 件）。
        #
        # 帧率低于该周期时（每帧间隔更长），``due`` 每帧都会成立，等效于
        # 「每帧采一个点」——这是采样能力的上限，不必也不该再缩放。
        sample_period_s = every / _PICK_SAMPLE_CALIBRATION_FPS
        # 离盘后攒够 n 个采样的时限 + 一点余量；超时仍不足则按空手处理。
        settle_s = (keep + 1) * sample_period_s
        live_hands: set[int] = set()
        # 先把「手部 ID 在进盘途中被重建」的过程转交给真正接手的那只手。
        self._hand_over_split_visits(tray_box, ts)
        for hand in self._hand_tracks:
            if not hand.matched or hand.hits < max(
                1, self._hand_tracker.min_hits
            ):
                # 单帧误检不参与进盘判定（min_hits 之前对手部无效）。
                continue
            live_hands.add(hand.track_id)
            # 判定用本帧原始检测框：滤波框在快速移动时滞后（200px 跳变只
            # 走到 155px），边界判定会晚一帧。
            hand_box = hand.observation_box()
            cx, cy = hand_box.center
            inside_raw = (
                tray_box.x1 <= cx <= tray_box.x2
                and tray_box.y1 <= cy <= tray_box.y2
            )
            holding = self._holding_screw(hand_box)
            holding_now = holding is not None
            visit = self._visits.get(hand.track_id)

            if visit is None:
                if not inside_raw:
                    self._record_outside_holding(hand.track_id, holding, ts)
                empty_outside = self._mostly_empty_outside(
                    hand.track_id, ts, default_empty=True
                )
                visit = _HandVisit(
                    hand_track_id=hand.track_id,
                    entered_tray_ts=ts,
                    holding_on_entry=holding_now and not empty_outside,
                    inside_tray=inside_raw,
                    last_seen_ts=ts,
                    # 新建的这一帧就是第一个采样点（与 ``due`` 的首帧语义一致）。
                    last_sample_ts=ts,
                )
                self._visits[hand.track_id] = visit

            visit.last_seen_ts = ts
            visit.last_hand_box = hand_box
            visit.last_outside_holding = None if inside_raw else holding_now
            # 二级判定的输入：累积**原始检测框**的测量。只喂模型，
            # 规则判定完全走下面的原有逻辑，不受影响。
            if visit.entered_confirmed and not visit.holding_on_entry:
                frame = self._raw_frame(ts)
                if visit.raw_entry_rods is None:
                    visit.raw_entry_rods = sum(
                        1 for det in frame.screws if frame.in_tray(det.box)
                    )
                visit.raw.add(measure_pick(frame, hand.track_id, hand_box))
            # 记录「本趟第一次进盘」。起点不取首次见到手的时刻（那会把盘外
            # 跟踪时长算进去），也不逐帧累加盘内帧（手在盘内整段漏检时
            # 会严重低估，示例①第 3 次取料只有 4 帧命中盘区）。
            if inside_raw:
                if visit.first_inside_ts is None:
                    visit.first_inside_ts = ts
                visit.dwell_in_tray_s += max(0.0, dt)
            # 采样按**时间**节拍走（见 ``_HandVisit.due``）：按帧数数点会让
            # 同一段动作在 20fps 下攒到 1/3 更多的样本，判定随帧率漂移。
            due = visit.due(ts, sample_period_s)
            if due:
                visit.push_inside(inside_raw, keep)
                # 盘内读数不可用（会蹭到静置杆），只在盘外采握持。
                if not inside_raw and visit.exited_at_ts is None:
                    self._record_outside_holding(hand.track_id, holding, ts)

            # ---- 离盘后：攒盘外握持采样，攒够再定论 ----
            if visit.exited_at_ts is not None:
                if inside_raw:
                    # 又回到盘内：视为同一段取料的中途抖动，取消离盘待定。
                    visit.exited_at_ts = None
                    visit.restart_holding()
                    visit.holding_seen = False
                    visit.carry.reset()
                    continue
                # 手已出盘、手里的杆还没出盘的那几帧**既不采样也不定论**。
                #
                # 杆心比手晚出盘是常态：示例④ 第 3 次取料手在 f348 出盘、
                # 杆在 f352 才出盘，中间 4 帧按「杆必须出盘」都是「手里没杆」。
                # 把这几帧记成 False，窗口 [F,F,T] 多数票就是「空手」，
                # 整件取料丢掉；掉一帧窗口错位，结论更随机。跳过这几帧后，
                # 采样窗口只包含杆真正出盘之后的读数。
                #
                # 也不能让这几帧照常走「攒够就定论 / 超时就定论」：示例⑤ 第 2、
                # 3 次取料手在 6.53s / 13.13s 出盘，手里那根杆一直在盘区下沿
                # 慢慢往下带，杆心出盘要 1s 以上（第 2 次 2.2s）。按离盘后的
                # 固定时长计算，0.4s 的结算期一到，采样窗口还是空的、「见过
                # 没见过手握杆」也还是假的，于是当场按空手定论，整件取料丢掉。
                # 等待的时限按 ``_carry_wait_s()``，不能用离盘后的固定时长。
                #
                # 「离盘」本身由采样窗口确认（可能比手真正出盘晚一两帧，
                # 所以这一段在 exited 之后才判得到）。判据是「手边有随手动、
                # 但杆心仍在盘区内的杆」——那是这一件正在出门，不是空手离开；
                # 进盘时就在盘里的存货不算（``entry_inventory``）。
                if self._carrying_out_of_tray(
                    hand_box, visit.entry_inventory, visit.carry, ts=ts
                ):
                    if ts - visit.exited_at_ts <= self._carry_wait_s():
                        # 「手边有随手动、杆心还在盘区内」是**带杆出门**的
                        # 暂定证据：工人把杆停在盘区下沿刷完再带走时，杆心一
                        # 直没出盘，holding_now 全程为假，不记这一条整件取料
                        # 会丢（示例⑥ 第 2 件 10.07s、示例⑦ 第 2/3 件）。
                        #
                        # 但「手在盘外 + 手边有盘内杆」本身不够：手框很大，
                        # 盘里/夹具上的杆都会落进来。示例① 工人按夹具、示例⑧
                        # 工人把杆装上夹具、示例⑨ 手从夹具边划过，都会被凭空
                        # 判成一次取料。
                        #
                        # 差别在**这根杆是不是真的被手带着走**：真取料时杆
                        # 跟着手动（自身位移 ≥0.017）或明显落在手框里；夹具上
                        # 的杆全程不动、落手面积也不够（见 ``_CarryEvidence``）。
                        visit.carry_frames += 1
                        if visit.carry.carried:
                            visit.holding_seen = True
                        # 等杆出盘的这段里，手确实在盘外：照常记「盘外空手」
                        # 历史，供**下一次**进盘定性。旧实现这几帧已经结算、
                        # 重建了进盘过程，每帧都在记；现在过程还挂着，不补
                        # 记录就断了窗口（示例③在 7.5fps 下下一件被误判成
                        # 「带料进盘」而漏掉）。
                        if due:
                            self._record_outside_holding(
                                hand.track_id, holding, ts
                            )
                        continue
                    # 等太久了还没把杆带出去（贴着盘沿刷、或检测框一直在盘区
                    # 内抖动）：不再等，用已经攒到的盘外读数定论。
                # 正面证据不挑采样点：采样按 k 帧一个，手离盘后很快就
                # 出画面时窗口可能一个点都攒不到。
                if holding_now:
                    visit.holding_seen = True
                if due:
                    visit.push_holding(holding_now, keep)
                # 攒满窗口才定论（和之前一致）。提前定论会改变后面几段的
                # 「进盘时手上有没有杆」：实测 105708 在 7.5fps 下从 4 取
                # 掉到 3 取。点数不足的情形由 holding_verdict 和「手消失」
                # 分支兜底，不需要抢跑。
                ready = len(visit.holding_samples) >= keep
                timed_out = ts - visit.exited_at_ts > settle_s
                if not (ready or timed_out):
                    continue
                self._settle_pick(
                    visit=visit,
                    hand_track_id=hand.track_id,
                    hand_box=hand_box,
                    ts=ts,
                    holding=holding,
                    keep=keep,
                )
                self._visits.pop(hand.track_id, None)
                continue

            # ---- 尚未离盘：判断进盘 / 仍在盘内 ----
            if not visit.entered_confirmed:
                if any(visit.inside_samples):
                    visit.entered_confirmed = True
                    # 进盘定性：看进盘前的盘外历史。
                    visit.holding_on_entry = not self._mostly_empty_outside(
                        hand.track_id, ts, default_empty=True
                    )
                    # 带料进盘时记下手里那根杆的位置：离盘时它与离盘手里
                    # 那根对得上，才是「带进来又带走」，不判取料；对不上
                    # 说明带进来一根、又取走了盘里的另一根。
                    if visit.holding_on_entry and holding is not None:
                        visit.entry_held = holding.box.center
                    visit.inside_tray = True
                    visit.restart_holding()
                    visit.carry.reset()
                    # 记下这一趟进盘**之前**盘区里那批丝杆（存货快照）。
                    # 离盘后区分「手里带着走的那根」与「盘里的存货」用
                    # （见 ``_carrying_out_of_tray``）。
                    if self._screw_history:
                        visit.entry_inventory = self._screw_history[-1][1]
                elif not inside_raw:
                    self._record_outside_holding(hand.track_id, holding, ts)
                continue

            if not visit.left_majority:
                visit.inside_tray = True
                continue

            # 确认离盘：从这里开始攒盘外握持采样。
            visit.inside_tray = False
            visit.exited_at_ts = ts
            visit.restart_holding()
            visit.carry.reset()
            if not inside_raw:
                # 手此刻确实在盘外：这个读数就是离盘后的第一个证据。
                visit.last_outside_holding = holding_now
                if holding_now:
                    visit.holding_seen = True
            # 离盘这一帧不采样：此刻手的读数还可能是「手刚跨过盘边线、
            # 盘里那根还在手框里」的过渡态（杆心没出盘时
            # ``_holding_screw`` 会否掉盘内的杆，但离盘那一帧的盘内/盘外
            # 判定本身就有 ±1 帧抖动）。离盘后的采样从下一帧起算，
            # 与帧率、采样相位都无关。
            visit.last_sample_ts = ts

        # 手消失：先用已经攒到的采样定论，再结束进盘过程。
        trackers = {track.track_id: track for track in self._hand_tracks}
        for hand_id in list(self._visits):
            if hand_id in live_hands:
                continue
            visit = self._visits[hand_id]
            if ts - visit.last_seen_ts < self.cfg.holding_lost_timeout_s:
                continue
            if visit.exited_at_ts is None and self._vanished_outside(visit):
                # 手在盘外直接消失，离盘还没来得及按采样窗口确认（窗口
                # 按 k 帧一个点，走得太快就采不到）。最后观测确实在盘外，
                # 就按那一刻离盘结算，别把这段整段丢掉。
                visit.exited_at_ts = visit.last_seen_ts
                if visit.last_outside_holding:
                    visit.push_holding(True, keep)
                    visit.holding_seen = True
            if visit.exited_at_ts is not None:
                # 手已经确认离盘：用已攒到的采样定论，哪怕不足 n 个。
                # 之前要求「窗口里至少有一个采样」才结算，低帧率/很快就
                # 出画面时窗口是空的，这段进盘过程被直接丢掉，取料事件
                # 永远发不出来。
                #
                # 手框优先用当前轨迹，轨迹已被跟踪器删除（手出画面超过
                # 保留期）时用快照：手一消失就删轨迹的话，这里会拿到
                # None 而静默跳过，取料同样发不出来。
                hand = trackers.get(hand_id)
                hand_box = (
                    hand.observation_box()
                    if hand is not None
                    else visit.last_hand_box
                )
                if hand_box is not None:
                    self._settle_pick(
                        visit=visit,
                        hand_track_id=hand_id,
                        hand_box=hand_box,
                        ts=visit.exited_at_ts,
                        holding=None,
                        keep=keep,
                    )
            self._visits.pop(hand_id, None)
        for hand_id in list(self._outside_holding):
            if hand_id not in live_hands and hand_id not in self._visits:
                self._outside_holding.pop(hand_id, None)
                self._outside_holding_last_ts.pop(hand_id, None)

    @staticmethod
    def _vanished_outside(visit: _HandVisit) -> bool:
        """手消失时最后观测在盘外，且已经确认进过盘。"""

        return visit.entered_confirmed and visit.last_outside_holding is not None

    def _settle_pick(
        self,
        *,
        visit: _HandVisit,
        hand_track_id: int,
        hand_box: BBox,
        ts: float,
        holding: Detection | None,
        keep: int,
    ) -> None:
        """攒够采样（或不得不定论）时给这一件下定论。"""

        # 盘内停留 = 首次进盘 → 结算时刻的区间。
        # 原实现的起点是「首次见到这只手」：手在盘外被跟踪 3 秒再蹭一下
        # 盘区，dwell 就有 3 秒多，「盘边划过不判取料」的守卫形同虚设。
        # 起点改锚到真正进盘的那一帧，终点仍用结算时刻——离盘后攒采样
        # 的这段时间本来就计入阈值（现场标定的 pick_confirm_s 是在这套
        # 口径下调出来的），只换起点才不改变既有的取料行为。
        if visit.first_inside_ts is not None:
            dwell = max(0.0, ts - visit.first_inside_ts)
        else:
            dwell = visit.dwell_in_tray_s
        if visit.holding_on_entry:
            # 带料进盘：只有在「带出去的就是带进来的那根」时才不判取料。
            #
            # 现场（122925 84.33s）工人拿一根杆进盘、在盘里整理，随后把
            # **盘里的另一根**带出去——那是一次真取料。原先一刀切按
            # 「带料进盘」跳过，整件丢掉，后面那件还被顶成漏涂 NG。
            if self._same_rod_as_entry(visit, holding, hand_box):
                self._logger.debug(
                    "手 %s 带着丝杆进盘、又带着同一根离开，不判取料",
                    hand_track_id,
                )
                return
            self._logger.debug(
                "手 %s 带料进盘，但带走的是盘里的另一根，按取料处理",
                hand_track_id,
            )
        if dwell < self.cfg.pick_confirm_s:
            # 进盘到离盘太快，多半是手从盘边划过。
            self._logger.debug(
                "手 %s 在盘内停留过短（%.2fs），不判取料",
                hand_track_id,
                dwell,
            )
            return
        values = self._pick_features(visit, ts=ts, dwell=dwell)
        verdict = visit.holding_verdict(keep)
        judge_result = self._judge_pick(values, rule=verdict)
        if self._training_pick_hook is not None:
            self._training_pick_hook(values, ts)
        if judge_result is not None and not judge_result.used:
            self._logger.debug(
                "手 %s 取料判定：规则=%s，逻辑回归 p=%.3f（未参与）",
                hand_track_id,
                verdict,
                judge_result.probability,
            )
        if (
            verdict
            and judge_result is not None
            and judge_result.used
            and not judge_result.decision
        ):
            # 规则投「是」，但模型**明确判否**（``_judge_pick`` 的否决门，
            # 概率低于 ``_RULE_TRUE_VETO_P``）。这一票必须在这里落地：
            # 否决门只把 ``decision`` 改成 False，若下游仍按 ``verdict=True``
            # 继续，否决等于没生效。
            #
            # 现场 122925 第 1 秒即如此：工人把夹具/工件摆到工位上，手在盘区
            # 里停留 3.1 秒，规则凭「杆落在手框里 + 杆动过」投真、模型给
            # p=0.141 强烈判否，原先仍报了取料，还连带一次漏涂 NG。
            self._logger.debug(
                "手 %s 规则投是、模型判否（p=%.3f < %.2f），不判取料",
                hand_track_id,
                judge_result.probability,
                _RULE_TRUE_VETO_P,
            )
            return
        if not verdict and (judge_result is None or not judge_result.decision):
            # 多数票判否、但**确实见过手握杆**：不能直接丢掉。
            #
            # 离盘后的采样窗口很小（默认 3 点），真实取料的形态常常是
            # [空, 空, 有]——前两点落在杆还没出盘的那几帧，多数票判空手，
            # 而那一帧「有」是硬证据（盘外手持杆检出率 96~99%）。示例
            # 170216 第 1 件（12.53~13.8 基准）就是这样被丢掉的。
            #
            # 这与上面 ``_judge_pick`` 的「``hs_true > 0`` 算中间带」同一
            # 口径：只有**一帧正面证据都没有**时才按空手处理（那时才是真的
            # 空手离开，见 ``test_empty_hand_after_exit_still_does_not_pick``）。
            if not (visit.holding_seen and values["hs_true"] > 0.0):
                self._logger.debug(
                    "手 %s 离盘后 %d 个采样点里没有见到手里的杆，不判取料%s",
                    hand_track_id,
                    len(visit.holding_samples),
                    ""
                    if judge_result is None
                    else f"（逻辑回归 p={judge_result.probability:.3f}）",
                )
                return
            self._logger.debug(
                "手 %s 离盘后多数采样为空、但见过握持（%d/%d），按取料处理",
                hand_track_id,
                int(values["hs_true"]),
                int(values["hs_n"]),
            )
        screw = holding
        if screw is None:
            # 定论这一帧恰好没检出杆：用占位证据，置信度记 0 便于追溯。
            screw = _holding_placeholder(hand_box)
        self._confirm_pick(
            ts=ts,
            hand_track_id=hand_track_id,
            screw=screw,
            dwell_s=dwell,
            judge=judge_result,
        )

    # ------------------------------------------------------------------

def _holding_placeholder(hand_box: BBox) -> Detection:
    """离盘帧没有杆检测时，用握手位置附近的一个占位「握持证据」。

    只在「离盘前不久已确认过握持」时才用到；占位框贴着手框下缘（手握杆
    时杆就在那个方向），置信度记 0 便于在事件里区分「实检」与「沿用」。
    """

    cx = hand_box.center[0]
    cy = hand_box.center[1] + hand_box.height * 0.5
    return Detection(
        label="",
        confidence=0.0,
        box=BBox(cx - 1.0, cy - 1.0, cx + 1.0, cy + 1.0),
    )


