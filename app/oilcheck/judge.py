"""轻量逻辑回归：取料/涂油判定的二级复核。

一线判定仍然是 ``pipeline.py`` 里的规则（几何门限 + 连续帧数），它们
可解释、可逐条追到现场实测。但规则是「任一条件不满足就否掉」，几条
边界情形（示例⑧ 手按夹具、示例⑨ 盘边划过）只能靠越收越紧的门限去
排除，代价是真实取料也开始走钢丝。

这一层的作用是**在规则给不出把握的那一小段里补一票**：把规则已经算出
来的那几维证据（落手面积、杆/手位移比、握持帧数、停留时长……）喂给一个
逻辑回归，输出「这是一次真取料」的概率。规则明确成立（多数握持采样为真、
杆心已出盘）时不经过它；规则明确否定（一帧握持证据都没有）时也不经过
它。只有处在中间带时才让它参与。

权重由 ``tools/train_judge.py`` 用示例视频 + ``tools/detect_operations.py``
的独立清点结果训练，存成 JSON（``config/judge_*.json``）。没有权重文件时
本层整体不生效，判定退化成纯规则——装上代码不会因为缺模型文件而改变行为。

设计约束：
- 只用 numpy，不引入 sklearn（现场离线机器装依赖麻烦）；
- 特征先标准化再乘权重，标准化参数随权重一起存；
- 概率要能解释：每次判定都记下特征值与最终概率，进复核页。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger("oilcheck.judge")

# 低于这个概率判「不是」，高于 ``HIGH`` 判「是」，中间随规则走。
# 取 0.35/0.65：逻辑回归在示例视频上的输出分布是双峰的（负样本 <0.2、
# 正样本 >0.85），中间带只留给规则拿不准的个例。
DEFAULT_LOW = 0.35
DEFAULT_HIGH = 0.65


@dataclass
class JudgeResult:
    """一次二级判定的结果。"""

    probability: float
    decision: bool
    used: bool
    features: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "probability": round(self.probability, 4),
            "decision": self.decision,
            "used": self.used,
            "features": {k: round(v, 4) for k, v in self.features.items()},
        }


class LogisticJudge:
    """读 JSON 权重的逻辑回归判别器。

    权重文件格式::

        {
          "version": 1,
          "features": ["dwell", "inhand_frames", ...],
          "mean": [...], "scale": [...],
          "weights": [...], "bias": -1.2,
          "low": 0.35, "high": 0.65
        }

    ``scale`` 里的 0 会被当作 1（常数列）。
    """

    def __init__(
        self,
        features: list[str],
        mean: list[float],
        scale: list[float],
        weights: list[float],
        bias: float,
        *,
        low: float = DEFAULT_LOW,
        high: float = DEFAULT_HIGH,
        source: str = "",
    ) -> None:
        self.features = list(features)
        self.mean = np.asarray(mean, dtype=float)
        self.scale = np.asarray(
            [s if abs(s) > 1e-12 else 1.0 for s in scale], dtype=float
        )
        self.weights = np.asarray(weights, dtype=float)
        self.bias = float(bias)
        self.low = float(low)
        self.high = float(high)
        self.source = source

    @classmethod
    def load(cls, path: Path | str) -> LogisticJudge | None:
        """从 JSON 读权重；文件不存在 / 格式不对时返回 None 并记日志。"""

        path = Path(path)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            features = [str(x) for x in data["features"]]
            judge = cls(
                features=features,
                mean=[float(x) for x in data["mean"]],
                scale=[float(x) for x in data["scale"]],
                weights=[float(x) for x in data["weights"]],
                bias=float(data["bias"]),
                low=float(data.get("low", DEFAULT_LOW)),
                high=float(data.get("high", DEFAULT_HIGH)),
                source=str(path),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("逻辑回归权重 %s 读取失败（%s），本次不启用", path, exc)
            return None
        if not (
            len(judge.mean) == len(features)
            and len(judge.scale) == len(features)
            and len(judge.weights) == len(features)
        ):
            logger.warning(
                "逻辑回归权重 %s 的维度不一致（特征 %d 维），本次不启用",
                path,
                len(features),
            )
            return None
        logger.info(
            "已加载逻辑回归复核：%s（%d 维特征，门限 %.2f/%.2f）",
            path.name,
            len(features),
            judge.low,
            judge.high,
        )
        return judge

    def probability(self, values: dict[str, float]) -> float:
        """给定特征字典，返回正类概率。缺的特征按均值（标准化后为 0）处理。"""

        raw = np.asarray(
            [float(values.get(name, self.mean[i])) for i, name in enumerate(self.features)],
            dtype=float,
        )
        z = (raw - self.mean) / self.scale
        logit = float(z @ self.weights + self.bias)
        # 数值稳定的 sigmoid
        if logit >= 0:
            return 1.0 / (1.0 + math.exp(-logit))
        exp = math.exp(logit)
        return exp / (1.0 + exp)

    def decide(
        self, values: dict[str, float], *, rule: bool | None
    ) -> JudgeResult:
        """结合规则结论与模型概率给出最终判定。

        ``rule`` 是规则层的结论：

        - ``True``：规则已明确判「是」，模型只用于记录（不改结论）；
        - ``False``：规则已明确判「否」，同理；
        - ``None``：规则拿不准，完全由模型决定。
        """

        probability = self.probability(values)
        if rule is None:
            decision = probability >= 0.5
            return JudgeResult(probability, decision, True, dict(values))
        return JudgeResult(probability, bool(rule), False, dict(values))


@dataclass
class Episode:
    """逻辑回归判定出来的一段动作，带起止时刻。"""

    start_ts: float
    end_ts: float
    peak: float
    frames: int

    @property
    def duration(self) -> float:
        return max(0.0, self.end_ts - self.start_ts)

    def to_dict(self) -> dict[str, object]:
        return {
            "start_ts": round(self.start_ts, 3),
            "end_ts": round(self.end_ts, 3),
            "duration_s": round(self.duration, 3),
            "peak_probability": round(self.peak, 3),
            "frames": self.frames,
        }


class EpisodeSegmenter:
    """把逐帧概率切成一段段动作，给出开始与结束时刻。

    规则判定给的是「一次动作正好落在这一帧」，而现场关心的是**这件事
    从头到尾发生在什么时间**：取料从手进盘到杆出盘，涂油从刷头第一次
    贴上到真正离开。两者都不该由结算那一帧反推（取料结算要等采样窗口
    攒满，示例⑤ 会晚 2 秒以上）。

    做法是带滞回的双门限分段：

    - 概率 ≥ ``on`` 进入动作，起止帧往前回退到概率首次越过 ``low``
      的地方（动作起点比「攒够证据」早）；
    - 概率 ≤ ``off`` 退出动作；
    - 中间的空档不足 ``bridge_s`` 就并进同一段（刷手抬一下仍是同一次）。

    这样输出的 ``start_ts`` / ``end_ts`` 是动作本身的边界，而不是
    「证据够了」的时刻。
    """

    def __init__(
        self,
        *,
        on: float = 0.60,
        off: float = 0.35,
        low: float = 0.20,
        bridge_s: float = 0.8,
        min_frames: int = 2,
    ) -> None:
        self.on = on
        self.off = off
        self.low = low
        self.bridge_s = bridge_s
        self.min_frames = min_frames
        self._active: _ActiveEpisode | None = None
        self._closed: list[Episode] = []

    def update(self, ts: float, probability: float, *, dt: float = 0.0) -> None:
        active = self._active
        if active is None:
            if probability >= self.low:
                self._active = _ActiveEpisode(
                    start_ts=ts, last_on_ts=ts, peak=probability, frames=1
                )
            return
        active.frames += 1
        active.peak = max(active.peak, probability)
        if probability >= self.low:
            active.last_on_ts = ts
        if probability >= self.on:
            active.confirmed = True
        if probability < self.off and ts - active.last_on_ts > self.bridge_s:
            # 空档超过 bridge_s 才收段；不够就并进同一段（刷手抬一下
            # 再放回来仍是同一次）。
            self._close_active()

    def drain(self) -> list[Episode]:
        """取出已经收掉的段。"""

        ready = self._closed
        self._closed = []
        return ready

    def close(self) -> None:
        """视频结束 / 复位：把还挂着的段收掉。"""

        self._close_active()

    def take_active(self) -> Episode | None:
        """就地封口并返回进行中的那一段（没有则 None）。

        判定常常早于动作结束（涂油攒够时长时人还在继续刷），事件那一刻要
        的就是**这时已经攒到的起点**。封口之后这一段不再参与后续帧，
        否则继续刷会把它一直往后拖，下一次判定拿到错误的区间。
        """

        active = self._active
        self._active = None
        if active is None:
            return None
        if active.frames < 1:
            return None
        return Episode(
            start_ts=active.start_ts,
            end_ts=active.last_on_ts,
            peak=active.peak,
            frames=active.frames,
        )

    def _close_active(self) -> None:
        active = self._active
        self._active = None
        if active is None:
            return
        if not active.confirmed or active.frames < self.min_frames:
            return
        self._closed.append(
            Episode(
                start_ts=active.start_ts,
                end_ts=active.last_on_ts,
                peak=active.peak,
                frames=active.frames,
            )
        )


@dataclass
class _ActiveEpisode:
    start_ts: float
    last_on_ts: float
    peak: float
    frames: int
    confirmed: bool = False



def train_logistic(
    samples: list[dict[str, float]],
    labels: list[int],
    *,
    sample_weight: list[float] | None = None,
    iterations: int = 4000,
    lr: float = 0.1,
    l2: float = 1.0,
) -> tuple[list[str], list[float], list[float], list[float], float]:
    """纯 numpy 的逻辑回归训练（牛顿法 + L2），返回 (特征名, 均值, 尺度, 权重, 偏置)。

    样本量小（几十条到几百条）、特征维度低（十几维），用批量牛顿法（IRLS）
    几轮就收敛，比梯度下降稳定，也不需要 sklearn。

    ``l2`` 取 1.0 而不是 1e-3：小样本上几组特征完全可线性分开时，不加够
    正则权重会冲到几千——那样的模型换一段视频就崩。L2 把每个权重朝 0 拉，
    泛化更稳，代价是训练集准确率不再正好 100%（那本来也不是目标）。

    ``sample_weight`` 用于类别不平衡：逐帧密集采样后负样本远多于正样本，
    按类别总权重拉平（见 ``tools/train_judge.py`` 的 ``balanced_weights``）。
    """

    if not samples:
        raise ValueError("没有训练样本")
    features = sorted(samples[0].keys())
    x = np.asarray(
        [[float(s.get(name, 0.0)) for name in features] for s in samples], dtype=float
    )
    y = np.asarray(labels, dtype=float)
    if sample_weight is None:
        weight = np.ones(len(samples))
    else:
        weight = np.asarray(sample_weight, dtype=float)
        if weight.shape != y.shape:
            raise ValueError("sample_weight 长度与样本数不一致")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    z = (x - mean) / scale
    z = np.hstack([z, np.ones((z.shape[0], 1))])
    w = np.zeros(z.shape[1])
    reg = np.eye(z.shape[1]) * l2
    reg[-1, -1] = 0.0  # 偏置不正则
    zw = z * weight[:, None]
    for _ in range(iterations):
        logits = z @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        grad = zw.T @ (p - y) + reg @ w
        hessian = (z * (weight * p * (1 - p))[:, None]).T @ z + reg
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            step = lr * grad
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return (
        features,
        mean.tolist(),
        scale.tolist(),
        w[:-1].tolist(),
        float(w[-1]),
    )


def save_judge(
    path: Path | str,
    trained: tuple[list[str], list[float], list[float], list[float], float],
    *,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
    note: str = "",
) -> Path:
    """把训练结果写成判别器能读的 JSON。"""

    features, mean, scale, weights, bias = trained
    path = Path(path)
    payload = {
        "version": 1,
        "note": note,
        "features": features,
        "mean": [round(v, 6) for v in mean],
        "scale": [round(v, 6) for v in scale],
        "weights": [round(v, 6) for v in weights],
        "bias": round(bias, 6),
        "low": low,
        "high": high,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


class ForestJudge:
    """读 JSON 的随机森林判别器（树结构直接存成数组，推理不依赖 sklearn）。

    为什么用随机森林而不是逻辑回归：现场实测（9 段示例视频、30 个取料候选
    事件、31 个涂油候选事件，按视频分折）两者在取料上都能 9/9 对齐脚本，
    但涂油上随机森林 6/7、逻辑回归 5/7；更重要的是**决策余量**——随机森林
    在涂油上的最小余量 0.750，逻辑回归只有 0.184（示例④ 有一件差一点就
    翻盘）。余量大意味着换一段录像、光照或工人习惯略有不同时更不容易翻车。

    逻辑回归在边界上是线性的，而「杆是否真被带走」这些证据是分段的
    （要么完全没有、要么很明确），树模型天然的轴对齐切分正好合适。

    树结构存成扁平的数组（``children_left`` / ``feature`` / ``threshold`` /
    ``value``），推理只是一次数组遍历，不引入 sklearn 依赖。

    权重文件格式::

        {
          "version": 1,
          "kind": "forest",
          "features": [...],
          "trees": [
            {"left": [...], "right": [...], "feature": [...],
             "threshold": [...], "value": [...]},   # value 是正类概率
            ...
          ],
          "low": 0.35, "high": 0.65
        }
    """

    def __init__(
        self,
        features: list[str],
        trees: list[dict[str, list]],
        *,
        low: float = DEFAULT_LOW,
        high: float = DEFAULT_HIGH,
        source: str = "",
    ) -> None:
        self.features = list(features)
        self.trees = trees
        self.low = float(low)
        self.high = float(high)
        self.source = source
        self._index = {name: i for i, name in enumerate(self.features)}

    @classmethod
    def load(cls, path: Path | str) -> ForestJudge | None:
        path = Path(path)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("kind") != "forest":
                return None
            judge = cls(
                features=[str(x) for x in data["features"]],
                trees=data["trees"],
                low=float(data.get("low", DEFAULT_LOW)),
                high=float(data.get("high", DEFAULT_HIGH)),
                source=str(path),
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning("随机森林权重 %s 读取失败（%s），本次不启用", path, exc)
            return None
        if not judge.trees:
            return None
        logger.info(
            "已加载随机森林复核：%s（%d 棵树，门限 %.2f/%.2f）",
            path.name, len(judge.trees), judge.low, judge.high,
        )
        return judge

    def probability(self, values: dict[str, float]) -> float:
        """所有树的正类概率取平均。"""

        row = [float(values.get(name, 0.0)) for name in self.features]
        total = 0.0
        for tree in self.trees:
            total += self._tree_probability(tree, row)
        return total / len(self.trees)

    @staticmethod
    def _tree_probability(tree: dict[str, list], row: list[float]) -> float:
        node = 0
        left = tree["left"]
        right = tree["right"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        value = tree["value"]
        while True:
            index = feature[node]
            if index < 0:            # 叶子
                return float(value[node])
            if row[index] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]

    def decide(
        self, values: dict[str, float], *, rule: bool | None
    ) -> JudgeResult:
        probability = self.probability(values)
        if rule is None:
            return JudgeResult(probability, probability >= 0.5, True, dict(values))
        return JudgeResult(probability, bool(rule), False, dict(values))


def save_forest(
    path: Path | str,
    features: list[str],
    trees: list[dict[str, list]],
    *,
    low: float = DEFAULT_LOW,
    high: float = DEFAULT_HIGH,
    note: str = "",
) -> Path:
    """把随机森林写成 ``ForestJudge`` 能读的 JSON。"""

    path = Path(path)
    payload = {
        "version": 1,
        "kind": "forest",
        "note": note,
        "features": features,
        "trees": trees,
        "low": low,
        "high": high,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def export_forest(estimator, features: list[str]) -> list[dict[str, list]]:
    """把 sklearn 的 RandomForestClassifier 导出成扁平数组。"""

    trees: list[dict[str, list]] = []
    for est in estimator.estimators_:
        tree = est.tree_
        values = tree.value[:, 0, :]
        totals = values.sum(axis=1)
        positive = [
            float(v[1] / t) if t > 0 else 0.0 for v, t in zip(values, totals)
        ]
        trees.append({
            "left": [int(x) for x in tree.children_left],
            "right": [int(x) for x in tree.children_right],
            "feature": [int(x) for x in tree.feature],
            "threshold": [round(float(x), 6) for x in tree.threshold],
            "value": [round(v, 6) for v in positive],
        })
    return trees
