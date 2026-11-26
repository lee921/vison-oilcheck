"""采集原始检测框特征并训练取料/涂油二级判定模型。

用法::

    uv run python tools/train_judge.py --collect     # 只采集，写出 JSONL 供检查
    uv run python tools/train_judge.py               # 采集 + 训练 + 写出权重

特征来自 ``app/oilcheck/raw_features.py``：**只用 YOLO 检测框 + 标定区域**，
不读 pipeline 已经算好的规则结论（``seen`` / ``hs_true`` / carry 累积帧数）。
采集逻辑与 ``tools/eval_raw_features.py`` 共用同一份实现，保证「评测用
什么特征，上线就用什么特征」。

标签来自 ``tools/detect_operations.py`` 的独立清点结果
（``data/operations_preview/*.json``）：它不建作业周期，两个工序各判各的，
是这套判定的「人工核对基准」。

产物：

- ``data/raw_events.jsonl``：每个候选事件的特征与标签（排查用）；
- ``config/judge_pick.json`` / ``config/judge_oiling.json``：权重，监控进程
  启动时自动加载（见 ``app/oilcheck/judge.py``）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import numpy as np  # noqa: E402

from app.oilcheck.raw_features import (  # noqa: E402
    OILING_FEATURE_NAMES,
    PICK_FEATURE_NAMES,
)

OILING_SELECTED = [
    "cum_s", "frames", "samples", "held_ratio", "rod_moved_ratio",
    "rod_parked_ratio", "brush_hand_dist_min", "rod_in_hand_max",
    "rod_moved_max", "brush_found_ratio", "contact_ratio",
]
PICK_SELECTED = [
    "rod_moved_max", "rod_iou_max", "dwell", "samples", "hand_w",
    "tray_rod_n_last", "found_ratio", "hand_below_tray_last", "tray_rod_delta",
]

FEATURES = {
    "pick": [f for f in PICK_SELECTED if f in PICK_FEATURE_NAMES],
    "oiling": [f for f in OILING_SELECTED if f in OILING_FEATURE_NAMES],
}


def balanced_weights(labels: list[int]) -> list[float]:
    """两类样本总数相同的权重（涂油 15 正 / 122 负，不配平会全判否）。"""

    total = len(labels)
    positives = sum(labels)
    negatives = total - positives
    if positives == 0 or negatives == 0:
        return [1.0] * total
    w_pos = total / (2.0 * positives)
    w_neg = total / (2.0 * negatives)
    return [w_pos if label else w_neg for label in labels]


def _vectors(rows: list[dict], features: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(row.get(f, 0.0)) for f in features] for row in rows], dtype=float
    )


def grouped_holdout(rows: list[dict], features: list[str], *, algorithm: str) -> None:
    """按视频分折：留出整段视频预测，才是可信的准确率。"""

    from compare_models import build_models  # noqa: PLC0415

    factory = build_models().get(algorithm)
    if factory is None:
        return
    x = _vectors(rows, features)
    y = np.asarray([int(r["label"]) for r in rows])
    groups = np.asarray([int(r["video"]) for r in rows])
    print(f"  {algorithm} 按视频分折（{len(set(groups.tolist()))} 折）：")
    tp = fp = tn = fn = exact = 0
    for video in sorted(set(groups.tolist())):
        train, test = groups != video, groups == video
        if y[train].sum() < 2 or len(y[train]) - y[train].sum() < 2:
            print(f"    v{video}: 训练折单类，跳过")
            continue
        model = factory()
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        t = y[test]
        tp += int(((pred == 1) & (t == 1)).sum())
        fp += int(((pred == 1) & (t == 0)).sum())
        tn += int(((pred == 0) & (t == 0)).sum())
        fn += int(((pred == 0) & (t == 1)).sum())
        ok = int(pred.sum()) == int(t.sum())
        exact += ok
        print(
            f"    v{video}: 测试 {len(t)} 条 acc={float((pred == t).mean()):.3f} "
            f"判出 {int(pred.sum())} 真 {int(t.sum())} {'✓' if ok else '✗'}"
        )
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else 0.0
    print(
        f"    ⇒ 合并 acc={accuracy:.3f} TP{tp} FP{fp} TN{tn} FN{fn}；"
        f"计数完全一致 {exact}/{len(set(groups.tolist()))}"
    )


def train_forest(rows: list[dict], kind: str, out_dir: Path) -> Path | None:
    """训练 ExtraTrees 并导出成纯 JSON 树数组（``judge.ForestJudge``）。

    为什么是 ExtraTrees：按视频分折、事件级评测下，纯原始特征（无规则
    结论、无绝对位置）上它取料 9/9 全对且最小判别余量 0.588，是唯一一个
    既能全对、余量又足够大的；随机森林差一件，LR/SVM 有余量 0.13 的
    危险样本。
    """

    from sklearn.ensemble import ExtraTreesClassifier

    from app.oilcheck.judge import export_forest, save_forest  # noqa: PLC0415

    features = FEATURES[kind]
    x = _vectors(rows, features)
    y = np.asarray([int(r["label"]) for r in rows])
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    if positives < 2 or negatives < 2:
        print(f"{kind}: 正 {positives} / 负 {negatives} 样本不足（各需 ≥2），不训练")
        return None

    model = ExtraTreesClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=2,
        class_weight="balanced", random_state=0, n_jobs=1,
    ).fit(x, y)
    pred = model.predict(x)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    accuracy = (tp + tn) / max(1, len(y))
    print(f"  {kind} ExtraTrees 训练集 acc={accuracy:.3f} TP{tp} FP{fp} TN{tn} FN{fn}")

    return save_forest(
        out_dir / f"judge_{kind}.json",
        features,
        export_forest(model, features),
        note=(
            f"tools/train_judge.py 生成（ExtraTrees 300 棵 / 深度 6，"
            f"纯原始检测框特征）：{positives} 正 / {negatives} 负，"
            f"训练集 acc={accuracy:.3f} TP{tp} FP{fp} TN{tn} FN{fn}"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="采集原始特征并训练二级判定模型")
    parser.add_argument("--collect", action="store_true", help="只采集，不训练")
    parser.add_argument("--limit", type=int, default=0, help="每段视频只跑前 N 帧")
    parser.add_argument("--out-dir", default="config", help="权重输出目录")
    parser.add_argument(
        "--algorithm", default="ExtraTrees",
        help="按视频分折校验用的算法名（见 tools/compare_models.py）",
    )
    args = parser.parse_args()

    from eval_raw_features import collect  # noqa: PLC0415

    out_dir = Path(args.out_dir)
    # compare_models 里 "ExtraTrees" 的工厂；校验用同族配置
    report = {"pick": [], "oiling": []}
    for kind in ("pick", "oiling"):
        rows = collect(kind, args.limit)
        report[kind] = rows
        positives = sum(int(r["label"]) for r in rows)
        print(f"{kind}: {len(rows)} 个候选事件（正 {positives}），"
              f"{len(FEATURES[kind])} 维原始特征")

    sample_path = Path(__file__).resolve().parent.parent / "data" / "raw_events.jsonl"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8") as fp:
        for rows in report.values():
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"样本明细: {sample_path}")
    if args.collect:
        return 0

    for kind, rows in report.items():
        usable = [r for r in rows if r["label"] in (0, 1)]
        grouped_holdout(usable, FEATURES[kind], algorithm=args.algorithm)
        target = train_forest(usable, kind, out_dir)
        if target is not None:
            print(f"{kind} → {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
