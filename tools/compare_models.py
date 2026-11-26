"""多种机器学习算法对比：按视频分折，同一份候选帧样本。

目标不是「谁训练集准确率高」，而是**留出一整段视频时还能不能判对**——
现场换一段录像就崩的模型没有意义。所以统一用 leave-one-video-out。

用法::

    uv run python tools/compare_models.py                 # 取料 + 涂油
    uv run python tools/compare_models.py --kind pick
    uv run python tools/compare_models.py --json data/frames.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

warnings.filterwarnings("ignore")

PICK_FEATURES = [
    "dwell", "hs_n", "hs_true", "carry_frames", "seen", "inventory_n",
    "carry_inhand_frames", "carry_inhand_max", "carry_rod_move",
    "carry_hand_move", "carry_align", "rods_in_tray_delta",
]
OILING_FEATURES = [
    "cum_s", "frames", "min_distance", "parts", "held_ratio", "rod_move",
    "contact_ratio",
]


def load(path: Path, kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [r for r in rows if r.get("kind") == kind]
    feats = PICK_FEATURES if kind == "pick" else OILING_FEATURES
    x = np.asarray([[float(r.get(f, 0.0)) for f in feats] for r in rows], dtype=float)
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)
    groups = np.asarray([int(r.get("video", 0)) for r in rows], dtype=int)
    return x, y, groups


def build_models() -> dict:
    """返回 name -> factory。每个 factory 每次新建一个未训练的模型。"""

    from sklearn.ensemble import (
        AdaBoostClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    models = {
        "LR(L2)": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)
        ),
        "LR(L1)": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, penalty="l1", solver="liblinear", max_iter=5000),
        ),
        "SVM-RBF": lambda: make_pipeline(
            StandardScaler(), SVC(C=2.0, gamma="scale", probability=True, random_state=0)
        ),
        "SVM-linear": lambda: make_pipeline(
            StandardScaler(), SVC(kernel="linear", C=1.0, probability=True, random_state=0)
        ),
        "决策树": lambda: DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=3, class_weight="balanced", random_state=0
        ),
        "随机森林": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=0, n_jobs=1,
        ),
        "ExtraTrees": lambda: ExtraTreesClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=2,
            class_weight="balanced", random_state=0, n_jobs=1,
        ),
        "GBDT": lambda: GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=3, random_state=0
        ),
        "AdaBoost": lambda: AdaBoostClassifier(
            n_estimators=120, learning_rate=0.5, random_state=0
        ),
        "MLP": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=3000, random_state=0),
        ),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = lambda: XGBClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=3,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
            scale_pos_weight=1.0, eval_metric="logloss",
            random_state=0, n_jobs=1, verbosity=0,
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        models["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=200, learning_rate=0.08, max_depth=4,
            num_leaves=15, random_state=0, n_jobs=1, verbose=-1,
        )
    except ImportError:
        pass
    return models


def leave_one_video_out(factory, x, y, groups) -> tuple[float, int, int, int, int]:
    """留出一段视频做测试，其余训练。返回 (acc, tp, fp, tn, fn) 合并结果。"""

    tp = fp = tn = fn = 0
    for video in sorted(set(groups.tolist())):
        train = groups != video
        test = groups == video
        if not test.any():
            continue
        y_train = y[train]
        if y_train.sum() < 2 or len(y_train) - y_train.sum() < 2:
            continue
        model = factory()
        try:
            model.fit(x[train], y_train)
            pred = model.predict(x[test])
        except Exception:  # noqa: BLE001 - 单个算法失败不该中断整轮对比
            continue
        for p, t in zip(pred, y[test]):
            if p and t:
                tp += 1
            elif p and not t:
                fp += 1
            elif not p and not t:
                tn += 1
            else:
                fn += 1
    total = tp + fp + tn + fn
    return (tp + tn) / total if total else float("nan"), tp, fp, tn, fn


def main() -> int:
    parser = argparse.ArgumentParser(description="按视频分折对比多种算法")
    parser.add_argument("--json", default="data/judge_all_frames.jsonl",
                        help="候选帧样本（默认全量候选帧）")
    parser.add_argument("--kind", choices=("pick", "oiling", "both"), default="both")
    args = parser.parse_args()

    path = Path(args.json)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.is_file():
        print(f"找不到样本文件: {path}")
        return 1

    kinds = ["pick", "oiling"] if args.kind == "both" else [args.kind]
    models = build_models()
    for kind in kinds:
        x, y, groups = load(path, kind)
        positives = int(y.sum())
        print(f"\n=== {kind}: {len(y)} 条（正 {positives} / 负 {len(y) - positives}），"
              f"{len(set(groups.tolist()))} 段视频")
        results = []
        for name, factory in models.items():
            acc, tp, fp, tn, fn = leave_one_video_out(factory, x, y, groups)
            results.append((acc, name, tp, fp, tn, fn))
        results.sort(key=lambda item: (-item[0] if item[0] == item[0] else 1))
        for acc, name, tp, fp, tn, fn in results:
            print(f"  {name:<12} acc={acc:.3f}  TP{tp:>4} FP{fp:>4} TN{tn:>4} FN{fn:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
