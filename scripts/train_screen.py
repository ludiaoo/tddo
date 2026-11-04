"""训练候选预筛模型（M9.1）：可行性 + Δcost 两个 LightGBM。

    python -m scripts.train_screen --in results/ml/candidates_v1.csv.gz --test-sets 7,13

**防泄漏**：训练/测试按**算例**划分（同一算例的候选高度相关，按行划分会虚高）。
测试集默认取 `7,13`（一个 25 客户、一个 100 客户，且都不在训练里）。

输出 `models/screen_v1/`：`feas.txt`、`delta.txt`、`meta.json`（特征名、阈值、指标、训练配置）。
模型文件很小（< 1 MB），随仓库走；`--no-ml` / 模型缺失时求解器自动退回全量评估。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.io import PROJECT_ROOT
from src.ml.features import FEATURE_NAMES

FEAS_THRESHOLD = 0.5


def _load(paths: list[Path]) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=[c for c in FEATURE_NAMES if c in df.columns])
    return df


def _hit_rate_at_m(df: pd.DataFrame, m: int, score_fn, rng) -> float:
    """组 = (set, seed, step)；组内 ≥ m 个候选且至少一个可行才算一组。

    score_fn(X) -> 分数（越大越好）；返回「真最优可行候选被选进前 m」的比例。
    """
    hits, total = 0, 0
    for _, g in df.groupby(["**set", "**seed", "**step"], sort=False):
        if len(g) < m or g.feasible.sum() == 0:
            continue
        X = g[list(FEATURE_NAMES)].to_numpy(dtype=float)
        picked = set(score_fn(X, g))
        best = g.index[g.feasible == 1][np.argmin(g.loc[g.feasible == 1, "delta"].to_numpy())]
        total += 1
        hits += int(best in {g.index[i] for i in picked})
    return hits / max(total, 1), total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/ml/candidates_v1.csv.gz")
    ap.add_argument("--test-sets", default="7,13")
    ap.add_argument("--out", default="models/screen_v1")
    ap.add_argument("--m", type=int, default=5, help="hit-rate 的 m（与 ML_SCREEN_TOP_M 对齐）")
    ap.add_argument("--rounds", type=int, default=400)
    args = ap.parse_args()

    import lightgbm as lgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    inp = PROJECT_ROOT / args.inp
    df = _load([inp])
    test_sets = {int(x) for x in args.test_sets.split(",")}
    tr = df[~df["**set"].isin(test_sets)]
    te = df[df["**set"].isin(test_sets)]
    print(f"样本 {len(df)}（训练 {len(tr)} / 测试 {len(te)}），"
          f"可行率 train {tr.feasible.mean():.3f} / test {te.feasible.mean():.3f}")
    print(f"训练算例 {sorted(tr['**set'].unique())}，测试算例 {sorted(te['**set'].unique())}")

    Xtr = tr[list(FEATURE_NAMES)].to_numpy(dtype=float)
    Xte = te[list(FEATURE_NAMES)].to_numpy(dtype=float)
    ytr, yte = tr.feasible.to_numpy(), te.feasible.to_numpy()

    common = dict(n_estimators=args.rounds, learning_rate=0.05, num_leaves=31,
                  min_child_samples=100, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, n_jobs=10, verbose=-1, random_state=0)

    t0 = time.time()
    feas = lgb.LGBMClassifier(objective="binary", is_unbalance=True, **common)
    feas.fit(Xtr, ytr)
    p_te = feas.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p_te)
    ap = average_precision_score(yte, p_te)
    print(f"\n[可行性] AUC={auc:.4f}  PR-AUC={ap:.4f}  （{time.time() - t0:.0f}s）")
    print("特征重要性 top8：")
    imp = sorted(zip(FEATURE_NAMES, feas.feature_importances_), key=lambda x: -x[1])[:8]
    for name, v in imp:
        print(f"  {name:20s} {v:8.0f}")

    ftr = tr[tr.feasible == 1]
    fte = te[te.feasible == 1]
    delta = lgb.LGBMRegressor(objective="regression_l1", **common)
    delta.fit(ftr[list(FEATURE_NAMES)].to_numpy(dtype=float),
              np.log1p(ftr.delta.to_numpy()))
    pred = np.expm1(delta.predict(fte[list(FEATURE_NAMES)].to_numpy(dtype=float)))
    mae = np.mean(np.abs(pred - fte.delta.to_numpy()))
    print(f"[Δcost] 测试 MAE={mae:.3f}（真实 Δcost 中位 {fte.delta.median():.3f}）")

    # ---- 线性可行性门（第一级，微秒级）：阈值按「可行样本召回 ≥ 99%」标定 ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    Ztr = scaler.transform(Xtr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    lr.fit(Ztr, ytr)
    p_lin_tr = lr.predict_proba(Ztr)[:, 1]
    thr = float(np.quantile(p_lin_tr[ytr == 1], 0.01))       # 可行样本 99% 召回
    Zte = scaler.transform(Xte)
    p_lin_te = lr.predict_proba(Zte)[:, 1]
    gate_keep = float((p_lin_te >= thr).mean())
    gate_recall = float((p_lin_te[yte == 1] >= thr).mean())
    print(f"[线性门] 阈值={thr:.4f}（按可行召回 99% 标定）→ "
          f"测试集通过率 {gate_keep:.1%}，可行样本召回 {gate_recall:.1%}")


    # ---- hit-rate@m：模型 vs 随机 vs 「直飞距离最短」启发式 ----
    rng = np.random.default_rng(0)

    def model_fn(X, g):
        p = feas.predict_proba(X)[:, 1]
        d = np.expm1(delta.predict(X))
        s = np.where(p >= FEAS_THRESHOLD, -d, 1e18)      # 不可行的大罚
        return np.argsort(s)[: args.m]

    def random_fn(X, g):
        return rng.choice(len(X), size=min(args.m, len(X)), replace=False)

    def heuristic_fn(X, g):
        return np.argsort(g.seq_flight_len.to_numpy())[: args.m]

    print("\n[hit-rate@%d]（测试算例，组内 ≥ %d 个候选且有可行解）" % (args.m, args.m))
    for name, fn in [("模型", model_fn), ("随机", random_fn), ("直飞距离启发式", heuristic_fn)]:
        hr, n = _hit_rate_at_m(te, args.m, fn, rng)
        print(f"  {name:10s} {hr:.3f}   （{n} 组）")

    out = PROJECT_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    # 用 model_to_string 落盘：LightGBM 的 C 层 fopen 打不开含中文的路径，文本由 Python 写绕开
    (out / "feas.txt").write_text(feas.booster_.model_to_string(), encoding="utf-8")
    (out / "delta.txt").write_text(delta.booster_.model_to_string(), encoding="utf-8")
    meta = {
        "feature_names": list(FEATURE_NAMES),
        "feas_threshold": FEAS_THRESHOLD,
        "trained_on": str(inp.name),
        "train_sets": sorted(int(x) for x in tr["**set"].unique()),
        "test_sets": sorted(int(x) for x in te["**set"].unique()),
        "n_rows": int(len(df)),
        "metrics": {"auc": round(float(auc), 4), "pr_auc": round(float(ap), 4),
                    "delta_mae": round(float(mae), 4)},
        "linear": {
            "w": [float(x) for x in lr.coef_[0]],
            "b": float(lr.intercept_[0]),
            "mean": [float(x) for x in scaler.mean_],
            "std": [float(x) for x in scaler.scale_],
            "threshold": thr,
        },
        "params": {k: v for k, v in common.items()},
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}\\{{feas.txt, delta.txt, meta.json}}")


if __name__ == "__main__":
    main()
