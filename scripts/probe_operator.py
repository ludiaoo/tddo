"""M9.3 探路实验：算子选择到底有没有「可学的空间」？

**问题**：ALNS 每轮在 16 个 (破坏, 修复) 组合里选一个。现在靠轮盘赌 + 手工四档打分。
如果换成「看状态选算子」的学习策略，天花板在哪？值不值得做？

**做法**（廉价、一天内出结论，先证伪再投入）：

1. 用**均匀随机**策略跑若干算例（`op_policy="uniform"`），每轮记录
   状态（迭代进度、当前/最优比、停滞计数、温度、解结构）+ 动作 + 结果；
   均匀策略保证 16 个动作的样本覆盖均衡，避免「只试过好算子」的选择偏差。
2. 三个诊断量：
   - **动作本身的差异**：16 个动作的改进率 spread 有多大（如果都差不多，先说明动作不重要）；
   - **状态 × 动作的交互**：`action only` 模型 vs `action + state` 模型的样本外 AUC ——
     后者不赢，说明「看状态选算子」没有可学的信号；
   - **oracle 上界**：按 (规模 × 迭代阶段) 分桶，桶内最优动作的改进率 vs 全局最优动作，
     差距就是「完美策略」相对「固定最优算子」的收益上限。
3. 判据（写在 M9 方案里）：AUC 提升 < 0.02 且 oracle 增益 < 1.5 个百分点 → 不值得做，
   转向 M9.2 热启动。

用法：
    python -m scripts.probe_operator --sets 5,7,9,12,13,15 --seeds 4 --iters 1500
    python -m scripts.probe_operator --analyze-only
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.io import PROJECT_ROOT, load_config, load_sets

_CFG = None
_SETS = None
POLICY = "uniform"          # 由 main 设置：uniform（无偏覆盖）/ adaptive（当前线上策略）
OUT = PROJECT_ROOT / "results" / "ml" / "oplog.csv.gz"

STATE_FEATS = [
    "it_frac", "cur_over_best", "log_no_improve", "log_T_over_t0",
    "n_routes", "n_sorties", "drone_frac", "destroyed_frac",
]


def _init() -> None:
    global _CFG, _SETS
    _CFG = load_config()
    _SETS = {s["id"]: s for s in load_sets()}


def _run(task: tuple) -> list[dict]:
    from src.alns import solve
    from src.generator import build_instance

    sid, seed, iters, policy = task
    rows: list[dict] = []
    inst = build_instance(_SETS[sid], _CFG)
    t0 = time.time()

    def trace(r: dict) -> None:
        r["it_frac"] = r["it"] / iters
        r["log_no_improve"] = np.log1p(r["no_improve"])
        r["log_T_over_t0"] = np.log(max(r["T"], 1e-9) / 300.0)
        r["drone_frac"] = r["drone_served"] / max(inst.n_customers, 1)
        r["destroyed_frac"] = r["destroyed"] / max(inst.n_customers, 1)
        r["set"], r["seed"] = sid, seed
        rows.append(r)

    solve(inst, seed=seed, iters=iters, trace=trace, op_policy=policy)
    print(f"  Set{sid:2d} seed={seed} → {len(rows)} 轮，{time.time() - t0:.0f}s", flush=True)
    return rows


def collect(sets: list[int], seeds: int, iters: int) -> None:
    tasks = [(sid, seed, iters, POLICY) for sid in sets for seed in range(seeds)]
    print(f"探路采集：{len(tasks)} 次运行 × {iters} 轮（policy={POLICY}）")
    all_rows: list[dict] = []
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=min(len(tasks), 10), initializer=_init) as pool:
        futs = [pool.submit(_run, t) for t in tasks]
        for fut in as_completed(futs):
            all_rows.extend(fut.result())

    import pandas as pd

    df = pd.DataFrame(all_rows).drop(columns=["delta"], errors="ignore")
    df["policy"] = POLICY
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"\n共 {len(df)} 轮决策，耗时 {(time.time() - t_all) / 60:.1f} min → {OUT.name}")


def analyze() -> None:
    import lightgbm as lgb
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    df = pd.read_csv(OUT)
    df["success"] = df["status"].isin(["best", "improve"]).astype(int)
    print(f"样本 {len(df)}，运行组 {df.groupby(['set', 'seed']).ngroups}，整体改进率 {df.success.mean():.3f}")

    # ---------- 1. 动作本身有多大差别 ----------
    g = df.groupby("action").agg(n=("success", "size"), rate=("success", "mean"))
    g["ci95"] = 1.96 * np.sqrt(g.rate * (1 - g.rate) / g.n)
    g = g.sort_values("rate", ascending=False)
    print("\n[1] 16 个动作的改进率（均匀策略下的无偏估计）")
    print(g.round(3).to_string())
    spread = g.rate.max() - g.rate.min()
    print(f"    最高 {g.rate.max():.3f}（{g.index[0]}）最低 {g.rate.min():.3f}（{g.index[-1]}）"
          f"  极差 {spread * 100:.1f}pp")

    # ---------- 2. 状态 × 动作 有没有可学信号 ----------
    acts = sorted(df.action.unique())
    act_idx = {a: i for i, a in enumerate(acts)}
    A = np.zeros((len(df), len(acts)), dtype=np.float32)
    A[np.arange(len(df)), df.action.map(act_idx).to_numpy()] = 1.0
    S = df[STATE_FEATS].to_numpy(dtype=np.float32)
    y = df.success.to_numpy()
    groups = (df["set"].astype(str) + "_" + df["seed"].astype(str)).to_numpy()

    params = dict(objective="binary", n_estimators=300, learning_rate=0.05,
                  num_leaves=31, min_child_samples=200, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, n_jobs=10, verbose=-1, random_state=0)

    auc_a, auc_b = [], []
    for tr, te in GroupKFold(n_splits=4).split(A, y, groups):
        ma = lgb.LGBMClassifier(**params).fit(A[tr], y[tr])
        mb = lgb.LGBMClassifier(**params).fit(np.hstack([A, S])[tr], y[tr])
        auc_a.append(roc_auc_score(y[te], ma.predict_proba(A[te])[:, 1]))
        auc_b.append(roc_auc_score(y[te], mb.predict_proba(np.hstack([A, S])[te])[:, 1]))
    print("\n[2] 样本外 AUC（按运行组切分，4 折）")
    print(f"    action only          {np.mean(auc_a):.4f}")
    print(f"    action + state       {np.mean(auc_b):.4f}")
    print(f"    提升                 {(np.mean(auc_b) - np.mean(auc_a)) * 100:+.2f}pp")

    # ---------- 3. oracle 上界（分桶取最优动作） ----------
    df["stage"] = pd.qcut(df.it_frac, 4, labels=False, duplicates="drop")
    buckets = df.groupby(["n_customers", "stage"])
    rows = []
    for (n, st), b in buckets:
        if len(b) < 400:
            continue
        best = b.groupby("action").success.mean().max()
        rows.append({"n": n, "stage": st, "share": len(b) / len(df), "best_rate": best})
    tab = pd.DataFrame(rows)
    oracle = float((tab.share * tab.best_rate).sum() / tab.share.sum())
    best_fixed = float(g.rate.max())
    print(f"\n[3] 分桶最优（规模 × 迭代阶段，桶内 ≥400 样本）")
    print(f"    完美策略的改进率上界 {oracle:.3f}   全局最优固定动作 {best_fixed:.3f}"
          f"   上限增益 {(oracle - best_fixed) * 100:+.1f}pp")
    print(f"    参考：整体平均 {df.success.mean():.3f}（= 当前随机策略的量级）")

    print("\n[结论判据] AUC 提升 < 0.02 且上限增益 < 1.5pp → 学习型算子选择不值得做")
    print(f"    实测：AUC 提升 {(np.mean(auc_b) - np.mean(auc_a)):.3f}，"
          f"上限增益 {(oracle - best_fixed) * 100:.1f}pp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="5,7,9,12,13,15")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--policy", default="uniform", choices=["uniform", "adaptive"])
    ap.add_argument("--tag", default="", help="输出文件名后缀（区分 uniform / adaptive）")
    args = ap.parse_args()

    global POLICY, OUT
    POLICY = args.policy
    if args.tag:
        OUT = PROJECT_ROOT / "results" / "ml" / f"oplog_{args.tag}.csv.gz"

    if not args.analyze_only:
        collect([int(x) for x in args.sets.split(",")], args.seeds, args.iters)
    analyze()


if __name__ == "__main__":
    main()
