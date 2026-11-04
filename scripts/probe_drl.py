"""状态-动作优势分布探针（M9.4 判定实验，不训练任何模型）。

要回答的问题
------------
在**同一个搜索状态**下，换一对算子、跑 K 轮，成本能差多少？
把这个差值和「同一个动作换种子重跑的噪声」放在一起比：

- 若 优势 / 噪声 < 1 → 任何学习型控制器（DRL、bandit、LLM）都在拟合噪声，不可能回本；
- 若 优势大但**与状态无关**（各状态的算子排序高度一致）→ 那就不是学习问题，是「换一个固定算子」
  的问题，成本低两个数量级；
- 只有「优势大 + 排序随状态翻转」，才真正需要状态相关的策略。

做法
----
1. 用当前自适应策略跑一次，在若干迭代点快照 `(解, T)`；
2. 每个快照固定 16 组「破坏|修复」各跑 K 轮，**同一种子**（公共随机数，配对比较）；
3. 同一动作再换 3 个种子跑，量出「同一个动作自己会漂多少」；
4. 比较两者，并检查 16 个动作的排序在不同状态之间是否稳定。

用法
----
    python -m scripts.probe_drl --sets 5,7,9 --seeds 2 --marks 200,700,1300 --K 50
    python -m scripts.probe_drl --sets 13 --seeds 1 --marks 400,1000 --K 50 --append
    python -m scripts.probe_drl --analyze-only
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd

from src.alns import DESTROY, REPAIR, solve
from src.generator import build_instance
from src.io import PROJECT_ROOT, load_config, load_sets

OUT = PROJECT_ROOT / "results" / "ml" / "drl_probe.csv"

_CFG = None
_SETS = None
_MAIN_ITERS = 1500
NOISE_SEED_OFFSETS = (1, 2, 3)      # 同一动作的重复种子
NOISE_PAIRS = (0, 5, 10, 15)        # 抽 4 个动作量噪声（覆盖 4 个修复算子）


def _all_pairs() -> list[str]:
    return [f"{d}|{r}" for d in DESTROY for r in REPAIR]


def collect_states(inst, seed: int, marks: tuple[int, ...], iters: int) -> dict:
    """跑一次常规 ALNS，在指定迭代点快照解与温度。"""
    got: dict[int, dict] = {}

    def tr(rec):
        if rec["it"] in marks:
            got[rec["it"]] = {
                "sol": copy.deepcopy(rec["sol"]),
                "cur": float(rec["cur_cost"]),
                "best": float(rec["best_cost"]),
                "T": float(rec["T"]),
            }

    solve(inst, seed=seed, iters=iters, trace=tr)
    return got


def rollout(inst, st: dict, pair: str, K: int, seed: int, alpha: float) -> float:
    r = solve(
        inst, seed=seed, iters=K,
        init_sol=copy.deepcopy(st["sol"]),
        t0=st["T"], alpha=alpha, fixed_pair=pair,
    )
    return float(r.best_cost)


def run(sets: list[int], seeds: int, marks: tuple[int, ...], K: int) -> pd.DataFrame:
    global _CFG, _SETS
    if _CFG is None:
        _CFG = load_config()
        _SETS = {s["id"]: s for s in load_sets()}
    pairs = _all_pairs()
    alpha = (1e-3) ** (1.0 / _MAIN_ITERS)
    rows = []
    t_start = time.time()

    for sid in sets:
        inst = build_instance(_SETS[sid], _CFG)
        for seed in range(seeds):
            sts = collect_states(inst, seed, marks, _MAIN_ITERS)
            for it, st in sorted(sts.items()):
                base = rollout(inst, st, pairs[0], K, seed, alpha)      # 只为占位计时
                t0 = time.time()
                costs = {}
                for j, pr in enumerate(pairs):
                    costs[pr] = base if j == 0 else rollout(inst, st, pr, K, seed, alpha)
                for pr, c in costs.items():
                    rows.append(dict(set=sid, seed=seed, it=it, pair=pr, kind="main",
                                     noise_seed=seed, cost=c,
                                     cur=st["cur"], best=st["best"]))
                for j in NOISE_PAIRS:
                    pr = pairs[j]
                    for off in NOISE_SEED_OFFSETS:
                        c = rollout(inst, st, pr, K, seed + off * 1000, alpha)
                        rows.append(dict(set=sid, seed=seed, it=it, pair=pr, kind="noise",
                                         noise_seed=seed + off * 1000, cost=c,
                                         cur=st["cur"], best=st["best"]))
                dt = time.time() - t0
                print(f"  set{sid} seed{seed} it{it}: 16+{len(NOISE_PAIRS) * 3} 次 rollout "
                      f"耗时 {dt:.1f}s（状态成本 {st['cur']:.2f}）", flush=True)
        print(f"[set{sid}] 完成，累计 {time.time() - t_start:.0f}s", flush=True)
    return pd.DataFrame(rows)


def analyze() -> None:
    df = pd.read_csv(OUT)
    main = df[df.kind == "main"]
    key = ["set", "seed", "it"]

    # 1) 每个状态：16 个动作的成本分布
    rows = []
    for k, g in main.groupby(key):
        c = g.set_index("pair").cost
        cur = float(g.cur.iloc[0])
        rows.append(dict(
            set=k[0], seed=k[1], it=k[2], cur=cur,
            best16=c.min(), worst16=c.max(), med=c.median(), mean=c.mean(),
            spread_pct=(c.mean() / c.min() - 1) * 100,       # 平均动作相对最优动作的代价
            med_pct=(c.median() / c.min() - 1) * 100,
            worst_pct=(c.max() / c.min() - 1) * 100,
            gain_vs_cur=(cur / c.min() - 1) * 100,           # 最优动作相对「不动作」的收益
        ))
    st = pd.DataFrame(rows).sort_values(key)

    # 2) 噪声：同一 (状态, 动作) 换种子的相对标准差
    noise = df[df.kind == "noise"].copy()
    nz = []
    for k, g in noise.groupby(key + ["pair"]):
        if len(g) >= 3:
            nz.append(dict(zip(key + ["pair"], k), cv=g.cost.std(ddof=1) / g.cost.mean() * 100))
    nz = pd.DataFrame(nz)
    per_state_noise = nz.groupby(key).cv.mean().rename("noise_pct").reset_index()
    st = st.merge(per_state_noise, on=key, how="left")
    st["snr"] = st.spread_pct / st.noise_pct

    pd.set_option("display.width", 200)
    print("\n=== 每个状态：换算子 vs 换种子（单位 %）===")
    print(st[["set", "seed", "it", "cur", "best16", "spread_pct", "worst_pct",
              "noise_pct", "snr"]].round(2).to_string(index=False))

    print("\n=== 汇总 ===")
    print(f"状态数 {len(st)}；K 轮 rollout")
    print(f"平均动作相对最优动作： +{st.spread_pct.mean():.2f}%（中位 {st.spread_pct.median():.2f}%）")
    print(f"最差动作相对最优动作： +{st.worst_pct.mean():.2f}%")
    print(f"最优动作相对不动作：   +{st.gain_vs_cur.mean():.2f}%")
    print(f"同一动作换种子的噪声： {st.noise_pct.mean():.2f}%")
    print(f"信噪比 SNR：           {st.snr.mean():.2f}")

    # 3) 排序稳定性：动作排序是否随状态翻转
    piv = main.pivot_table(index=key, columns="pair", values="cost")
    ranks = piv.rank(axis=1, method="average")
    print(f"\n各动作排名跨状态标准差（均值，16 个动作平均）：{ranks.std().mean():.2f} 名")
    st_by_size = st.groupby("set")[["spread_pct", "noise_pct", "snr"]].mean().round(2)
    print("\n按算例：")
    print(st_by_size.to_string())

    # 4) 静态策略：永远用「平均排名最好」的那一个算子对
    best_static = ranks.mean().idxmin()
    reg_static = (piv[best_static] / piv.min(axis=1) - 1) * 100
    print(f"\n静态最优算子对 `{best_static}`：相对「每状态最优」的后悔 +{reg_static.mean():.2f}%"
          f"（中位 {reg_static.median():.2f}%，最好 {reg_static.min():.2f}%，最差 {reg_static.max():.2f}%）")

    # 5) 赢家分布
    win = piv.idxmin(axis=1).value_counts()
    print(f"\n每个状态的最优动作分布（共 {len(piv)} 个状态）：")
    print(win.to_string())

    # 6) 与「固定算子」的既有实测对照
    print("\n注：既有实测（bench_fixedpair）把 `worst|equal` 固定后配对 +17.18%，"
          "与上表「静态最优算子」的后悔方向一致。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="5,7,9")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--marks", default="200,700,1300")
    ap.add_argument("--K", type=int, default=50, help="每次 rollout 的迭代数")
    ap.add_argument("--main-iters", type=int, default=1500, help="快照所用主运行的迭代数")
    ap.add_argument("--append", action="store_true", help="追加到已有 CSV")
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    global _MAIN_ITERS
    _MAIN_ITERS = args.main_iters

    if not args.analyze_only:
        marks = tuple(int(x) for x in args.marks.split(","))
        df = run([int(x) for x in args.sets.split(",")], args.seeds, marks, args.K)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        if args.append and OUT.exists():
            old = pd.read_csv(OUT)
            df = pd.concat([old, df], ignore_index=True).drop_duplicates(
                subset=["set", "seed", "it", "pair", "kind", "noise_seed"])
        df.to_csv(OUT, index=False)
        print(f"\n写出 {OUT}（{len(df)} 行）")
    analyze()


if __name__ == "__main__":
    main()
