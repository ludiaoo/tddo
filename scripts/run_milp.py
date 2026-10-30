"""10 客户算例：MILP 精确解 vs ALNS（论文 Table 3 的 Gurobi 列）。

用法：
    python -m scripts.run_milp                 # 默认 300 s / 算例、3 个种子
    python -m scripts.run_milp --tl 600 --seeds 5

⚠ **两边的算例必须一致**：MILP 不建模禁飞区，所以 ALNS 也要用 `use_nfz=False`。
拿「有禁飞区的 ALNS」去和「无禁飞区的 MILP」比，会把建模差异当成算法差距。

输出 `results/milp_check.csv`。
"""

from __future__ import annotations

import argparse
import csv
import time

from src.alns import disable_nfz, solve
from src.evaluator import evaluate
from src.generator import build_instance
from src.io import PROJECT_ROOT, load_config, load_sets
from src.milp import solve_milp

OUT = PROJECT_ROOT / "results" / "milp_check.csv"
SMALL = [1, 2, 3, 4]          # 论文里 Gurobi 只解了这四个 10 客户算例


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl", type=float, default=300.0, help="MILP 时间上限（秒/算例）")
    ap.add_argument("--seeds", type=int, default=3, help="ALNS 独立运行次数，取最优")
    ap.add_argument("--iters", type=int, default=400, help="ALNS 迭代轮数")
    args = ap.parse_args()

    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}
    rows = []

    print(f"{'Set':>3} {'n':>3} {'K':>2} | {'MILP':>9} {'gap':>6} {'CPU':>6} | "
          f"{'ALNS':>9} {'CPU':>6} {'vs MILP':>8}")
    print("-" * 66)

    for sid in SMALL:
        inst = disable_nfz(build_instance(by_id[sid], cfg))

        t0 = time.time()
        best, best_seed = float("inf"), -1
        for seed in range(args.seeds):
            r = solve(inst, seed=seed, iters=args.iters)
            if r.best_cost < best:
                best, best_seed = r.best_cost, seed
        alns_cpu = time.time() - t0

        m = solve_milp(inst, time_limit=args.tl)
        if m.solution is not None:
            res, c_check = evaluate(inst, m.solution)
            assert res.feasible, f"Set{sid} MILP 解在评估器里不可行：{res.all_violations()[:3]}"
            assert abs(c_check - m.cost) < 1e-4 * max(1.0, m.cost), (
                f"Set{sid} MILP 目标 {m.cost:.4f} 与评估器 {c_check:.4f} 不一致"
            )

        gap = (best - m.cost) / m.cost * 100.0 if m.cost == m.cost else float("nan")
        rows.append(
            {
                "set": sid,
                "n_customers": inst.n_customers,
                "n_trucks": inst.n_trucks,
                "milp_cost": round(m.cost, 4),
                "milp_bound": round(m.bound, 4),
                "milp_gap_pct": round(m.gap * 100, 4),
                "milp_cpu_s": round(m.cpu, 1),
                "milp_status": m.status,
                "alns_cost": round(best, 4),
                "alns_seed": best_seed,
                "alns_cpu_s": round(alns_cpu, 1),
                "alns_vs_milp_pct": round(gap, 3),
            }
        )
        print(
            f"{sid:>3} {inst.n_customers:>3} {inst.n_trucks:>2} | {m.cost:>9.3f} "
            f"{m.gap*100:>5.2f}% {m.cpu:>6.0f} | {best:>9.3f} {alns_cpu:>6.1f} {gap:>7.2f}%"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写出 {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
