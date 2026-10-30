"""跑全部 16 个算例，输出 Table 3 口径的三列对照（不含 Gurobi / PSO）。

    python -m scripts.run_alns                 # 默认：按规模递减轮数，单种子
    python -m scripts.run_alns --seeds 3       # 多种子取最优（论文用 10）
    python -m scripts.run_alns --full          # 论文的 2000 轮（很慢）

三列的语义（与论文 Table 3 一致）：
- **ALNS**：完整模型。Saving% = (Initial − Best)/Initial
- **Truck-only**：不派无人机（`use_drone=False`，车队规模不变）
- **Without-detour**：禁飞区照查，但**不许绕行**（`use_detour=False`）：穿圆的航段直接不可飞
后两列的 Increasing% = (该列成本 − ALNS 成本)/ALNS 成本

轮数按规模递减：Python 版的评估是顺序模拟，100 客户单轮约 0.5 s，
2000 轮 × 16 算例 × 10 种子在本机要跑十几小时，先固定「能跑完」的预算。
"""

from __future__ import annotations

import argparse
import csv
import time

from src.alns import solve
from src.generator import build_instance
from src.io import PROJECT_ROOT, load_config, load_sets

OUT = PROJECT_ROOT / "results" / "table3_like.csv"


def _iters(n: int) -> int:
    return 300 if n <= 25 else (150 if n <= 50 else 60)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="使用论文的 2000 轮")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--skip-extra", action="store_true", help="只跑 ALNS 主列")
    args = ap.parse_args()

    cfg = load_config()
    rows = []
    t_all = time.time()

    print(
        f"{'Set':>3} {'n':>4} {'K':>3} | {'ALNS':>9} {'Init':>9} {'Save':>6} {'CPU':>6} | "
        f"{'TruckOnly':>9} {'+%':>6} | {'NoDetour':>9} {'+%':>6}"
    )
    print("-" * 92)

    for s in load_sets():
        inst = build_instance(s, cfg)
        iters = 2000 if args.full else _iters(inst.n_customers)

        def best_of(**kw) -> tuple[float, float]:
            best, cpu = float("inf"), 0.0
            for seed in range(args.seeds):
                t0 = time.time()
                r = solve(inst, seed=seed, iters=iters, **kw)
                cpu += time.time() - t0
                best = min(best, r.best_cost)
            return best, cpu

        t0 = time.time()
        r0 = solve(inst, seed=0, iters=iters)
        cpu0 = time.time() - t0
        best, _ = (best_of() if args.seeds > 1 else (r0.best_cost, cpu0))

        row = {
            "set": inst.set_id,
            "problem": inst.base.name,
            "city": inst.base.city,
            "n_customers": inst.n_customers,
            "n_trucks": inst.n_trucks,
            "n_nfz": len(inst.nfz),
            "initial_cost": round(r0.initial_cost, 2),
            "alns_cost": round(best, 2),
            "saving_pct": round((r0.initial_cost - best) / r0.initial_cost * 100, 2),
            "alns_cpu_s": round(cpu0, 1),
            "iters": iters,
        }

        if args.skip_extra:
            rows.append(row)
            print(f"{inst.set_id:>3} {inst.n_customers:>4} {inst.n_trucks:>3} | "
                  f"{best:>9.2f} {r0.initial_cost:>9.2f} {row['saving_pct']:>5.1f}% {cpu0:>5.1f}s")
            continue

        truck, cpu_t = best_of(use_drone=False)
        nodet, cpu_n = best_of(use_detour=False)
        row.update(
            truckonly_cost=round(truck, 2),
            truckonly_increase_pct=round((truck - best) / best * 100, 2),
            truckonly_cpu_s=round(cpu_t, 1),
            nodetour_cost=round(nodet, 2),
            nodetour_increase_pct=round((nodet - best) / best * 100, 2),
            nodetour_cpu_s=round(cpu_n, 1),
        )
        rows.append(row)
        print(
            f"{inst.set_id:>3} {inst.n_customers:>4} {inst.n_trucks:>3} | {best:>9.2f} "
            f"{r0.initial_cost:>9.2f} {row['saving_pct']:>5.1f}% {cpu0:>5.1f}s | "
            f"{truck:>9.2f} {row['truckonly_increase_pct']:>5.1f}% | "
            f"{nodet:>9.2f} {row['nodetour_increase_pct']:>5.1f}%"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写出 {OUT.relative_to(PROJECT_ROOT)}（{len(rows)} 行，{time.time()-t_all:.0f}s）")


if __name__ == "__main__":
    main()
