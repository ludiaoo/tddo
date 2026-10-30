"""跑全部 16 个算例的 ALNS，输出 results/alns_bench.csv。

用法：
    python -m scripts.run_alns            # 默认轮数：25 客户以内 300、50 客户 150、100 客户 60
    python -m scripts.run_alns --full     # 按论文参数 2000 轮（很慢）

轮数按规模递减是因为 Python 版的评估器比论文的实现慢，2000 轮 × 16 算例 × 10 种子
在本机要跑很久；先固定单种子跑通链路，多周期对比留给 M6。
"""

from __future__ import annotations

import argparse
import csv
import time

from src.alns import solve
from src.generator import build_instance
from src.io import PROJECT_ROOT, load_config, load_sets

OUT = PROJECT_ROOT / "results" / "alns_bench.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="使用论文的 2000 轮")
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config()
    rows = []
    t_all = time.time()

    for s in load_sets():
        inst = build_instance(s, cfg)
        if args.full:
            iters = 2000
        else:
            iters = 300 if inst.n_customers <= 25 else (150 if inst.n_customers <= 50 else 60)
        for seed in range(args.seeds):
            t0 = time.time()
            r = solve(inst, seed=seed, iters=iters)
            rows.append(
                {
                    "set": inst.set_id,
                    "problem": inst.base.name,
                    "city": inst.base.city,
                    "n_customers": inst.n_customers,
                    "n_trucks": inst.n_trucks,
                    "n_nfz": len(inst.nfz),
                    "initial_cost": round(r.initial_cost, 2),
                    "best_cost": round(r.best_cost, 2),
                    "saving_pct": round(r.saving * 100, 2),
                    "iters": iters,
                    "seed": seed,
                    "cpu_s": round(time.time() - t0, 2),
                }
            )
            print(
                f"Set{inst.set_id:2d} n={inst.n_customers:3d} 初始 {r.initial_cost:8.2f} "
                f"→ {r.best_cost:8.2f}  {r.saving:5.1%}  {rows[-1]['cpu_s']:6.1f}s"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写出 {OUT.relative_to(PROJECT_ROOT)}（{len(rows)} 行，{time.time()-t_all:.0f}s）")


if __name__ == "__main__":
    main()
