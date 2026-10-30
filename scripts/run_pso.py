"""PSO 对照列（Table 3 的 PSO 列），在 ALNS 结果之上补一列。

    python -m scripts.run_pso            # 默认 2 个种子取最优
    python -m scripts.run_pso --seeds 5

读 `results/table3_like.csv`（ALNS / Truck-only / Without-detour），补上 PSO 成本与
Increasing%，写出 `results/table3_with_pso.csv`。这样不需要把 ALNS 那三列重跑一遍。

⚠ 编码是本项目自造的：论文只给了 PSO 的参数（学习因子 2.05、惯性权重 0.9→0.4），
   **没给粒子编码与解码方式**。这里用 random-key（优先级列表）+ 与初始解相同的贪心插入解码，
   详见 `src/pso.py` 的文件头。因此这一列只能作为「同预算下的另一个元启发式」参考，
   不宜当作论文 PSO 的严格复刻。
"""

from __future__ import annotations

import argparse
import csv
import time

from src.generator import build_instance
from src.io import PROJECT_ROOT, load_config, load_sets
from src.pso import solve_pso

SRC = PROJECT_ROOT / "results" / "table3_like.csv"
OUT = PROJECT_ROOT / "results" / "table3_with_pso.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"缺 {SRC.name}，先跑 python -m scripts.run_alns")

    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}
    with open(SRC, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    print(f"{'Set':>3} {'n':>4} | {'ALNS':>9} {'PSO':>9} {'+%':>7} {'解码次数':>8} {'CPU':>6}")
    print("-" * 62)

    for row in rows:
        sid = int(row["set"])
        iters = int(row["iters"])
        inst = build_instance(by_id[sid], cfg)

        best, cpu = float("inf"), 0.0
        decodes = 0
        for seed in range(args.seeds):
            t0 = time.time()
            r = solve_pso(inst, seed=seed, iters=iters)
            cpu += time.time() - t0
            decodes = r.decodes
            best = min(best, r.best_cost)

        alns = float(row["alns_cost"])
        row["pso_cost"] = round(best, 2)
        row["pso_increase_pct"] = round((best - alns) / alns * 100, 2)
        row["pso_cpu_s"] = round(cpu, 1)
        row["pso_decodes"] = decodes
        print(
            f"{sid:>3} {inst.n_customers:>4} | {alns:>9.2f} {best:>9.2f} "
            f"{row['pso_increase_pct']:>6.1f}% {decodes:>8} {cpu:>5.1f}s"
        )

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写出 {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
