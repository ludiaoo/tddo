"""画路线图（论文 Fig.8 的同款内容）：卡车实线 + 无人机虚线。

    python -m scripts.plot_routes            # 四个代表算例（每档规模一个），2×2
    python -m scripts.plot_routes --all      # 16 个算例全画
    python -m scripts.plot_routes --refresh  # 忽略缓存，重新求解

解会缓存在 `results/solutions/setXX.json`。**求解口径与 Table 3 完全一致**：轮数取 2000，
种子取 `bench_main.csv` 里该算例达到最优的那个，所以图上标题的成本就是表里那一格。
改了 ALNS 代码后记得 `--refresh`。
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt

from src import style
from src.alns import solve
from src.evaluator import evaluate
from src.generator import build_instance
from src.io import PROJECT_ROOT, load_config, load_sets
from src.solution import solution_from_dict, solution_to_dict
from src.visualizer import attach_legend, plot_solution, solution_legend_handles

CACHE = PROJECT_ROOT / "results" / "solutions"
FIG_DIR = PROJECT_ROOT / "results" / "figs"
REPRESENTATIVE = [1, 5, 9, 13]          # 10 / 25 / 50 / 100 客户各一个，对应 Fig.8 的四档
BENCH = PROJECT_ROOT / "results" / "bench_main.csv"
PAPER_ITERS = 2000


def _best_seed(set_id: int) -> tuple[int, int]:
    """取该算例在 `bench_main.csv` 里达到最优的 (种子, 轮数)，保证图与表同源。

    缺文件时退回 (seed 0, 2000 轮)——成本可能比表里略高，但仍是合法解。
    """
    if not BENCH.exists():
        return 0, PAPER_ITERS
    best: tuple[float, int, int] | None = None
    with open(BENCH, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if int(row["set"]) != set_id or int(row["tune"]) != 1:
                continue
            cand = (float(row["best_cost"]), int(row["seed"]), int(row["iters"]))
            if best is None or cand[0] < best[0]:
                best = cand
    return (best[1], best[2]) if best else (0, PAPER_ITERS)


def get_solution(inst, *, refresh: bool = False, seed: int | None = None) -> tuple:
    """取（或算）一个算例的 ALNS 最优解，返回 (sol, cost, from_cache)。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"set{inst.set_id:02d}.json"
    if path.exists() and not refresh:
        d = json.loads(path.read_text(encoding="utf-8"))
        sol = solution_from_dict(d["solution"])
        _, cost = evaluate(inst, sol)
        return sol, cost, True

    auto_seed, iters = _best_seed(inst.set_id)
    seed = auto_seed if seed is None else seed
    t0 = time.time()
    r = solve(inst, seed=seed, iters=iters)
    cpu = time.time() - t0
    path.write_text(
        json.dumps(
            {
                "set": inst.set_id,
                "seed": seed,
                "iters": iters,
                "cost": round(r.best_cost, 4),
                "cpu_s": round(cpu, 2),
                "solution": solution_to_dict(r.best),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    return r.best, r.best_cost, False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="画全部 16 个算例")
    ap.add_argument("--refresh", action="store_true", help="忽略缓存重新求解")
    args = ap.parse_args()

    style.apply()
    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}
    ids = sorted(by_id) if args.all else REPRESENTATIVE

    cols = 2                       # 论文 Fig.8 就是 2×2，图例挂在每个面板右侧
    n = len(ids)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.4 * rows))
    axes = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for ax, sid in zip(axes, ids):
        inst = build_instance(by_id[sid], cfg)
        sol, cost, cached = get_solution(inst, refresh=args.refresh)
        plot_solution(
            inst, sol, ax=ax, legend=False,
            title=f"Set{sid}: {inst.n_customers} customers, K={inst.n_trucks}, cost {cost:.1f}",
        )
        attach_legend(ax, inst, sol, font_size=5.5)      # Fig.8 式图例，挂在右侧框外
        tag = "缓存" if cached else "新解"
        print(f"Set{sid:2d} n={inst.n_customers:3d} 成本 {cost:9.2f}  [{tag}]")

    for ax in axes[n:]:
        ax.set_axis_off()

    fig.suptitle("Truck & Drone Routes (ALNS solutions)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG_DIR / ("routes_all.png" if args.all else "routes_fig8.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
