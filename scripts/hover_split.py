"""B 项的机制诊断：悬停时间花在哪儿了。

悬停有两个来源，物理含义完全不同：

| 来源 | 场景 | 能不能避免 |
|---|---|---|
| `hover_recov` | 无人机先到回收点，在卡车上方盘旋等车 | 取决于是否允许落地等待 |
| `hover_window` | 无人机早到客户点，等时间窗开启 | 同上，但**本项目的自造时间窗让它成为主项** |

把两者混在一起报一个"悬停 12 min"没有解释力，所以单独量一遍。这段诊断只跑 4 个算例 × 3 种子，
目的是解释机制，不做统计推断（与 M9.11 的单算例诊断同一口径）。

用法：
    python -m scripts.hover_split --sets 1,5,9,13 --seeds 3
"""

from __future__ import annotations

import argparse
import copy
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src.io import PROJECT_ROOT, load_config, load_sets

CFG = None
SETS = None


def _run(task):
    sid, span, seed = task
    import src.operators as ops
    from src.alns import solve
    from src.generator import build_instance
    from src.evaluator import _simulate_route

    ops.SORTIE_SPAN = span
    cfg = copy.deepcopy(CFG)
    cfg["energy"]["hover_power_kw"] = 1.1
    inst = build_instance(SETS[sid], cfg)
    r = solve(inst, seed=seed, iters=2000)

    recov = win = 0.0
    n = 0
    for rt in r.best.routes:
        ev = _simulate_route(inst, rt, fast=False)
        recov += ev.hover_recov
        win += ev.hover_window
        n += ev.n_sorties
    return {
        "set": sid, "span": span, "seed": seed, "n_customers": inst.n_customers,
        "n_sorties": n, "cost": round(float(r.best_cost), 3),
        "hover_recov_min": round(recov / 60.0, 2),
        "hover_window_min": round(win / 60.0, 2),
    }


def _init(cfg, sets):
    global CFG, SETS
    CFG, SETS = cfg, sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="1,5,9,13")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--spans", default="1,3")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config()
    sets = {s["id"]: s for s in load_sets()}
    want = [int(x) for x in args.sets.split(",")]
    spans = [int(x) for x in args.spans.split(",")]
    tasks = [(sid, sp, seed) for sid in want for sp in spans for seed in range(args.seeds)]

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(cfg, sets)) as pool:
        for i, row in enumerate(pool.map(_run, tasks), 1):
            rows.append(row)
            print(f"  [{i}/{len(tasks)}] set{row['set']:3d} span={row['span']} "
                  f"回收点 {row['hover_recov_min']:6.1f} min / 窗口 {row['hover_window_min']:6.1f} min",
                  flush=True)

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "hover_split.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"\n总耗时 {(time.time()-t0)/60:.1f} min → {out.relative_to(PROJECT_ROOT)}\n")
    g = df.groupby("span")[["hover_recov_min", "hover_window_min", "n_sorties"]].mean().round(2)
    g["窗口占比%"] = (g["hover_window_min"] / (g["hover_recov_min"] + g["hover_window_min"]) * 100).round(1)
    print(g.to_string())
    print("\n分算例（span=1，均值）：")
    print(df[df["span"] == 1][["set", "n_customers", "n_sorties",
                               "hover_recov_min", "hover_window_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
