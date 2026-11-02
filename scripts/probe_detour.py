"""诊断：Without-detour 的幅度由什么决定？

背景：我们的 Without-detour 列只有 1.2%（论文 0–21.2%）。
已知的两个候选原因——
1. **时段覆盖率低**：禁飞只占班次 35–45%，大量飞行在窗口外；
2. **绕行本身很便宜**：中位 56 m，即使禁也很难省钱。

本脚本把两个因素同时扫描（时段覆盖率 × 半径倍数），每组测三件事：
- 几何层：需绕行航段比例、节点被禁比例（不求解，快）
- 决策层：允许绕 vs 禁止绕的**最优成本差**（Alns，1 种子固定预算）

输出 `results/probe_detour.csv`。用法：
    python -m scripts.probe_detour --seeds 1 --iters 600
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np

from src.io import PROJECT_ROOT, load_config, load_sets

_CFG = None
_SETS = None


def _init() -> None:
    global _CFG, _SETS
    _CFG = load_config()
    _SETS = {s["id"]: s for s in load_sets()}


def _variant(inst, coverage: float, radius_mult: float):
    """把禁飞时段改成 [0, coverage×span]，半径乘 radius_mult。"""
    if not inst.nfz:
        return inst
    span = max(w.l for w in inst.tw) * 1.1
    zones = tuple(
        replace(z, t_start=0.0, t_end=coverage * span, r=z.r * radius_mult)
        for z in inst.nfz
    )
    return replace(inst, nfz=zones)


def _geometry(inst, samples: int = 600, seed: int = 0) -> tuple[float, float]:
    """(需绕行航段比例, 节点被禁比例)。按时间均匀抽样，模拟真实飞行时刻分布。"""
    from src.cache import GeometryCache

    g = GeometryCache.of(inst)
    N = inst.base.n_nodes
    rng = np.random.default_rng(seed)
    span = max(w.l for w in inst.tw)

    det = tot = 0
    for _ in range(samples):
        i, j = (int(x) for x in rng.integers(1, N, 2))
        if i == j:
            continue
        t0 = float(rng.uniform(0.0, span))
        tot += 1
        leg = g.leg(i, j, t0, t0 + 600.0)
        if leg.feasible and leg.detour > 1e-9:
            det += 1

    blocked = tot_n = 0
    for node in range(1, N):
        for k in range(8):
            blocked += 0 if g.service_ok(node, span * k / 8.0)[0] else 1
            tot_n += 1
    return (det / max(tot, 1), blocked / max(tot_n, 1))


def _run(task: tuple) -> dict:
    from src.alns import solve
    from src.generator import build_instance

    sid, coverage, radius_mult, seeds, iters = task
    inst = _variant(build_instance(_SETS[sid], _CFG), coverage, radius_mult)
    det_rate, blk_rate = _geometry(inst)

    on = [solve(inst, seed=s, iters=iters) for s in range(seeds)]
    off = [solve(inst, seed=s, iters=iters, use_detour=False) for s in range(seeds)]
    best_on = min(r.best_cost for r in on)
    best_off = min(r.best_cost for r in off)
    sorties = np.mean([sum(len(x.sorties) for x in r.best.routes) for r in on])
    return {
        "set": sid,
        "n_customers": inst.n_customers,
        "coverage": coverage,
        "radius_mult": radius_mult,
        "detour_rate": round(det_rate * 100, 2),
        "blocked_rate": round(blk_rate * 100, 2),
        "sorties": round(float(sorties), 1),
        "cost_on": round(float(best_on), 2),
        "cost_off": round(float(best_off), 2),
        "nodetour_inc_pct": round((best_off / best_on - 1) * 100, 2),
    }


FIELDS = [
    "set", "n_customers", "coverage", "radius_mult", "detour_rate",
    "blocked_rate", "sorties", "cost_on", "cost_off", "nodetour_inc_pct",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="5,7,9,12,13,15")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    ids = [int(x) for x in args.sets.split(",")]
    # 覆盖率 × 半径：看哪个是主因
    grid = [(c, r) for r in (1.0, 1.5, 2.0) for c in (0.35, 0.60, 0.85, 1.0)]
    tasks = [(sid, c, r, args.seeds, args.iters) for sid in ids for c, r in grid]

    out = PROJECT_ROOT / "results" / "probe_detour.csv"
    workers = args.workers or min(max(os.cpu_count() // 2, 2), 12)
    print(f"{len(tasks)} 个组合（{len(ids)} 算例 × {len(grid)} 组），{workers} 进程", flush=True)

    rows: list[dict] = []
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futs = {pool.submit(_run, t): t for t in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            sid, c, r, _s, _i = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                print(f"  Set{sid} c={c} r={r} 失败：{exc}", flush=True)
                continue
            if k % 8 == 0 or k == len(tasks):
                print(f"  [{k}/{len(tasks)}] {(time.time()-t_all)/60:.1f}min", flush=True)

    rows.sort(key=lambda x: (x["set"], x["radius_mult"], x["coverage"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print("\n=== 明细 ===")
    print(f"{'Set':>3} {'覆盖':>5} {'半径×':>5} {'绕行率':>7} {'被禁率':>7} {'架次':>5} {'+%':>7}")
    for x in rows:
        print(
            f"{x['set']:>3} {x['coverage']:>5.2f} {x['radius_mult']:>5.1f} "
            f"{x['detour_rate']:>6.1f}% {x['blocked_rate']:>6.1f}% {x['sorties']:>5.1f} "
            f"{x['nodetour_inc_pct']:>+6.1f}%"
        )

    import statistics as st

    print("\n=== 按组合汇总（6 算例均值）===")
    print(f"{'覆盖':>5} {'半径×':>5} {'绕行率':>7} {'被禁率':>7} {'+%均值':>7} {'+%最大':>7}")
    for r in (1.0, 1.5, 2.0):
        for c in (0.35, 0.60, 0.85, 1.0):
            sub = [x for x in rows if x["coverage"] == c and x["radius_mult"] == r]
            if not sub:
                continue
            print(
                f"{c:>5.2f} {r:>5.1f} {st.mean(x['detour_rate'] for x in sub):>6.1f}% "
                f"{st.mean(x['blocked_rate'] for x in sub):>6.1f}% "
                f"{st.mean(x['nodetour_inc_pct'] for x in sub):>+6.1f}% "
                f"{max(x['nodetour_inc_pct'] for x in sub):>+6.1f}%"
            )
    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}（总耗时 {(time.time()-t_all)/60:.1f}min）")


if __name__ == "__main__":
    main()
