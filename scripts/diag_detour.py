"""诊断：禁止绕行到底改变了什么？

Without-detour 一栏我们只有 1.2%，论文最高 21.2%。要判断这是「实现问题」还是
「结构性差异」，光看成本不够——必须看**解的结构变了多少**：

- 有多少客户从「无人机服务」被迫改成「卡车直服」？
- 架次数、无人机里程掉了多少？
- 被禁用/需绕行的航段占总航段的比例？

如果结构几乎没变，说明「绕行」在这批算例里本来就是可有可无的选项，
成本差小是**真实结论**，不是实现缺陷。

用法：python -m scripts.diag_detour --iters 800 --seeds 1
输出：results/diag_detour.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from src.io import PROJECT_ROOT, load_config, load_sets

_CFG = None
_SETS = None


def _init() -> None:
    global _CFG, _SETS
    _CFG = load_config()
    _SETS = {s["id"]: s for s in load_sets()}


def _stats(inst, sol) -> dict:
    from src.evaluator import evaluate
    from src.solution import DEPOT

    res, cost = evaluate(inst, sol)
    drone_cust = {c for r in sol.routes for s in r.sorties for c in s.customers}
    truck_cust = {n for r in sol.routes for n in r.nodes if n != DEPOT}
    return {
        "cost": float(cost),
        "sorties": sum(len(r.sorties) for r in sol.routes),
        "drone_customers": len(drone_cust),
        "truck_customers": len(truck_cust),
        "drone_km": sum(r.drone_distance for r in res.routes) / 1000.0,
        "truck_km": sum(r.truck_distance for r in res.routes) / 1000.0,
        "driver_s": sum(r.finish_time for r in res.routes),
    }


def _run(task: tuple) -> dict:
    from src.alns import solve
    from src.generator import build_instance
    from src.cache import GeometryCache

    sid, seeds, iters = task
    inst = build_instance(_SETS[sid], _CFG)

    best_on = min((solve(inst, seed=s, iters=iters) for s in range(seeds)),
                  key=lambda r: r.best_cost)
    best_off = min((solve(inst, seed=s, iters=iters, use_detour=False) for s in range(seeds)),
                   key=lambda r: r.best_cost)
    on, off = _stats(inst, best_on.best), _stats(inst, best_off.best)

    # 几何：解的航段里有多少需要绕行、绕行占该航段多少
    g = GeometryCache.of(inst)
    n_legs = n_det = 0
    rel = []
    for r in best_on.best.routes:
        for s in r.sorties:
            seq = [r.nodes[s.launch], *s.customers, r.nodes[s.recover]]
            for a, b in zip(seq[:-1], seq[1:]):
                n_legs += 1
                # 取该航段在解里的实际起飞时刻不易复原，这里用「班次中段」做代表性判断
                leg = g.leg(a, b, 2 * 3600.0, 2 * 3600.0 + 300.0)
                if leg.feasible and leg.detour > 1e-9:
                    n_det += 1
                    rel.append(leg.detour / max(leg.straight, 1e-9))

    return {
        "set": sid,
        "n_customers": inst.n_customers,
        "cost_on": round(on["cost"], 2),
        "cost_off": round(off["cost"], 2),
        "inc_pct": round((off["cost"] / on["cost"] - 1) * 100, 2),
        "drone_cust_on": on["drone_customers"],
        "drone_cust_off": off["drone_customers"],
        "drone_cust_lost": on["drone_customers"] - off["drone_customers"],
        "sorties_on": on["sorties"],
        "sorties_off": off["sorties"],
        "drone_km_on": round(on["drone_km"], 1),
        "drone_km_off": round(off["drone_km"], 1),
        "truck_km_on": round(on["truck_km"], 1),
        "truck_km_off": round(off["truck_km"], 1),
        "driver_s_on": round(on["driver_s"], 0),
        "driver_s_off": round(off["driver_s"], 0),
        "legs": n_legs,
        "legs_detour": n_det,
        "detour_share_pct": round(n_det / max(n_legs, 1) * 100, 1),
        "detour_rel_p50_pct": round(st.median(rel) * 100, 2) if rel else 0.0,
    }


FIELDS = [
    "set", "n_customers", "cost_on", "cost_off", "inc_pct",
    "drone_cust_on", "drone_cust_off", "drone_cust_lost",
    "sorties_on", "sorties_off", "drone_km_on", "drone_km_off",
    "truck_km_on", "truck_km_off", "driver_s_on", "driver_s_off",
    "legs", "legs_detour", "detour_share_pct", "detour_rel_p50_pct",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    ids = [s["id"] for s in load_sets()]
    tasks = [(sid, args.seeds, args.iters) for sid in ids]
    workers = args.workers or min(max(os.cpu_count() // 2, 2), 10)
    print(f"{len(tasks)} 个算例，{workers} 进程", flush=True)

    rows: list[dict] = []
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futs = {pool.submit(_run, t): t for t in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            try:
                rows.append(fut.result())
            except Exception as exc:
                print("  失败：", exc, flush=True)
                continue
            print(f"  [{k}/{len(tasks)}]", flush=True)

    rows.sort(key=lambda x: x["set"])
    out = PROJECT_ROOT / "results" / "diag_detour.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{'Set':>3} {'n':>4} {'成本+%':>7} {'无人机客户 开→关':>16} {'丢失':>5} "
          f"{'架次 开→关':>12} {'无人机km 开→关':>15} {'需绕航段':>9}")
    for x in rows:
        print(
            f"{x['set']:>3} {x['n_customers']:>4} {x['inc_pct']:>+6.2f}% "
            f"{x['drone_cust_on']:>7} → {x['drone_cust_off']:<6} {x['drone_cust_lost']:>5} "
            f"{x['sorties_on']:>5} → {x['sorties_off']:<5} "
            f"{x['drone_km_on']:>7} → {x['drone_km_off']:<7} "
            f"{x['detour_share_pct']:>8.1f}%"
        )

    print("\n=== 汇总 ===")
    print(f"成本 +%：均值 {st.mean(x['inc_pct'] for x in rows):+.2f}%，"
          f"最大 {max(x['inc_pct'] for x in rows):+.2f}%")
    lost = [x['drone_cust_lost'] for x in rows]
    base = [x['drone_cust_on'] for x in rows]
    print(f"无人机客户：平均 {st.mean(base):.1f} → {st.mean(x['drone_cust_off'] for x in rows):.1f} "
          f"（平均少 {st.mean(lost):.1f} 个，占 {st.mean(lost)/max(st.mean(base),1e-9):.0%}）")
    print(f"需绕行航段占解内航段：均值 {st.mean(x['detour_share_pct'] for x in rows):.1f}%")
    print(f"绕行相对长度（中位的均值）：{st.mean(x['detour_rel_p50_pct'] for x in rows):.2f}%")
    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}（{(time.time()-t_all)/60:.1f}min）")


if __name__ == "__main__":
    main()
