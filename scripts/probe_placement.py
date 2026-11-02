"""诊断：禁飞区的**摆放方式**是否决定 Without-detour 的幅度。

## 假设

现有实现把圆心放在客户点上（偏移 0.25–0.55r）。后果是：
- 该客户在禁飞期内直接被禁（γ=1）→ 任务塌给卡车；
- 其他航段最多**擦过圆周**，绕行量只有几十米（实测中位 56 m）→ 绕不绕几乎不花钱。

于是「禁止绕行」这件事几乎没有代价，Without-detour 的 Increasing% 自然接近 0。
而论文 Fig.4/5 讲的机制是**航段穿越禁飞区**：圆夹在两个配送点之间，航段必须绕切线+圆弧，
绕行量可达航段的 30–50%。

## 做法

对比两种摆放，其余参数完全相同：
- `cluster`：现状（圆心贴客户点）
- `corridor`：圆心取两个客户连线的中点（圆夹在中间），且保证圆心不罩仓库

输出绕行量分位数与 Without-detour 的 Increasing%。

用法：python -m scripts.probe_placement --iters 600
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np

from src.io import PROJECT_ROOT, load_config, load_sets
from src.instance import NFZ

_CFG = None
_SETS = None


def _init() -> None:
    global _CFG, _SETS
    _CFG = load_config()
    _SETS = {s["id"]: s for s in load_sets()}


def _corridor_zones(inst, rng) -> tuple[NFZ, ...]:
    """把每个禁飞区平移到「两个客户连线的中点」，半径与时段保持不变。

    挑选条件：两点距离在 [2.2r, 5r] 之间（太近则圆罩住两端，太远则中段没意义），
    且圆心到 depot > r + 500 m。
    """
    xy = np.array([[nd.x, nd.y] for nd in inst.base.nodes])
    depot = xy[0]
    n = inst.base.n_nodes
    out: list[NFZ] = []
    for k, z in enumerate(inst.nfz):
        best = None
        for _ in range(400):
            i, j = (int(x) for x in rng.integers(1, n, 2))
            if i == j:
                continue
            d = float(np.hypot(*(xy[i] - xy[j])))
            if not (2.2 * z.r <= d <= 5.0 * z.r):
                continue
            c = (xy[i] + xy[j]) / 2.0 + rng.normal(0.0, 0.15 * z.r, 2)
            if np.linalg.norm(c - depot) <= z.r + 500.0:
                continue
            # 中点到两端都要在圆外，否则线段端点落在圆内（无切线）
            if np.linalg.norm(c - xy[i]) <= z.r * 1.05 or np.linalg.norm(c - xy[j]) <= z.r * 1.05:
                continue
            best = c
            break
        if best is None:
            out.append(z)                       # 退化为原位置
        else:
            out.append(replace(z, cx=float(best[0]), cy=float(best[1]), label=z.label))
    return tuple(out)


def _geometry(inst, samples: int = 2500, seed: int = 0) -> dict:
    """绕行量分布、需绕行比例、节点被禁比例。时间按班次均匀抽样。"""
    from src.cache import GeometryCache

    g = GeometryCache.of(inst)
    N = inst.base.n_nodes
    rng = np.random.default_rng(seed)
    span = max(w.l for w in inst.tw)
    dets, tot = [], 0
    for _ in range(samples):
        i, j = (int(x) for x in rng.integers(1, N, 2))
        if i == j:
            continue
        t0 = float(rng.uniform(0.0, span))
        tot += 1
        leg = g.leg(i, j, t0, t0 + 600.0)
        if leg.feasible and leg.detour > 1e-9:
            dets.append(leg.detour / max(leg.straight, 1e-9))

    blocked = tot_n = 0
    for node in range(1, N):
        for k in range(8):
            blocked += 0 if g.service_ok(node, span * k / 8.0)[0] else 1
            tot_n += 1
    dets.sort()
    rel = lambda q: (dets[int(q * (len(dets) - 1))] * 100) if dets else 0.0   # noqa: E731
    return {
        "detour_rate": round(len(dets) / max(tot, 1) * 100, 2),
        "detour_rel_p50": round(rel(0.5), 1),
        "detour_rel_p90": round(rel(0.9), 1),
        "blocked_rate": round(blocked / max(tot_n, 1) * 100, 2),
    }


def _run(task: tuple) -> dict:
    from src.alns import solve
    from src.generator import build_instance

    sid, placement, seeds, iters = task
    inst = build_instance(_SETS[sid], _CFG)
    # 统一时段为「覆盖整个班次」，把变量隔离到摆放方式上
    span = max(w.l for w in inst.tw) * 1.1
    inst = replace(inst, nfz=tuple(replace(z, t_start=0.0, t_end=span) for z in inst.nfz))
    if placement == "corridor":
        inst = replace(inst, nfz=_corridor_zones(inst, np.random.default_rng(12345 + sid)))

    geo = _geometry(inst)
    on = min(solve(inst, seed=s, iters=iters).best_cost for s in range(seeds))
    off = min(solve(inst, seed=s, iters=iters, use_detour=False).best_cost for s in range(seeds))
    sorties = np.mean([sum(len(x.sorties) for x in solve(inst, seed=0, iters=iters).best.routes)])
    return {
        "set": sid,
        "n_customers": inst.n_customers,
        "placement": placement,
        **geo,
        "sorties": round(float(sorties), 1),
        "cost_on": round(float(on), 2),
        "cost_off": round(float(off), 2),
        "nodetour_inc_pct": round((off / on - 1) * 100, 2),
    }


FIELDS = [
    "set", "n_customers", "placement", "detour_rate", "detour_rel_p50", "detour_rel_p90",
    "blocked_rate", "sorties", "cost_on", "cost_off", "nodetour_inc_pct",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="5,7,9,13,15")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    ids = [int(x) for x in args.sets.split(",")]
    tasks = [(sid, p, args.seeds, args.iters) for sid in ids for p in ("cluster", "corridor")]
    workers = args.workers or min(max(os.cpu_count() // 2, 2), 12)
    print(f"{len(tasks)} 个组合，{workers} 进程", flush=True)

    rows: list[dict] = []
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futs = {pool.submit(_run, t): t for t in tasks}
        for k, fut in enumerate(as_completed(futs), 1):
            sid, p, _s, _i = futs[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                print(f"  Set{sid} {p} 失败：{exc}", flush=True)
                continue
            print(f"  [{k}/{len(tasks)}] Set{sid} {p} 完成", flush=True)

    rows.sort(key=lambda x: (x["set"], x["placement"]))
    out = PROJECT_ROOT / "results" / "probe_placement.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print("\n=== 明细 ===")
    print(f"{'Set':>3} {'摆放':>9} {'绕行率':>7} {'相对p50':>8} {'相对p90':>8} {'被禁率':>7} {'架次':>5} {'+%':>7}")
    for x in rows:
        print(
            f"{x['set']:>3} {x['placement']:>9} {x['detour_rate']:>6.1f}% "
            f"{x['detour_rel_p50']:>7.1f}% {x['detour_rel_p90']:>7.1f}% "
            f"{x['blocked_rate']:>6.1f}% {x['sorties']:>5.1f} {x['nodetour_inc_pct']:>+6.1f}%"
        )

    print("\n=== 按摆放汇总 ===")
    for p in ("cluster", "corridor"):
        sub = [x for x in rows if x["placement"] == p]
        if not sub:
            continue
        print(
            f"{p:>9} 绕行率 {st.mean(x['detour_rate'] for x in sub):5.1f}%  "
            f"相对p50 {st.mean(x['detour_rel_p50'] for x in sub):5.1f}%  "
            f"被禁 {st.mean(x['blocked_rate'] for x in sub):5.1f}%  "
            f"架次 {st.mean(x['sorties'] for x in sub):4.1f}  "
            f"+% 均值 {st.mean(x['nodetour_inc_pct'] for x in sub):+5.1f}%  "
            f"最大 {max(x['nodetour_inc_pct'] for x in sub):+5.1f}%"
        )
    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}（{(time.time()-t_all)/60:.1f}min）")


if __name__ == "__main__":
    main()
