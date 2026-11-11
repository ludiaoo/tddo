"""空间分布检验：客户「聚簇」时，换电的收益还在不在。

## 为什么这一维度特殊

M9.17 补了四组生成参数扰动（时间窗 / 取件 / 禁飞区），但**客户坐标不是本项目生成的**——
它来自 Murray & Raj (2020) 的原始 mFSTSP 数据（Seattle / Buffalo 真实经纬度）。
所以"空间分布"严格来说不属于"自造参数"的风险；不过审稿人仍可能问：
「你的 2.3% 是不是靠这批算例的客户铺得比较开？」

## 怎么造出干净的对照

不能直接把客户坐标挪一挪就完事：卡车的 `truck_dist_m` 是原始路网矩阵，按节点编号索引，
坐标一挪，路网矩阵与几何就脱钩了（无人机按新坐标飞、卡车按旧矩阵开），那是自相矛盾的算例。

所以这里做的是**整套底图替换**：两种形态都用同一套「欧氏距离 × 绕行系数」的解析路网，

```
truck_dist(i,j) = 欧氏距离(i,j) × road_factor
truck_time(i,j) = truck_dist(i,j) / truck_speed
```

`road_factor` 与 `truck_speed` 从**原始矩阵反标定**（见 `_calibrate`），所以 `uniform` 形态
尽可能贴近真实算例；`clustered` 形态只改客户坐标（向 k 个中心收缩），其余（需求、车队规则、
时间窗生成规则）完全一样。两种形态的差因此只来自空间分布。

## 两个形态

| 形态 | 客户坐标 |
|---|---|
| `uniform` | 原始坐标（欧氏 × 绕行系数的解析路网） |
| `clustered` | 每个客户向最近的簇心收缩 `pull`（默认 0.55），簇心保持不动 |

用法：
    python -m scripts.spatial_check --sets 5,9,13,15 --seeds 5 --workers 10
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.instance import BaseProblem, FleetParams, Instance, Node, TimeWindow
from src.io import PROJECT_ROOT, load_base_problem, load_config, load_sets

CFG = None
BY_ID = None


def _calibrate(base: BaseProblem) -> tuple[float, float]:
    """从原始路网矩阵反标定「绕行系数」与「卡车平均速度」。"""
    xy = np.array([[nd.x, nd.y] for nd in base.nodes], dtype=float)
    d_euc = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1))
    n = len(xy)
    iu = np.triu_indices(n, k=1)
    mask = d_euc[iu] > 200.0
    ratio = base.truck_dist_m[iu][mask] / d_euc[iu][mask]
    road_factor = float(np.median(ratio)) if ratio.size else 1.35

    d = base.truck_dist_m[iu]
    t = base.truck_time_s[iu]
    ok = (d > 200.0) & (t > 0)
    speed = float(np.median(d[ok] / t[ok])) if ok.any() else 11.0
    return road_factor, speed


def _synthetic_base(base: BaseProblem, form: str, pull: float, rng) -> BaseProblem:
    """按形态重建底图：坐标（可选收缩）+ 解析路网矩阵。"""
    xy = np.array([[nd.x, nd.y] for nd in base.nodes], dtype=float)
    n_cust = len(xy) - 1

    if form == "clustered":
        k = max(2, min(6, n_cust // 8))
        centers = xy[rng.choice(np.arange(1, n_cust + 1), size=min(k, n_cust), replace=False)]
        for idx in range(1, n_cust + 1):
            dist = np.linalg.norm(centers - xy[idx], axis=1)
            c = centers[int(np.argmin(dist))]
            xy[idx] = xy[idx] + pull * (c - xy[idx])       # 向最近簇心收缩

    road_factor, speed = _calibrate(base)
    diff = xy[:, None, :] - xy[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1)) * road_factor
    np.fill_diagonal(d, 0.0)
    t = d / speed

    nodes = tuple(
        Node(idx=nd.idx, kind=nd.kind, lat=nd.lat, lon=nd.lon,
             x=float(xy[i, 0]), y=float(xy[i, 1]))
        for i, nd in enumerate(base.nodes)
    )
    return BaseProblem(name=f"{base.name}__{form}", city=base.city, nodes=nodes,
                       truck_dist_m=d, truck_time_s=t, parcel_kg=base.parcel_kg)


def _build(sid: int, form: str, pull: float) -> Instance:
    """复刻 generator.build_instance 的流程，只把底图换成合成底图（且不带禁飞区）。"""
    from src.generator import _fleet_size, _make_pickup, _make_time_windows
    from src.initial import routes_needed_by_windows

    set_cfg = BY_ID[sid]
    gen = CFG["generation"]
    p = FleetParams.from_config(CFG)

    raw = load_base_problem(set_cfg["problems"][0], set_cfg["city"])
    seed = int(gen["seed"]) + sid
    rng = np.random.default_rng(seed)
    base = _synthetic_base(raw, form, pull, rng)

    f = np.array(base.parcel_kg, dtype=float)
    g = _make_pickup(base, rng, *gen["pickup_ratio"], float(gen["demand_rounding_lb"]))
    tw = _make_time_windows(base, rng, float(gen["time_window_offset"]),
                            tuple(gen["time_window_spread_h"]),
                            tuple(gen["time_window_width_min"]))
    day_end = float(gen["day_length_h"]) * 3600.0
    tw = tuple(TimeWindow(e=w.e, l=min(w.l, day_end)) for w in tw)   # 夹到班次内（合成路网时距变了）

    fs = CFG["fleet_size"]
    k = _fleet_size(float(f.sum()), p.truck_capacity, float(fs["capacity_slack"]),
                    int(fs["min_trucks"]), int(fs["max_trucks"]))
    inst = Instance(set_id=sid, base=base, f=f, g=g, tw=tw, nfz=(), n_trucks=k,
                    params=p, seed=seed)
    inst.n_trucks = int(max(k, routes_needed_by_windows(inst)))
    return inst


def _run(task):
    sid, form, pull, seed, hotswap = task
    from dataclasses import replace
    from src.alns import solve

    inst = _build(sid, form, pull)
    if hotswap:
        inst = replace(inst, params=replace(inst.params, allow_hotswap=True))
    t0 = time.time()
    r = solve(inst, seed=seed, iters=2000)
    return {
        "set": sid, "form": form, "seed": seed, "hotswap": int(hotswap),
        "n_customers": inst.n_customers, "n_trucks": inst.n_trucks,
        "best_cost": round(float(r.best_cost), 4), "cpu_s": round(time.time() - t0, 1),
    }


def _init(cfg, by_id):
    global CFG, BY_ID
    CFG, BY_ID = cfg, by_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="5,9,13,15")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pull", type=float, default=0.55, help="向簇心收缩的比例（0=不动）")
    ap.add_argument("--forms", default="uniform,clustered")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}
    sets = [int(x) for x in args.sets.split(",")]
    forms = args.forms.split(",")

    tasks = [(sid, form, args.pull, seed, hs)
             for form in forms for sid in sets
             for seed in range(args.seeds) for hs in (False, True)]
    print(f"空间分布检验：{len(tasks)} 次求解（{len(forms)} 形态 × {len(sets)} 算例 × "
          f"{args.seeds} 种子 × 2 臂）", flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(cfg, by_id)) as pool:
        for i, row in enumerate(pool.map(_run, tasks), 1):
            rows.append(row)
            if i % 8 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] set{row['set']:2d} {row['form']:<9} "
                      f"hs={row['hotswap']} cost={row['best_cost']:8.2f} "
                      f"ETA {(time.time()-t0)/i*(len(tasks)-i)/60:.1f}min", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "spatial_check.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"\n总耗时 {(time.time()-t0)/60:.1f} min → {out.relative_to(PROJECT_ROOT)}\n")
    for form in forms:
        d = df[df["form"] == form]
        p = d.pivot_table(index=["set", "seed"], columns="hotswap", values="best_cost").dropna()
        pct = ((p[1] - p[0]) / p[0] * 100.0)
        per = pct.groupby(level=0).mean()
        print(f"{form:<10}换电 {pct.mean():+6.2f}%  中位 {pct.median():+6.2f}%  "
              f"更好 {int((pct<0).sum())}/{len(pct)}  分算例 "
              + " ".join(f"S{s}:{v:+.1f}%" for s, v in per.items()))


if __name__ == "__main__":
    main()
