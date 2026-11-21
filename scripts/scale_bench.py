"""规模压测：把现有 100 客户底图复制成 200/500 客户聚簇算例，只报工程指标。

⚠ 口径声明（写死在结果里，避免被误读成论文结论）：

- 10/25/50/100 用现有 mFSTSP 派生算例；200/500 用**合成聚簇**算例（复制同一张底图、旋转平移），
  卡车矩阵为「欧氏 × 1.25 / 35 km/h」的合成路网，不是任何真实城市的路网。
- 固定种子、固定低迭代预算，只观察**墙钟随规模的增长**与解是否可行；
  不跨规模比较成本百分比，不做显著性检验，不写进论文表格。

    python -m scripts.scale_bench --iters 100
    python -m scripts.scale_bench --sizes 200,500 --iters 100 --out scale_bench_large
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.alns import solve
from src.evaluator import evaluate
from src.generator import build_instance
from src.instance import BaseProblem, Node
from src.io import PROJECT_ROOT, load_base_problem, load_config, load_sets

RESULT_DIR = PROJECT_ROOT / "results"


def _latlon(x: float, y: float, lat0: float, lon0: float) -> tuple[float, float]:
    """合成坐标的近似经纬度；只用于序列化，不参与任何距离计算。"""
    lat = lat0 + y / 111_320.0
    lon = lon0 + x / (111_320.0 * max(0.2, math.cos(math.radians(lat0))))
    return lat, lon


def build_scaled_base(
    target_customers: int,
    source_problem: str,
    city: str,
    cluster_radius_m: float = 20_000.0,
    intra_scale: float = 0.7,
) -> BaseProblem:
    """把一张 100 客户底图复制成 target_customers 个客户（单仓库 + k 个旋转簇）。"""
    src = load_base_problem(source_problem, city)
    src_xy = np.array([[n.x, n.y] for n in src.nodes], dtype=float)
    depot = src_xy[0]
    lat0, lon0 = src.nodes[0].lat, src.nodes[0].lon

    n_clusters = math.ceil(target_customers / src.n_customers)
    nodes = [Node(0, "depot", lat0, lon0, 0.0, 0.0)]
    parcel = [0.0]
    xy = [(0.0, 0.0)]

    for c in range(n_clusters):
        angle = 2.0 * math.pi * c / n_clusters + math.pi / 6.0
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        center = np.array([
            cluster_radius_m * math.cos(angle),
            cluster_radius_m * math.sin(angle),
        ])
        take = min(src.n_customers, target_customers - len(xy) + 1)
        for j in range(1, take + 1):
            p = (src_xy[j] - depot) * intra_scale
            q = np.array([cos_a * p[0] - sin_a * p[1],
                          sin_a * p[0] + cos_a * p[1]]) + center
            la, lo = _latlon(float(q[0]), float(q[1]), lat0, lon0)
            nodes.append(Node(len(nodes), "customer", la, lo, float(q[0]), float(q[1])))
            parcel.append(src.parcel_kg[j])
            xy.append((float(q[0]), float(q[1])))

    coords = np.array(xy, dtype=float)
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1) * 1.25
    np.fill_diagonal(dist, 0.0)
    time_s = dist / (35.0 / 3.6)

    return BaseProblem(
        name=f"synthetic-cluster-{target_customers}",
        city="synthetic",
        nodes=tuple(nodes),
        truck_dist_m=dist,
        truck_time_s=time_s,
        parcel_kg=tuple(parcel),
    )


def _representative_set(n: int, sets: list[dict]) -> dict:
    hit = next((s for s in sets if int(s["n_customers"]) == n), None)
    if hit is None:
        raise ValueError(f"configs/instances.toml 里没有 {n} 客户算例")
    return hit


def _build(n: int, cfg: dict, sets: list[dict]) -> tuple[object, str, int]:
    """返回 (Instance, 来源标签, set_id)。"""
    if n <= 100:
        set_cfg = _representative_set(n, sets)
        return build_instance(set_cfg, cfg), "mFSTSP-derived", int(set_cfg["id"])

    ref = _representative_set(100, sets)
    base = build_scaled_base(n, ref["problems"][0], str(ref["city"]))
    local_cfg = deepcopy(cfg)
    fleet = local_cfg["fleet_size"]
    fleet["max_trucks"] = max(int(fleet["max_trucks"]), math.ceil(n / 8.0))
    local_cfg["generation"]["nfz_count_max"] = min(20, max(8, n // 25))
    set_cfg = {"id": 900 + n, "n_customers": n, "city": "synthetic",
               "problems": [ref["problems"][0]]}
    inst = build_instance(set_cfg, local_cfg, base=base)
    # 合成簇的时间窗会把路线数顶到 K 上界；贪心初始解实测需要 k_window+7 条，
    # 留 20 辆冗余避免开局不可行（只影响规模压测，不进论文口径）。
    from src.initial import routes_needed_by_windows
    inst.n_trucks = max(inst.n_trucks, routes_needed_by_windows(inst) + 20)
    return inst, "synthetic-cluster", 900 + n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", default="10,25,50,100,200,500")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="scale_bench")
    args = ap.parse_args()

    cfg = load_config()
    sets = load_sets()
    rows: list[dict] = []

    print(f"{'n':>5} {'source':<18} {'K':>4} {'CPU(s)':>8} {'cost':>9} {'feas':>5} {'sorties':>7}")
    print("-" * 64)
    for n in [int(x) for x in args.sizes.split(",") if x.strip()]:
        inst, source, set_id = _build(n, cfg, sets)
        t0 = time.perf_counter()
        result = solve(inst, seed=args.seed, iters=args.iters)
        cpu = time.perf_counter() - t0
        evs, total = evaluate(inst, result.best)
        sorties = sum(r.n_sorties for r in evs.routes)

        row = {
            "n_customers": n,
            "source": source,
            "set_id": set_id,
            "iters": args.iters,
            "seed": args.seed,
            "cpu_s": round(cpu, 2),
            "initial_cost": round(result.initial_cost, 3),
            "best_cost": round(total, 3),
            "feasible": int(evs.feasible),
            "violations": len(evs.all_violations()),
            "n_trucks": inst.n_trucks,
            "n_sorties": sorties,
            "n_routes": len(evs.routes),
        }
        rows.append(row)
        print(f"{n:>5} {source:<18} {inst.n_trucks:>4} {cpu:>8.2f} "
              f"{total:>9.2f} {str(evs.feasible):>5} {sorties:>7}")

    out = RESULT_DIR / f"{args.out}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
