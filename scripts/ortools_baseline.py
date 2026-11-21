"""外部基线：用 OR-Tools RoutingModel 跑同批算例的纯卡车 VRPTW。

定位：**外部求解器对照**，不是本项目模型的等价实现。OR-Tools 的路由模型无法直接表达
「无人机从卡车顶起飞 / 在另一个停靠点回收 / 换电 / 单架次电量 / 时空禁飞区」，
因此这里只构造卡车侧子问题：

- 节点、时间窗、路网距离/时间矩阵、车数、服务时间、卡车容量 —— 与 `Instance` 完全一致；
- 成本口径与评估器一致：卡车燃油 `e_truck * 距离(km) * c_truck` + 司机工时 `c_customer * 返仓时刻`；
- **放松项**：取件载荷 `g_i` 不计入容量（只计派件 `f_i`）；不派无人机，因此禁飞区/电量/换电都不出现。
  所以 OR-Tools 的数字是「卡车侧松弛子问题」的解，不能直接当作原问题的下界来宣传。

用法：
    python -m scripts.ortools_baseline                      # 默认 8 个算例（西雅图 4 + 天河 4）
    python -m scripts.ortools_baseline --sets 13,104 --limit 60

输出：`results/ortools_baseline.csv` 与 `results/ortools_baseline.md`。
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

try:
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2
except ImportError as exc:  # pragma: no cover - 环境缺依赖时给出明确指引
    raise SystemExit(
        "缺少 OR-Tools。先执行：pip install -r requirements-dev.txt\n"
        "（或 pip install 'ortools>=9.11'）"
    ) from exc

from src.energy import driver_cost, truck_cost
from src.generator import build_instance
from src.io import PROJECT_ROOT, load_config, load_sets

# 目标函数放大成整数：0.0063 $/s * 1e6 = 6300，避免 OR-Tools 取整丢掉司机成本
SCALE = 1_000_000
# 容量按 0.1 kg 取整（需求本身按 0.5 lb 取整，0.1 kg 足够）
KG_SCALE = 10

FIELDS = [
    "set", "city", "n_customers", "n_trucks", "time_limit_s", "wall_s",
    "status", "objective_cost", "recomputed_cost", "abs_diff",
    "distance_m", "driver_s", "routes_used", "alns_truck_best",
    "alns_full_best", "gap_vs_alns_truck_pct", "alns_full_saving_pct",
]


def default_limit(n_customers: int) -> int:
    """给不同规模的算例分配搜索时间（秒）。保持整体在几分钟量级。"""
    if n_customers <= 10:
        return 10
    if n_customers <= 25:
        return 15
    if n_customers <= 50:
        return 25
    return 40


def _solve(inst, time_limit_s: int, day_h: float) -> dict:
    p = inst.params
    dist = inst.base.truck_dist_m
    tmat = inst.base.truck_time_s
    f = inst.f
    service = p.service_time
    manager = pywrapcp.RoutingIndexManager(inst.base.n_nodes, inst.n_trucks, 0)
    routing = pywrapcp.RoutingModel(manager)

    fuel_per_m = p.e_truck * p.c_truck / 1000.0

    def arc_cb(i_idx: int, j_idx: int) -> int:
        i, j = manager.IndexToNode(i_idx), manager.IndexToNode(j_idx)
        return int(round(fuel_per_m * float(dist[i][j]) * SCALE))

    routing.SetArcCostEvaluatorOfAllVehicles(routing.RegisterTransitCallback(arc_cb))

    def time_cb(i_idx: int, j_idx: int) -> int:
        i, j = manager.IndexToNode(i_idx), manager.IndexToNode(j_idx)
        t = float(tmat[i][j]) + (service if i != 0 else 0.0)
        return int(round(t))

    time_id = routing.RegisterTransitCallback(time_cb)
    horizon = int(round(day_h * 3600.0))
    # slack_max 必须给足：客户早到要能原地等窗口开启，否则时间窗几乎必然不可行
    routing.AddDimension(time_id, horizon, horizon, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    # span = End - Start，Start 固定为 0 → span 就是该车返仓时刻；
    # 系数取 c_customer * SCALE，于是目标里精确包含司机工时（含时间窗等待）。
    time_dim.SetSpanCostCoefficientForAllVehicles(int(round(p.c_customer * SCALE)))
    for node, w in enumerate(inst.tw, start=1):
        cumul = time_dim.CumulVar(manager.NodeToIndex(node))
        cumul.SetRange(int(math.floor(w.e)), int(math.ceil(w.l)))

    # 容量：只计派件 f_i，取件 g_i 放松（见模块 docstring）
    def load_cb(_i_idx: int, j_idx: int) -> int:
        j = manager.IndexToNode(j_idx)
        return int(round(float(f[j]) * KG_SCALE)) if j != 0 else 0

    routing.AddDimension(
        routing.RegisterTransitCallback(load_cb), 0,
        int(round(p.truck_capacity * KG_SCALE)), True, "Load",
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.FromSeconds(time_limit_s)
    params.log_search = False

    t0 = time.time()
    sol = routing.SolveWithParameters(params)
    wall = time.time() - t0
    if sol is None:
        return {
            "status": "no_solution", "wall_s": round(wall, 2),
            "objective_cost": None, "recomputed_cost": None,
            "distance_m": 0.0, "driver_s": 0.0, "routes_used": 0,
            "routes": [],
        }

    routes: list[list[int]] = []
    total_dist = 0.0
    total_finish = 0.0
    used = 0
    for v in range(inst.n_trucks):
        idx = routing.Start(v)
        seq = [manager.IndexToNode(idx)]
        while not routing.IsEnd(idx):
            idx = sol.Value(routing.NextVar(idx))
            seq.append(manager.IndexToNode(idx))
        if len(seq) > 2:
            used += 1
        routes.append(seq)
        total_dist += sum(float(dist[a][b]) for a, b in zip(seq, seq[1:]))
        total_finish += float(sol.Min(time_dim.CumulVar(routing.End(v))))

    obj = float(sol.ObjectiveValue()) / SCALE
    recalc = truck_cost(total_dist, p) + driver_cost(total_finish, p)
    return {
        "status": "ok", "wall_s": round(wall, 2),
        "objective_cost": round(obj, 4),
        "recomputed_cost": round(recalc, 4),
        "abs_diff": round(abs(obj - recalc), 4),
        "distance_m": round(total_dist, 1),
        "driver_s": round(total_finish, 1),
        "routes_used": used,
        "routes": routes,
    }


def _best_by_set(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    out: dict[int, float] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                sid = int(row["set"])
                val = float(row["best_cost"])
            except (KeyError, TypeError, ValueError):
                continue
            if sid not in out or val < out[sid]:
                out[sid] = val
    return out


def _best_by_set(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    out: dict[int, float] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                sid = int(row["set"])
                val = float(row["best_cost"])
            except (KeyError, TypeError, ValueError):
                continue
            if sid not in out or val < out[sid]:
                out[sid] = val
    return out


def _fmt(v: float | None, digits: int = 2) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sets", default="1,5,9,13,101,102,103,104",
        help="逗号分隔的 set id；默认西雅图 4 个 + 天河 4 个",
    )
    ap.add_argument("--limit", type=int, default=0, help="覆盖全部算例的搜索秒数（0=按规模自动）")
    ap.add_argument("--out", default="ortools_baseline", help="结果文件名（不含扩展名）")
    args = ap.parse_args()

    cfg = load_config()
    set_map = {s["id"]: s for s in load_sets()}
    ids = [int(x) for x in args.sets.split(",") if x.strip()]
    day_h = float(cfg["generation"]["day_length_h"])

    full_sources = ["bench_main.csv", "bench_th_base.csv"]
    # 后面的来源覆盖前面的：fullpair（10 种子 × 2000 轮）必须是最后一项，
    # 早期快跑（3 种子 × 1000 轮）只作回退。
    truck_sources = ["bench_truckonly.csv", "bench_th_truck_ortools.csv", "bench_th_truck_fullpair.csv"]
    full_best: dict[int, float] = {}
    truck_best: dict[int, float] = {}
    for name in full_sources:
        full_best.update(_best_by_set(PROJECT_ROOT / "results" / name))
    for name in truck_sources:
        truck_best.update(_best_by_set(PROJECT_ROOT / "results" / name))

    rows: list[dict] = []
    for sid in ids:
        if sid not in set_map:
            print(f"跳过未知 set {sid}")
            continue
        inst = build_instance(set_map[sid], cfg)
        limit = args.limit or default_limit(inst.n_customers)
        res = _solve(inst, limit, day_h)
        row = {
            "set": sid,
            "city": inst.base.city,
            "n_customers": inst.n_customers,
            "n_trucks": inst.n_trucks,
            "time_limit_s": limit,
            "wall_s": res["wall_s"],
            "status": res["status"],
            "objective_cost": res["objective_cost"],
            "recomputed_cost": res["recomputed_cost"],
            "abs_diff": res.get("abs_diff"),
            "distance_m": res["distance_m"],
            "driver_s": res["driver_s"],
            "routes_used": res["routes_used"],
            "alns_truck_best": truck_best.get(sid),
            "alns_full_best": full_best.get(sid),
        }
        if row["objective_cost"] is not None and truck_best.get(sid):
            row["gap_vs_alns_truck_pct"] = round(
                (row["objective_cost"] - truck_best[sid]) / truck_best[sid] * 100, 2
            )
        else:
            row["gap_vs_alns_truck_pct"] = None
        if full_best.get(sid) and truck_best.get(sid):
            row["alns_full_saving_pct"] = round(
                (full_best[sid] - truck_best[sid]) / truck_best[sid] * 100, 2
            )
        else:
            row["alns_full_saving_pct"] = None
        rows.append(row)
        print(
            f"Set{sid:>3} n={inst.n_customers:>3} K={inst.n_trucks:>2} "
            f"OR-Tools={_fmt(row['objective_cost'])} "
            f"ALNS纯卡车={_fmt(row['alns_truck_best'])} "
            f"墙钟={row['wall_s']}s status={row['status']}", flush=True,
        )

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.out}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# OR-Tools 外部基线（纯卡车 VRPTW）",
        "",
        "> 模型范围：节点/时间窗/有向路网矩阵/车数/服务时间/容量与 `Instance` 一致；",
        "> 成本口径与评估器一致（燃油 + 司机工时，含时间窗等待）。",
        "> **放松**：不计取件载荷，不派无人机，因此禁飞区/电量/换电都不出现。",
        "> 这是「卡车侧松弛子问题」，不是原问题的等价解或严格下界。",
        "",
        "| Set | 城市 | n | K | 限时(s) | OR-Tools | 墙钟(s) | ALNS 纯卡车 | 差距 | ALNS 全模型 | 全模型节省 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['set']} | {r['city']} | {r['n_customers']} | {r['n_trucks']} | "
            f"{r['time_limit_s']} | {_fmt(r['objective_cost'])} | {r['wall_s']} | "
            f"{_fmt(r['alns_truck_best'])} | {_fmt(r['gap_vs_alns_truck_pct'])}% | "
            f"{_fmt(r['alns_full_best'])} | {_fmt(r['alns_full_saving_pct'])}% |"
        )
    lines += [
        "",
        "- 差距 = (OR-Tools − ALNS 纯卡车) / ALNS 纯卡车。正值表示 OR-Tools 更贵，负值表示更便宜。",
        "- 全模型节省 = (ALNS 全模型 − ALNS 纯卡车) / ALNS 纯卡车，负值表示无人机方案更省。",
        "- ALNS 数值取各 CSV 中同 set 的 `best_cost` 最小值（西雅图 `bench_main.csv`，天河 `bench_th_base.csv`；",
        "  纯卡车西雅图 `bench_truckonly.csv`，天河 `bench_th_truck_fullpair.csv`（10 种子 × 2000 轮，",
        "  与全模型同口径；`bench_th_truck_ortools.csv` 是早期 3 种子 × 1000 轮的快跑对照）。",
        "- OR-Tools 使用 `PATH_CHEAPEST_ARC` 初始解 + `GUIDED_LOCAL_SEARCH`，单线程，限时见上表。",
        "- `objective_cost` 是 OR-Tools 目标/1e6；`recomputed_cost` 是用项目评估器公式重算的成本，",
        "  两列应一致（CSV 里的 `abs_diff` 是差值），用来防止目标口径接错。",
    ]
    md_path = out_dir / f"{args.out}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写出 {csv_path} 与 {md_path}")


if __name__ == "__main__":
    main()
