"""算例生成：把 mFSTSP 底图改造成论文语境的双向 + 时间窗 + 时空禁飞区算例。

⚠ **本模块的改造规则不是论文给的**。论文 §5 只有一句：
    "These cases were modified to include additional pickup demands, time window
     and no-fly zone constraints."
具体生成规则、NFZ 数量/半径/时段一律未公开（见 M0_论文要点笔记.md §9）。
因此以下规则全部是本项目自定义，参数放在 configs/config.toml 的 [generation]，
固定随机种子，保证「换台机器能复现我自己的结果」。

规则摘要：
  取件需求   g_i = round_0.5lb( U(0, ratio · f_i) )
  时间窗     e_i = offset · τ_truck(depot→i)，窗宽 l_i − e_i ~ U(width_lo, width_hi)
  禁飞区     每算例 1–2 个圆：随机取一个客户作种子，取其 k 近邻的质心为圆心，
             半径 = 该簇包围半径 × U(0.6, 1.0)；时段固定在「工作时段」起点后 4–8 小时
  车队规模   K = ceil(slack · Σf / Q_truck)，夹在 [min_trucks, max_trucks]
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .instance import (
    LB_TO_KG,
    BaseProblem,
    FleetParams,
    Instance,
    NFZ,
    TimeWindow,
)
from .io import (
    PROJECT_ROOT,
    load_base_problem,
    load_config,
    load_sets,
)

INSTANCE_DIR = PROJECT_ROOT / "data" / "instances"


def _round_to(step_lb: float, value_lb: float) -> float:
    """按 step 向上取整（lb 域），返回 lb。"""
    return math.ceil(value_lb / step_lb) * step_lb


def _make_pickup(
    base: BaseProblem, rng: np.random.Generator, ratio_lo: float, ratio_hi: float, step_lb: float
) -> np.ndarray:
    """取件需求 g_i ~ U(ratio_lo, ratio_hi) · f_i，向上取整到 step_lb。"""
    g = np.zeros(base.n_nodes, dtype=float)
    for node in range(1, base.n_nodes):
        f_lb = base.parcel_kg[node] / LB_TO_KG
        ratio = rng.uniform(ratio_lo, ratio_hi)
        g[node] = _round_to(step_lb, ratio * f_lb) * LB_TO_KG
    return g


def _make_time_windows(
    base: BaseProblem,
    rng: np.random.Generator,
    offset: float,
    spread_h: tuple[float, float],
    width_range: tuple[float, float],
) -> tuple[TimeWindow, ...]:
    """e_i = offset · τ_truck(depot→i) + U(spread_lo, spread_hi) 小时，窗宽 U(width_lo, width_hi) 分钟。

    **为什么要加随机散布**：若窗宽只挂在「从仓库直达该点的时间」上，所有窗口都会挤在
    首 1–2 小时内，100 节点算例每车要跑十几家客户，必然不可行（初版实测最晚窗口只到 1.7 h）。
    加上 0–3 h 的散布后，窗口铺开在一个工作日里，同时保留时间约束的紧度。
    """
    lo, hi = width_range
    s_lo, s_hi = spread_h
    windows: list[TimeWindow] = []
    for node in range(1, base.n_nodes):
        base_e = offset * float(base.truck_time_s[0, node])
        e = base_e + rng.uniform(s_lo, s_hi) * 3600.0
        width = rng.uniform(lo, hi) * 60.0
        windows.append(TimeWindow(e=e, l=e + width))
    return tuple(windows)


def _make_nfz(
    base: BaseProblem,
    rng: np.random.Generator,
    n_target: int,
    radius_range: tuple[float, float] = (250.0, 600.0),
    start_range: tuple[float, float] = (1.0, 2.0),
    duration_range: tuple[float, float] = (1.2, 2.2),
) -> tuple[NFZ, ...]:
    """生成圆形时空禁飞区。

    - 数量：按客户规模给定（`n_target`），对齐论文 Fig.8 的「多而小」形态
    - 圆心：**落在某个客户点附近（偏移 0.25–0.55 r）**，保证区内至少有一个客户 → γ≡1 可达；
      初版取「k 近邻质心」，簇一散开圆里就一个客户也没有，`γ≡0` 使 Fig.3 的
      「禁飞期内不能起降」判定永远走不到（被单测抓出来的）；圆心也不压在节点上，
      否则 M2 的切线长 sqrt(d^2-r^2) 在 d=0 处无定义
    - 半径：**绝对** U(250, 600) m
    - 时段：[U(start_lo,start_hi), +U(dur_lo,dur_hi)] 小时，以班次起点 t=0 为基准
    - 约束：圆心到 depot 的距离必须 > r + 500 m（禁飞区罩住仓库会让算例失去意义）

    半径用绝对米数而非「簇半径 × 比例」，是因为 Fig.8 里无论 10 还是 100 个客户，
    圆的绝对尺寸都在几百米量级；按比例算会让 100 客户算例出现 3–4 km 的大圆，
    把整片客户在禁飞期内直接禁掉，绕行机制就退化成了任务转移。
    """
    xy = np.array([[nd.x, nd.y] for nd in base.nodes])
    depot = xy[0]

    zones: list[NFZ] = []
    used_seeds: set[int] = set()

    for slot in range(n_target):
        for _ in range(300):
            seed = int(rng.integers(1, base.n_nodes))
            if seed in used_seeds:
                continue
            r = float(rng.uniform(*radius_range))
            # 圆心从客户点偏移 0.25–0.55 r：保证该客户仍在区内，但圆心不等于节点。
            # 若圆心正好压在节点上，切线长 sqrt(d^2 - r^2) 在 d=0 处无定义（M2 的绕行几何会出 nan）
            theta = rng.uniform(0.0, 2.0 * np.pi)
            off = rng.uniform(0.25, 0.55) * r
            center = xy[seed] + np.array([np.cos(theta), np.sin(theta)]) * off
            if np.linalg.norm(center - depot) <= r + 500.0:
                continue

            t_start = rng.uniform(*start_range) * 3600.0
            t_end = t_start + rng.uniform(*duration_range) * 3600.0
            zones.append(
                NFZ(
                    cx=float(center[0]),
                    cy=float(center[1]),
                    r=r,
                    t_start=t_start,
                    t_end=t_end,
                    label=f"nfz{slot + 1}",
                )
            )
            used_seeds.add(seed)
            break

    return tuple(zones)


def _fleet_size(
    total_kg: float, capacity_kg: float, slack: float, lo: int, hi: int
) -> int:
    k = math.ceil(total_kg * slack / capacity_kg)
    return int(min(max(k, lo), hi))


def build_instance(set_cfg: dict, cfg: dict) -> Instance:
    params = FleetParams.from_config(cfg)
    gen = cfg["generation"]
    fs = cfg["fleet_size"]

    base = load_base_problem(set_cfg["problems"][0], set_cfg["city"])
    seed = int(gen["seed"]) + int(set_cfg["id"])
    rng = np.random.default_rng(seed)

    f = np.array(base.parcel_kg, dtype=float)
    ratio_lo, ratio_hi = gen["pickup_ratio"]
    g = _make_pickup(base, rng, ratio_lo, ratio_hi, float(gen["demand_rounding_lb"]))

    tw_lo, tw_hi = gen["time_window_width_min"]
    tw = _make_time_windows(
        base,
        rng,
        float(gen["time_window_offset"]),
        tuple(gen["time_window_spread_h"]),
        (tw_lo, tw_hi),
    )

    # 校核：任何窗口都不得越过班次长度
    day_end = float(gen["day_length_h"]) * 3600.0
    over = [i + 1 for i, w in enumerate(tw) if w.l > day_end]
    if over:
        raise ValueError(f"Set{set_cfg['id']}: 节点 {over[:5]} 的时间窗越过班次长度 {day_end/3600:.0f} h")

    n_target = int(round(base.n_customers / float(gen["nfz_per_customers"])))
    n_target = min(max(n_target, int(gen["nfz_count_min"])), int(gen["nfz_count_max"]))
    nfz = _make_nfz(
        base,
        rng,
        n_target,
        radius_range=tuple(gen["nfz_radius_m"]),
        start_range=tuple(gen["nfz_start_h"]),
        duration_range=tuple(gen["nfz_duration_h"]),
    )

    n_trucks = _fleet_size(
        float(f.sum()),
        params.truck_capacity,
        float(fs["capacity_slack"]),
        int(fs["min_trucks"]),
        int(fs["max_trucks"]),
    )

    inst = Instance(
        set_id=int(set_cfg["id"]),
        base=base,
        f=f,
        g=g,
        tw=tw,
        nfz=nfz,
        n_trucks=n_trucks,
        params=params,
        seed=seed,
    )

    # 车队规模不能只看容量：时间窗可能逼出更多路线。
    # K = max(容量下界, 松弛优先所需路线数)，否则初始解会出现「用了 3 条路线但 K=1」的矛盾。
    from .initial import routes_needed_by_windows

    k_window = routes_needed_by_windows(inst)
    inst.n_trucks = int(max(n_trucks, k_window))
    return inst


def instance_to_dict(inst: Instance) -> dict:
    """序列化为 JSON（全部 SI 单位，字段名带单位后缀）。"""
    b = inst.base
    p = inst.params
    return {
        "set_id": inst.set_id,
        "problem": b.name,
        "city": b.city,
        "seed": inst.seed,
        "n_customers": b.n_customers,
        "n_trucks": inst.n_trucks,
        "nodes": [
            {"id": nd.idx, "kind": nd.kind, "lat": nd.lat, "lon": nd.lon, "x_m": nd.x, "y_m": nd.y}
            for nd in b.nodes
        ],
        "demand": {
            "f_kg": inst.f.tolist(),
            "g_kg": inst.g.tolist(),
        },
        "time_windows": [
            {"node": i + 1, "e_s": w.e, "l_s": w.l} for i, w in enumerate(inst.tw)
        ],
        "nfz": [
            {
                "label": z.label,
                "cx_m": z.cx,
                "cy_m": z.cy,
                "r_m": z.r,
                "t_start_s": z.t_start,
                "t_end_s": z.t_end,
            }
            for z in inst.nfz
        ],
        "truck": {
            "dist_m": b.truck_dist_m.tolist(),
            "time_s": b.truck_time_s.tolist(),
        },
        "params": {
            "drone_speed_ms": p.drone_speed,
            "drone_capacity_kg": p.drone_capacity,
            "drone_battery_kwh": p.drone_battery,
            "eta_soc": p.eta_soc,
            "truck_capacity_kg": p.truck_capacity,
            "launch_s": p.launch_time,
            "recovery_s": p.recovery_time,
            "service_s": p.service_time,
            "e_drone_kwh_per_kg_km": p.e_drone,
            "b_kwh_per_km": p.b,
            "e_truck_l_per_km": p.e_truck,
            "c_drone_per_kwh": p.c_drone,
            "c_truck_per_l": p.c_truck,
            "c_customer_per_s": p.c_customer,
        },
    }


def main() -> None:
    import json

    cfg = load_config()
    sets = load_sets()
    INSTANCE_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for s in sets:
        inst = build_instance(s, cfg)
        path = INSTANCE_DIR / f"set{inst.set_id:02d}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(instance_to_dict(inst), fh, ensure_ascii=False)

        tw = np.array([w.width() for w in inst.tw]) / 60.0
        rows.append(
            (
                inst.set_id,
                inst.n_customers,
                inst.base.city,
                inst.n_trucks,
                int(inst.f.sum() / LB_TO_KG),
                int(inst.g.sum() / LB_TO_KG),
                len(inst.nfz),
                float(np.median(tw)),
                max((z.r for z in inst.nfz), default=0.0),
            )
        )

    print(f"{'Set':>3} {'n':>4} {'城市':<8} {'K':>3} {'派件lb':>7} {'取件lb':>7} {'NFZ':>4} {'窗宽中位':>8} {'最大半径':>9}")
    print("-" * 68)
    for r in rows:
        print(f"{r[0]:>3} {r[1]:>4} {r[2]:<8} {r[3]:>3} {r[4]:>7} {r[5]:>7} {r[6]:>4} {r[7]:>7.0f}m {r[8]:>8.0f}m")
    print(f"\n已写入 {INSTANCE_DIR}（{len(rows)} 个算例）")


if __name__ == "__main__":
    main()
