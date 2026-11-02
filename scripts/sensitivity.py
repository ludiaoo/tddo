"""M7 灵敏度分析（复刻论文 Fig.9–12）。

    python -m scripts.sensitivity --which all          # 全跑（约 20–30 分钟）
    python -m scripts.sensitivity --which fig9 --quick # 冒烟测试（点少、轮数低）

四组实验与论文的对应关系：

| 图 | 扫什么 | 用哪个算例 | 论文结论（做对照用） |
|---|---|---|---|
| Fig.9  | 卡车容量 150–500 lb × 需求缩放 0.5–1.5 | Set5（25 客户） | 容量↑ 成本与车数单调↓；150→300 最陡；350 以上趋平；300–400 是均衡区间 |
| Fig.10 | 无人机载荷 5–15 lb × 续航 0.60–0.90 kWh；载荷 × 速度 55–80 km/h | Set5 | 载荷↑ 成本↓；5–8 lb 降最陡；10–12 lb 后趋平；续航/速度越高，载荷收益越大 |
| Fig.11 | 禁飞区半径 ×1–4；禁飞时段（无/工作时段/全天） | Set5–Set8（25 客户的四个） | 半径影响强于时段；「无→工作时段」只小幅升，「全天」明显跳 |
| Fig.12 | 同 Fig.11 的任务数（架次数） | 同上 | 非单调：半径「半包围」簇时架次变多，整簇断开后变少 |

**口径说明**（写报告时要带上）：
- 需求缩放与容量变化都会改变车队规模，所以每次都按
  `K = max(容量下界, 时间窗所需路线数)` 重算（与 M1 的规则一致），车数本身也是 Fig.9 的输出；
- **每点跑 `SEEDS` 个种子取均值**。这一点是被实测逼出来的：同一算例换种子，
  成本会差 4–18 美元，而灵敏度要测的效应本身只有 5–15 美元——单种子曲线基本是噪声。
  实测 600 轮并不能降低这个离散度，**增加种子数才是有效手段**；
- 轮数按规模递减（10→300、25→200、50→120、100→60），这是 Python 评估器的速度限制，
  不是论文的 2000 轮 × 10 种子；
- 结果落 CSV（`results/sens_*.csv`，含均值与极差），重跑默认读缓存，`--refresh` 强制重算。
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import style
from src.alns import solve
from src.generator import build_instance
from src.initial import routes_needed_by_windows
from src.instance import Instance, NFZ, TimeWindow
from src.io import PROJECT_ROOT, load_config, load_sets

RES = PROJECT_ROOT / "results"
FIG = RES / "figs"
LB = 0.45359237
KMH = 1.0 / 3.6

CAPS = [150, 200, 250, 300, 350, 400, 450, 500]
DEMAND = [0.5, 0.75, 1.0, 1.25, 1.5]
PAYLOADS = [5, 7, 8, 10, 12, 15]           # lb
ENDURANCE = [0.60, 0.70, 0.80, 0.90]       # kWh
SPEEDS = [55, 65, 75, 80]                  # km/h
RADII = [1.0, 1.5, 2.0, 3.0, 4.0]          # 基线半径的倍数
PERIODS = ["None", "Work-time", "Full-time"]
SEEDS = 5                                   # 每点独立种子数，取均值（见上文口径说明）


def iters_for(n: int) -> int:
    return 300 if n <= 10 else (200 if n <= 25 else (120 if n <= 50 else 60))


# ---------------------------------------------------------------- 变体构造
def _resize_fleet(inst: Instance) -> Instance:
    """按 M1 的规则重算车队规模：K = max(容量下界, 时间窗所需路线数)。"""
    cap_k = math.ceil(inst.f.sum() / inst.params.truck_capacity)
    win_k = routes_needed_by_windows(inst)
    return replace(inst, n_trucks=max(1, min(14, max(cap_k, win_k))))


def scale_demand(inst: Instance, factor: float) -> Instance:
    """需求按比例缩放（派件与取件同乘），并重算车队规模。"""
    new = replace(inst, f=inst.f * factor, g=inst.g * factor)
    return _resize_fleet(new)


def with_truck_capacity(inst: Instance, capacity_lb: float) -> Instance:
    params = replace(inst.params, truck_capacity=capacity_lb * LB)
    return _resize_fleet(replace(inst, params=params))


def with_drone(inst: Instance, *, payload_lb=None, battery_kwh=None, speed_kmh=None) -> Instance:
    p = inst.params
    params = replace(
        p,
        drone_capacity=p.drone_capacity if payload_lb is None else payload_lb * LB,
        drone_battery=p.drone_battery if battery_kwh is None else battery_kwh,
        drone_speed=p.drone_speed if speed_kmh is None else speed_kmh * KMH,
    )
    return replace(inst, params=params)


def with_radius(inst: Instance, mult: float) -> Instance:
    return replace(inst, nfz=tuple(replace(z, r=z.r * mult) for z in inst.nfz))


def with_period(inst: Instance, mode: str) -> Instance:
    if mode == "None":
        return replace(inst, nfz=())
    if mode == "Work-time":
        return inst
    if mode == "Full-time":
        return replace(inst, nfz=tuple(replace(z, t_start=0.0, t_end=1.0e6) for z in inst.nfz))
    raise ValueError(mode)


# ---------------------------------------------------------------- 求解与缓存
def run_point(inst: Instance, seeds: int = SEEDS) -> dict:
    """同一算例跑 `seeds` 个独立种子，返回均值与极差。

    单种子的离散度与要测的效应同量级（实测 4–18 vs 5–15），所以这里必须平均。
    """
    t0 = time.time()
    costs, sorties = [], []
    for sd in range(seeds):
        r = solve(inst, seed=sd, iters=iters_for(inst.n_customers))
        costs.append(float(r.best_cost))
        sorties.append(sum(len(x.sorties) for x in r.best.routes))
    return {
        "cost": round(float(np.mean(costs)), 3),
        "cost_min": round(float(np.min(costs)), 3),
        "cost_max": round(float(np.max(costs)), 3),
        "trucks": inst.n_trucks,
        "sorties": round(float(np.mean(sorties)), 2),
        "cpu_s": round(time.time() - t0, 1),
    }


RESULT_KEYS = ("cost", "cost_min", "cost_max", "trucks", "sorties", "cpu_s")


def cached_rows(path: Path, key_fields: tuple[str, ...], build, combos, refresh: bool):
    """带缓存的批量求值：`build(combo) -> (inst, row_dict)`。

    结果**逐行落盘**（先写表头再追加），中途中断不会丢掉已算好的点——
    这一组实验单点要十几秒，全跑半小时，丢了很肉疼。
    """
    done: dict[tuple, dict] = {}
    if path.exists() and not refresh:
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done[tuple(str(row[k]) for k in key_fields)] = row

    path.parent.mkdir(parents=True, exist_ok=True)
    # ⚠ 0 字节也算不 fresh：被中断的进程会留下空文件，若按「存在即追加」就永远不写表头
    fresh = refresh or (not path.exists()) or path.stat().st_size == 0
    fh = open(path, "w" if fresh else "a", newline="", encoding="utf-8")
    writer = None
    rows = []
    try:
        for combo in combos:
            inst, base = build(combo)
            key = tuple(str(base[k]) for k in key_fields)
            if key in done:
                rows.append(done[key])
                print(f"  {base}  [缓存]")
                continue
            got = run_point(inst)
            row = {**base, **{k: str(v) for k, v in got.items()}}
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                if fresh:
                    writer.writeheader()
            writer.writerow(row)
            fh.flush()
            rows.append({k: str(v) for k, v in row.items()})
            print(f"  {base}  cost={got['cost']:.1f} "
                  f"[{got['cost_min']:.1f},{got['cost_max']:.1f}] "
                  f"trucks={got['trucks']} sorties={got['sorties']} ({got['cpu_s']}s)")
    finally:
        fh.close()
    return rows


def _f(rows, **kw):
    out = []
    for r in rows:
        if all(abs(float(r[k]) - v) < 1e-9 for k, v in kw.items()):
            out.append(r)
    return out


# ---------------------------------------------------------------- Fig.9
def fig9(cfg, sets, refresh: bool) -> None:
    sid = 5
    inst0 = build_instance({s["id"]: s for s in sets}[sid], cfg)
    combos = [(c, d) for d in DEMAND for c in CAPS]

    def build(combo):
        c, d = combo
        inst = with_truck_capacity(scale_demand(inst0, d), c)
        return inst, {"capacity_lb": c, "demand_factor": d, "set": sid}

    print(f"Fig.9：卡车容量 × 需求缩放（Set{sid}）")
    rows = cached_rows(RES / "sens_fig9.csv", ("capacity_lb", "demand_factor"), build, combos, refresh)

    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for i, d in enumerate(DEMAND):
        sub = sorted(_f(rows, demand_factor=d), key=lambda r: float(r["capacity_lb"]))
        xs = [float(r["capacity_lb"]) for r in sub]
        style.line(axes[0], xs, [float(r["cost"]) for r in sub], i, label=f"{d:g}")
        style.line(axes[1], xs, [float(r["trucks"]) for r in sub], i, label=f"{d:g}")

    axes[0].set_xlabel("Load Capacity (lb)")
    axes[0].set_ylabel("Total Cost ($)")
    axes[0].set_title("Impact of Truck Capacity on Cost")
    axes[0].legend(title="Scaling Factor", loc="upper right", title_fontsize=7)
    axes[1].set_xlabel("Load Capacity (lb)")
    axes[1].set_ylabel("Number of Trucks")
    axes[1].set_title("Impact of Truck Capacity on Fleet Size")
    axes[1].legend(title="Scaling Factor", loc="upper right", title_fontsize=7)
    fig.tight_layout()
    out = FIG / "fig9_capacity_demand.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已写出 {out.relative_to(PROJECT_ROOT)}\n")


# ---------------------------------------------------------------- Fig.10
def fig10(cfg, sets, refresh: bool) -> None:
    sid = 5
    inst0 = build_instance({s["id"]: s for s in sets}[sid], cfg)
    combos = [("endurance", e, p) for e in ENDURANCE for p in PAYLOADS] + \
             [("speed", v, p) for v in SPEEDS for p in PAYLOADS]

    def build(combo):
        kind, x, payload = combo
        if kind == "endurance":
            inst = with_drone(inst0, payload_lb=payload, battery_kwh=x)
            base = {"kind": kind, "x": x, "payload_lb": payload, "set": sid}
        else:
            inst = with_drone(inst0, payload_lb=payload, speed_kmh=x)
            base = {"kind": kind, "x": x, "payload_lb": payload, "set": sid}
        return inst, base

    print(f"Fig.10：无人机载荷 × 续航 / 速度（Set{sid}）")
    rows = cached_rows(RES / "sens_fig10.csv", ("kind", "x", "payload_lb"), build, combos, refresh)

    style.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for i, e in enumerate(ENDURANCE):
        sub = sorted([r for r in rows if r["kind"] == "endurance" and abs(float(r["x"]) - e) < 1e-9],
                     key=lambda r: float(r["payload_lb"]))
        style.line(axes[0], [float(r["payload_lb"]) for r in sub],
                   [float(r["cost"]) for r in sub], i, label=f"{e:.2f} kWh")
    for i, v in enumerate(SPEEDS):
        sub = sorted([r for r in rows if r["kind"] == "speed" and abs(float(r["x"]) - v) < 1e-9],
                     key=lambda r: float(r["payload_lb"]))
        style.line(axes[1], [float(r["payload_lb"]) for r in sub],
                   [float(r["cost"]) for r in sub], i, label=f"{v} km/h")

    for ax, title, leg in zip(
        axes,
        ["Impact of Drone Payload on Cost (battery capacity fixed)",
         "Impact of Drone Payload on Cost (speed fixed)"],
        ["Battery", "Speed"],
    ):
        ax.set_xlabel("Drone Payload Capacity (lb)")
        ax.set_ylabel("Total Cost ($)")
        ax.set_title(title)
        ax.legend(title=leg, loc="upper right", title_fontsize=7)
    fig.tight_layout()
    out = FIG / "fig10_drone_payload.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已写出 {out.relative_to(PROJECT_ROOT)}\n")


# ---------------------------------------------------------------- Fig.11 / 12
def fig11(cfg, sets, refresh: bool) -> None:
    ids = [5, 6, 7, 8]                     # 四个 25 客户算例，对应论文的 Case 1–4
    by_id = {s["id"]: s for s in sets}
    combos = []
    for k, sid in enumerate(ids, start=1):
        combos += [("radius", sid, k, m) for m in RADII]
        combos += [("period", sid, k, p) for p in PERIODS]

    def build(combo):
        kind, sid, case, x = combo
        inst0 = build_instance(by_id[sid], cfg)
        inst = with_radius(inst0, x) if kind == "radius" else with_period(inst0, x)
        return inst, {"kind": kind, "set": sid, "case": case, "x": x}

    print("Fig.11/12：禁飞区半径与禁飞时段")
    rows = cached_rows(RES / "sens_fig11.csv", ("kind", "set", "x"), build, combos, refresh)

    style.apply()
    fig, axes = plt.subplots(4, 2, figsize=(9.2, 11.0))
    for r_i, case in enumerate(range(1, 5)):
        rad = sorted([r for r in rows if r["kind"] == "radius" and int(r["case"]) == case],
                     key=lambda r: float(r["x"]))
        per = [next(r for r in rows if r["kind"] == "period" and int(r["case"]) == case
                    and r["x"] == name) for name in PERIODS]

        ax = axes[r_i, 0]
        ax.plot([float(r["x"]) for r in rad], [float(r["cost"]) for r in rad],
                color=style.ACCENT, marker="s", markersize=4.5, linewidth=1.3)
        ax.set_xlabel("NFZ Radius (× baseline)")
        ax.set_ylabel("Total Cost ($)")
        ax.set_title(f"Case {case} - Radius")

        ax = axes[r_i, 1]
        ax.plot(PERIODS, [float(r["cost"]) for r in per],
                color=style.ACCENT, marker="s", markersize=4.5, linewidth=1.3,
                linestyle="--")
        ax.set_xlabel("No-Fly Period")
        ax.set_ylabel("Total Cost ($)")
        ax.set_title(f"Case {case} - No-fly Period")

    fig.tight_layout()
    out = FIG / "fig11_nfz_sensitivity.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已写出 {out.relative_to(PROJECT_ROOT)}")

    # ---- Fig.12：任务数（架次数）随半径的变化 ----
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for r_i, case in enumerate(range(1, 5)):
        rad = sorted([r for r in rows if r["kind"] == "radius" and int(r["case"]) == case],
                     key=lambda r: float(r["x"]))
        style.line(axes[0], [float(r["x"]) for r in rad],
                   [float(r["sorties"]) for r in rad], r_i, label=f"Case {case}")
        style.line(axes[1], [float(r["x"]) for r in rad],
                   [float(r["trucks"]) for r in rad], r_i, label=f"Case {case}")
    axes[0].set_xlabel("NFZ Radius (× baseline)")
    axes[0].set_ylabel("Number of Drone Sorties")
    axes[0].set_title("Mission Count vs. NFZ Radius")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].set_xlabel("NFZ Radius (× baseline)")
    axes[1].set_ylabel("Number of Trucks")
    axes[1].set_title("Fleet Size vs. NFZ Radius")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIG / "fig12_mission_count.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已写出 {out.relative_to(PROJECT_ROOT)}\n")


def fig11b_fullday(cfg, sets, refresh: bool) -> None:
    """补充实验：把禁飞时段设成**全天**，再看半径的影响。

    动机：按基线时段（开工后 1–2 h 起、持续 1.2–2.2 h）扫半径，曲线几乎是平的——
    因为大量飞行发生在禁飞窗口之外，半径放大与否根本不影响它们。
    把时段改成全天，绕行/禁飞才在所有飞行上都生效，这时半径的斜率才能看出来。
    这一条同时解释了为什么论文「半径影响强于时段」而我们相反：时段覆盖率不同。
    """
    ids = [5, 6, 7, 8]
    by_id = {s["id"]: s for s in sets}
    combos = [(sid, m) for sid in ids for m in RADII]

    def build(combo):
        sid, m = combo
        inst = with_radius(with_period(build_instance(by_id[sid], cfg), "Full-time"), m)
        return inst, {"set": sid, "radius": m}

    print("补充：全天禁飞下的半径效应")
    rows = cached_rows(RES / "sens_fig11_fullday.csv", ("set", "radius"), build, combos, refresh)

    style.apply()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, case in enumerate(range(1, 5)):
        sub = sorted([r for r in rows if int(r["set"]) == ids[case - 1]],
                     key=lambda r: float(r["radius"]))
        style.line(ax, [float(r["radius"]) for r in sub],
                   [float(r["cost"]) for r in sub], i, label=f"Case {case}")
    ax.set_xlabel("NFZ Radius (× baseline)")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title("Radius Effect under Full-time No-Fly Period")
    # 图例挂在框外，避免压住 Case 4 的曲线（之前 loc="best" 正好落在它上面）
    ax.legend(title="Case", loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0.0, title_fontsize=7)
    fig.tight_layout()
    out = FIG / "fig11b_radius_fullday.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已写出 {out.relative_to(PROJECT_ROOT)}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all",
                    choices=["all", "fig9", "fig10", "fig11", "fig11b"])
    ap.add_argument("--refresh", action="store_true", help="忽略 CSV 缓存")
    args = ap.parse_args()

    cfg = load_config()
    sets = load_sets()
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.which in ("all", "fig9"):
        fig9(cfg, sets, args.refresh)
    if args.which in ("all", "fig10"):
        fig10(cfg, sets, args.refresh)
    if args.which in ("all", "fig11"):
        fig11(cfg, sets, args.refresh)
    if args.which in ("all", "fig11b"):
        fig11b_fullday(cfg, sets, args.refresh)
    print(f"总耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
