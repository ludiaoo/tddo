"""并行跑 ALNS 实验（多进程）。

单机 20 核、单次求解 1–3 分钟，160 次运行串行要几小时；ALNS 的「算例 × 种子」之间
完全独立，是这个场景里最典型的易并行结构。

用法：
    python -m scripts.bench --exp main   --seeds 10 --iters 2000   # 新参数（默认）
    python -m scripts.bench --exp ablate --seeds 10 --iters 2000 --no-tune   # 论文原参数
    python -m scripts.bench --exp main --seeds 3 --sets 5,9,13     # 快速子集

结果逐行追加到 `results/bench_<exp>.csv`，中断可续（按 set+seed+tune+iters 去重）。
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.io import PROJECT_ROOT, load_config, load_sets

_CFG = None
_SETS = None
_OP_POLICY = "adaptive"
_FIXED_PAIR = None
_INTERV = "none"
_INTERV_EVERY = 50
_ACCEPT = "sa"
_SELECTOR = "tier"
_DEGREE = "uniform"
_UCB_VALUE = False
_POLISH = "off"


def _meters(inst, sol) -> dict:
    """最优解的时效与结构指标（每条路线重放一次，约毫秒级）。"""
    from src.evaluator import _simulate_route

    if sol is None:
        return {}
    fins, sorties, served, hover_s, hover_kwh = [], 0, 0, 0.0, 0.0
    for rt in sol.routes:
        ev = _simulate_route(inst, rt, fast=False)
        fins.append(ev.finish_time)
        sorties += ev.n_sorties
        served += sum(len(x.customers) for x in rt.sorties)
        hover_s += ev.hover_time
        hover_kwh += ev.hover_energy
    return {
        "makespan_s": round(max(fins) if fins else 0.0, 1),
        "driver_h": round(sum(fins) / 3600.0, 3),
        "n_sorties": int(sorties),
        "drone_served": int(served),
        "hover_min": round(hover_s / 60.0, 2),      # B 项：悬停时长与悬停耗电（诊断与校验用）
        "hover_kwh": round(hover_kwh, 5),
    }


def _init(ml_screen: bool = False, op_policy: str = "adaptive",
          fixed_pair: str | None = None, interv: str = "none",
          interv_every: int = 50, extra_destroy: str | None = None,
          accept: str = "sa", selector: str = "tier", degree: str = "uniform",
          ucb_value: bool = False, drone_topk: int = 0,
          drones_per_truck: int = 1, polish: str = "off",
          extra_trucks: int = 0, hotswap: bool = False,
          sortie_span: int = 1, hover_power_kw: float = 0.0,
          spare_batteries: int = -1, battery_mass_kg: float = -1.0,
          swap_time: float = -1.0,
          nfz_discount: float = -1.0, worst_window: float = -1.0) -> None:
    """每个子进程只加载一次配置（读 16 个算例的元数据）。"""
    global _CFG, _SETS
    _CFG = load_config()
    if drones_per_truck != 1:              # M9.12：放宽「每车 m 架机」
        _CFG["fleet"]["drones_per_truck"] = drones_per_truck
    if extra_trucks:                       # M9.14：等资本对照，多派几辆车
        _CFG["fleet_size"]["extra_trucks"] = extra_trucks
    if hotswap:                            # M9.15：允许同节点先回收再起飞（零资本放松）
        _CFG["fleet"]["allow_hotswap"] = True
    if sortie_span != 1:                   # M9.15：新建架次可跨多段锚点（解锁并行）
        import src.operators as _ops
        _ops.SORTIE_SPAN = sortie_span
    if hover_power_kw > 0.0:               # B 项：悬停计电（默认 0 = 论文口径）
        _CFG["energy"]["hover_power_kw"] = hover_power_kw
    if spare_batteries >= 0:               # E 项：每车带几块备用电池（默认 -1 = 不限）
        _CFG["fleet"]["spare_batteries"] = spare_batteries
    if battery_mass_kg >= 0.0:             # E 项：单块电池重量（<0 = 用配置值）
        _CFG["fleet"]["battery_mass_kg"] = battery_mass_kg
    if swap_time >= 0.0:                   # M9.21：换电本身的耗时（<0 = 用配置值）
        _CFG["fleet"]["swap_s"] = swap_time
    if nfz_discount >= 0.0 or worst_window > 0.0:   # M9.22：算子内部策略消融
        import src.operators as _ops
        if nfz_discount >= 0.0:
            _ops.NFZ_DISCOUNT = nfz_discount
        if worst_window > 0.0:
            _ops.WORST_WINDOW = worst_window
    _SETS = {s["id"]: s for s in load_sets()}
    if ml_screen:
        from src.ml import state
        state.set_mode("screen")
    global _OP_POLICY, _FIXED_PAIR, _INTERV, _INTERV_EVERY
    _OP_POLICY = op_policy
    _FIXED_PAIR = fixed_pair
    _INTERV = interv
    _INTERV_EVERY = interv_every
    global _POLISH
    _POLISH = polish
    global _ACCEPT, _SELECTOR, _DEGREE
    _ACCEPT, _SELECTOR, _DEGREE = accept, selector, degree
    global _UCB_VALUE
    _UCB_VALUE = ucb_value
    if drone_topk:                         # 无人机候选预筛（M9.9）
        import src.operators as _ops
        _ops.DRONE_TOPK = drone_topk
    if extra_destroy:                      # 把候选破坏算子并进算子池（M9.7）
        import importlib
        from src.operators import DESTROY
        mod = importlib.import_module(f"src.{extra_destroy}")
        DESTROY.update(mod.CANDIDATES)


def _run(task: tuple) -> dict:
    from src.alns import solve
    from src.generator import build_instance
    from src.ml import state

    from src.intervention import make_policy

    sid, seed, iters, tune, use_drone, use_detour = task
    t0 = time.time()
    inst = build_instance(_SETS[sid], _CFG)
    pol = make_policy(_INTERV, seed)
    r = solve(
        inst, seed=seed, iters=iters, tune=tune,
        use_drone=use_drone, use_detour=use_detour,
        op_policy=_OP_POLICY, fixed_pair=_FIXED_PAIR,
        interv_policy=pol, interv_every=_INTERV_EVERY,
        accept=_ACCEPT, selector=_SELECTOR, degree=_DEGREE, ucb_value=_UCB_VALUE,
        polish=_POLISH,
    )
    cnt = state.counters()
    return {
        "set": sid,
        "n_customers": inst.n_customers,
        "n_trucks": inst.n_trucks,
        "seed": seed,
        "iters": iters,
        "tune": int(tune),
        "use_drone": int(use_drone),
        "use_detour": int(use_detour),
        "initial_cost": round(float(r.initial_cost), 4),
        "best_cost": round(float(r.best_cost), 4),
        "saving_pct": round(float(r.saving) * 100, 3),
        "t0": round(float(r.t0_used), 2),
        "restarts": int(r.restarts),
        "cpu_s": round(time.time() - t0, 2),
        # 时效与结构指标（M9.14）：成本 = 司机工时 × 费率，所以成本降幅与
        # 「工时总和」同源；makespan 是另一条价值轴（客户最晚收货时刻）。
        **_meters(inst, r.best),
        "ml_attempts": cnt.get("attempts", 0),
        "ml_evals": cnt.get("evals", 0),
    }


FIELDS = [
    "set", "n_customers", "n_trucks", "seed", "iters", "tune",
    "use_drone", "use_detour", "initial_cost", "best_cost",
    "saving_pct", "t0", "restarts", "cpu_s", "ml_attempts", "ml_evals",
    "makespan_s", "driver_h", "n_sorties", "drone_served",
    "hover_min", "hover_kwh",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="main")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--sets", default="", help="逗号分隔的 set id，默认全部 16 个")
    ap.add_argument("--no-tune", action="store_true", help="用论文原参数（消融基线）")
    ap.add_argument("--no-drone", action="store_true", help="Truck-only 对照")
    ap.add_argument("--no-detour", action="store_true", help="Without-detour 对照")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--refresh", action="store_true", help="忽略已有 CSV，重跑")
    ap.add_argument("--op-policy", default="adaptive",
                    choices=["adaptive", "uniform", "fixed"],
                    help="算子选择策略（诊断用；默认 adaptive = 论文的轮盘赌+权重）")
    ap.add_argument("--fixed-pair", default=None, help="op-policy=fixed 时的 '坏|修' 组合")
    ap.add_argument("--interv", default="none", choices=["none", "random", "rule"],
                    help="周期性有界干预的决策者（M9.6 对照；默认 none = 与历史逐位一致）")
    ap.add_argument("--interv-every", type=int, default=50, help="每多少轮干预一次")
    ap.add_argument("--accept", default="sa", choices=["sa", "rel"],
                    help="接受准则：sa=绝对退火（现状），rel=相对劣化（M9.8）")
    ap.add_argument("--selector", default="tier", choices=["tier", "ucb", "dqn"],
                    help="算子选择：tier=四档加分（现状），ucb=UCB1（M9.8）")
    ap.add_argument("--degree", default="uniform", choices=["uniform", "big"],
                    help="破坏度：uniform=U(10%%,40%%)（现状），big=6%% 概率大破坏（M9.8）")
    ap.add_argument("--ucb-value", action="store_true",
                    help="UCB 用「降幅」而非「成功档位」当回报（配合 --selector ucb）")
    ap.add_argument("--polish", default="off", choices=["off", "anchors", "anchors_cur"],
                    help="局部搜索：刷新最优时做锚点平移（M9.13）")
    ap.add_argument("--sortie-span", type=int, default=1,
                    help="新建架次可跨越的卡车路段数（1=历史行为；M9.15 解锁并行）")
    ap.add_argument("--hotswap", action="store_true",
                    help="允许同节点先回收再起飞（M9.15，零资本放松；默认关闭）")
    ap.add_argument("--extra-trucks", type=int, default=0,
                    help="额外增加的卡车数（等资本对照用；M9.14）")
    ap.add_argument("--drones-per-truck", type=int, default=1,
                    help="每车可同时在飞的架次数（1=论文原模型；M9.12 扩展）")
    ap.add_argument("--drone-topk", type=int, default=0,
                    help="无人机候选预筛上限（0=全量，现状；M9.9）")
    ap.add_argument("--hover-power", type=float, default=0.0,
                    help="悬停功率 kW（0=论文口径，悬停不计电；B 项建议 0.45）")
    ap.add_argument("--spare-batteries", type=int, default=-1,
                    help="每车备用电池块数（-1=不限，论文口径；E 项，一条路线最多 b+m 个架次）")
    ap.add_argument("--battery-mass", type=float, default=-1.0,
                    help="单块电池质量 kg（<0=用 configs 里的值；0=关掉重量项，只留数量约束）")
    ap.add_argument("--swap-time", type=float, default=-1.0,
                    help="换电耗时 s（<0=用配置值；0=现行口径；建议测 120 / 300）")
    ap.add_argument("--nfz-discount", type=float, default=-1.0,
                    help="related 算子里禁飞区客户的距离权重（<0=用默认 0.6；1.0=关掉该偏好）")
    ap.add_argument("--worst-window", type=float, default=-1.0,
                    help="worst 算子的候选窗 = k×该系数（<0=用默认 1.0=严格取前 k；2.0=在前 2k 内随机）")
    ap.add_argument("--extra-destroy", default=None,
                    help="额外加载的破坏算子模块（src/<name>.py，需暴露 CANDIDATES），如 nfz_ops")
    ap.add_argument("--ml-screen", action="store_true",
                    help="启用 M9.1 候选预筛（默认关；关时与历史结果逐位一致）")
    args = ap.parse_args()

    tune = not args.no_tune
    sets = [int(x) for x in args.sets.split(",")] if args.sets else [s["id"] for s in load_sets()]
    out = PROJECT_ROOT / "results" / f"bench_{args.exp}.csv"

    done: set[tuple] = set()
    if out.exists() and not args.refresh:
        with open(out, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add((int(row["set"]), int(row["seed"]), int(row["iters"]), int(row["tune"])))

    tasks = [
        (sid, seed, args.iters, tune, not args.no_drone, not args.no_detour)
        for sid in sets
        for seed in range(args.seeds)
    ]
    todo = [t for t in tasks if (t[0], t[1], t[2], int(t[3])) not in done]

    workers = args.workers or min(max(os.cpu_count() // 2, 2), 10)
    print(
        f"实验 {args.exp}：{len(tasks)} 个任务，已完成 {len(tasks) - len(todo)}，"
        f"待跑 {len(todo)}，{workers} 进程",
        flush=True,
    )
    if not todo:
        print("没有待跑任务")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    fresh = args.refresh or not out.exists() or out.stat().st_size == 0
    fh = open(out, "w" if fresh else "a", newline="", encoding="utf-8")
    writer = None
    t_all = time.time()
    n_done = 0
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                 initargs=(args.ml_screen, args.op_policy, args.fixed_pair,
                                           args.interv, args.interv_every,
                                           args.extra_destroy,
                                           args.accept, args.selector, args.degree,
                                           args.ucb_value, args.drone_topk,
                                           args.drones_per_truck, args.polish,
                                           args.extra_trucks,
                                           args.hotswap,
                                           args.sortie_span,
                                           args.hover_power,
                                           args.spare_batteries,
                                           args.battery_mass,
                                           args.swap_time,
                                           args.nfz_discount,
                                           args.worst_window)) as pool:
            futs = {pool.submit(_run, t): t for t in todo}
            for fut in as_completed(futs):
                sid, seed, iters, _tune, _d, _dt = futs[fut]
                try:
                    row = fut.result()
                except Exception as exc:                       # 单个任务失败不拖垮整批
                    print(f"  Set{sid} seed={seed} 失败：{exc}", flush=True)
                    continue
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=FIELDS)
                    if fresh:
                        writer.writeheader()
                writer.writerow(row)
                fh.flush()
                n_done += 1
                eta = (time.time() - t_all) / n_done * (len(todo) - n_done)
                print(
                    f"  [{n_done}/{len(todo)}] Set{sid:2d} seed={seed} "
                    f"cost={row['best_cost']:8.2f} saving={row['saving_pct']:5.1f}% "
                    f"T0={row['t0']:5.1f} 重启={row['restarts']} "
                    f"{row['cpu_s']:5.1f}s  ETA {eta/60:.1f}min",
                    flush=True,
                )
    finally:
        fh.close()

    print(f"\n总耗时 {(time.time()-t_all)/60:.1f} min，已写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
