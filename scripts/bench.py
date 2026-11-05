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


def _init(ml_screen: bool = False, op_policy: str = "adaptive",
          fixed_pair: str | None = None, interv: str = "none",
          interv_every: int = 50, extra_destroy: str | None = None,
          accept: str = "sa", selector: str = "tier", degree: str = "uniform",
          ucb_value: bool = False) -> None:
    """每个子进程只加载一次配置（读 16 个算例的元数据）。"""
    global _CFG, _SETS
    _CFG = load_config()
    _SETS = {s["id"]: s for s in load_sets()}
    if ml_screen:
        from src.ml import state
        state.set_mode("screen")
    global _OP_POLICY, _FIXED_PAIR, _INTERV, _INTERV_EVERY
    _OP_POLICY = op_policy
    _FIXED_PAIR = fixed_pair
    _INTERV = interv
    _INTERV_EVERY = interv_every
    global _ACCEPT, _SELECTOR, _DEGREE
    _ACCEPT, _SELECTOR, _DEGREE = accept, selector, degree
    global _UCB_VALUE
    _UCB_VALUE = ucb_value
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
        "ml_attempts": cnt.get("attempts", 0),
        "ml_evals": cnt.get("evals", 0),
    }


FIELDS = [
    "set", "n_customers", "n_trucks", "seed", "iters", "tune",
    "use_drone", "use_detour", "initial_cost", "best_cost",
    "saving_pct", "t0", "restarts", "cpu_s", "ml_attempts", "ml_evals",
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
    ap.add_argument("--selector", default="tier", choices=["tier", "ucb"],
                    help="算子选择：tier=四档加分（现状），ucb=UCB1（M9.8）")
    ap.add_argument("--degree", default="uniform", choices=["uniform", "big"],
                    help="破坏度：uniform=U(10%%,40%%)（现状），big=6%% 概率大破坏（M9.8）")
    ap.add_argument("--ucb-value", action="store_true",
                    help="UCB 用「降幅」而非「成功档位」当回报（配合 --selector ucb）")
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
                                           args.ucb_value)) as pool:
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
