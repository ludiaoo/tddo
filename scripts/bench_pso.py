"""并行跑 PSO 对照列（Table 3 的 PSO 列）。

    python -m scripts.bench_pso --seeds 10 --iters 2000      # 论文协议
    python -m scripts.bench_pso --seeds 3 --sets 5,13        # 快速子集
    python -m scripts.bench_pso --seeds 1 --iters 200 --exp quick   # 冒烟，不碰主结果

与 `scripts/bench.py` 同构：任务 = (set, seed)，多进程、逐行落盘、按
`set+seed+iters` 去重可续跑。输出 `results/bench_<exp>.csv`，默认 `bench_pso.csv`。

**预算口径**（与 `src/pso.py` 文件头一致）：论文只说「与 ALNS 相同预算」，
ALNS 每轮评估 1 个候选解，所以这里把预算定义成**解码次数 = 2000**（= ALNS 迭代数），
再除以种群规模 20 得到 PSO 迭代轮数 100。若按「轮数相同」算，PSO 会多评估 20 倍解，
那不是公平比较。

⚠ 编码是自造的 random-key（论文未给），这一列只能当「同预算下的另一个元启发式」参考。
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
SWARM = 20


def _init() -> None:
    global _CFG, _SETS
    _CFG = load_config()
    _SETS = {s["id"]: s for s in load_sets()}


def _run(task: tuple) -> dict:
    from src.generator import build_instance
    from src.pso import solve_pso

    sid, seed, iters = task
    t0 = time.time()
    inst = build_instance(_SETS[sid], _CFG)
    r = solve_pso(inst, seed=seed, iters=iters, swarm=SWARM)
    return {
        "set": sid,
        "n_customers": inst.n_customers,
        "n_trucks": inst.n_trucks,
        "seed": seed,
        "iters": iters,
        "best_cost": round(float(r.best_cost), 4),
        "decodes": int(r.decodes),
        "failed_decodes": int(r.failed_decodes),
        "cpu_s": round(time.time() - t0, 2),
    }


FIELDS = [
    "set", "n_customers", "n_trucks", "seed", "iters",
    "best_cost", "decodes", "failed_decodes", "cpu_s",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--iters", type=int, default=2000, help="解码次数（= ALNS 迭代数）")
    ap.add_argument("--sets", default="")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--exp", default="pso", help="输出文件名后缀：results/bench_<exp>.csv")
    args = ap.parse_args()

    sets = [int(x) for x in args.sets.split(",")] if args.sets else [s["id"] for s in load_sets()]
    out = PROJECT_ROOT / "results" / f"bench_{args.exp}.csv"

    done: set[tuple] = set()
    if out.exists() and not args.refresh:
        with open(out, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add((int(row["set"]), int(row["seed"]), int(row["iters"])))

    tasks = [(sid, seed, args.iters) for sid in sets for seed in range(args.seeds)]
    todo = [t for t in tasks if t not in done]

    workers = args.workers or min(max(os.cpu_count() // 2, 2), 10)
    print(
        f"PSO：{len(tasks)} 个任务，已完成 {len(tasks) - len(todo)}，待跑 {len(todo)}，"
        f"{workers} 进程（每任务 {args.iters} 次解码 = {args.iters // SWARM} 轮 × 种群 {SWARM}）",
        flush=True,
    )
    if not todo:
        print("没有待跑任务")
        return

    out.parent.mkdir(parents=True, exist_ok=True)
    fresh = args.refresh or not out.exists() or out.stat().st_size == 0
    fh = open(out, "w" if fresh else "a", newline="", encoding="utf-8")
    writer = None
    t_all, n_done = time.time(), 0
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
            futs = {pool.submit(_run, t): t for t in todo}
            for fut in as_completed(futs):
                sid, seed, _ = futs[fut]
                try:
                    row = fut.result()
                except Exception as exc:
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
                    f"cost={row['best_cost']:8.2f} 解码={row['decodes']} "
                    f"失败={row['failed_decodes']:3d} {row['cpu_s']:5.1f}s  ETA {eta/60:.1f}min",
                    flush=True,
                )
    finally:
        fh.close()

    print(f"\n总耗时 {(time.time()-t_all)/60:.1f} min，已写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
