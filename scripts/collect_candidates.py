"""采集无人机插入候选的「特征 + 标签」（M9.1 预筛的训练数据）。

    python -m scripts.collect_candidates --sets 5,7,9,13 --seeds 2 --iters 300
    python -m scripts.collect_candidates --sets 5 --seeds 1 --iters 60 --out results/ml/smoke.csv.gz

采集模式（`src/ml/state.set_mode("collect")`）下候选**照常全量评估**，解与 off 模式
逐位一致——只是把每个候选的「廉价特征 + 是否可行 + 真实 Δcost」落盘。
所以这个脚本同时是一次回归测试：跑完对比 `--check` 可验证 collect 与 off 同成本。

输出：`results/ml/candidates_<tag>.csv.gz`（列 = 特征 27 + meta 5 + 标签 2）。
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from src.io import PROJECT_ROOT, load_config, load_sets

_CFG = None
_SETS = None


def _init() -> None:
    global _CFG, _SETS
    _CFG = load_config()
    _SETS = {s["id"]: s for s in load_sets()}


def _run(task: tuple) -> tuple[list[dict], dict]:
    from src.alns import solve
    from src.generator import build_instance
    from src.ml import state

    sid, seed, iters = task
    rows: list[dict] = []
    state.set_mode("collect", sink=rows.append, ctx={"set_id": sid, "seed": seed})

    inst = build_instance(_SETS[sid], _CFG)
    t0 = time.time()
    r = solve(inst, seed=seed, iters=iters)
    cpu = time.time() - t0

    meta = {"set": sid, "n_customers": inst.n_customers, "seed": seed,
            "iters": iters, "best_cost": round(float(r.best_cost), 4), "cpu_s": round(cpu, 2),
            **state.counters()}
    return rows, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="5,7,9,13,15")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tag", default="", help="输出文件名后缀")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    sets = [int(x) for x in args.sets.split(",")]
    tasks = [(sid, seed, args.iters) for sid in sets for seed in range(args.seeds)]
    workers = args.workers or min(max(len(tasks), 2), 10)

    out = PROJECT_ROOT / "results" / "ml" / f"candidates_{args.tag}.csv.gz" if args.tag \
        else PROJECT_ROOT / "results" / "ml" / "candidates.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    metas: list[dict] = []
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init) as pool:
        futs = {pool.submit(_run, t): t for t in tasks}
        for fut in as_completed(futs):
            sid, seed, _ = futs[fut]
            rows, meta = fut.result()
            all_rows.extend(rows)
            metas.append(meta)
            print(f"  Set{sid:2d} seed={seed} cost={meta['best_cost']:9.2f} "
                  f"候选={meta.get('attempts', 0):7d} 评估={meta.get('evals', 0):7d} "
                  f"特征错误={meta.get('feat_errors', 0)} {meta['cpu_s']:6.1f}s", flush=True)

    import pandas as pd

    df = pd.DataFrame(all_rows)
    df.to_csv(out, index=False, compression="gzip")
    print(f"\n任务 {len(tasks)} 个，候选 {len(df)} 行，"
          f"可行率 {df.feasible.mean():.3f}，总耗时 {(time.time() - t_all) / 60:.1f} min")
    print("各任务：")
    for m in sorted(metas, key=lambda x: (x['set'], x['seed'])):
        print(f"  Set{m['set']:2d} seed={m['seed']} best={m['best_cost']:9.2f} "
              f"attempts={m.get('attempts', 0):7d} evals={m.get('evals', 0):7d}")
    print(f"已写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
