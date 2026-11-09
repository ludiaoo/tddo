"""M9.17：结论对「数据分布」的敏感性检验。

## 为什么要做

我们的 16 个算例覆盖 10/25/50/100 客户、两座城市，但**生成参数只有一套**：
时间窗宽度 U(30,90) min、禁飞区 600–1200 m、取件比例 U(0,1)、车队 slack 1.0。
也就是说"多算例"并不等于"多分布"——所有算例其实是同一个生成器的 16 次抽样。

如果「换电降 2.26%」只在默认参数下成立，那它是这套算例的性质，不是问题的性质。
这个检验逐项拧动生成参数，看结论会不会翻。

## 四组扰动（只改一组，其余不变）

| 标签 | 改动 | 想验证什么 |
|---|---|---|
| `default` | 无 | 复现默认口径（对照） |
| `tight_tw` | 时间窗 15–45 min | 窗口更紧时，换电的价值是否还在 |
| `no_pickup` | 取件比例 0 | 去掉双向载荷，退化成纯派送 |
| `dense_nfz` | 禁飞区数量 ÷5、半径 900–1500 m | 禁飞区更密更大时是否仍成立 |

用法：
    python -m scripts.dist_check --seeds 5 --sets 5,9,10,13 --workers 10
"""

from __future__ import annotations

import argparse
import copy
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from src.io import PROJECT_ROOT, load_config, load_sets

SETS = None
CFG = None

CONFIGS = {
    "default": {},
    "tight_tw": {"generation": {"time_window_width_min": [15, 45]}},
    "no_pickup": {"generation": {"pickup_ratio": [0.0, 0.0]}},
    "dense_nfz": {"generation": {"nfz_per_customers": 5, "nfz_radius_m": [900, 1500]}},
}


def _run(task):
    sid, seed, hotswap, tag = task
    from src.alns import solve
    from src.generator import build_instance

    cfg = copy.deepcopy(CFG)
    cfg["fleet"]["allow_hotswap"] = hotswap
    for sec, kv in CONFIGS[tag].items():
        cfg[sec].update(kv)

    inst = build_instance(SETS[sid], cfg)
    t0 = time.time()
    r = solve(inst, seed=seed, iters=2000)
    return {
        "tag": tag, "set": sid, "seed": seed, "hotswap": int(hotswap),
        "n_customers": inst.n_customers, "nfz": len(inst.nfz), "n_trucks": inst.n_trucks,
        "best_cost": round(float(r.best_cost), 4), "cpu_s": round(time.time() - t0, 1),
    }


def _init(cfg, sets):
    global CFG, SETS
    CFG, SETS = cfg, sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sets", default="5,9,10,13")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tags", default=",".join(CONFIGS))
    args = ap.parse_args()

    cfg = load_config()
    sets = {s["id"]: s for s in load_sets()}
    ids = [int(x) for x in args.sets.split(",")]
    tags = [t.strip() for t in args.tags.split(",")]

    tasks = [(sid, seed, hs, tag) for tag in tags for sid in ids
             for seed in range(args.seeds) for hs in (False, True)]
    print(f"共 {len(tasks)} 次求解：{len(tags)} 组扰动 × {len(ids)} 算例 × "
          f"{args.seeds} 种子 × 2 臂", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(cfg, sets)) as pool:
        for i, row in enumerate(pool.map(_run, tasks), 1):
            rows.append(row)
            if i % 10 == 0 or i == len(tasks):
                print(f"  [{i}/{len(tasks)}] {row['tag']} Set{row['set']} "
                      f"seed={row['seed']} hs={row['hotswap']} {row['best_cost']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "dist_check.csv"
    df.to_csv(out, index=False)

    print("\n=== 每组扰动下：换机 vs 不换机（同种子配对）===")
    print(f"{'扰动':10s} {'算例':>4s} {'配对均值':>9s} {'中位':>8s} {'更好/更差':>10s} {'算例级':>7s}")
    for tag in tags:
        d = df[df.tag == tag]
        a = d[d.hotswap == 0].set_index(["set", "seed"]).best_cost
        b = d[d.hotswap == 1].set_index(["set", "seed"]).best_cost
        rel = ((b / a - 1) * 100).dropna()
        per = (b.groupby("set").min() / a.groupby("set").min() - 1) * 100
        print(f"{tag:10s} {len(per):4d} {rel.mean():+8.2f}% {rel.median():+7.2f}% "
              f"{int((rel < 0).sum()):5d}/{int((rel > 0).sum()):<4d} "
              f"{int((per < 0).sum()):5d}/{len(per):<3d}")
    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
