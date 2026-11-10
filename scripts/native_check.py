"""C 项：把算例还原成 mFSTSP 的**原生形态**，看换电结论还在不在。

## 为什么必须做

本项目的算例是自造的：取件比例 U(0,1)、时间窗 U(30,90) min、每算例 2–8 个禁飞区。
换电（`allow_hotswap`）的收益有 77% 来自"同节点不能既降落又起飞"这个建模缺陷（M9.15），
但**触发换电机会的却是这些自造约束**——取件让无人机在同一个节点既送货又取货、
紧时间窗让架次必须接得很密、禁飞区让无人机绕远、落地时间点变得更不可控。

审稿人一句话就能问倒：「把取件、时间窗、禁飞区全去掉，回到 Murray & Raj (2020) 的
原形态，你的结论还剩多少？」这个脚本就是回答它。

## 两条臂 × 两种形态

| 形态 | 取件 | 禁飞区 | 时间窗 | 说明 |
|---|---|---|---|---|
| `default` | U(0,1) | 2–8 个，600–1200 m | U(30,90) min | 本项目现行口径（对照） |
| `native` | 0 | 无 | 8 h（等于不约束） | mFSTSP 原生形态 |

两条臂：`span1`（论文模型）与 `hotswap`（放开同节点换电）。同种子配对。

用法：
    python -m scripts.native_check --seeds 5 --sets 5,9,13 --workers 10
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

# 只改生成参数，求解器与评估器完全不动 —— 这样两种形态的差只可能来自算例本身
FORMS = {
    "default": {},
    "native": {
        "generation": {
            "pickup_ratio": [0.0, 0.0],          # 纯派送，无双向载荷
            # 禁飞区关掉：nfz_per_customers 是分母，不能给 0（会除零），给一个极大的数
            # 让 round(n/9999)=0，再由 count_max=0 夹成 0 个圆
            "nfz_per_customers": 9999,
            "nfz_count_min": 0,
            "nfz_count_max": 0,
            "time_window_offset": 1.0,           # 窗口下界 = 最快可到时刻
            "time_window_spread_h": [0.0, 0.0],  # 不再随机散布
            # 7 h ≈ 整个班次（8 h 会让 e_i + 宽度越过班次末，生成器会直接报错）
            "time_window_width_min": [420, 420],
        }
    },
}


def _run(task):
    sid, seed, hotswap, form = task
    from src.alns import solve
    from src.generator import build_instance

    cfg = copy.deepcopy(CFG)
    cfg["fleet"]["allow_hotswap"] = hotswap
    for sec, kv in FORMS[form].items():
        cfg[sec].update(kv)

    inst = build_instance(SETS[sid], cfg)
    t0 = time.time()
    r = solve(inst, seed=seed, iters=2000)
    routes = r.best.routes if r.best else []
    return {
        "form": form, "set": sid, "seed": seed, "hotswap": int(hotswap),
        "n_customers": inst.n_customers, "n_trucks": inst.n_trucks, "nfz": len(inst.nfz),
        "pickup_kg": round(float(inst.g.sum()), 1),
        "tw_width_min": round(float(min(w.width() for w in inst.tw)) / 60.0, 1),
        "best_cost": round(float(r.best_cost), 4),
        "n_sorties": sum(len(rt.sorties) for rt in routes),
        "drone_served": sum(len(x.customers) for rt in routes for x in rt.sorties),
        "cpu_s": round(time.time() - t0, 1),
    }


def _init(cfg, sets):
    global CFG, SETS
    CFG, SETS = cfg, sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sets", default="", help="默认全部 16 个")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--forms", default="default,native")
    args = ap.parse_args()

    cfg = load_config()
    sets = [s for s in load_sets() if not args.sets or s["id"] in
            {int(x) for x in args.sets.split(",")}]
    forms = args.forms.split(",")

    tasks = [(s["id"], seed, hs, f) for f in forms for s in sets
             for seed in range(args.seeds) for hs in (False, True)]
    print(f"C 项：{len(tasks)} 次求解（{len(forms)} 形态 × {len(sets)} 算例 × "
          f"{args.seeds} 种子 × 2 臂），{args.workers} 进程", flush=True)

    rows: list[dict] = []
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(cfg, {s["id"]: s for s in sets})) as pool:
        for i, row in enumerate(pool.map(_run, tasks), 1):
            rows.append(row)
            if i % 10 == 0 or i == len(tasks):
                eta = (time.time() - t_all) / i * (len(tasks) - i)
                print(f"  [{i}/{len(tasks)}] set{row['set']:2d} {row['form']:<7} "
                      f"hs={row['hotswap']} cost={row['best_cost']:8.2f} "
                      f"ETA {eta/60:.1f}min", flush=True)

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results" / "native_check.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"\n总耗时 {(time.time()-t_all)/60:.1f} min → {out.relative_to(PROJECT_ROOT)}\n")
    print("## 换电收益（同种子配对，负 = 更好）\n")
    for form in forms:
        d = df[df["form"] == form]
        p = d.pivot_table(index=["set", "seed"], columns="hotswap", values="best_cost")
        if 0 not in p or 1 not in p:
            continue
        p = p.dropna()
        pct = (p[1] - p[0]) / p[0] * 100.0
        per_set = pct.groupby(level=0).mean()
        n_better = int((pct < 0).sum())
        print(f"{form:<9} 配对均值 {pct.mean():+6.2f}%  中位 {pct.median():+6.2f}%  "
              f"{n_better}/{len(pct)} 个 (set,seed) 更好  "
              f"算例级 {int((per_set < 0).sum())}/{len(per_set)}")
    print("\n分算例（负 = 换电更好）：")
    for form in forms:
        d = df[df["form"] == form]
        p = d.pivot_table(index=["set", "seed"], columns="hotswap", values="best_cost").dropna()
        pct = ((p[1] - p[0]) / p[0] * 100.0).groupby(level=0).mean()
        print(f"  {form:<9} " + "  ".join(f"S{s}:{v:+6.2f}%" for s, v in pct.items()))


if __name__ == "__main__":
    main()
