"""汇总并行实验，产出新版 Table 3 与四行消融表。

输入（`python -m scripts.bench` 的输出）：
- `bench_main.csv`      ALNS，新参数，10 种子
- `bench_ablate.csv`    ALNS，**论文原参数**，5 种子（消融用）
- `bench_truckonly.csv` 纯卡车，5 种子
- `bench_nodetour.csv`  Without-detour，5 种子

输出：
- `results/table3_v2.csv`    与论文 Table 3 同口径的对照表
- 终端打印：与旧表（150–300 轮 × 2 种子）的差异、四行消融

约定：**每个单元格取该算例所有种子里的最优**（论文也是「10 种子取最优」）。
消融比较为了公平，两边都只用 seed 0–4。
"""

from __future__ import annotations

import pandas as pd

from src.io import PROJECT_ROOT

RES = PROJECT_ROOT / "results"


def _best(path: str, seeds: int | None = None) -> pd.DataFrame:
    d = pd.read_csv(RES / path)
    if seeds is not None:
        d = d[d.seed < seeds]
    g = d.groupby("set").agg(
        n_customers=("n_customers", "first"),
        n_trucks=("n_trucks", "first"),
        initial=("initial_cost", "median"),
        best=("best_cost", "min"),
        worst=("best_cost", "max"),
        cpu_mean=("cpu_s", "mean"),
    )
    g["spread_pct"] = (g.worst / g.best - 1) * 100
    return g


def main() -> None:
    main10 = _best("bench_main.csv")
    main5 = _best("bench_main.csv", seeds=5)
    abl5 = _best("bench_ablate.csv", seeds=5)
    truck = _best("bench_truckonly.csv", seeds=5)
    nodet = _best("bench_nodetour.csv", seeds=5)

    t = pd.DataFrame(
        {
            "n": main10.n_customers,
            "K": main10.n_trucks,
            "initial": main10.initial.round(2),
            "alns_best": main10.best.round(2),
            "saving_pct": ((1 - main10.best / main10.initial) * 100).round(2),
            "seed_spread_pct": main10.spread_pct.round(1),
            "alns_cpu_s": main10.cpu_mean.round(1),
            "truckonly": truck.best.round(2),
            "truckonly_inc_pct": ((truck.best / main10.best - 1) * 100).round(2),
            "nodetour": nodet.best.round(2),
            "nodetour_inc_pct": ((nodet.best / main10.best - 1) * 100).round(2),
            "ablate5_best": abl5.best.round(2),
            "tune_gain_pct": ((abl5.best / main5.best - 1) * 100).round(2),
        }
    )
    out = RES / "table3_v2.csv"
    t.to_csv(out, encoding="utf-8", index_label="set")

    print("=== 新版 Table 3（论文协议：2000 轮 × 10 种子）===")
    print(t.to_string())

    print("\n=== 汇总指标 ===")
    for col, name in [
        ("saving_pct", "Saving%（初始→最优）"),
        ("truckonly_inc_pct", "相对 Truck-only"),
        ("nodetour_inc_pct", "相对 Without-detour"),
        ("tune_gain_pct", "参数修正带来的改进（5 种子同口径）"),
        ("seed_spread_pct", "种子间极差（越大多样性越重要）"),
    ]:
        v = t[col]
        print(f"{name:34s} 均值 {v.mean():7.2f}   中位 {v.median():7.2f}   范围 {v.min():6.2f} ~ {v.max():6.2f}")

    print("\n=== 与旧表对照（旧：150–300 轮 × 2 种子）===")
    old = pd.read_csv(RES / "table3_with_pso.csv").set_index("set")
    cmp = pd.DataFrame(
        {
            "旧_alns": old.alns_cost.round(2),
            "新_alns": main10.best.round(2),
            "成本变化%": ((main10.best / old.alns_cost - 1) * 100).round(2),
            "旧_saving": old.saving_pct.round(1),
            "新_saving": t.saving_pct,
            "旧_truckonly%": old.truckonly_increase_pct.round(1),
            "新_truckonly%": t.truckonly_inc_pct,
            "旧_nodetour%": old.nodetour_increase_pct.round(1),
            "新_nodetour%": t.nodetour_inc_pct,
        }
    )
    print(cmp.to_string())
    print(f"\n成本平均变化 {cmp['成本变化%'].mean():+.2f}%，"
          f"最大 {cmp['成本变化%'].min():+.2f}%")
    print(f"旧表 CPU 均值（16 算例） {old.alns_cpu_s.mean():.1f}s → "
          f"新 {main10.cpu_mean.mean():.1f}s（轮数从 150–300 提到 2000）")
    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
