"""等资本对照与盈亏平衡分析（M9.14）。

回答一个审稿人必问的问题：m=2 比 m=1 多了一倍的无人机，凭什么说它更好？

三条臂，成本口径完全一致（都是「一个班次的运营成本」，单位 $）：

| 臂 | 车队 | 相对 m=1 的额外投入 |
|---|---|---|
| `m1`      | K 车 × 1 机 | —（基线） |
| `m2`      | K 车 × 2 机 | **K 架无人机**（无新增司机、无新增车） |
| `truck1`  | K+1 车 × 1 机 | **1 辆车 + 1 名司机**（司机成本已在目标函数里） |

`m2` 与 `truck1` 的「运力增量」可比：一个是每车多带一架机，一个是多派一辆车。
资本差异用**盈亏平衡**表达，不预设无人机价格：

    c* = (C_m1 − C_m2) / (K·(m−1))        # 每架无人机每天最多值多少钱

c* 是「让 m=2 与 m=1 总成本相等」的临界日成本。c* 越大，结论越稳。
同理，`truck1` 的临界日成本只针对**车辆本身**（司机已计入运营成本）。

用法：
    python -m scripts.equal_capital               # 用 results/bench_*.csv
    python -m scripts.equal_capital --sets 5,9,13
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.io import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
ARMS = {
    "m1": "bench_main.csv",
    "m2": "bench_m2.csv",
    "m3": "bench_m3.csv",
    "truck1": "bench_truck1.csv",
}
# 时效指标（makespan）单独一臂：bench 后来才记录 makespan_s，所以只在这两份里可比
MS_ARMS = {"m1": "bench_ms_m1.csv", "m2": "bench_ms_m2.csv"}


def _load(name: str) -> pd.DataFrame | None:
    path = RESULTS / ARMS[name]
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="", help="只统计这些算例（逗号分隔，默认全部）")
    args = ap.parse_args()
    want = [int(x) for x in args.sets.split(",")] if args.sets.strip() else None

    base = _load("m1")
    if base is None:
        raise SystemExit("缺少 results/bench_main.csv")
    if want:
        base = base[base["set"].isin(want)]

    arms = {}
    for nm in ("m2", "m3", "truck1"):
        d = _load(nm)
        if d is not None:
            arms[nm] = d[d["set"].isin(base["set"].unique())] if want else d

    # 每个算例的车队规模 K（以 m=1 基线为准；truck1 记录的是 K+1）
    K = base.groupby("set")["n_trucks"].first()

    print("=" * 78)
    print("等资本对照：三臂的运营成本（每算例 10 种子最小值 / 配对均值）")
    print("=" * 78)
    rows = []
    for sid in sorted(base["set"].unique()):
        b = base[base["set"] == sid]
        rec = {
            "set": sid,
            "n": int(b["n_customers"].iloc[0]),
            "K": int(K[sid]),
            "m1": b["best_cost"].min(),
        }
        for nm, d in arms.items():
            if sid not in set(d["set"]):
                continue
            t = d[d["set"] == sid]
            x = t.merge(b, on=["set", "seed"], suffixes=("_t", "_b"))
            rec[f"{nm}_min"] = t["best_cost"].min()
            rec[f"{nm}_pair%"] = (x["best_cost_t"] / x["best_cost_b"] - 1).mean() * 100
        rows.append(rec)
    tab = pd.DataFrame(rows).set_index("set")
    cols = ["n", "K", "m1"]
    for nm in arms:
        if f"{nm}_min" in tab:
            cols += [f"{nm}_min", f"{nm}_pair%"]
    print(tab[cols].round(2).to_string())

    print("\n" + "=" * 78)
    print("汇总（按算例求和后比较；配对均值是逐种子差的中位数口径）")
    print("=" * 78)
    for nm, d in arms.items():
        if f"{nm}_min" not in tab or f"{nm}_pair%" not in tab:
            continue
        # 逐种子配对：所有 (set, seed) 一起算
        x = d.merge(base, on=["set", "seed"], suffixes=("_t", "_b"))
        delta = x["best_cost_t"] / x["best_cost_b"] - 1
        n_better = int((delta < 0).sum())
        n_worse = int((delta > 0).sum())
        sum_t = tab[f"{nm}_min"].sum()
        sum_b = tab["m1"].sum()
        print(f"  {nm:7s} 配对均值 {delta.mean()*100:+6.2f}% | 中位 {delta.median()*100:+6.2f}% | "
              f"更好 {n_better:3d} / 更差 {n_worse:3d} | 算例求和 {sum_t:8.1f} vs {sum_b:8.1f} "
              f"({(sum_t/sum_b-1)*100:+5.2f}%) | 算例级最优更好 "
              f"{int((tab[f'{nm}_min'] < tab['m1']).sum())}/{len(tab)}")

    print("\n" + "=" * 78)
    print("盈亏平衡：额外投入每天最多值多少钱（c* = 运营成本节省 / 额外设备数）")
    print("=" * 78)
    print("  m2   ：每车多 1 架无人机，额外设备 = K 架")
    print("  truck1：多 1 辆车 + 1 名司机（司机成本已在运营成本里，故 c* 只针对车辆）")
    for nm, per_unit in (("m2", K), ("truck1", 1)):
        if f"{nm}_min" not in tab:
            continue
        save = tab["m1"] - tab[f"{nm}_min"]          # 正 = 省钱
        if nm == "truck1":
            unit = pd.Series(1, index=tab.index)
        else:
            unit = per_unit
        cstar = save / unit
        print(f"\n  [{nm}]")
        print(f"    总体：节省 {save.sum():+7.1f} $/班次，额外设备 {unit.sum():.0f} 台，"
              f"c* = {save.sum()/unit.sum():+.2f} $/台/天")
        print(f"    分算例 c*：中位 {cstar.median():+.2f}，范围 {cstar.min():+.2f} ~ {cstar.max():+.2f}")
        pos = (save > 0).sum()
        print(f"    省钱的算例 {pos}/{len(tab)}")
        print("    " + "  ".join(f"Set{s}:{v:+.1f}" for s, v in cstar.items()))

    print("\n" + "=" * 78)
    print("拐点检验：m=3 相对 m=2 是否还有额外收益")
    print("=" * 78)
    if "m3" in arms and "m2" in arms:
        x = arms["m3"].merge(arms["m2"], on=["set", "seed"], suffixes=("_3", "_2"))
        dd = x["best_cost_3"] / x["best_cost_2"] - 1
        print(f"  m3 vs m2：配对均值 {dd.mean()*100:+.2f}% | 中位 {dd.median()*100:+.2f}% | "
              f"更好 {int((dd<0).sum())} / 更差 {int((dd>0).sum())}")
        per = x.groupby("set").apply(
            lambda g: (g["best_cost_3"].min() / g["best_cost_2"].min() - 1) * 100, include_groups=False
        )
        print(f"  算例级最优：m3 更好 {int((per < 0).sum())}/{len(per)} 个算例")
        print("  " + "  ".join(f"Set{s}:{v:+.1f}" for s, v in per.items()))
        print("  读法：若 m3 不优于 m2，说明收益来自「并行能力」而非「多一架飞机」——")
        print("        这是挡住「你只是加了运力」这一质疑的第一层证据。")

    # ---- 时效轴 ----
    ms = {}
    for nm, f in MS_ARMS.items():
        path = RESULTS / f
        if path.exists():
            ms[nm] = pd.read_csv(path)
    if len(ms) == 2:
        print("\n" + "=" * 78)
        print("第二条价值轴：时效（makespan = 最晚一辆车收车的时刻）")
        print("=" * 78)
        x = ms["m2"].merge(ms["m1"], on=["set", "seed"], suffixes=("_2", "_1"))
        g = x.groupby("set").agg(
            n=("n_customers_1", "first"),
            m1_h=("makespan_s_1", lambda v: v.min() / 3600),
            m2_h=("makespan_s_2", lambda v: v.min() / 3600),
            pair=("makespan_s_2", "size"),
        )
        x["d%"] = (x["makespan_s_2"] / x["makespan_s_1"] - 1) * 100
        g["pair%"] = x.groupby("set")["d%"].mean()
        g["min%"] = (g["m2_h"] / g["m1_h"] - 1) * 100
        print(g[["n", "m1_h", "m2_h", "min%", "pair%"]].round(2).to_string())
        d = x["d%"]
        print(f"\n  全算例：配对均值 {d.mean():+.2f}% | 中位 {d.median():+.2f}% | "
              f"更短 {int((d<0).sum())} / 更长 {int((d>0).sum())} | "
              f"算例级更短 {int((g['min%'] < 0).sum())}/{len(g)}")
        # 同一批解里，成本降幅与时效降幅的关系
        print("  读法：成本 = 司机工时 × 费率，若 makespan 降幅明显大于成本降幅，")
        print("        说明 m=2 把「总工时」摊到了更多车上——成本只降一点，时效降很多。")

    print("\n" + "-" * 78)
    print("读法：c* 是「每架无人机每天最多值多少钱」的临界值。")
    print("      真实无人机按 2 万元 / 1000 个班次摊销 ≈ 20 $/天，若 c* 显著低于该值，")
    print("      说明纯成本口径下 m=2 回不了本——价值必须记在时效/运力上（见上一节）。")


if __name__ == "__main__":
    main()
