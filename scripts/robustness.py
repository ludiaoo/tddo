"""单参数敏感性总表：换电耗时 / 悬停功率 / 电池库存。

三组实验都是「同一批算例、同种子、只拧一个旋钮」，所以放在一个脚本里出表，
避免三份各自为政的小脚本。每个小节都能单独运行（缺文件的就跳过）。

| 小节 | 旋钮 | 回答什么 |
|---|---|---|
| A | `fleet.swap_s` = 0 / 120 / 300 s | 换电降 2.3% 是不是吃了「换电不要时间」的便宜 |
| B | `energy.hover_power_kw` = 0.45 / 1.10 / 1.80 kW | C9 的「不成立」是单点判断还是区间结论 |
| C | `fleet.spare_batteries` = 0 / 1 / 2 / 3 / ∞ | 「每车 ≥2 块电池」这条建议的饱和点在哪 |

用法：
    python -m scripts.robustness
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.io import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"


def _load(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists():
        return None
    d = pd.read_csv(p)
    cols = [c for c in ("set", "seed", "best_cost", "n_sorties") if c in d.columns]
    return d[d["iters"] == 2000][cols]


def _pair(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    m = a.merge(b, on=["set", "seed"], suffixes=("_a", "_b"))
    return (m["best_cost_a"] - m["best_cost_b"]) / m["best_cost_b"] * 100.0


def _line(d: pd.Series) -> str:
    k = int((d < 0).sum())
    return f"{d.mean():+7.2f}%  [{k:2d} 好 / {len(d)-k:2d} 差]  中位 {d.median():+6.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="")
    args = ap.parse_args()
    want = {int(x) for x in args.sets.split(",")} if args.sets.strip() else None

    def F(df):
        return df if (df is None or want is None) else df[df["set"].isin(want)]

    base_full = _load("bench_main.csv")
    hs_full = _load("bench_hotswap.csv")
    if base_full is None or hs_full is None:
        raise SystemExit("缺少 bench_main / bench_hotswap")

    # 全部对比统一到「6 算例 × 5 种子」这 30 个配对上：bench_main / bench_hotswap 是 16×10，
    # 直接拿来比会混样本（本项目踩过四次的口径错误）。
    keys = _load("bench_e_b0.csv")
    if keys is None:
        raise SystemExit("缺少 bench_e_b0.csv（用来确定 30 个配对）")
    kk = keys[["set", "seed"]]
    base = F(base_full.merge(kk, on=["set", "seed"]))
    hs0 = F(hs_full.merge(kk, on=["set", "seed"]))

    rows: list[dict] = []

    # ---------- A. 换电耗时 ----------
    print("# 一、换电耗时（`fleet.swap_s`）\n")
    print(f"{'换电耗时':<12}{'换电 vs 基准':<46}{'相对 swap=0 的换电臂'}")
    print("（全部对比统一到 6 算例 × 5 种子的 30 个配对；bench_main/hotswap 的 16×10 不直接参与）")
    print(f"{'0 s（现行）':<10}{_line(_pair(hs0, base)):<44}{'—'}")
    for sw, tag in ((120, "sw120_hs"), (300, "sw300_hs")):
        a = F(_load(f"bench_{tag}.csv"))
        if a is None:
            print(f"{sw:>4} s       （缺结果）")
            continue
        d_benefit = _pair(a, base)
        d_self = _pair(a, hs0)
        print(f"{sw:>4} s      {_line(d_benefit):<44}{_line(d_self)}")
        rows.append({"knob": "swap_s", "value": sw, "benefit_pct": round(d_benefit.mean(), 3),
                     "benefit_n_better": int((d_benefit < 0).sum()), "n": len(d_benefit),
                     "vs_zero_pct": round(d_self.mean(), 3)})

    # ---------- B. 悬停功率 ----------
    print("\n# 二、悬停功率（`energy.hover_power_kw`）\n")
    print("B1. C9：跨段锚点的边际收益（m=2 臂：m2_span3 vs m2）")
    c9 = {1.10: None}
    a = F(_load("bench_b_m2_span3.csv")); b = F(_load("bench_b_m2.csv"))
    if a is not None and b is not None:
        c9[1.10] = _pair(a, b)
        print(f"  1.10 kW（基准）  {_line(c9[1.10])}")
    for kw, tag in ((0.45, "h45"), (1.80, "h18")):
        a = F(_load(f"bench_{tag}_m2span3.csv")); b = F(_load(f"bench_{tag}_m2.csv"))
        if a is None or b is None:
            print(f"  {kw:.2f} kW          （缺结果）")
            continue
        d = _pair(a, b)
        c9[kw] = d
        print(f"  {kw:.2f} kW          {_line(d)}")
        rows.append({"knob": "hover_m2span3", "value": kw, "benefit_pct": round(d.mean(), 3),
                     "benefit_n_better": int((d < 0).sum()), "n": len(d)})

    print("\nB2. C2：换电 vs 基准（同一悬停功率下）")
    for kw, tag in ((0.45, "h45"), (1.10, "b"), (1.80, "h18")):
        a = F(_load(f"bench_{tag}_hs.csv")); b = F(_load(f"bench_{tag}_main.csv"))
        if a is None or b is None:
            continue
        d = _pair(a, b)
        print(f"  {kw:.2f} kW          {_line(d)}")
        rows.append({"knob": "hover_hs", "value": kw, "benefit_pct": round(d.mean(), 3),
                     "benefit_n_better": int((d < 0).sum()), "n": len(d)})

    print("\nB3. 跨段单独用（m=1 臂：span3 vs span1）")
    a = F(_load("bench_h45_span3.csv")); b = F(_load("bench_h45_main.csv"))
    if a is not None and b is not None:
        print(f"  0.45 kW          {_line(_pair(a, b))}")
    a = F(_load("bench_b_span3.csv")); b = F(_load("bench_b_main.csv"))
    if a is not None and b is not None:
        print(f"  1.10 kW          {_line(_pair(a, b))}")

    # ---------- C. 电池库存 ----------
    print("\n# 三、电池库存（`fleet.spare_batteries`）\n")
    for b_, tag in ((0, "e_b0"), (1, "e_b1"), (2, "e_b2"), (3, "e_b3")):
        a = F(_load(f"bench_{tag}_hs.csv")); c = F(_load(f"bench_{tag}.csv"))
        if a is None or c is None:
            continue
        d = _pair(a, c)                              # 同库存下：换电 vs 不换电
        cost = _pair(c, base)                        # 该库存的基准臂相对「不限」
        print(f"b={b_}: 换电 {_line(d)}   基准臂成本 {cost.mean():+6.2f}%")
        rows.append({"knob": "battery_hs", "value": b_, "benefit_pct": round(d.mean(), 3),
                     "benefit_n_better": int((d < 0).sum()), "n": len(d),
                     "base_cost_pct": round(cost.mean(), 3)})
    print(f"b=∞: 换电 {_line(_pair(hs0, base))}   基准臂成本  +0.00%")

    out = RESULTS / "robustness.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
