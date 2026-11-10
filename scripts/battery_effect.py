"""E 项：电池库存对「换电」收益的影响（W2 最小版的量化）。

## 问题

论文全篇隐含一个假设：**每飞完一个架次就凭空换上一块满电电池**（无限电池、零重量）。
本项目原来的 `allow_hotswap` 沿用了它——落地就能立刻再起飞，不要电池也不要时间。

现实中一辆车顶多带几块备用电池，于是出现一个很自然的追问：

> 换电降的那 2.4%，需要车上带几块电池才吃得到？只带 1 块呢？不带呢？

## 臂

| 臂 | 备用电池 b | 一条路线最多几个架次 |
|---|---|---|
| `e_b0` / `e_b0_hs` | 0 | 1（= b + m，m=1） |
| `e_b1` / `e_b1_hs` | 1 | 2 |
| `e_b2` / `e_b2_hs` | 2 | 3 |
| `bench_main` / `bench_hotswap` | 不限（-1） | 不限 |

`_hs` = 打开换电。重量项在这一轮**关掉**（`--battery-mass 0`），只留数量约束，
这样曲线只反映"手头有几块电池"，不掺重量。

用法：
    python -m scripts.battery_effect
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.io import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
INF_ARM = {"base": "bench_main.csv", "hs": "bench_hotswap.csv"}
LIM_ARM = {
    0: {"base": "bench_e_b0.csv", "hs": "bench_e_b0_hs.csv"},
    1: {"base": "bench_e_b1.csv", "hs": "bench_e_b1_hs.csv"},
    2: {"base": "bench_e_b2.csv", "hs": "bench_e_b2_hs.csv"},
}


def _load(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d["iters"] == 2000]
    cols = [c for c in ("set", "seed", "best_cost", "n_sorties", "drone_served") if c in d.columns]
    return d[cols]


def _pair(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    m = a.merge(b, on=["set", "seed"], suffixes=("_a", "_b"))
    return (m["best_cost_a"] - m["best_cost_b"]) / m["best_cost_b"] * 100.0


def _line(d: pd.Series) -> str:
    k = int((d < 0).sum())
    return (f"{d.mean():+7.2f}%  [{k:2d} 好 / {len(d)-k:2d} 差]  "
            f"中位 {d.median():+6.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="", help="只统计这些算例")
    args = ap.parse_args()

    want = {int(x) for x in args.sets.split(",")} if args.sets.strip() else None

    def _filter(df):
        return df if (df is None or want is None) else df[df["set"].isin(want)]

    inf = {k: _filter(_load(v)) for k, v in INF_ARM.items()}
    lim = {b: {k: _filter(_load(v)) for k, v in arms.items()} for b, arms in LIM_ARM.items()}
    if any(v is None for v in inf.values()):
        raise SystemExit("缺少 bench_main / bench_hotswap")

    print("# E 项：电池库存 vs 换电收益（重量项已关闭）\n")
    print("## 表 A：不同电池库存下的成本（相对「不限电池」的基准臂）\n")
    print(f"{'备用电池':<10}{'基准臂':<44}{'换电臂':<44}")
    for b, arms in lim.items():
        if arms["base"] is None:
            print(f"b={b:<8}（缺结果）")
            continue
        d0 = _pair(arms["base"], inf["base"])
        d1 = _pair(arms["hs"], inf["hs"]) if arms["hs"] is not None else None
        print(f"b={b:<8}{_line(d0):<40}" + (_line(d1) if d1 is not None else "—"))
    print(f"{'不限':<10}{0.0:+7.2f}%  （基线）" + " " * 28 + f"{_line(_pair(inf['hs'], inf['base']))}")

    print("\n## 表 B：同一电池库存下，「换电 vs 不换电」的收益 —— 这条就是要的曲线\n")
    rows = []
    for b, arms in lim.items():
        if arms["hs"] is None or arms["base"] is None:
            continue
        d = _pair(arms["hs"], arms["base"])
        rows.append({"b": b, "mean_pct": round(d.mean(), 3),
                     "median_pct": round(d.median(), 3),
                     "n_better": int((d < 0).sum()), "n": len(d)})
        print(f"b={b:<3} {_line(d)}")
    d = _pair(inf["hs"], inf["base"])
    rows.append({"b": -1, "mean_pct": round(d.mean(), 3), "median_pct": round(d.median(), 3),
                 "n_better": int((d < 0).sum()), "n": len(d)})
    print(f"不限  {_line(d)}")

    print("\n## 表 C：架次数与无人机服务客户数（均值）\n")
    for lab, df in [("b=0 基准", lim[0]["base"]), ("b=0 换电", lim[0]["hs"]),
                    ("b=1 基准", lim[1]["base"]), ("b=1 换电", lim[1]["hs"]),
                    ("b=2 基准", lim[2]["base"]), ("b=2 换电", lim[2]["hs"]),
                    ("不限 基准", inf["base"]), ("不限 换电", inf["hs"])]:
        if df is None:
            continue
        ns = f"{df['n_sorties'].mean():5.2f}" if "n_sorties" in df else "  n/a"
        ds = f"{df['drone_served'].mean():5.2f}" if "drone_served" in df else "  n/a"
        print(f"{lab:<10}架次 {ns}  无人机服务 {ds}")

    out = RESULTS / "battery_effect.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
