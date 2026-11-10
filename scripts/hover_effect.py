"""B 项：悬停能耗计价的效应（C9 的补验）。

## 为什么要做

C9（跨段锚点 `SORTIE_SPAN>1`）在 M9.15 里是"与多机配对才有收益"的结论，但那条结论
是在**悬停免费**的口径下得到的：论文式(39) 只按飞行距离计电，无人机在回收点上方
等卡车的那 8.7 → 57.9 min 完全不花钱、也不占电量。

跨段锚点把悬停时长抬高 6.7 倍，所以它的收益有多少是"真收益"、多少是"建模漏洞套利"，
必须把悬停计价后重跑才知道。这个脚本就是那张对照表。

## 四条臂的两次跑法

| 臂 | 免费悬停（M9.15 已有） | 计电（本次重跑） |
|---|---|---|
| `span1` | `bench_main.csv` | `bench_b_main.csv` |
| `span3` | `bench_m1_span3.csv` | `bench_b_span3.csv` |
| `hs` | `bench_hotswap.csv` | `bench_b_hs.csv` |
| `hs+span3` | `bench_hs_span3.csv` | `bench_b_hs_span3.csv` |
| `m2` | `bench_m2.csv` | `bench_b_m2.csv` |
| `m2+span3` | `bench_m2_span3.csv` | `bench_b_m2_span3.csv` |

全部 2 000 轮、同种子配对（`--seeds 5`，seed 0–4），所以两次跑法的差就是**悬停口径**的差。

用法：
    python -m scripts.hover_effect
    python -m scripts.hover_effect --hover-power 1.1
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.io import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"

# (标签, 说明, 免费悬停的臂, 计电的同一臂)
FREE = {
    "span1": "bench_main.csv",
    "span3": "bench_m1_span3.csv",
    "hs": "bench_hotswap.csv",
    "hs_span3": "bench_hs_span3.csv",
    "m2": "bench_m2.csv",
    "m2_span3": "bench_m2_span3.csv",
}
HOVER = {
    "span1": "bench_b_main.csv",
    "span3": "bench_b_span3.csv",
    "hs": "bench_b_hs.csv",
    "hs_span3": "bench_b_hs_span3.csv",
    "m2": "bench_b_m2.csv",
    "m2_span3": "bench_b_m2_span3.csv",
}
# 要回答的两个问题各用哪一对臂
PAIRS = [
    ("span3_vs_span1", "m=1：跨段锚点 vs 相邻锚点", "span3", "span1"),
    ("hsspan3_vs_hs", "换电：跨段锚点 vs 相邻锚点", "hs_span3", "hs"),
    ("m2span3_vs_m2", "m=2：跨段锚点 vs 相邻锚点", "m2_span3", "m2"),
]


def _load(name: str) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["iters"] == 2000]
    return df[["set", "seed", "best_cost", "saving_pct", "n_customers"] +
              (["hover_min", "hover_kwh"] if "hover_min" in df.columns else [])]


def _paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """按 (set, seed) 配对，返回相对差（负 = 前者更好）。"""
    m = a.merge(b, on=["set", "seed"], suffixes=("_a", "_b"))
    m["delta_pct"] = (m["best_cost_a"] - m["best_cost_b"]) / m["best_cost_b"] * 100.0
    return m


def _fmt(m: pd.DataFrame) -> str:
    d = m["delta_pct"]
    n_better = int((d < 0).sum())
    n_worse = int((d > 0).sum())
    return (
        f"{d.mean():+7.2f}%  "
        f"[{n_better:2d} 好 / {n_worse:2d} 差 / {len(d)-n_better-n_worse:2d} 平]  "
        f"中位 {d.median():+6.2f}%  最差 {d.max():+6.2f}%"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hover-power", type=float, default=1.1, help="本次计电采用的悬停功率 kW")
    args = ap.parse_args()

    free = {k: _load(v) for k, v in FREE.items()}
    hov = {k: _load(v) for k, v in HOVER.items()}
    missing = [HOVER[k] for k, v in hov.items() if v is None]
    if missing:
        raise SystemExit(f"缺少计电臂结果：{missing}（先跑 scripts.bench --exp b_* --hover-power {args.hover_power}）")

    print(f"# B 项：悬停能耗计价（{args.hover_power} kW = b·v 的 {args.hover_power/1.30:.2f} 倍）\n")
    print("## 表 A：跨段锚点的收益，在两种悬停口径下\n")
    print(f"{'对照':<28}{'① 悬停免费':<62}{'② 悬停计电':<62}")
    rows = []
    for tag, label, num, den in PAIRS:
        if free[num] is None or free[den] is None:
            print(f"{label:<28}（缺免费悬停口径的结果，跳过）")
            continue
        m1 = _paired(free[num], free[den])
        m2 = _paired(hov[num], hov[den])
        print(f"{label:<28}{_fmt(m1):<58}{_fmt(m2):<58}")
        rows.append({
            "comparison": tag, "label": label,
            "free_mean_pct": round(m1["delta_pct"].mean(), 3),
            "free_n_better": int((m1["delta_pct"] < 0).sum()),
            "hover_mean_pct": round(m2["delta_pct"].mean(), 3),
            "hover_n_better": int((m2["delta_pct"] < 0).sum()),
            "n_pairs": len(m2),
        })

    print("\n## 表 B：悬停计价本身让各臂贵了多少（同臂、同种子，只换口径）\n")
    for k in ("span1", "span3", "hs", "hs_span3", "m2", "m2_span3"):
        if free[k] is None or hov[k] is None:
            continue
        m = _paired(hov[k], free[k])
        extra = ""
        col = "hover_min_a" if "hover_min_a" in m.columns else ("hover_min" if "hover_min" in m.columns else None)
        if col:
            extra = f"  悬停时长 均值 {m[col].mean():5.2f} min"
        print(f"{k:<10}{_fmt(m)}{extra}")
        rows.append({
            "comparison": f"price_{k}", "label": f"{k}：计电 vs 免费",
            "free_mean_pct": 0.0, "free_n_better": 0,
            "hover_mean_pct": round(m["delta_pct"].mean(), 3),
            "hover_n_better": int((m["delta_pct"] < 0).sum()),
            "n_pairs": len(m),
        })

    print("\n## 表 C：悬停时长分布（计电口径下的最优解）\n")
    for k in ("span1", "span3", "hs", "hs_span3", "m2", "m2_span3"):
        df = hov[k]
        if df is None or "hover_min" not in df.columns:
            continue
        h = df["hover_min"]
        print(
            f"{k:<10}均值 {h.mean():6.2f} min  中位 {h.median():6.2f}  "
            f"90 分位 {np.quantile(h, 0.9):6.2f}  最大 {h.max():6.2f}  "
            f"非零占比 {100.0*(h>1e-9).mean():5.1f}%"
        )

    out = RESULTS / "hover_effect.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
