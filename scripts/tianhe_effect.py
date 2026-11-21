"""第 1 项：把主结论搬到天河真实路网重跑。

## 为什么要做

`M9.23 §8 第 5 条` 写着一句难看的实话：**15.9%、1.5%–3.4%、换电 −2.26%、多机 −2.94%
这些数字全部来自西雅图算例**，天河路网的卡车距离比西雅图绕 20%–40%（卡车÷无人机
1.68–1.95 vs 1.40），换过去结论还在不在，没测过就不能说。

这个脚本只做一件事：在天河算例（set101–104，10/25/50/100 客户）上把同一批对照臂跑一遍，
按 (set, seed) 配对给出结论，并与西雅图的同口径数字并排放，标清样本不可混用。

## 臂

| 标签 | 参数 | 对应结论 |
|---|---|---|
| `base` | 默认（m=1、相邻锚点、悬停免费） | 节省率基线 |
| `hs` | `--hotswap` | C2 同节点换电 |
| `m2` | `--drones-per-truck 2` | C4 一车两机 |
| `span3` | `--sortie-span 3` | C9 跨段单独用 |
| `m2span3` | `--drones-per-truck 2 --sortie-span 3` | C9 跨段 + 多机 |
| `hov_*` | 以上四条再加 `--hover-power 1.1` | C13 悬停计电下的分层 |

全部 2 000 轮、同种子配对（`--seeds 10`），所以差就是参数本身的差。

用法：
    python -m scripts.tianhe_effect
"""

from __future__ import annotations

import argparse
from math import comb

import pandas as pd

from src.io import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"

# 天河：免费悬停 / 计电
FREE = {
    "base": "bench_th_base.csv",
    "hs": "bench_th_hs.csv",
    "m2": "bench_th_m2.csv",
    "span3": "bench_th_span3.csv",
    "m2span3": "bench_th_m2span3.csv",
}
HOVER = {
    "base": "bench_th_hov_base.csv",
    "hs": "bench_th_hov_hs.csv",
    "m2": "bench_th_hov_m2.csv",
    "m2span3": "bench_th_hov_m2span3.csv",
}
# 西雅图同口径（仅用于并排对照，样本不同）
SEATTLE = {
    "base": "bench_main.csv",
    "hs": "bench_hotswap.csv",
    "m2": "bench_m2.csv",
    "span3": "bench_m1_span3.csv",
    "m2span3": "bench_m2_span3.csv",
}
SEATTLE_HOVER = {
    "base": "bench_b_main.csv",
    "hs": "bench_b_hs.csv",
    "m2": "bench_b_m2.csv",
    "m2span3": "bench_b_m2_span3.csv",
}
PAIRS = [
    ("hs_vs_base", "同节点换电 vs 基准", "hs", "base"),
    ("m2_vs_base", "一车两机 vs 基准", "m2", "base"),
    ("span3_vs_base", "跨段单独用 vs 基准", "span3", "base"),
    ("m2span3_vs_m2", "跨段边际（m=2 臂内）", "m2span3", "m2"),
]


def _load(name: str, iters: int = 2000) -> pd.DataFrame | None:
    path = RESULTS / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["iters"] == iters] if "iters" in df.columns else df
    if df.empty:
        return None
    cols = ["set", "seed", "best_cost", "initial_cost", "saving_pct", "n_trucks",
            "n_sorties", "drone_served", "cpu_s"]
    cols = [c for c in cols if c in df.columns]
    return df[cols]


def _sign_p(k: int, n: int) -> float:
    """双侧精确符号检验（并列已剔除）。"""
    if n <= 0:
        return 1.0
    k = min(k, n - k)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    m = a.merge(b, on=["set", "seed"], suffixes=("_a", "_b"))
    m["delta_pct"] = (m["best_cost_a"] - m["best_cost_b"]) / m["best_cost_b"] * 100.0
    return m


def _fmt(m: pd.DataFrame) -> str:
    d = m["delta_pct"]
    better = int((d < -1e-9).sum())
    worse = int((d > 1e-9).sum())
    p = _sign_p(better, better + worse)
    return (f"{d.mean():+6.2f}%  [{better:2d} 好/{worse:2d} 差/{len(d)-better-worse:2d} 平]  "
            f"中位 {d.median():+6.2f}%  p={p:.3g}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()

    free = {k: _load(v, args.iters) for k, v in FREE.items()}
    hov = {k: _load(v, args.iters) for k, v in HOVER.items()}
    if free["base"] is None:
        raise SystemExit("缺少 bench_th_base.csv（先跑 scripts.bench --exp th_base）")

    rows: list[dict] = []

    # ---------------- 表 A：求解质量
    print("# 第 1 项：主结论在天河算例上的重跑\n")
    print("## 表 A：求解质量（默认臂）\n")
    b = free["base"]
    per = b.groupby("set").agg(n=("saving_pct", "size"), 节省率=("saving_pct", "mean"),
                               初始=("initial_cost", "mean"), 最终=("best_cost", "mean"),
                               CPU=("cpu_s", "mean"))
    print(per.round(2).to_string())
    print(f"\n合计 {len(b)} 次求解：节省率均值 {b['saving_pct'].mean():.2f}%  "
          f"中位 {b['saving_pct'].median():.2f}%  区间 "
          f"{b['saving_pct'].min():.1f}%–{b['saving_pct'].max():.1f}%")

    # ---------------- 表 B：天河上的四组配对（免费悬停）
    print("\n## 表 B：天河上的结论（悬停免费，配对 n = 4 算例 × 10 种子）\n")
    print("（天河 4 算例 × 10 种子 = 40 配对；西雅图 16 算例 × 10 种子 = 160 配对。"
          "两边都用自己的全量，只比方向与量级）\n")
    print(f"{'对照':<26}{'天河 n=40':<62}{'西雅图 n=160（同口径，样本不同）'}")
    for tag, label, num, den in PAIRS:
        if free.get(num) is None or free.get(den) is None:
            print(f"{label:<26}（缺 {FREE[num]}，跳过）")
            continue
        m = _paired(free[num], free[den])
        seat = ""
        s_num, s_den = SEATTLE.get(num), SEATTLE.get(den)
        if s_num and s_den and _load(s_num) is not None and _load(s_den) is not None:
            ms = _paired(_load(s_num), _load(s_den))
            seat = _fmt(ms)
        print(f"{label:<26}{_fmt(m):<62}{seat}")
        rows.append({"table": "B", "comparison": tag, "label": label,
                     "n": len(m), "mean_pct": round(m["delta_pct"].mean(), 3),
                     "better": int((m["delta_pct"] < -1e-9).sum()),
                     "worse": int((m["delta_pct"] > 1e-9).sum()),
                     "p": round(_sign_p(int((m["delta_pct"] < -1e-9).sum()),
                                        len(m) - int((m["delta_pct"].abs() < 1e-9).sum())), 5)})

    # ---------------- 表 C：悬停计电下同一批配对
    if any(v is not None for v in hov.values()):
        print("\n## 表 C：悬停计电 1.1 kW 后\n")
        for tag, label, num, den in PAIRS:
            if hov.get(num) is None or hov.get(den) is None:
                print(f"{label:<26}（缺计电臂，跳过）")
                continue
            m = _paired(hov[num], hov[den])
            print(f"{label:<26}{_fmt(m)}")
            rows.append({"table": "C", "comparison": tag, "label": label,
                         "n": len(m), "mean_pct": round(m["delta_pct"].mean(), 3),
                         "better": int((m["delta_pct"] < -1e-9).sum()),
                         "worse": int((m["delta_pct"] > 1e-9).sum()),
                         "p": round(_sign_p(int((m["delta_pct"] < -1e-9).sum()),
                                            len(m) - int((m["delta_pct"].abs() < 1e-9).sum())), 5)})

        print("\n## 表 D：悬停计价本身让各臂贵了多少（同臂同种子，只换口径）\n")
        for k in ("base", "hs", "m2", "m2span3"):
            if free.get(k) is None or hov.get(k) is None:
                continue
            m = _paired(hov[k], free[k])
            extra = ""
            if "hover_min_a" in m.columns:
                extra = f"  悬停 {m['hover_min_a'].mean():5.2f} min"
            print(f"{k:<10}{_fmt(m)}{extra}")
            rows.append({"table": "D", "comparison": f"price_{k}", "label": f"{k}：计电 vs 免费",
                         "n": len(m), "mean_pct": round(m["delta_pct"].mean(), 3),
                         "better": int((m["delta_pct"] < -1e-9).sum()),
                         "worse": int((m["delta_pct"] > 1e-9).sum()),
                         "p": round(_sign_p(int((m["delta_pct"] < -1e-9).sum()),
                                            len(m) - int((m["delta_pct"].abs() < 1e-9).sum())), 5)})

    # ---------------- 结构变化
    print("\n## 表 E：换电/多机带来的结构变化（均值）\n")
    for k in ("base", "hs", "m2", "m2span3"):
        if free.get(k) is None:
            continue
        d = free[k]
        print(f"{k:<10}卡车 {d['n_trucks'].mean():5.2f}  架次 {d['n_sorties'].mean():5.2f}  "
              f"无人机服务 {d['drone_served'].mean():5.2f}  CPU {d['cpu_s'].mean():6.1f} s")

    out = RESULTS / "tianhe_effect.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
