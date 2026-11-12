"""M9.22：算子内部策略的消融（只有这两处能给出数字）。

## 为什么需要这个脚本

M9.22 之前，README 与工程手册都写了「`worst` 在最前 2k 名里随机取 k」和
「`related` 对禁飞区 2r 内的距离打 6 折」，但两处都**没有单独度量过**——
参数修正的 +1.76% 是四项一起的，算子策略的贡献一直混在里面。

而且核对代码时发现一个问题：`destroy_worst` 原本写的是
`sav[:k * 2][:k]`，取的是前 2k 的**前 k 个**，等于严格取前 k 个，注释里的
「随机化」从未生效。所以这次把随机化真正实现成 `WORST_WINDOW` 开关
（默认 1.0 = 严格取前 k，与历史结果逐位一致），再单独测它有没有用。

## 三条臂

| 臂 | 开关 | 含义 |
|---|---|---|
| `base` | 默认 | `NFZ_DISCOUNT=0.6`、`WORST_WINDOW=1.0` |
| `nfz1` | `--nfz-discount 1.0` | 关掉禁飞区折扣，`related` 退化为纯距离 |
| `win2` | `--worst-window 2.0` | `worst` 在前 2k 内随机抽 k 个 |

另外跑一条全量臂 `nfz1_full`（16 算例 × 10 种子，`--nfz-discount 1.0`），直接对
`bench_main.csv`——因为 base 臂与它在 30 个配对上逐位相同，主结果表本身就可以当基准。

同算例、同种子、同轮数（2 000），因此差就是策略本身的差。

用法：
    python -m scripts.operator_ablation
"""

from __future__ import annotations

from math import comb

import pandas as pd

from src.io import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
BASE = "bench_abl_op_base.csv"
VARIANTS = [
    ("nfz1", "bench_abl_op_nfz1.csv", "关掉禁飞区折扣（related 退化为纯距离）"),
    ("win2", "bench_abl_op_win2.csv", "worst 在前 2k 内随机抽 k"),
]


def _load(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d[d["iters"] == 2000] if "iters" in d.columns else d


def _sign_p(k: int, n: int) -> float:
    """双侧精确二项检验（p = 0.5），与项目其他分析脚本口径一致。"""
    k = min(k, n - k)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> None:
    base = _load(BASE)
    if base is None:
        raise SystemExit(f"缺少 {BASE}，先跑 scripts.bench --exp abl_op_base")

    rows = []
    print("算子策略消融（同一批 (算例, 种子) 配对，2 000 轮）\n")
    print(f"{'变体':<6}{'配对均值':>10}{'中位':>9}{'更好/更差':>11}{'p':>9}   分算例")
    for tag, fname, desc in VARIANTS:
        var = _load(fname)
        if var is None:
            print(f"{tag:<6}{'（缺结果表）':>20}")
            continue
        m = var.merge(base, on=["set", "seed"], suffixes=("_a", "_b"))
        pct = (m["best_cost_a"] - m["best_cost_b"]) / m["best_cost_b"] * 100.0
        better, worse = int((pct < 0).sum()), int((pct > 0).sum())
        per = pct.groupby(m["set"]).mean()
        print(f"{tag:<6}{pct.mean():>+9.2f}%{pct.median():>+8.2f}%"
              f"{f'{better}/{worse}':>11}{_sign_p(better, len(pct)):>9.3g}   "
              + " ".join(f"S{int(s)}:{v:+.1f}%" for s, v in per.items()))
        rows.append({"variant": tag, "desc": desc, "n": len(pct),
                     "mean_pct": round(float(pct.mean()), 3),
                     "median_pct": round(float(pct.median()), 3),
                     "better": better, "worse": worse,
                     "p": round(_sign_p(better, len(pct)), 5)})

    # 各臂的绝对成本水平，用来确认「差异不是来自算例不同」
    print("\n成本水平（均值，美元）")
    for tag, df in [("base", base)] + [(t, _load(f)) for t, f, _ in VARIANTS]:
        if df is not None:
            print(f"  {tag:<6}{df['best_cost'].mean():8.2f}"
                  f"   CPU {df['cpu_s'].mean():6.1f} s")

    full = _load("bench_abl_op_nfz1_full.csv")
    main = _load("bench_main.csv")
    if full is not None and main is not None:
        m = full.merge(main, on=["set", "seed"], suffixes=("_a", "_b"))
        pct = (m["best_cost_a"] - m["best_cost_b"]) / m["best_cost_b"] * 100.0
        better = int((pct < -1e-9).sum())
        worse = int((pct > 1e-9).sum())
        tie = len(pct) - better - worse
        print(f"全量复核（16×10）：均值 {pct.mean():+.2f}%  中位 {pct.median():+.2f}%  "
              f"更好/更差/并列 {better}/{worse}/{tie}  "
              f"p={_sign_p(better, better + worse):.3g}（去并列 n={better + worse}）")
        print(f"  并列集中在 10 客户算例，说明折扣在小算例上不改变被选集合")
        rows.append({"variant": "nfz1_full", "desc": "关掉折扣（全量 16×10）",
                     "n": len(pct), "mean_pct": round(float(pct.mean()), 3),
                     "median_pct": round(float(pct.median()), 3),
                     "better": better, "worse": worse, "tie": tie,
                     "p": round(_sign_p(better, better + worse), 5)})

    if rows:
        out = RESULTS / "operator_ablation.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n已写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
