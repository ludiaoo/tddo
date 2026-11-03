"""汇总全部并行实验，产出 Table 3（论文同口径）与四行消融。

输入（`scripts/bench.py` / `scripts/bench_pso.py` 的输出）：

| 文件 | 内容 | 种子 |
|---|---|---|
| `bench_main.csv` | ALNS，本项目参数（tuned） | 10 |
| `bench_main.csv`（前 5 种子） | 消融的对照臂 | 5 |
| `bench_ablate.csv` | ALNS，**论文原参数**（`--no-tune`） | 5 |
| `bench_truckonly.csv` | 纯卡车（`--no-drone`） | 10 |
| `bench_nodetour.csv` | Without-detour（`--no-detour`） | 10 |
| `bench_pso.csv` | PSO 对照（random-key 编码，自造） | 10 |

输出：
- `results/table3_v2.csv` / `.md`  与论文 Table 3 同口径的对照表
- 终端打印：汇总指标、与旧表（150–300 轮 × 2 种子）的差异、四行消融

约定：**每个单元格取该算例所有种子里的最优**（论文也是「多随机种子取最优」）。
消融比较为了公平，两边都只用 seed 0–4（tuned 与 no-tune 同种子数）。
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


def _best_pso(path: str) -> pd.DataFrame:
    d = pd.read_csv(RES / path)
    g = d.groupby("set").agg(
        pso_best=("best_cost", "min"),
        pso_worst=("best_cost", "max"),
        pso_cpu=("cpu_s", "mean"),
        pso_decodes=("decodes", "first"),
    )
    return g


def _paired(path: str, base: pd.DataFrame, iters_base: int = 2000) -> pd.DataFrame:
    """**同种子配对**比较：每个种子下对照组相对基线臂的差距，再对种子取均值。

    为什么需要它：表里的 `xxx_inc_pct` 是「两边各种子取最优再比」，两个最小值都带
    极值偏差（10 个种子的最小值必然被往下拽），两个偏差相减会放大噪声。
    配对比较用同一随机流下的两次运行，抵消了种子好坏，剩下的才是机制本身的效应。
    """
    d = pd.read_csv(RES / path)
    d = d[d.iters == iters_base][["set", "seed", "best_cost"]]
    b = pd.read_csv(RES / "bench_main.csv")
    b = b[(b.iters == iters_base) & (b.tune == 1)][["set", "seed", "best_cost"]]
    m = d.merge(b, on=["set", "seed"], suffixes=("_c", "_b"))
    m["gap_pct"] = (m.best_cost_c / m.best_cost_b - 1) * 100
    return m.groupby("set").agg(
        paired_gap_pct=("gap_pct", "mean"),
        paired_pos=("gap_pct", lambda s: int((s > 0).sum())),
        paired_n=("gap_pct", "size"),
    )


def _tune_paired() -> pd.DataFrame:
    """参数消融的配对比较：同一算例同一历史上，论文原参数 vs 本项目修正参数。"""
    a = pd.read_csv(RES / "bench_ablate.csv")
    b = pd.read_csv(RES / "bench_main.csv")
    b = b[(b.tune == 1) & (b.seed < 5) & (b.iters == 2000)]
    a = a[a.seed < 5]
    m = a.merge(b, on=["set", "seed"], suffixes=("_a", "_t"))
    m["gap_pct"] = (m.best_cost_a / m.best_cost_t - 1) * 100
    return m.groupby("set").agg(
        tune_paired_pct=("gap_pct", "mean"),
        tune_pos=("gap_pct", lambda s: int((s > 0).sum())),
    )


def main() -> None:
    main10 = _best("bench_main.csv")
    main5 = _best("bench_main.csv", seeds=5)
    abl5 = _best("bench_ablate.csv", seeds=5)
    truck = _best("bench_truckonly.csv")
    nodet = _best("bench_nodetour.csv")
    pso = _best_pso("bench_pso.csv")
    truck_p = _paired("bench_truckonly.csv", main10)
    nodet_p = _paired("bench_nodetour.csv", main10)
    pso_p = _paired("bench_pso.csv", main10)
    tune_p = _tune_paired()

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
            "truckonly_paired_pct": truck_p.paired_gap_pct.round(2),
            "nodetour": nodet.best.round(2),
            "nodetour_inc_pct": ((nodet.best / main10.best - 1) * 100).round(2),
            "nodetour_paired_pct": nodet_p.paired_gap_pct.round(2),
            "nodetour_pos_seeds": nodet_p.paired_pos.astype(int),
            "pso": pso.pso_best.round(2),
            "pso_inc_pct": ((pso.pso_best / main10.best - 1) * 100).round(2),
            "pso_paired_pct": pso_p.paired_gap_pct.round(2),
            "pso_cpu_s": pso.pso_cpu.round(1),
            "ablate5_best": abl5.best.round(2),
            "tune_gain_pct": ((abl5.best / main5.best - 1) * 100).round(2),
            "tune_paired_pct": tune_p.tune_paired_pct.round(2),
        }
    )
    out = RES / "table3_v2.csv"
    t.to_csv(out, encoding="utf-8", index_label="set")

    # Markdown 版本（README / 报告直接贴）：不引入 tabulate 依赖，手写对齐
    cols = ["n", "K", "alns_best", "saving_pct", "truckonly_inc_pct",
            "nodetour_inc_pct", "nodetour_paired_pct", "pso_inc_pct",
            "alns_cpu_s", "pso_cpu_s"]
    head = {
        "n": "n", "K": "K", "alns_best": "ALNS ($)", "saving_pct": "Saving (%)",
        "truckonly_inc_pct": "Truck-only inc (%)", "nodetour_inc_pct": "No-detour inc (%)",
        "nodetour_paired_pct": "No-detour paired (%)",
        "pso_inc_pct": "PSO inc (%)", "alns_cpu_s": "ALNS CPU (s)", "pso_cpu_s": "PSO CPU (s)",
    }
    lines = ["| Set | " + " | ".join(head[c] for c in cols) + " |",
             "|---" * (len(cols) + 1) + "|"]
    for sid, row in t.iterrows():
        lines.append(f"| {sid} | " + " | ".join(f"{row[c]:g}" for c in cols) + " |")
    (RES / "table3_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== Table 3（论文协议：ALNS 2000 轮 × 10 种子；PSO 2000 次解码 × 10 种子）===")
    print(t.to_string())

    print("\n=== 汇总指标 ===")
    for col, name in [
        ("saving_pct", "Saving%（初始解→最优）"),
        ("truckonly_inc_pct", "相对 Truck-only 的增加"),
        ("nodetour_inc_pct", "相对 Without-detour 的增加"),
        ("pso_inc_pct", "PSO 相对 ALNS 的劣化"),
        ("tune_gain_pct", "参数修正带来的改进（5 种子同口径）"),
        ("seed_spread_pct", "种子间极差（最差/最优）"),
    ]:
        v = t[col]
        print(f"{name:32s} 均值 {v.mean():7.2f}   中位 {v.median():7.2f}   范围 {v.min():6.2f} ~ {v.max():6.2f}")
    print(f"{'ALNS CPU 均值 / 最大':32s} {t.alns_cpu_s.mean():7.1f} / {t.alns_cpu_s.max():.1f} s")
    print(f"{'PSO  CPU 均值':32s} {t.pso_cpu_s.mean():7.1f} s")

    print("\n=== 同种子配对比较（消除「取最优」的极值偏差，更可信）===")
    for col, name in [
        ("truckonly_paired_pct", "Truck-only 相对 ALNS"),
        ("nodetour_paired_pct", "Without-detour 相对 ALNS"),
        ("pso_paired_pct", "PSO 相对 ALNS"),
        ("tune_paired_pct", "论文原参数 相对 修正参数"),
    ]:
        v = t[col]
        print(f"{name:26s} 配对均值 {v.mean():6.2f}   中位 {v.median():6.2f}   "
              f"范围 {v.min():6.2f} ~ {v.max():6.2f}")
    print(f"{'Without-detour 配对为正':26s} {int((t.nodetour_paired_pct > 0).sum())}/16 个算例；"
          f"逐种子看 {(t.nodetour_pos_seeds > 5).sum()}/16 个算例过半种子为正")

    old_path = RES / "archive" / "table3_with_pso.csv"
    if old_path.exists():
        print("\n=== 与旧表对照（旧：150–300 轮 × 2 种子，且早于架次完整性修复）===")
        old = pd.read_csv(old_path).set_index("set")
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
        print(f"\n成本平均变化 {cmp['成本变化%'].mean():+.2f}%，最大 {cmp['成本变化%'].min():+.2f}%")

    print(f"\n已写出 {out.relative_to(PROJECT_ROOT)} 与 table3_v2.md")


if __name__ == "__main__":
    main()
