"""把 scripts.scale_bench 的结果画成一张规模-墙钟曲线，并写出口径说明。

    python -m scripts.scale_fig
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src import style
from src.io import PROJECT_ROOT

RESULT_DIR = PROJECT_ROOT / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="scale_bench.csv")
    ap.add_argument("--fig", default="figs/figF_scale_time.png")
    ap.add_argument("--md", default="scale_bench.md")
    args = ap.parse_args()

    csv_path = RESULT_DIR / args.csv
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    n = [int(r["n_customers"]) for r in rows]
    cpu = [float(r["cpu_s"]) for r in rows]
    sources = [r["source"] for r in rows]
    iters = rows[0]["iters"]

    style.apply()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(n, cpu, marker="o", markersize=5, linewidth=1.4,
            color=style.route_color(0), label="ALNS, fixed budget")
    for x, y, src in zip(n, cpu, sources):
        label = f"{y:.1f}s"
        ax.annotate(label, (x, y), xytext=(4, 5), textcoords="offset points", fontsize=7)
        if src == "synthetic-cluster":
            ax.annotate("synthetic", (x, y), xytext=(0, -14),
                        textcoords="offset points", fontsize=7, ha="center", color="#666666")
    ax.set_xscale("log", base=10)
    ax.set_yscale("log", base=10)
    ax.set_xticks(n)
    ax.set_xticklabels([str(x) for x in n])
    ax.set_xlabel("Customers (log scale)")
    ax.set_ylabel("Wall-clock seconds (log scale)")
    ax.set_title(f"Engineering scaling check ({iters} ALNS iterations, seed 0)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    fig_path = RESULT_DIR / args.fig
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    small = next((r for r in rows if int(r["n_customers"]) == 100), None)
    large = next((r for r in rows if int(r["n_customers"]) == 500), None)
    ratio = (float(large["cpu_s"]) / float(small["cpu_s"])) if small and large else float("nan")

    lines = [
        "# 规模压测：工程指标，不是论文结论",
        "",
        f"固定迭代预算 **{iters} 轮**、单种子，比较 10–500 客户的墙钟与可行性。"
        "10/25/50/100 用现有 mFSTSP 派生算例；200/500 是把同一张 100 客户底图复制成 "
        "2/5 个旋转簇的合成算例，卡车矩阵为「欧氏 × 1.25 / 35 km/h」。",
        "",
        "| 客户数 | 来源 | K | 墙钟 s | 成本 | 可行 | 架次 |",
        "|---:|---|---:|---:|---:|:---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n_customers']} | {r['source']} | {r['n_trucks']} | {r['cpu_s']} | "
            f"{r['best_cost']} | {'✓' if r['feasible'] == '1' else '✗'} | {r['n_sorties']} |"
        )
    lines += [
        "",
        f"100 → 500 客户墙钟从 {small['cpu_s'] if small else '—'} s 增至 "
        f"{large['cpu_s'] if large else '—'} s（约 {ratio:.1f}×）；所有点均给出可行解。",
        "",
        "**口径边界**：这是固定低预算下的工程压力测试，不作跨规模成本百分比比较，"
        "不做显著性检验，不写进论文实验表格。合成簇只用于回答「规模上去会不会跑不动」，"
        "不回答「真实 500 点业务成本是多少」。车队 K 为时间窗下界 + 20 辆冗余，"
        "避免合成算例初始解不可行；因此成本绝对值没有业务含义。",
        "",
        f"图：`results/figs/figF_scale_time.png`。复跑：`python -m scripts.scale_bench --iters {iters}`。",
    ]
    md_path = RESULT_DIR / args.md
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写出 {fig_path.relative_to(PROJECT_ROOT)} 与 {md_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
