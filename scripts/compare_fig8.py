"""生成「论文 Fig.8 vs 本项目算例」的对照图。

用法：
    python -m scripts.compare_fig8

依赖本地的论文 PDF（不随仓库分发）。输出 `results/figs/compare_with_paper.png`，
该文件**含 Elsevier 原图，已加入 .gitignore，仅本地查看**。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

from src.generator import build_instance
from src.io import PROJECT_ROOT, load_config, load_sets
from scripts.plot_routes import get_solution
from src.visualizer import plot_solution

PDF = PROJECT_ROOT / "Zeng 等 - 2026 - Optimizing bidirectional delivery with multiple drones and trucks a mixed-integer nonlinear model t.pdf"
PAGE_PNG = PROJECT_ROOT / ".tmp" / "hi20-20.png"
OUT = PROJECT_ROOT / "results" / "figs" / "compare_with_paper.png"

# Fig.8 在第 20 页，四面板 2×2；下面这个框是从 400 dpi 渲染图上量出来的
FIG_BOX = (280, 1500, 2820, 3830)

# 与论文四个面板对应的自建算例（同客户规模，seattle 居中那组）
PICK = [1, 5, 9, 13]


def render_page() -> None:
    if PAGE_PNG.exists():
        return
    PAGE_PNG.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftoppm", "-f", "20", "-l", "20", "-r", "400", "-png", str(PDF), str(PAGE_PNG.with_suffix(""))],
        check=True,
    )


def main() -> None:
    render_page()
    paper = Image.open(PAGE_PNG).crop(FIG_BOX)
    w, h = paper.size
    # 四面板 2×2：按中线切成四块，顺序为 (a)(b)(c)(d)
    quads = [
        paper.crop((0, 0, w // 2, h // 2)),
        paper.crop((w // 2, 0, w, h // 2)),
        # 下半两块要切掉底部的 "Fig. 8. Route planning results..." 图注，避免被中线劈成两半
        paper.crop((0, h // 2, w // 2, h - 70)),
        paper.crop((w // 2, h // 2, w, h - 70)),
    ]

    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 9.2), gridspec_kw={"height_ratios": [1.05, 1], "wspace": 0.12, "hspace": 0.22}
    )

    for col, (sid, quad) in enumerate(zip(PICK, quads)):
        ax_top = axes[0, col]
        ax_top.imshow(quad)
        ax_top.set_axis_off()

        inst = build_instance(by_id[sid], cfg)
        sol, cost, _ = get_solution(inst)
        ax_bot = axes[1, col]
        plot_solution(inst, sol, ax=ax_bot, legend=(sid != PICK[-1]))
        ax_bot.set_title(
            f"本项目 Set{sid}  K={inst.n_trucks}  成本 {cost:.1f}", fontsize=9
        )

        axes[0, col].set_title(
            f"{inst.n_customers} 客户", fontsize=10, pad=4,
        )

    fig.text(0.02, 0.955, "论文 Fig.8（含路线）", fontsize=10, weight="normal")
    fig.text(0.02, 0.475, "本项目：ALNS 求解结果（同配色同画风）", fontsize=10, weight="normal")
    fig.suptitle("算例对照：论文 Fig.8 与本项目生成结果", fontsize=13, y=0.995)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已写出 {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
