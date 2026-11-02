"""配色/画风核对图：把我们按论文风格画出来的示意曲线，和论文原图并排放在一起看。

    python -m scripts.style_preview

产出两张：
- `results/figs/style_preview.png` —— 只含本项目生成的内容（色板 + 两个样式示意面板）
- `results/figs/style_vs_paper.png` —— 左：论文 Fig.9 原图；右：同样式示意（**含 Elsevier 原图，不入库**）

⚠ 示意面板里的数据是**编的**，只为看画风（线宽、标记、图例、脊线、网格）。
   真实灵敏度曲线由 M7 的扫描脚本产出。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from src import style
from src.io import PROJECT_ROOT

PDF = PROJECT_ROOT / "Zeng 等 - 2026 - Optimizing bidirectional delivery with multiple drones and trucks a mixed-integer nonlinear model t.pdf"
OUT_DIR = PROJECT_ROOT / "results" / "figs"
TMP = PROJECT_ROOT / ".tmp" / "style"

X = np.array([150, 200, 250, 300, 350, 400, 450, 500])
# 仅用于看画风的示意数据（形状模仿论文 Fig.9 的上图）
CURVES = {
    "0.5": 300 * np.exp(-0.006 * (X - 150)) + 150,
    "0.75": 330 * np.exp(-0.006 * (X - 150)) + 165,
    "1.0": 380 * np.exp(-0.007 * (X - 150)) + 175,
    "1.25": 460 * np.exp(-0.008 * (X - 150)) + 190,
    "1.5": 520 * np.exp(-0.009 * (X - 150)) + 200,
}


def mock_sensitivity(ax, *, legend_title: str = "Scaling Factor") -> None:
    for i, (k, y) in enumerate(CURVES.items()):
        style.line(ax, X, y, i, label=k)
    ax.set_xlabel("Load Capacity (lb)")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title("Impact of Truck Capacity and Demand Scaling on Cost")
    ax.legend(title=legend_title, loc="upper right", title_fontsize=7)


def mock_single(ax, *, dashed: bool, title: str) -> None:
    xs = ["No NFZ", "Work-time NFZ", "Full-time NFZ"]
    ys = [222, 227, 239]
    ax.plot(xs, ys, color=style.ACCENT, marker="s", markersize=4.5, linewidth=1.3,
            linestyle="--" if dashed else "-")
    ax.set_xlabel("No-Fly Period")
    ax.set_ylabel("Total Cost ($)")
    ax.set_title(title)


def swatches(ax) -> None:
    ax.set_axis_off()
    ax.set_title("Paper palette (extracted from the PDF)", loc="left")
    labels = [f"{h}  #{h.lstrip('#')}" for h in style.PALETTE]
    for i, (c, lab) in enumerate(zip(style.PALETTE, labels)):
        y = 0.86 - i * 0.17
        ax.add_patch(Rectangle((0.02, y), 0.12, 0.11, transform=ax.transAxes,
                               facecolor=c, edgecolor="#CCCCCC", linewidth=0.6))
        ax.text(0.18, y + 0.045, lab, transform=ax.transAxes, va="center", fontsize=8)
    ax.text(0.02, 0.02, "Orange #EF8B4C: single-curve panels (Fig.11)", transform=ax.transAxes,
            fontsize=7, color="#666666")


def main() -> None:
    style.apply()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.9], wspace=0.32)
    ax0 = fig.add_subplot(gs[0, 0])
    mock_sensitivity(ax0)
    ax1 = fig.add_subplot(gs[0, 1])
    mock_single(ax1, dashed=False, title="Case 1 - Radius")
    ax2 = fig.add_subplot(gs[0, 2])
    swatches(ax2)
    fig.suptitle("Style preview (mock data, for line/marker/legend check only)", fontsize=10)
    out = OUT_DIR / "style_preview.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"已写出 {out.relative_to(PROJECT_ROOT)}")

    # 与论文原图并排：两边同宽，才能比较线宽/字号的实际观感
    if not PDF.exists():
        print("缺论文 PDF，跳过并排图")
        return
    TMP.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["pdftoppm", "-f", "21", "-l", "21", "-r", "200", "-png", str(PDF), str(TMP / "p")],
        check=True,
    )
    from PIL import Image

    page = Image.open(TMP / "p-21.png")
    w, h = page.size
    paper = page.crop((int(0.10 * w), int(0.10 * h), int(0.62 * w), int(0.30 * h)))

    # 单独画一张同尺寸的样式示意（4.6 x 1.6 in ≈ 论文那种单栏小图）
    style.apply(font_size=7.0)
    fig = plt.figure(figsize=(4.6, 1.7))
    ax = fig.add_subplot(111)
    mock_sensitivity(ax)
    fig.tight_layout(pad=0.4)
    TMP.mkdir(parents=True, exist_ok=True)
    mock_png = TMP / "mock.png"
    fig.savefig(mock_png, dpi=200)
    plt.close(fig)
    style.apply()

    mock = Image.open(mock_png).convert("RGB")
    scale = paper.width / mock.width
    mock = mock.resize((paper.width, int(mock.height * scale)), Image.LANCZOS)

    W = paper.width
    H = paper.height + mock.height + 24
    sheet = Image.new("RGB", (W, H), "white")
    sheet.paste(paper, (0, 0))
    sheet.paste(mock, (0, paper.height + 24))
    out2 = OUT_DIR / "style_vs_paper.png"
    sheet.save(out2)
    print(f"已写出 {out2.relative_to(PROJECT_ROOT)}（含 Elsevier 原图，不入库）")


if __name__ == "__main__":
    main()
