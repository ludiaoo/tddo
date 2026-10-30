"""论文同款配色与画风（Fig.8–12）。

## 配色是怎么来的

不是目测，是从 PDF 里取的：把第 20–22 页的图区渲染成 300 dpi，取**饱和度最高**的像素
按色相聚类，每个色相桶取代表色。结果与 matplotlib 的 tab10 在一档"部分透明"上吻合
（平均通道误差 18/255），但 Fig.9/10/11 的五条曲线明显更柔和，是论文自己挑的浅色调。

因此分两套：

| 用途 | 调色板 | 说明 |
|---|---|---|
| 灵敏度曲线（Fig.9/10/11，Fig.12 类似） | `PALETTE` 5 色柔和系 | 曲线细、方形标记、无边框图例 |
| 路线图（Fig.8，需要十几条不同路线） | `ROUTES`（tab10） | 论文那张图用的是饱和色，浅色分不开 12 条线 |

单曲线图（Fig.11 的每个小图）统一用 `ACCENT`（那个橙色）——图里左边实线、右边虚线，
虚线一侧正是用了同一个橙色的 dashed 版本。

## 画风（从图里量出来的）

- 白底、**无网格**（Fig.9/10/11 都没有网格；Fig.11 有极浅的网格，接近看不见）
- 四边都有黑色细脊线（box），刻度朝外、很短
- 线条细（≈1.2 pt），标记是**小方块**（`s`），边长 ≈4 pt
- 图例带细边框、字号小、标题行在框内（Fig.9 的 "Scaling Factor"）
- 没有粗体，标题与轴标签同字号量级
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

# ---- 论文里那 5 条曲线的颜色（按 Fig.9 图例的顺序：0.5 / 0.75 / 1.0 / 1.25 / 1.5）----
PALETTE = [
    "#7FC3C3",   # 青（teal）
    "#BCBCDA",   # 薰衣草紫
    "#EF8B4C",   # 橙  ← 单曲线图统一用它
    "#82AAD2",   # 浅蓝
    "#93BC80",   # 浅绿
]
ACCENT = PALETTE[2]

# ---- 路线图用色：Fig.8 里十几条路线是饱和色，浅色区分不开 ----
ROUTES = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
    "#AEC7E8", "#FFBB78", "#98DF8A", "#FF9896",
]

# ---- 要素色 ----
NFZ_FILL = "#F0A8A8"      # 禁飞区填充（论文那张图里的粉色）
NFZ_EDGE = "#D98C8C"
CUSTOMER = "#5B7A99"      # 客户点：低饱和蓝灰，不与路线抢视觉
DEPOT = "#2C2C2A"         # 仓库：近黑

GRID = "#E8E8E8"          # Fig.11 那种"几乎看不见"的网格

# 论文图里用的是衬线体（Times 那一路）。matplotlib ≥3.6 支持字体回退列表，
# 所以拉丁字符走 Times New Roman、中文自动回退。
#
# ⚠ 中文回退**不能用 SimSun/NSimSun**：这两个字体内嵌了点阵字模，小于约 8 pt 时
# FreeType 走点阵路径、matplotlib 的 Agg 渲染出来是**空白**（图例 6 pt 必现，
# 标题 9 pt 正常，非常难查）。微软雅黑/黑体没有点阵表，小字号正常。
_FONT_SERIF = ["Times New Roman", "Microsoft YaHei", "DejaVu Serif"]
_FONT_SANS = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
SERIF = True          # 想换成无衬线体改这里


def apply(font_size: float = 9.0, grid: bool = False) -> None:
    """套用论文画风。`grid=True` 只给 Fig.11 那种小图用。"""
    matplotlib.use("Agg", force=False)
    plt.rcParams.update({
        "font.serif": _FONT_SERIF,
        "font.sans-serif": _FONT_SANS,
        # ⚠ 这里必须给**列表**：matplotlib ≥3.6 只在 font.family 是列表时做逐字回退，
        # 写 "serif" 只会去 font.serif 里挑第一个能用的字体，中文就成方框了（踩过）。
        "font.family": _FONT_SERIF if SERIF else _FONT_SANS,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": font_size,
        "axes.titlesize": font_size + 1.5,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1.0,
        "ytick.labelsize": font_size - 1.0,
        "legend.fontsize": font_size - 1.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.spines.top": True,
        "axes.spines.right": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.size": 3.0,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "lines.markeredgewidth": 0.6,
        "axes.grid": grid,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "legend.frameon": True,
        "legend.edgecolor": "#CCCCCC",
        "legend.framealpha": 1.0,
        "legend.borderpad": 0.35,
        "legend.labelspacing": 0.25,
        "legend.handlelength": 1.6,
    })


def line(ax, x, y, i: int = 0, *, label: str | None = None, dashed: bool = False):
    """按论文风格画一条曲线：细线 + 小方块标记。"""
    return ax.plot(
        x, y,
        color=PALETTE[i % len(PALETTE)],
        marker="s",
        markersize=4.0,
        linewidth=1.2,
        linestyle="--" if dashed else "-",
        label=label,
    )


def route_color(i: int) -> str:
    return ROUTES[i % len(ROUTES)]
