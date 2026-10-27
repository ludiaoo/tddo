"""静态网络可视化：节点、仓库、圆形时空禁飞区（对应论文 Fig.8 的网络示意）。

不含路线——路线要等 M3 评估器 / M5 ALNS 出解之后再画。
图的作用是**肉眼校验算例**：禁飞区有没有罩住仓库、半径是否离谱、簇分布是否合理。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from .generator import build_instance
from .instance import Instance
from .io import PROJECT_ROOT, load_config, load_sets

FIG_DIR = PROJECT_ROOT / "results" / "figs"

# 中文字体：Windows 下优先雅黑/黑体，缺失时退回默认（图注会变方框但不报错）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_network(inst: Instance, ax=None, title: str | None = None):
    """画单个算例：节点 + 仓库 + 禁飞区。"""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    xy = np.array([[nd.x, nd.y] for nd in inst.base.nodes]) / 1000.0  # km

    # 禁飞区
    for z in inst.nfz:
        c = Circle(
            (z.cx / 1000.0, z.cy / 1000.0),
            z.r / 1000.0,
            facecolor="#E24B4A",
            alpha=0.12,
            edgecolor="#A32D2D",
            linewidth=1.0,
            linestyle="--",
            zorder=1,
        )
        ax.add_patch(c)
        ax.annotate(
            f"{z.label}\n{z.t_start / 3600:.1f}–{z.t_end / 3600:.1f} h",
            (z.cx / 1000.0, z.cy / 1000.0),
            ha="center",
            va="center",
            fontsize=6.5,
            color="#A32D2D",
            zorder=2,
        )

    # 客户：面积随派件量
    demand = inst.f[1:]
    sizes = 12 + 60 * np.sqrt(demand / max(demand.max(), 1e-9))
    ax.scatter(xy[1:, 0], xy[1:, 1], s=sizes, c="#378ADD", edgecolors="#0C447C",
               linewidths=0.5, alpha=0.85, zorder=3, label="客户")

    # 仓库
    ax.scatter(*xy[0], s=140, marker="s", c="#2C2C2A", edgecolors="white",
               linewidths=0.8, zorder=4, label="仓库")

    # 等比例
    lo = xy.min(axis=0) - 1.0
    hi = xy.max(axis=0) + 1.0
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.set_xlabel("东向 (km)", fontsize=8)
    ax.set_ylabel("北向 (km)", fontsize=8)
    ax.tick_params(labelsize=7)

    if title is None:
        title = (
            f"Set{inst.set_id} · {inst.base.city} · {inst.n_customers} 客户 · "
            f"K={inst.n_trucks} · NFZ={len(inst.nfz)}"
        )
    ax.set_title(title, fontsize=9)
    return ax


def plot_all(instances: list[Instance], path: Path) -> Path:
    """4×4 总览图。"""
    fig, axes = plt.subplots(4, 4, figsize=(13, 13))
    for ax, inst in zip(axes.ravel(), instances):
        plot_network(inst, ax=ax)
    fig.suptitle("16 个复现算例：节点分布与时空禁飞区", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    cfg = load_config()
    sets = load_sets()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    instances = []
    for s in sets:
        inst = build_instance(s, cfg)
        instances.append(inst)
        out = FIG_DIR / f"set{inst.set_id:02d}_network.png"
        fig, ax = plt.subplots(figsize=(6, 6))
        plot_network(inst, ax=ax)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)

    overview = plot_all(instances, FIG_DIR / "all_sets.png")
    print(f"已写出 {len(instances)} 张单图 + 1 张总览：{FIG_DIR}")
    print(f"总览：{overview.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
