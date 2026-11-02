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
from matplotlib.patches import Circle, Patch

from . import style
from .generator import build_instance
from .instance import Instance
from .io import PROJECT_ROOT, load_config, load_sets

FIG_DIR = PROJECT_ROOT / "results" / "figs"

style.apply()


def _draw_nfz(inst: Instance, ax) -> None:
    """禁飞区：浅粉填充 + 虚线边（论文 Fig.8 的样式）。"""
    for z in inst.nfz:
        c = Circle(
            (z.cx / 1000.0, z.cy / 1000.0),
            z.r / 1000.0,
            facecolor=style.NFZ_FILL,
            alpha=0.30,
            edgecolor=style.NFZ_EDGE,
            linewidth=0.9,
            linestyle="--",
            zorder=1,
        )
        ax.add_patch(c)
        ax.annotate(
            z.label,
            (z.cx / 1000.0, z.cy / 1000.0),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color=style.NFZ_EDGE,
            zorder=2,
        )


def _setup_axes(inst: Instance, ax, title: str | None) -> None:
    xy = np.array([[nd.x, nd.y] for nd in inst.base.nodes]) / 1000.0
    lo = xy.min(axis=0) - 1.0
    hi = xy.max(axis=0) + 1.0
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (km)")
    ax.set_ylabel("North (km)")
    if title is None:
        title = (
            f"Set{inst.set_id} · {inst.base.city} · {inst.n_customers} customers · "
            f"K={inst.n_trucks} · NFZ={len(inst.nfz)}"
        )
    ax.set_title(title)


def plot_network(inst: Instance, ax=None, title: str | None = None, legend: bool = True):
    """画单个算例：节点 + 仓库 + 禁飞区。

    蓝点 = 客户（大小 ∝ √派件量），黑方块 = 仓库，粉色虚线圈 = 时空禁飞区。
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    xy = np.array([[nd.x, nd.y] for nd in inst.base.nodes]) / 1000.0  # km

    _draw_nfz(inst, ax)

    # 客户：面积随派件量
    demand = inst.f[1:]
    sizes = 12 + 60 * np.sqrt(demand / max(demand.max(), 1e-9))
    ax.scatter(xy[1:, 0], xy[1:, 1], s=sizes, c=style.CUSTOMER, edgecolors="white",
               linewidths=0.4, alpha=0.9, zorder=3, label="Customers (size ∝ √demand)")

    # 仓库
    ax.scatter(*xy[0], s=60, marker="D", c=style.DEPOT, edgecolors="white",
               linewidths=0.7, zorder=4, label="Depot")

    if legend:
        nfz_proxy = Patch(facecolor=style.NFZ_FILL, alpha=0.45, edgecolor=style.NFZ_EDGE,
                          linewidth=0.9, linestyle="--", label="No-fly zone (space-time)")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [nfz_proxy], labels + ["No-fly zone (space-time)"],
                  loc="upper left", fontsize=5.5, framealpha=0.9,
                  handletextpad=0.4, borderpad=0.3, labelspacing=0.25,
                  handlelength=1.4)

    _setup_axes(inst, ax, title)
    return ax


def solution_legend_handles(inst: Instance) -> list:
    """路线图的**通用**图例（不按卡车逐条列，卡车颜色看图即可）。"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], color="#555555", linewidth=1.3, label="Truck route (solid)"),
        Line2D([], [], color="#555555", linewidth=0.9, linestyle=(0, (3, 2)),
               label="Drone sortie (dashed)"),
        Line2D([], [], color=style.CUSTOMER, marker="o", linestyle="none",
               markersize=4.5, label="Truck-served customer"),
        Line2D([], [], color="#555555", marker="o", markerfacecolor="white",
               markeredgecolor="#555555", markeredgewidth=0.8, linestyle="none",
               markersize=4.5, label="Drone-served customer"),
        Line2D([], [], color=style.DEPOT, marker="D", linestyle="none",
               markersize=5.5, label="Depot"),
    ]
    if inst.nfz:
        handles.append(Patch(facecolor=style.NFZ_FILL, alpha=0.4,
                             edgecolor=style.NFZ_EDGE, linewidth=0.8,
                             linestyle="--", label="No-fly zone (space-time)"))
    return handles


def plot_solution(
    inst: Instance,
    sol: "Solution",
    ax=None,
    title: str | None = None,
    legend: bool = False,
) -> None:
    """画完整解：卡车实线 + 无人机虚线 + 仓库 + 禁飞区（论文 Fig.8 的画法）。

    - 每辆卡车一色（`style.ROUTES`），卡车路径是实线；
    - 该车无人机的每个架次用**同色虚线**：起飞点 → 客户序列 → 回收点；
    - 无人机服务的客户画空心小圆，和卡车直服的实心点区分开；
    - 仓库用小菱形（大黑方块太抢眼，也容易压住路线）。

    图例默认**不画在面板里**（会挡路线）：多面板脚本用
    `solution_legend_handles()` + `fig.legend()` 在图外放一条共享图例。
    """
    from .solution import DEPOT, Solution  # noqa: F401  避免循环导入

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    _draw_nfz(inst, ax)

    used = 0
    for r in sol.routes:
        if len(r.nodes) <= 2 and not r.sorties:
            continue                                   # 空车不画
        c = style.route_color(r.truck)
        xs = [inst.base.nodes[n].x / 1000.0 for n in r.nodes]
        ys = [inst.base.nodes[n].y / 1000.0 for n in r.nodes]
        ax.plot(xs, ys, color=c, linewidth=1.2, alpha=0.95, zorder=3,
                solid_capstyle="round", label=f"Truck {r.truck}")
        used += 1

        for s in r.sorties:
            seq = [r.nodes[s.launch], *s.customers, r.nodes[s.recover]]
            dxs = [inst.base.nodes[n].x / 1000.0 for n in seq]
            dys = [inst.base.nodes[n].y / 1000.0 for n in seq]
            ax.plot(dxs, dys, color=c, linewidth=0.9, linestyle=(0, (3, 2)),
                    alpha=0.95, zorder=3)
            cx = [inst.base.nodes[n].x / 1000.0 for n in s.customers]
            cy = [inst.base.nodes[n].y / 1000.0 for n in s.customers]
            ax.scatter(cx, cy, s=9, facecolor="white", edgecolor=c, linewidths=0.7,
                       zorder=4)

    # 客户（卡车直服）：比无人机服务的点更小
    xy = np.array([[nd.x, nd.y] for nd in inst.base.nodes]) / 1000.0
    truck_served = {n for r in sol.routes for n in r.nodes if n != DEPOT}
    if truck_served:
        idx = sorted(truck_served)
        ax.scatter(xy[idx, 0], xy[idx, 1], s=13, c=style.CUSTOMER, edgecolors="white",
                   linewidths=0.4, zorder=5, label="Truck-served customers")
    ax.scatter(*xy[0], s=42, marker="D", c=style.DEPOT, edgecolors="white",
               linewidths=0.7, zorder=6, label="Depot")

    if legend:
        ax.legend(handles=solution_legend_handles(inst), loc="upper left", fontsize=6,
                  framealpha=0.95, handletextpad=0.4, borderpad=0.35,
                  labelspacing=0.3, handlelength=1.8)

    _setup_axes(inst, ax, title)


def plot_all(instances: list[Instance], path: Path) -> Path:
    """4×4 总览图：面板不带图例，底部一条共享图例。"""
    fig, axes = plt.subplots(4, 4, figsize=(13, 13.6))
    for ax, inst in zip(axes.ravel(), instances):
        plot_network(inst, ax=ax, legend=False)
    fig.suptitle("16 instances: customers, depot and space-time no-fly zones", fontsize=12)
    handles = _network_legend_handles(instances[0])
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.035, 1, 0.98))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _network_legend_handles(inst: Instance) -> list:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], color=style.CUSTOMER, marker="o", linestyle="none",
               markersize=4.5, label="Customers (size ∝ √demand)"),
        Line2D([], [], color=style.DEPOT, marker="D", linestyle="none",
               markersize=5.5, label="Depot"),
    ]
    if inst.nfz:
        handles.append(Patch(facecolor=style.NFZ_FILL, alpha=0.4,
                             edgecolor=style.NFZ_EDGE, linewidth=0.8,
                             linestyle="--", label="No-fly zone (space-time)"))
    return handles


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
