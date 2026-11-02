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


def _draw_nfz(inst: Instance, ax, labels: bool = False) -> None:
    """禁飞区：鲑鱼粉**实心**圆（论文 Fig.8 的样子：无描边、无文字）。"""
    for z in inst.nfz:
        ax.add_patch(
            Circle(
                (z.cx / 1000.0, z.cy / 1000.0),
                z.r / 1000.0,
                facecolor=style.NFZ_FILL,
                alpha=0.55,
                edgecolor="none",
                zorder=1,
            )
        )
        if labels:
            ax.annotate(
                z.label, (z.cx / 1000.0, z.cy / 1000.0),
                xytext=(0, 6), textcoords="offset points",
                ha="center", va="bottom", fontsize=5.5,
                color=style.NFZ_EDGE, zorder=2,
            )


def _setup_axes(inst: Instance, ax, title: str | None, square: bool = True) -> None:
    """统一坐标范围。`square=True` 时把短边补到长边，保证多面板里每个框一样大。"""
    xy = np.array([[nd.x, nd.y] for nd in inst.base.nodes]) / 1000.0
    lo = xy.min(axis=0) - 1.0
    hi = xy.max(axis=0) + 1.0
    if square:
        span = float(max(hi[0] - lo[0], hi[1] - lo[1]))
        cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
        lo = np.array([cx - span / 2, cy - span / 2])
        hi = np.array([cx + span / 2, cy + span / 2])
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
        nfz_proxy = Patch(facecolor=style.NFZ_FILL, alpha=0.55, edgecolor="none",
                          label="No-Fly Zone")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [nfz_proxy], labels + ["No-Fly Zone"],
                  loc="upper left", fontsize=5.5, framealpha=0.95,
                  handletextpad=0.5, borderpad=0.35, labelspacing=0.25,
                  handlelength=1.6)

    _setup_axes(inst, ax, title)
    return ax


def solution_legend_handles(inst: Instance, sol: "Solution") -> list:
    """论文 Fig.8 的图例内容：Nodes / Truck k / Drone k / Depot / No-Fly Zone。

    每辆车占**两行**：实线（带小圆点）= 卡车路径，同色虚线 = 该车无人机的架次。
    这是 Fig.8 的做法，图上一眼就能把「哪条虚线和哪条实线是同一辆车」对上。
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from .solution import DEPOT

    handles = [
        Line2D([], [], color=style.CUSTOMER, marker="o", linestyle="none",
               markersize=3.6, label="Nodes"),
    ]
    for r in sorted(sol.routes, key=lambda x: x.truck):      # 图例按车号排序，便于对照
        if len(r.nodes) <= 2 and not r.sorties:
            continue
        c = style.route_color(r.truck)
        handles.append(
            Line2D([], [], color=c, linewidth=1.2, marker="o", markersize=2.6,
                   label=f"Truck {r.truck}")
        )
        if r.sorties:
            handles.append(
                Line2D([], [], color=c, linewidth=0.9, linestyle=(0, (3, 2)),
                       label=f"Drone {r.truck}")
            )
    handles.append(
        Line2D([], [], color=style.DEPOT, marker="D", linestyle="none",
               markersize=4.5, label="Depot")
    )
    if inst.nfz:
        handles.append(
            Patch(facecolor=style.NFZ_FILL, alpha=0.55, edgecolor="none",
                  label="No-Fly Zone")
        )
    return handles


def generic_route_legend_handles(inst: Instance) -> list:
    """不逐车列色的通用图例（多个面板车队规模不同时用这一套）。"""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], color=style.CUSTOMER, marker="o", linestyle="none",
               markersize=3.6, label="Nodes"),
        Line2D([], [], color="#555555", linewidth=1.2, marker="o", markersize=2.6,
               label="Truck route (one colour per truck)"),
        Line2D([], [], color="#555555", linewidth=0.9, linestyle=(0, (3, 2)),
               label="Drone sortie (dashed, same colour)"),
        Line2D([], [], color=style.DEPOT, marker="D", linestyle="none",
               markersize=4.5, label="Depot"),
    ]
    if inst.nfz:
        handles.append(Patch(facecolor=style.NFZ_FILL, alpha=0.55, edgecolor="none",
                             label="No-Fly Zone"))
    return handles


def attach_legend(ax, inst: Instance, sol: "Solution", font_size: float = 6.0) -> None:
    """把 Fig.8 式图例挂在**坐标区右侧外面**——不遮挡任何路线。

    论文把图例塞在面板左上角，路线会从图例底下穿过；这里放到框外，
    面板里只留数据。`bbox_to_anchor=(1.015, 1.0)` + `loc="upper left"` 即贴在右侧。
    """
    ax.legend(
        handles=solution_legend_handles(inst, sol),
        loc="upper left",
        bbox_to_anchor=(1.015, 1.0),
        borderaxespad=0.0,
        fontsize=font_size,
        frameon=True,
        framealpha=1.0,
        edgecolor="#D0D0D0",
        handletextpad=0.5,
        borderpad=0.45,
        labelspacing=0.28,
        handlelength=2.0,
    )


def plot_solution(
    inst: Instance,
    sol: "Solution",
    ax=None,
    title: str | None = None,
    legend: bool = False,
    font_size: float = 6.0,
) -> None:
    """画完整解：卡车实线（带节点小圆点）+ 无人机同色虚线（论文 Fig.8 的画法）。

    - 每辆卡车一色（`style.ROUTES`），路径实线并在经过的节点处画小圆点；
    - 该车无人机的每个架次用**同色虚线**：起飞点 → 客户序列 → 回收点；
    - 所有节点统一画成黑点（论文里就是 "Nodes" 一项，不区分谁服务的）；
    - 仓库用小菱形，禁飞区是鲑鱼粉实心圆。

    `legend=True` 时把 Fig.8 式图例挂在坐标区右侧外面（不遮路线）；
    多面板脚本请自行调用 `attach_legend()`，方便统一布局。
    """
    from .solution import DEPOT

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    _draw_nfz(inst, ax)

    for r in sol.routes:
        if len(r.nodes) <= 2 and not r.sorties:
            continue                                   # 空车不画
        c = style.route_color(r.truck)
        xs = [inst.base.nodes[n].x / 1000.0 for n in r.nodes]
        ys = [inst.base.nodes[n].y / 1000.0 for n in r.nodes]
        ax.plot(xs, ys, color=c, linewidth=1.2, zorder=3, marker="o",
                markersize=2.6, markerfacecolor=c, markeredgecolor=c,
                solid_capstyle="round", label=f"Truck {r.truck}")

        for s in r.sorties:
            seq = [r.nodes[s.launch], *s.customers, r.nodes[s.recover]]
            dxs = [inst.base.nodes[n].x / 1000.0 for n in seq]
            dys = [inst.base.nodes[n].y / 1000.0 for n in seq]
            ax.plot(dxs, dys, color=c, linewidth=0.85, linestyle=(0, (3, 2)),
                    zorder=2.5)

    # 节点：统一黑点（论文 "Nodes"）
    xy = np.array([[nd.x, nd.y] for nd in inst.base.nodes]) / 1000.0
    ax.scatter(xy[1:, 0], xy[1:, 1], s=9, c=style.CUSTOMER, edgecolors="none",
               zorder=4, label="Nodes")
    # 仓库：小菱形
    ax.scatter(*xy[0], s=38, marker="D", c=style.DEPOT, edgecolors="white",
               linewidths=0.7, zorder=5, label="Depot")

    if legend:
        attach_legend(ax, inst, sol, font_size=font_size)

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
               markersize=3.8, label="Nodes (size ∝ √demand)"),
        Line2D([], [], color=style.DEPOT, marker="D", linestyle="none",
               markersize=5.5, label="Depot"),
    ]
    if inst.nfz:
        handles.append(Patch(facecolor=style.NFZ_FILL, alpha=0.55, edgecolor="none",
                             label="No-Fly Zone"))
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
