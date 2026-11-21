"""论文用的四张图（A–D）：把「换电」这件事从机制讲到边界。

| 图 | 画什么 | 数据来源 |
|---|---|---|
| A | 同一节点「先落地、换电、再起飞」的时刻结构，与论文 `w_i` 的冲突 | 现场求解一条含换电的路线 |
| B | 四把刀下的收益区间（算例形态 / 换电耗时 / 电池库存 / 悬停功率 / 空间分布） | `results/bench_*.csv` |
| C | 电池库存曲线：换电收益 vs 每车备用电池数（双轴：基准臂代价） | 同上 |
| D | 解的结构变化：架次数、无人机服务客户数 | 同上 |

口径：所有百分比都是**同种子配对的相对差**（负 = 换电更好），置信区间用配对值的
bootstrap 百分位（2 000 次重采样）。不同条件的算例数不同，图里逐行标 n。

    python -m scripts.paper_figs            # 四张全画
    python -m scripts.paper_figs --only B C # 只画指定几张
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src import style
from src.io import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results"
FIGS = PROJECT_ROOT / "results" / "figs"
RNG = np.random.default_rng(20261111)


# ---------------------------------------------------------------- 数据

def _load(name: str) -> pd.DataFrame | None:
    p = RESULTS / name
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d["iters"] == 2000] if "iters" in d.columns else d
    cols = [c for c in ("set", "seed", "best_cost", "n_sorties", "drone_served") if c in d.columns]
    return d[cols]


def _raw(name: str) -> pd.DataFrame | None:
    """整表读入（需要 form / hotswap 这类分组列时用）。"""
    p = RESULTS / name
    if not p.exists():
        return None
    d = pd.read_csv(p)
    return d[d["iters"] == 2000] if "iters" in d.columns else d


def _pct(a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    m = a.merge(b, on=["set", "seed"], suffixes=("_a", "_b"))
    return ((m["best_cost_a"] - m["best_cost_b"]) / m["best_cost_b"] * 100.0).to_numpy()


def _ci(x: np.ndarray, n_boot: int = 2000) -> tuple[float, float, float]:
    """返回（均值, 2.5 分位, 97.5 分位）。配对重采样，比 t 区间稳。"""
    idx = RNG.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _keys() -> pd.DataFrame:
    """统一到 6 算例 × 5 种子的 30 个配对（与 M9.18–M9.21 一致）。"""
    d = _load("bench_e_b0.csv")
    if d is None:
        d = _load("bench_hotswap.csv").head(30)
    return d[["set", "seed"]]


# ---------------------------------------------------------------- 图 A

def _hotswap_route(seed_max: int = 5):
    """解一条含「同节点落地即再起飞」的路线，返回 (inst, route, ev)。"""
    from dataclasses import replace

    from src.alns import solve
    from src.evaluator import _simulate_route
    from src.generator import build_instance
    from src.io import load_config, load_sets

    cfg = load_config()
    cfg["fleet"]["allow_hotswap"] = True
    by_id = {s["id"]: s for s in load_sets()}
    inst = build_instance(by_id[5], cfg)
    for seed in range(seed_max):
        r = solve(inst, seed=seed, iters=2000)
        if r.best is None:
            continue
        for rt in r.best.routes:
            if len(rt.sorties) < 2:
                continue
            for s1, s2 in zip(rt.sorties, rt.sorties[1:]):
                if s1.recover == s2.launch:            # 同节点：先落地、再起飞
                    ev = _simulate_route(inst, rt, fast=False)
                    if ev.feasible:
                        return inst, rt, ev
    raise RuntimeError("没找到含换电的路线；换个算例或调大 seed_max")


def fig_a() -> None:
    inst, route, ev = _hotswap_route()
    specs = route.sorties
    s1, s2 = next((a, b) for a, b in zip(specs, specs[1:]) if a.recover == b.launch)
    wins = ev.sortie_windows
    t_l1, t_r1 = wins[0][0] / 60, wins[0][1] / 60
    t_l2, t_r2 = wins[1][0] / 60, wins[1][1] / 60
    t_end = ev.finish_time / 60
    node = route.nodes[s1.recover]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True, sharex=True)
    for ax, title in zip(axes, ["(a) Paper model: $w_i$ is a single flag",
                                "(b) This work: land $\\rightarrow$ swap $\\rightarrow$ relaunch"]):
        ax.set_title(title, fontsize=8.6)
        ax.axhspan(0.6, 1.4, color="#F4F4F4", zorder=0)
        ax.plot([0, t_end], [1, 1], color="#9A9A9A", lw=6, solid_capstyle="butt", zorder=1)
        ax.text(0.4, 1.24, "truck", fontsize=7, color="#666666")
        ax.barh(0, t_r1 - t_l1, left=t_l1, height=0.32, color=style.PALETTE[2], zorder=3)
        ax.text((t_l1 + t_r1) / 2, 0, "sortie 1", ha="center", va="center",
                fontsize=7.2, color="white", zorder=4)
        ax.axvline(t_r1, color="#333333", lw=0.7, ls=":", zorder=2)
        ax.annotate(f"node {node}", xy=(t_r1, 0.5), xytext=(t_r1, 0.72), fontsize=6.8,
                    ha="center", color="#333333")

    # (a) 论文模型：同一节点只能"降落"或"起飞"，第二个架次在模型里不存在
    ax = axes[0]
    ax.barh(0, t_r2 - t_l2, left=t_l2, height=0.32, facecolor="none",
            edgecolor="#B0B0B0", ls="--", lw=1.0, zorder=3)
    ax.plot([t_l2, t_r2], [0.16, -0.16], color="#C0392B", lw=1.0, zorder=5)
    ax.plot([t_l2, t_r2], [-0.16, 0.16], color="#C0392B", lw=1.0, zorder=5)
    ax.text((t_l2 + t_r2) / 2, 0.28, "forbidden: $w_i$ already used for recovery",
            fontsize=6.8, color="#C0392B", ha="center")
    ax.text(t_r1 + 0.8, -0.34, "recover only", fontsize=6.8, color="#C0392B", ha="left")

    # (b) 本文：落地 → 换电 → 原地再起飞
    ax = axes[1]
    swap = 2.0
    ax.add_patch(Rectangle((t_r1, -0.16), swap, 0.32, facecolor="#93BC80",
                           edgecolor="#5E7F4E", hatch="xx", lw=0.6, zorder=5))
    ax.text(t_r1 + swap / 2, -0.34, "swap", fontsize=6.8, ha="center", color="#3E5A32")
    ax.barh(0, t_r2 - t_l2, left=t_l2, height=0.32, color=style.PALETTE[4], zorder=3)
    ax.text((t_l2 + t_r2) / 2, 0, "sortie 2", ha="center", va="center",
            fontsize=7.2, color="white", zorder=4)
    ax.annotate("", xy=(t_l2 - 0.2, 0.26), xytext=(t_r1 + 0.2, 0.26),
                arrowprops=dict(arrowstyle="-|>", color="#5E7F4E", lw=0.8, ls="--"))
    ax.text((t_r1 + t_l2) / 2, 0.33, "same node", fontsize=6.6, ha="center", color="#3E5A32")

    for ax in axes:
        ax.set_ylim(-0.5, 1.5)
        ax.set_yticks([])
        ax.set_xlim(0, t_end * 1.02)
        ax.set_xlabel("time since departure (min)")
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    out = FIGS / "figA_hotswap_timeline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.relative_to(PROJECT_ROOT)}  （Set5 实解：{len(specs)} 个架次，"
          f"共享节点 {node}，换电时刻 {t_r1:.0f} min）")


# ---------------------------------------------------------------- 图 B

def fig_b() -> None:
    keys = _keys()
    base = _load("bench_main.csv").merge(keys, on=["set", "seed"])
    hs0 = _load("bench_hotswap.csv").merge(keys, on=["set", "seed"])

    rows: list[tuple[str, str, np.ndarray]] = []
    rows.append(("example", "default (6 sets x 5)", _pct(hs0, base)))
    nat = _raw("native_check.csv")
    if nat is not None:
        nat_d = nat[(nat["form"] == "default")]
        nat_n = nat[(nat["form"] == "native")]
        # 同一 (set, seed) 在两个 form 下各有一对臂，直接 merge 会 4 倍膨胀 → 先按臂拆开
        rows.append(("example", "mFSTSP-native form",
                     _pct(nat_n[nat_n["hotswap"] == 1], nat_d[nat_d["hotswap"] == 0])))

    for sw, tag in ((120, "sw120_hs"), (300, "sw300_hs")):
        a = _load(f"bench_{tag}.csv")
        if a is not None:
            rows.append(("swap", f"swap {sw} s", _pct(a, base)))

    for b, tag in ((2, "e_b2"), (3, "e_b3")):
        a, c = _load(f"bench_{tag}_hs.csv"), _load(f"bench_{tag}.csv")
        if a is not None and c is not None:
            rows.append(("battery", f"{b} spare packs", _pct(a, c)))
    a, c = _load("bench_e_b1_hs.csv"), _load("bench_e_b1.csv")
    if a is not None:
        rows.append(("battery", "1 spare pack", _pct(a, c)))

    for kw, tag in ((0.45, "h45"), (1.10, "b")):
        a, c = _load(f"bench_{tag}_hs.csv"), _load(f"bench_{tag}_main.csv")
        if a is not None:
            rows.append(("hover", f"hover {kw:.2f} kW", _pct(a, c)))

    sp = _raw("spatial_check.csv")
    if sp is not None:
        for form, lab in (("uniform", "uniform layout"), ("clustered", "clustered layout")):
            d = sp[sp["form"] == form]
            pv = d.pivot_table(index=["set", "seed"], columns="hotswap", values="best_cost").dropna()
            rows.append(("space", lab, ((pv[1] - pv[0]) / pv[0] * 100.0).to_numpy()))

    group_names = {"example": "instance form", "swap": "swap time",
                   "battery": "battery stock", "hover": "hover price", "space": "spatial layout"}
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    y, yticks, ylabels = 0, [], []
    prev_group = None
    for grp, lab, x in rows:
        if prev_group is not None and grp != prev_group:
            ax.axhline(y - 0.5, color="#DDDDDD", lw=0.7)
            y += 0.35
        m, lo, hi = _ci(x)
        color = style.ACCENT if lo < 0 and hi < 0 else ("#7F7F7F" if lo < 0 < hi else "#C0392B")
        ax.plot([lo, hi], [y, y], color=color, lw=1.1, solid_capstyle="butt")
        ax.plot(m, y, "s", color=color, ms=4.5)
        ax.text(0.55, y, f"n={len(x)}", fontsize=6.4, color="#999999", va="center")
        yticks.append(y)
        ylabels.append(f"{lab}")
        prev_group = grp
        y -= 1.0

    ax.axvline(0, color="#333333", lw=0.8, ls="--")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7.6)
    ax.set_xlabel("hot-swap benefit vs. baseline (%)   \u2190 better")
    ax.invert_yaxis()
    ax.set_xlim(-6.2, 1.6)
    ax.grid(axis="y", visible=False)
    ax.set_title("Fig. B  One conclusion, four knobs: where the benefit lives", fontsize=9)
    for grp, name in group_names.items():
        ys = [yy for (g, _l, _x), yy in zip(rows, yticks) if g == grp]
        if ys:
            ax.text(-6.0, min(ys) - 0.02, name, fontsize=7.2, color="#555555",
                    style="italic", va="center")
    fig.tight_layout()
    out = FIGS / "figB_benefit_bounds.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.relative_to(PROJECT_ROOT)}  （{len(rows)} 个条件，95% bootstrap CI）")


# ---------------------------------------------------------------- 图 C

def fig_c() -> None:
    keys = _keys()
    base = _load("bench_main.csv").merge(keys, on=["set", "seed"])
    labels = ["0", "1", "2", "3", "\u221e"]
    arms = [("e_b0", "e_b0_hs"), ("e_b1", "e_b1_hs"), ("e_b2", "e_b2_hs"),
            ("e_b3", "e_b3_hs"), None]
    benefit, lo_err, hi_err, pen = [], [], [], []
    for arm in arms:
        if arm is None:
            b = base
            h = _load("bench_hotswap.csv").merge(keys, on=["set", "seed"])
            pen.append(0.0)
        else:
            b = _load(f"bench_{arm[0]}.csv")
            h = _load(f"bench_{arm[1]}.csv")
            pen.append(float(_pct(b, base).mean()))
        m, lo, hi = _ci(_pct(h, b))
        benefit.append(m)
        lo_err.append(m - lo)
        hi_err.append(hi - m)

    # 双轴容易误读（右轴 0 会落在图底），所以拆成上下两栏共用 x
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.4, 4.0), sharex=True,
                                  gridspec_kw={"height_ratios": [2.1, 1.0]})
    ax.errorbar(range(len(labels)), benefit, yerr=[lo_err, hi_err], color=style.ACCENT,
                marker="s", lw=1.2, capsize=3, elinewidth=1.0)
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.set_ylabel("hot-swap benefit (%)")
    ax.annotate("structurally\nimpossible", xy=(0.08, -0.25), xytext=(0.55, -1.9),
                fontsize=7, color="#666666",
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.8))
    ax.set_title("Fig. C  Hot-swap needs batteries before it pays", fontsize=9)

    ax2.bar(range(len(labels)), pen, width=0.55, color="#BCBCDA")
    for i, v in enumerate(pen):
        ax2.text(i, v + 0.25, f"{v:+.1f}%", ha="center", fontsize=6.6, color="#555555")
    ax2.set_ylabel("baseline cost penalty (%)")
    ax2.set_xlabel("spare battery packs per truck")
    ax2.set_ylim(0, max(pen) * 1.35)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels)

    fig.tight_layout()
    out = FIGS / "figC_battery_stock.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------- 图 D

def fig_d() -> None:
    # 结构指标（架次数/服务客户数）只有新跑的表带：统一用 1.10 kW 那一对臂（30 配对）
    keys = _keys()
    base = _load("bench_b_main.csv").merge(keys, on=["set", "seed"])
    hs = _load("bench_b_hs.csv").merge(keys, on=["set", "seed"])

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax = axes[0]
    x = np.arange(2)
    w = 0.34
    so = [base["n_sorties"].mean(), hs["n_sorties"].mean()]
    dr = [base["drone_served"].mean(), hs["drone_served"].mean()]
    ax.bar(x - w / 2, so, w, color="#BCBCDA", label="sorties")
    ax.bar(x + w / 2, dr, w, color=style.ACCENT, label="customers served by drone")
    for xi, v in zip(x - w / 2, so):
        ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)
    for xi, v in zip(x + w / 2, dr):
        ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(["baseline", "hot-swap"])
    ax.set_ylim(0, 36)
    ax.set_ylabel("count (mean over 30 runs)")
    ax.legend(loc="upper left", fontsize=6.8)
    ax.set_title("(a) Hot-swap flies more, truck unchanged", fontsize=8.5)

    ax = axes[1]
    labels = ["0", "1", "2", "3", "\u221e"]
    arms = ["e_b0", "e_b1", "e_b2", "e_b3", None]
    vals = []
    for arm in arms:
        d = base if arm is None else _load(f"bench_{arm}.csv")
        vals.append(d["n_sorties"].mean())
    ax.plot(range(len(labels)), vals, color=style.PALETTE[2], marker="s", lw=1.2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.25, f"{v:.1f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("spare battery packs per truck")
    ax.set_ylabel("sorties (mean)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("(b) Battery stock caps the sortie count", fontsize=8.5)

    fig.tight_layout()
    out = FIGS / "figD_structure.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------- 图 E

def fig_e() -> None:
    """换电耗时 → 收益。五个测点，标出 300 s 那个方向反常的点。"""
    keys = _keys()
    base = _load("bench_main.csv")
    if base is None:
        raise SystemExit("缺少 bench_main.csv")
    base = base.merge(keys, on=["set", "seed"])
    arms = [(0, "bench_hotswap.csv"), (60, "bench_sw60_hs.csv"), (120, "bench_sw120_hs.csv"),
            (240, "bench_sw240_hs.csv"), (300, "bench_sw300_hs.csv")]

    xs, ys, err_lo, err_hi = [], [], [], []
    for sec, name in arms:
        d = _load(name)
        if d is None:
            raise SystemExit(f"缺少 {name}")
        pct = _pct(d.merge(keys, on=["set", "seed"]), base)
        m, lo, hi = _ci(pct)
        xs.append(sec / 60.0)          # 换成分钟，横轴更好读
        ys.append(m)
        err_lo.append(m - lo)
        err_hi.append(hi - m)

    fig, ax = plt.subplots(figsize=(3.9, 2.9))
    ax.axhline(0, color="#333333", lw=0.8, ls="--")
    ax.errorbar(xs, ys, yerr=[err_lo, err_hi], color=style.ACCENT, marker="o", ms=4,
                lw=1.2, capsize=3, elinewidth=1.0, zorder=3)
    ax.plot(xs[:4], ys[:4], color="#B0B0B0", lw=0.8, ls=":", zorder=2)
    ax.annotate("300 s breaks the\ntrend (p=0.016)", xy=(5.0, ys[-1]), xytext=(3.1, ys[-1] - 0.9),
                fontsize=6.8, color="#666666",
                arrowprops=dict(arrowstyle="-|>", color="#999999", lw=0.8))
    ax.set_xlabel("battery-swap duration (min)")
    ax.set_ylabel("cost saving vs. no hot-swap (%)")
    ax.set_title("Fig. E  Longer swaps eat the benefit", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}" for x in xs])
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    out = FIGS / "figE_swap_time.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.relative_to(PROJECT_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="A,B,C,D,E", help="只画指定几张，如 --only B C")
    args = ap.parse_args()
    style.apply(font_size=9.0)
    FIGS.mkdir(parents=True, exist_ok=True)
    want = {s.strip().upper() for s in args.only.replace(",", " ").split()}
    print("画图：")
    if "A" in want:
        fig_a()
    if "B" in want:
        fig_b()
    if "C" in want:
        fig_c()
    if "D" in want:
        fig_d()
    if "E" in want:
        fig_e()
    print("完成。")


if __name__ == "__main__":
    main()
