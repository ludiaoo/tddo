"""等待时间结构诊断（M9.11）：把成本大头拆开看。

回答两个问题：
1. 单车「等待」占收车时长的比例，以及其中多少是在**等无人机**、多少是在**等客户窗口**；
2. 如果把这两块分别抹掉，成本能降多少（这是协调的价值**上界**，不是可达值）。

    python -m scripts.wait_diag
    python -m scripts.wait_diag --sets 9,13 --iters 1200
"""

from __future__ import annotations

import argparse

from src.alns import solve
from src.evaluator import _simulate_route
from src.generator import build_instance
from src.io import load_config, load_sets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="5,7,9,12,13,15")
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}
    print("=== 1. 等待的构成（单车平均）===")
    print(f"{'算例':<6}{'makespan':>9}{'Σ收车':>9}{'等无人机':>10}{'等窗口':>9}"
          f"{'合计':>9}{'占比':>8}   构成")
    tot = [0.0, 0.0, 0.0]      # c0, c1, c2

    for sid in [int(x) for x in args.sets.split(",")]:
        inst = build_instance(by_id[sid], cfg)
        k = inst.params.c_customer
        r = solve(inst, seed=args.seed, iters=args.iters)
        wd = ww = wf = mk = 0.0
        c0 = c1 = c2 = 0.0
        for rt in r.best.routes:
            ev = _simulate_route(inst, rt, fast=False)
            wd += ev.wait_drone
            ww += ev.wait_window
            wf += ev.finish_time
            mk = max(mk, ev.finish_time)
            c0 += ev.cost(inst)
            c1 += ev.cost(inst) - k * ev.wait_drone
            c2 += ev.cost(inst) - k * ev.waiting_time
        tot[0] += c0; tot[1] += c1; tot[2] += c2
        frac = (wd + ww) / max(wf, 1) * 100
        print(f"Set{sid:<3d}{mk/3600:8.2f}h{wf/3600:8.2f}h{wd/3600:9.2f}h{ww/3600:8.2f}h"
              f"{(wd + ww)/3600:8.2f}h{frac:7.1f}%   "
              f"机 {wd / max(wd + ww, 1) * 100:.0f}% / 窗 {ww / max(wd + ww, 1) * 100:.0f}%")

    print("\n=== 2. 消除等待的价值上界（反事实，不可达）===")
    print(f"{'':<10}{'总成本':>10}{'去掉等无人机':>14}{'降幅':>8}{'去掉全部等待':>14}{'降幅':>8}")
    print(f"{'合计':<10}{tot[0]:10.2f}{tot[1]:14.2f}"
          f"{(tot[1] / tot[0] - 1) * 100:7.1f}%{tot[2]:14.2f}{(tot[2] / tot[0] - 1) * 100:7.1f}%")
    print("\n注：成本 = 无人机电费 + 卡车油费 + 司机工时（finish_time × 费率）。")
    print("    等待不是「解不好」，是模型强制的：卡车必须在回收点等无人机落地，客户窗口没开不能服务。")


if __name__ == "__main__":
    main()
