"""几何层诊断：统计 16 个算例里「绕行」与「节点禁飞」两条通道各起多大作用。

用法：python -m scripts.check_geometry

两条通道是论文里两个不同机制，不要混：
- **绕行**（Fig.4/Fig.5）：航段穿过圆且时段相交 → 多飞一段距离；
- **节点禁飞**（Fig.3）：起降点落在「正在禁飞」的圆内 → 该航段直接不可行，只能卡车服务。
"""

from __future__ import annotations

import math

import numpy as np

from src.distance import drone_leg
from src.generator import build_instance
from src.io import load_config, load_sets


def main() -> None:
    cfg = load_config()
    total = feasible = blocked = 0
    detours: list[float] = []
    per_set = []

    for s in load_sets():
        inst = build_instance(s, cfg)
        base, zones, n = inst.base, inst.nfz, inst.base.n_nodes
        st_total = st_block = 0
        st_det: list[float] = []
        for i in range(1, n):
            for j in range(1, n):
                if i == j:
                    continue
                for t0 in (0.0, 2000.0, 6000.0):
                    leg = drone_leg(base, zones, i, j, t0, t0 + 600.0)
                    total += 1
                    st_total += 1
                    assert math.isfinite(leg.distance) if leg.feasible else math.isinf(leg.distance)
                    if leg.feasible:
                        feasible += 1
                        assert leg.distance >= leg.straight - 1e-6
                        if leg.detour > 1e-9:
                            detours.append(leg.detour)
                            st_det.append(leg.detour)
                    else:
                        blocked += 1
                        st_block += 1
        per_set.append((inst.set_id, st_total, st_block / st_total, len(st_det) / st_total,
                        float(np.median(st_det)) if st_det else 0.0))

    print(f"{'Set':>3} {'航段':>7} {'节点禁飞%':>9} {'需绕行%':>8} {'绕行中位':>8}")
    print("-" * 40)
    for sid, tot, br, dr, dm in per_set:
        print(f"{sid:>3} {tot:>7} {br:>8.1%} {dr:>7.1%} {dm:>7.0f}m")

    arr = np.array(detours)
    print()
    print(f"合计 {total} 次航段计算，无异常")
    print(f"节点禁飞导致不可行：{blocked}/{total} = {blocked / total:.1%}")
    print(f"需几何绕行：{len(arr)}/{feasible} = {len(arr) / feasible:.1%}")
    if len(arr):
        print(f"绕行距离：中位 {np.median(arr):.0f} m，90 分位 {np.percentile(arr, 90):.0f} m，最大 {arr.max():.0f} m")


if __name__ == "__main__":
    main()
