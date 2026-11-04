"""无人机插入候选的**廉价**特征（M9.1 预筛，v3 O(1) 版）。

红线：任何特征都不得调用 `_route_eval` / `_simulate_route`——否则「省一次评估」就成了空话。

v3 的关键：把每个架次的「序列几何/时刻」在 `route_ctx` 里做成**前缀数组**，
插入位置 pos 的全部特征变成 O(1)（v2 是 O(架次长)，实测占掉一次评估的本钱）：

- `seg[j]`：原序列第 j 条直飞航段长；`prefix_dist / prefix_ft`：距离与「飞行+服务」时间前缀和；
- 插入 pos 的 `Δlen = d_prev + d_next − seg[pos]`，`t_c = t_launch + prefix_ft[pos] + d_prev/v`，
  `t_back = t_c + service + d_next/v + suffix_ft[pos+1] + recovery`；
- 时间用「无等待前缀时刻」近似；路线级精确量（finish/waiting，来自本来就要算的
  `before` 评估）作为特征让模型自己学修正。
"""

from __future__ import annotations

import math

from ..geometry import segment_intersects_circle
from ..solution import DEPOT

FEATURE_NAMES = (
    "kind",                # 0=并入已有架次, 1=新建架次
    "k_sortie",            # 该架次已有客户数
    "pos",                 # 插入位置
    "pos_rel",             # pos / max(k,1)
    "f_c", "g_c",          # 客户派件 / 取件
    "sortie_f", "sortie_g",        # 架次内已有的派件 / 取件合计
    "load_ratio",                  # (sortie_f + f_c) / 无人机载荷
    "pickup_ratio",                # (sortie_g + g_c) / 无人机载荷
    "d_prev", "d_next",            # 新序列里 c 与前驱 / 后继的直飞距离
    "seq_flight_len",              # 新序列全部直飞航段之和
    "delta_flight_len",            # 相对原序列的直飞增量（新建架次 = seq_flight_len）
    "anchor_truck_dist",           # 锚点段卡车路网距离
    "anchor_nfz_cross",            # 锚点（直线）穿过的禁飞圆个数
    "t_anchor_est",                # 锚点处的无等待前缀时刻 + 服务 + 起飞准备
    "slack_c_est",                 # 客户时间窗上界 − 无人机预计到达（可正可负）
    "delay_est",                   # 回收晚于卡车到达的估计等待（max(0, ·)）
    "min_slack_after",             # 回收点之后所有客户的最小 (l − t_est)
    "route_len", "route_n_sorties",
    "route_waiting", "route_finish", "route_drone_dist",   # before 评估里的精确量
    "n_customers", "n_trucks", "nfz_count",
)


def _leg_nfz(pa, pb, zones) -> int:
    return sum(
        1 for z in zones
        if segment_intersects_circle((pa.x, pa.y), (pb.x, pb.y), (z.cx, z.cy), z.r)
    )


def route_ctx(inst, r) -> dict:
    """每条路线算一次的上下文（含每个架次的前缀数组）。"""
    nodes = r.nodes
    p = inst.params
    f_list = inst.f.tolist()
    g_list = inst.g.tolist()
    tw_l = [w.l for w in inst.tw]
    xy = inst.base.nodes

    # 无等待前缀时刻（行驶 + 服务，不含无人机回收导致的等待）
    t = 0.0
    t_est = [0.0] * len(nodes)
    for k in range(len(nodes) - 1):
        i, j = nodes[k], nodes[k + 1]
        t += (p.service_time if i != DEPOT else 0.0) + float(inst.base.truck_time_s[i, j])
        t_est[k + 1] = t

    # min_slack_after[s]：位置 >= s 的客户里最小的 (l − t_est)；仓库不计
    min_slack_after = [math.inf] * (len(nodes) + 1)
    for s in range(len(nodes) - 1, -1, -1):
        nd = nodes[s]
        cur = math.inf if nd == DEPOT else tw_l[nd - 1] - t_est[s]
        min_slack_after[s] = min(cur, min_slack_after[s + 1])

    # 每条相邻段（卡车腿）的穿圆数——只依赖 (p, q) 点对，按算例全局记忆化
    # （路线每步都在变，但腿是固定节点对，跨步命中率很高）
    leg_memo = getattr(inst, "_ml_leg_nfz", None)
    if leg_memo is None:
        leg_memo = {}
        inst._ml_leg_nfz = leg_memo
    leg_nfz = []
    for k in range(len(nodes) - 1):
        key = (nodes[k], nodes[k + 1])
        got = leg_memo.get(key)
        if got is None:
            got = _leg_nfz(xy[nodes[k]], xy[nodes[k + 1]], inst.nfz)
            leg_memo[key] = got
        leg_nfz.append(got)

    # 每个架次：锚点、载荷合计、序列几何与时刻前缀
    specs_info = []
    for s in r.sorties:
        sp, sq, cust = s.launch, s.recover, s.customers
        pa, pb = xy[nodes[sp]], xy[nodes[sq]]
        pts = [pa] + [xy[x] for x in cust] + [pb]
        k = len(cust)
        seg = [
            math.hypot(pts[i].x - pts[i + 1].x, pts[i].y - pts[i + 1].y)
            for i in range(k + 1)
        ]
        old_len = math.fsum(seg)
        # prefix_ft[i] = 到达 seq[i] **之前**累计的（飞行 + 服务）时间；i = 0..k+1
        # 插入 pos：t_c = t_launch + prefix_ft[pos] + d_prev/v
        prefix_ft = [0.0] * (k + 2)
        for i in range(k + 1):
            prefix_ft[i + 1] = prefix_ft[i] + seg[i] / p.drone_speed + p.service_time
        specs_info.append({
            "sp": sp, "sq": sq, "cust": cust,
            "f_sum": math.fsum(f_list[x] for x in cust),
            "g_sum": math.fsum(g_list[x] for x in cust),
            "old_len": old_len,
            "nfz": _leg_nfz(pa, pb, inst.nfz),
            "truck_dist": float(inst.base.truck_dist_m[nodes[sp], nodes[sq]]),
            "t_launch": t_est[sp] + (p.service_time if nodes[sp] != DEPOT else 0.0) + p.launch_time,
            "seg": seg, "pts": pts,
            "prefix_ft": prefix_ft,
            "suffix_ft": [math.fsum(prefix_ft[i + 1:]) for i in range(k + 2)],
        })

    return {
        "t_est": t_est, "min_slack_after": min_slack_after, "nodes": nodes,
        "f_list": f_list, "g_list": g_list, "tw_l": tw_l,
        "leg_nfz": leg_nfz, "specs_info": specs_info,
    }


def drone_candidate_features(
    inst,
    ctx: dict,
    before,                      # 原路线的 RouteEval（本来就要算，白拿的精确量）
    *,
    kind: int,                   # 0=并入已有架次, 1=新建架次
    sp: int, sq: int,            # 锚点在 r.nodes 里的位置
    sortie: tuple[int, ...],     # 原架次客户序列（新建架次时为空）
    pos: int,                    # 插入位置
    c: int,
    anchor_nfz: int = 0,         # 锚点穿圆数（route_ctx 已预算）
    old_len: float = 0.0,        # 原架次飞行长度（merge 用，route_ctx 已预算）
    anchor_truck_dist: float = 0.0,
    sortie_f: float | None = None,   # 架次派件合计（route_ctx 已预算，省一次遍历）
    sortie_g: float | None = None,   # 架次取件合计（同上）
    si: dict | None = None,          # merge 时的 ctx["specs_info"][i]
    leg_k: int | None = None,        # new 时的卡车腿下标
) -> tuple:
    """返回与 `FEATURE_NAMES` 一一对应的特征元组。全 O(1)，无任何仿真。"""
    p = inst.params
    xy = inst.base.nodes
    f_list, g_list = ctx["f_list"], ctx["g_list"]
    pn, qn = ctx["nodes"][sp], ctx["nodes"][sq]

    f_c = f_list[c]
    g_c = g_list[c]
    if sortie_f is None:
        sortie_f = math.fsum(f_list[x] for x in sortie) if sortie else 0.0
    if sortie_g is None:
        sortie_g = math.fsum(g_list[x] for x in sortie) if sortie else 0.0

    pc = xy[c]
    if kind == 0 and si is not None:
        prev_pt = si["pts"][pos]                 # 原序列里 c 的前驱（pos=0 时是起飞锚点）
        next_pt = si["pts"][pos + 1]             # 后继（pos=k 时是回收锚点）
        d_prev = math.hypot(prev_pt.x - pc.x, prev_pt.y - pc.y)
        d_next = math.hypot(pc.x - next_pt.x, pc.y - next_pt.y)
        seg_replaced = si["seg"][pos]
        t_launch = si["t_launch"]
        pre_ft = si["prefix_ft"][pos]
        suf_ft = si["suffix_ft"][pos + 1]
    else:                                        # 新建架次：序列就是 [c]
        prev_pt, next_pt = xy[pn], xy[qn]
        d_prev = math.hypot(prev_pt.x - pc.x, prev_pt.y - pc.y)
        d_next = math.hypot(pc.x - next_pt.x, pc.y - next_pt.y)
        seg_replaced = 0.0
        t_launch = ctx["t_est"][sp] + (p.service_time if pn != DEPOT else 0.0) + p.launch_time
        pre_ft = 0.0
        suf_ft = 0.0

    delta_flight = d_prev + d_next - seg_replaced
    seq_flight_len = old_len + delta_flight if kind == 0 else d_prev + d_next

    t_c = t_launch + pre_ft + d_prev / p.drone_speed
    slack_c = ctx["tw_l"][c - 1] - t_c
    t_back = t_c + p.service_time + d_next / p.drone_speed + suf_ft + p.recovery_time
    t_q = ctx["t_est"][sq] + (p.service_time if qn != DEPOT else 0.0)
    delay_est = max(0.0, t_back - t_q)

    return (
        kind,
        len(sortie),
        pos,
        pos / max(len(sortie), 1),
        f_c, g_c,
        sortie_f, sortie_g,
        (sortie_f + f_c) / p.drone_capacity,
        (sortie_g + g_c) / p.drone_capacity,
        d_prev, d_next,
        seq_flight_len,
        delta_flight,
        anchor_truck_dist,
        anchor_nfz,
        t_launch,
        slack_c,
        delay_est,
        ctx["min_slack_after"][sq],
        len(ctx["nodes"]),
        before.n_sorties,
        before.waiting_time,
        before.finish_time,
        before.drone_distance,
        inst.n_customers,
        inst.n_trucks,
        len(inst.nfz),
    )
