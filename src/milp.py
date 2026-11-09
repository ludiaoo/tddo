"""MILP（Gurobi）：小算例的精确解，用于验证 ALNS 是否达到最优。

用途见 M6：论文 Table 3 里 10 客户的四个算例 ALNS 与 Gurobi **完全相等（gap 0%）**，
这里用同一个 10 客户算例检验我们的 ALNS 是否也能打到最优。

## 与论文 (2)–(79) 的三处刻意简化

1. **不建模禁飞区。** 绕行距离 `d_ijh` 依赖「航段与禁飞时段的相交关系」，也就是依赖到达时刻，
   是时间相关的分段距离；线性化需要 z_start / z_end / q 与 McCormick 四式（论文 75–78）。
   这里直接与 `use_nfz=False` 的 ALNS 对照——两边解的是**同一个**（无禁飞区）问题，
   比较依然严格。M2 的诊断显示 10 客户算例里需绕行的航段只占 3.7%，代价可接受。
2. **不建 p/u 序变量**，改用 MTZ 消子环；起降交替用「在飞标志 `al`」约束。
3. **无人机载荷用「到达时的在机量」递推**（`fa`/`ga`），等价于论文的 `f^a / f^t / g^a / g^t`，
   但省掉交接后的变量：本模型里交接只发生在起降点，落地量即交接量。

## 变量与约束（对应论文编号）

| 本模型 | 论文 | 说明 |
|---|---|---|
| `x[k,i,j]` | (2)–(12) | 卡车弧；流守恒 + MTZ 去子环 |
| `y[k,i,j]` / `wl` / `wr` | (6)(9) | 无人机弧；起降点必须是卡车访问的客户 |
| `al[k,i]` | (20)(21)(49) | 在飞标志：一架机同一时刻只能有一个架次 |
| `Tt` / `Td` | (22)–(31) | 两套时刻表，`w` 决定用什么基准 |
| `F` / `G` / `fa` / `ga` / `ft` | (52)–(73) | 双向载荷流与容量 |
| `E` / `ec` | (39)–(43) | 逐段能耗与单架次 SOC |
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .instance import Instance
from .solution import DEPOT, Route, Solution, Sortie

M_TIME = 1.0e5        # 时间用的大 M（秒）；算例时间尺度在 1e4 秒内
M_LOAD = 1.0e3        # 载荷用的大 M（kg）
M_ENERGY = 10.0       # 能耗用的大 M（kWh）


@dataclass
class MilpResult:
    cost: float
    bound: float
    gap: float
    cpu: float
    status: str
    solution: Solution | None
    n_vars: int
    n_cons: int
    # 目标的三项分解：无人机电费 / 卡车油费 / 司机工时（便于与评估器逐项对账）
    breakdown: dict[str, float] = None


def solve_milp(
    inst: Instance,
    *,
    time_limit: float = 900.0,
    mip_gap: float = 0.0,
    verbose: bool = False,
    allow_drone: bool = True,
    allow_hotswap: bool = False,
    fix: Solution | None = None,
) -> MilpResult:
    """`fix` 给定时把全部二元变量钉死到该解，只做**可行性校验**（M9.16 的诊断用）。"""
    """求解简化 MILP。`inst.nfz` 必须为空（禁飞区不在本模型内）。"""
    if inst.nfz:
        raise ValueError("MILP 版本不建模禁飞区，请先用 disable_nfz(inst)")

    p = inst.params
    base = inst.base
    C = list(inst.customers)                 # 1..n
    V = [DEPOT] + C
    KS = list(range(inst.n_trucks))
    n = len(C)

    dt_km = base.truck_dist_m / 1000.0
    tt_s = base.truck_time_s
    xy = np.array([[nd.x, nd.y] for nd in base.nodes], dtype=float)
    dd_km = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1)) / 1000.0
    td_s = dd_km * 1000.0 / p.drone_speed

    sv = {i: (p.service_time if i in C else 0.0) for i in V}
    e_win = {i: inst.tw[i - 1].e for i in C}
    l_win = {i: inst.tw[i - 1].l for i in C}

    m = gp.Model("td_nfz_milp")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.Threads = 0
    m.Params.MIPFocus = 1                    # 优先找可行解/证明最优，避免长时间卡在界上

    # ---------------- 变量 ----------------
    x, y, ec = {}, {}, {}
    for k in KS:
        for i in V:
            for j in V:
                if i != j:
                    x[k, i, j] = m.addVar(vtype=GRB.BINARY, name=f"x_{k}_{i}_{j}")
        for i in C:
            for j in C:
                if i != j:
                    y[k, i, j] = m.addVar(vtype=GRB.BINARY, name=f"y_{k}_{i}_{j}")
                    ec[k, i, j] = m.addVar(lb=0.0, name=f"ec_{k}_{i}_{j}")

    wl = {(k, i): m.addVar(vtype=GRB.BINARY, name=f"wl_{k}_{i}") for k in KS for i in C}
    wr = {(k, i): m.addVar(vtype=GRB.BINARY, name=f"wr_{k}_{i}") for k in KS for i in C}
    if not allow_drone:                      # 退化校验：全卡车，用来对照 Truck-only
        for d in (y, wl, wr):
            for var in d.values():
                var.UB = 0.0
    al = {(k, i): m.addVar(lb=0.0, ub=1.0, name=f"al_{k}_{i}") for k in KS for i in C}
    u = {(k, i): m.addVar(lb=0.0, ub=n + 1, name=f"u_{k}_{i}") for k in KS for i in C}

    Tt = {(k, i): m.addVar(lb=0.0, name=f"Tt_{k}_{i}") for k in KS for i in V}
    # ⚠ depot 在时间上也是两个时刻：出发（0）与返仓（目标里的 T_返回仓）。
    # 若共用一个变量，返仓时刻会被写回出发值，导致「先返仓再出发」的时序自相矛盾
    # （实测结论：IIS 直接指向 Tt_step_2_0_7 → Tt_step_2_4_0 的循环）。
    TtEnd = {k: m.addVar(lb=0.0, name=f"TtEnd_{k}") for k in KS}
    for k in KS:
        m.addConstr(Tt[k, DEPOT] == 0.0, name=f"Tt_start_{k}")
    Td = {(k, i): m.addVar(lb=0.0, name=f"Td_{k}_{i}") for k in KS for i in C}
    F = {(k, i): m.addVar(lb=0.0, name=f"F_{k}_{i}") for k in KS for i in V}
    G = {(k, i): m.addVar(lb=0.0, name=f"G_{k}_{i}") for k in KS for i in V}
    fa = {(k, i): m.addVar(lb=0.0, name=f"fa_{k}_{i}") for k in KS for i in C}
    ga = {(k, i): m.addVar(lb=0.0, name=f"ga_{k}_{i}") for k in KS for i in C}
    ft = {(k, i): m.addVar(lb=0.0, name=f"ft_{k}_{i}") for k in KS for i in C}
    E = {(k, i): m.addVar(lb=0.0, name=f"E_{k}_{i}") for k in KS for i in C}

    f = {i: float(inst.f[i]) for i in C}
    g = {i: float(inst.g[i]) for i in C}

    # ---------------- 路径 ----------------
    for k in KS:
        m.addConstr(gp.quicksum(x[k, DEPOT, j] for j in C) <= 1, name=f"depot_out_{k}")
        m.addConstr(
            gp.quicksum(x[k, DEPOT, j] for j in C) == gp.quicksum(x[k, i, DEPOT] for i in C),
            name=f"depot_bal_{k}",
        )
        for j in C:
            m.addConstr(
                gp.quicksum(x[k, i, j] for i in V if i != j)
                == gp.quicksum(x[k, j, i] for i in V if i != j),
                name=f"truck_flow_{k}_{j}",
            )

    # 每个客户恰好被服务一次（跨车）。
    # ⚠ 回收点要特殊处理：无人机飞到回收点是为了**被卡车收回**，不算「服务该客户」。
    # 因此右端是 `1 + Σ_k wr[k,j]`，正是论文 (18)(19) 里 `1 ± w_j` 的那个 w。
    for j in C:
        m.addConstr(
            gp.quicksum(x[k, i, j] for k in KS for i in V if i != j)
            + gp.quicksum(y[k, i, j] for k in KS for i in C if i != j)
            == 1 + gp.quicksum(wr[k, j] for k in KS),
            name=f"serve_once_{j}",
        )

    for k in KS:
        # 无人机流守恒：起降点作为源/汇
        for i in C:
            m.addConstr(
                gp.quicksum(y[k, i, j] for j in C if j != i)
                - gp.quicksum(y[k, j, i] for j in C if j != i)
                == wl[k, i] - wr[k, i],
                name=f"drone_flow_{k}_{i}",
            )
            # ⚠ M9.16：`lr_once` 是「同一节点不能既回收又起飞」的唯一来源——论文用单个
            # 二元变量 w_i 表示这两件事，天生表达不了「落地换电再起飞」。放开它（allow_hotswap）
            # 之后要同时改三处，否则模型要么不可行、要么把换电的时间成本抹掉：
            #   ① 卡车到访判据从 ≤1 改成 ≤1+Σx（一个节点可同时承担两件事）；
            #   ② Td 在「纯起飞点」清零，但热换点要保留落地时刻，否则 Tt 等不到机；
            #   ③ 起飞时刻仍从 Tt 递推，所以换电的时间成本由 Tt_wait_drone + launch_time 承担。
            if allow_hotswap:
                m.addConstr(
                    wl[k, i] + wr[k, i] <= 1 + gp.quicksum(x[k, j, i] for j in V if j != i),
                    name=f"lr_truck_visit_{k}_{i}",
                )
                m.addConstr(
                    Td[k, i] <= M_TIME * (1 - wl[k, i] + wr[k, i]),
                    name=f"Td_zero_at_launch_{k}_{i}",
                )
            else:
                m.addConstr(wl[k, i] + wr[k, i] <= 1, name=f"lr_once_{k}_{i}")
                m.addConstr(
                    wl[k, i] + wr[k, i] <= gp.quicksum(x[k, j, i] for j in V if j != i),
                    name=f"lr_truck_visit_{k}_{i}",
                )
                m.addConstr(Td[k, i] <= M_TIME * (1 - wl[k, i]), name=f"Td_zero_at_launch_{k}_{i}")
            m.addConstr(fa[k, i] <= M_LOAD * (1 - wr[k, i]), name=f"fa_zero_at_recover_{k}_{i}")
            # 拾取量只在「无人机到过这个点」时才可能非零（不是只在回收点）。
            # 初版写成 ga ≤ M·wr，等于禁止无人机途中收货，模型直接不可行。
            m.addConstr(
                ga[k, i] <= M_LOAD * gp.quicksum(y[k, j, i] for j in C if j != i),
                name=f"ga_only_if_visited_{k}_{i}",
            )
            m.addConstr(ft[k, i] <= p.drone_capacity + M_LOAD * (1 - wl[k, i]), name=f"ft_cap_{k}_{i}")

        m.addConstr(
            gp.quicksum(wl[k, i] for i in C) == gp.quicksum(wr[k, i] for i in C),
            name=f"lr_balance_{k}",
        )

        # MTZ 去子环
        for i in C:
            for j in C:
                if i != j:
                    m.addConstr(
                        u[k, j] >= u[k, i] + 1 - (n + 1) * (1 - x[k, i, j]),
                        name=f"mtz_{k}_{i}_{j}",
                    )
                    # 在飞标志：一架机同一时刻只能有一个架次
                    m.addConstr(
                        al[k, j]
                        >= al[k, i] + wl[k, i] - wr[k, i] - 1.0 * (1 - x[k, i, j]),
                        name=f"al_lb_{k}_{i}_{j}",
                    )
                    m.addConstr(
                        al[k, j]
                        <= al[k, i] + wl[k, i] - wr[k, i] + 1.0 * (1 - x[k, i, j]),
                        name=f"al_ub_{k}_{i}_{j}",
                    )

    # ---------------- 时间 ----------------
    for k in KS:
        for i in V:
            for j in V:
                if i == j:
                    continue
                if j == DEPOT:
                    # 返仓时刻单独记（对应目标里的 T_返回仓）
                    m.addConstr(
                        TtEnd[k]
                        >= Tt[k, i] + sv[i] + p.launch_time * (wl[k, i] if i in C else 0.0)
                        + tt_s[i][j] - M_TIME * (1 - x[k, i, j]),
                        name=f"Tt_return_{k}_{i}",
                    )
                    continue
                # 卡车时刻递推（j 为客户）
                m.addConstr(
                    Tt[k, j]
                    >= Tt[k, i]
                    + sv[i]
                    + p.launch_time * (wl[k, i] if i in C else 0.0)
                    + tt_s[i][j]
                    + (p.recovery_time * wr[k, j] if j in C else 0.0)
                    - M_TIME * (1 - x[k, i, j]),
                    name=f"Tt_step_{k}_{i}_{j}",
                )
                if j in C:
                    # 时间窗用「不等候到达」判（约束 35/36）
                    m.addConstr(
                        Tt[k, i]
                        + sv[i]
                        + p.launch_time * (wl[k, i] if i in C else 0.0)
                        + tt_s[i][j]
                        <= l_win[j] + M_TIME * (1 - x[k, i, j]),
                        name=f"Tt_window_{k}_{i}_{j}",
                    )
        for j in C:
            served_by_truck = gp.quicksum(x[k, i, j] for i in V if i != j)
            # 早到则等到窗口开启
            m.addConstr(Tt[k, j] >= e_win[j] - M_TIME * (1 - served_by_truck), name=f"Tt_wait_{k}_{j}")
            m.addConstr(Tt[k, j] >= Td[k, j] + p.recovery_time * wr[k, j] - M_TIME * (1 - wr[k, j]),
                        name=f"Tt_wait_drone_{k}_{j}")

        for i in C:
            for j in C:
                if i == j:
                    continue
                # 无人机时刻：从上一个无人机客户飞过来
                m.addConstr(
                    Td[k, j]
                    >= Td[k, i] + sv[i] + td_s[i][j] - M_TIME * (1 - y[k, i, j]) - M_TIME * wl[k, i],
                    name=f"Td_step_{k}_{i}_{j}",
                )
                # 无人机时刻：从起飞点出发（基准是卡车到达时刻）
                m.addConstr(
                    Td[k, j]
                    >= Tt[k, i]
                    + sv[i]
                    + p.launch_time
                    + td_s[i][j]
                    - M_TIME * (1 - y[k, i, j])
                    - M_TIME * (1 - wl[k, i]),
                    name=f"Td_launch_{k}_{i}_{j}",
                )

        for j in C:
            # ⚠ 只有「无人机到此**服务该客户**」才受时间窗约束。
            # 回收点是无人机落地的地方，不受该客户的窗口限制：
            # 评估器的顺序是「等机 → 回收(+recovery) → 若早到再等到 e_j」，
            # 若这里把窗口也压到回收点上，会得到 e_j + recovery_time，比评估器多一个回收耗时
            # （实测 Set2 就差在整整 30 s = recovery_time）。
            drone_served = gp.quicksum(y[k, i, j] for i in C if i != j) - wr[k, j]
            m.addConstr(Td[k, j] >= e_win[j] - M_TIME * (1 - drone_served), name=f"Td_wait_{k}_{j}")
            m.addConstr(Td[k, j] <= l_win[j] + M_TIME * (1 - drone_served), name=f"Td_window_{k}_{j}")

    # ---------------- 载荷 ----------------
    for k in KS:
        m.addConstr(
            F[k, DEPOT]
            == gp.quicksum(
                f[j] * (gp.quicksum(x[k, i, j] for i in V if i != j)
                        + gp.quicksum(y[k, i, j] for i in C if i != j)
                        - wr[k, j])          # 回收点由卡车服务，不重复计入本车的派件装载
                for j in C
            ),
            name=f"F_init_{k}",
        )
        m.addConstr(G[k, DEPOT] == 0.0, name=f"G_init_{k}")

        for i in V:
            for j in V:
                if i == j:
                    continue
                # ⚠ F/G 只在「客户 → 客户」的弧上递推。depot 处 F 表示**出车装载**（约束 52），
                # 若对「客户 → depot」也递推，会与 F_init 的等式冲突（回库剩余量恒为 0）。
                if i in C and j != DEPOT:
                    m.addConstr(
                        F[k, j] >= F[k, i] - f[i] - ft[k, i] - M_LOAD * (1 - x[k, i, j]),
                        name=f"F_lb_{k}_{i}_{j}",
                    )
                    m.addConstr(
                        F[k, j] <= F[k, i] - f[i] - ft[k, i] + M_LOAD * (1 - x[k, i, j]),
                        name=f"F_ub_{k}_{i}_{j}",
                    )
                    m.addConstr(
                        G[k, j] >= G[k, i] + g[i] + ga[k, i] - M_LOAD * (1 - x[k, i, j]),
                        name=f"G_lb_{k}_{i}_{j}",
                    )
                    m.addConstr(
                        G[k, j] <= G[k, i] + g[i] + ga[k, i] + M_LOAD * (1 - x[k, i, j]),
                        name=f"G_ub_{k}_{i}_{j}",
                    )
                # 卡车容量（约束 62）
                m.addConstr(
                    F[k, i] + G[k, i] <= p.truck_capacity + M_LOAD * (1 - x[k, i, j]),
                    name=f"truck_cap_{k}_{i}_{j}",
                )

        m.addConstr(
            F[k, DEPOT] <= p.truck_capacity,
            name=f"truck_cap_init_{k}",
        )

        # 无人机载荷递推（约束 63–66 的等价形式）
        for i in C:
            for j in C:
                if i == j:
                    continue
                # 情形一：i 是无人机客户（继续飞）
                m.addConstr(
                    fa[k, j] >= fa[k, i] - f[i] - M_LOAD * (1 - y[k, i, j]) - M_LOAD * wl[k, i],
                    name=f"fa_lb_{k}_{i}_{j}",
                )
                m.addConstr(
                    fa[k, j] <= fa[k, i] - f[i] + M_LOAD * (1 - y[k, i, j]) + M_LOAD * wl[k, i],
                    name=f"fa_ub_{k}_{i}_{j}",
                )
                m.addConstr(
                    ga[k, j] >= ga[k, i] + g[i] - M_LOAD * (1 - y[k, i, j]) - M_LOAD * wl[k, i],
                    name=f"ga_lb_{k}_{i}_{j}",
                )
                m.addConstr(
                    ga[k, j] <= ga[k, i] + g[i] + M_LOAD * (1 - y[k, i, j]) + M_LOAD * wl[k, i],
                    name=f"ga_ub_{k}_{i}_{j}",
                )
                # 情形二：i 是起飞点（换装载）
                m.addConstr(
                    fa[k, j] >= ft[k, i] - M_LOAD * (1 - y[k, i, j]) - M_LOAD * (1 - wl[k, i]),
                    name=f"fa_load_{k}_{i}_{j}",
                )
                m.addConstr(
                    fa[k, j] <= ft[k, i] + M_LOAD * (1 - y[k, i, j]) + M_LOAD * (1 - wl[k, i]),
                    name=f"fa_load_ub_{k}_{i}_{j}",
                )
                m.addConstr(
                    ga[k, j] <= M_LOAD * (1 - y[k, i, j]) + M_LOAD * (1 - wl[k, i]),
                    name=f"ga_empty_{k}_{i}_{j}",
                )
                # 无人机容量（约束 73）
                m.addConstr(
                    fa[k, j] + ga[k, j] <= p.drone_capacity + M_LOAD * (1 - y[k, i, j]),
                    name=f"drone_cap_{k}_{i}_{j}",
                )

                # 能耗：本段载荷 = 到达 j 时的在机量（约束 39/41）
                m.addConstr(
                    ec[k, i, j]
                    >= dd_km[i][j] * (p.e_drone * (fa[k, j] + ga[k, j]) + p.b) - M_ENERGY * (1 - y[k, i, j]),
                    name=f"ec_lb_{k}_{i}_{j}",
                )
                m.addConstr(ec[k, i, j] <= M_ENERGY * y[k, i, j], name=f"ec_ub_{k}_{i}_{j}")

                # 累计能耗：起飞点清零（约束 40），逐段累加（41/42）
                m.addConstr(
                    E[k, j] >= E[k, i] + ec[k, i, j] - M_ENERGY * (1 - y[k, i, j]),
                    name=f"E_step_{k}_{i}_{j}",
                )

            # 起飞点能耗清零（约束 40）：wl=1 时 E=0，其余节点不设上限（只受 SOC 约束）
            m.addConstr(E[k, i] <= M_ENERGY * (1 - wl[k, i]), name=f"E_reset_{k}_{i}")
            m.addConstr(E[k, i] <= p.eta_soc * p.drone_battery, name=f"soc_{k}_{i}")

    # ---------------- 目标（式 74 的三项） ----------------
    m.setObjective(
        p.c_drone * gp.quicksum(ec.values())
        + p.c_truck * p.e_truck * gp.quicksum(dt_km[i][j] * x[k, i, j] for k in KS for i in V for j in V if i != j)
        + p.c_customer * gp.quicksum(TtEnd[k] for k in KS),
        GRB.MINIMIZE,
    )

    if fix is not None:
        _pin(m, fix, inst, x, y, wl, wr)
        m.setObjective(0.0, GRB.MINIMIZE)      # 只判可行，不看目标

    t0 = time.time()
    m.optimize()
    cpu = time.time() - t0

    if m.SolCount == 0:
        if fix is not None:                    # 诊断：哪几条约束与「固定解」冲突
            try:
                m.computeIIS()
                bad = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
                print(f"  [IIS] 冲突约束 {len(bad)} 条：{bad[:8]}")
            except Exception as exc:           # noqa: BLE001
                print(f"  [IIS] 计算失败：{exc}")
        return MilpResult(float("nan"), float("nan"), float("nan"), cpu, f"no solution(status={m.Status})",
                          None, m.NumVars, m.NumConstrs)

    sol = _extract(inst, m, x, y, wl, wr)
    ec_sum = sum(v.X for v in ec.values())
    dist_km = sum(dt_km[i][j] * x[k, i, j].X for k in KS for i in V for j in V if i != j)
    t_sum = sum(TtEnd[k].X for k in KS)
    breakdown = {
        "drone_energy_kwh": ec_sum,
        "drone_cost": p.c_drone * ec_sum,
        "truck_distance_m": dist_km * 1000.0,
        "truck_cost": p.c_truck * p.e_truck * dist_km,
        "finish_time_s": t_sum,
        "driver_cost": p.c_customer * t_sum,
    }
    return MilpResult(
        cost=float(m.ObjVal),
        bound=float(m.ObjBound),
        gap=float(m.MIPGap),
        cpu=cpu,
        status={GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time limit"}.get(m.Status, str(m.Status)),
        solution=sol,
        n_vars=m.NumVars,
        n_cons=m.NumConstrs,
        breakdown=breakdown,
    )


def _pin(m, sol: Solution, inst: Instance, x, y, wl, wr) -> None:
    """把 `sol` 钉进模型：路径弧、无人机弧、起降标志全部取定值，其余二元变量置 0。

    用途：验证「我们的启发式解在 MILP 的约束下到底可不可行」。若不可行，
    `computeIIS` 能直接指出是时间、载荷还是能耗的哪一条被违反。
    """
    keep_x, keep_y, keep_l, keep_r = set(), set(), set(), set()
    for r in sol.routes:
        k = r.truck
        for a, b in zip(r.nodes, r.nodes[1:]):
            keep_x.add((k, a, b))
        for s in r.sorties:
            p, q = r.nodes[s.launch], r.nodes[s.recover]
            keep_l.add((k, p))
            keep_r.add((k, q))
            chain = [p, *s.customers, q]
            for a, b in zip(chain, chain[1:]):
                keep_y.add((k, a, b))

    for key, var in x.items():
        m.addConstr(var == (1.0 if key in keep_x else 0.0), name=f"fix_x_{key[0]}_{key[1]}_{key[2]}")
    for key, var in y.items():
        m.addConstr(var == (1.0 if key in keep_y else 0.0), name=f"fix_y_{key[0]}_{key[1]}_{key[2]}")
    for key, var in wl.items():
        m.addConstr(var == (1.0 if key in keep_l else 0.0), name=f"fix_wl_{key[0]}_{key[1]}")
    for key, var in wr.items():
        m.addConstr(var == (1.0 if key in keep_r else 0.0), name=f"fix_wr_{key[0]}_{key[1]}")


def _val(v: dict, key, tol: float = 0.5) -> bool:
    var = v.get(key)
    return var is not None and var.X > tol


def _extract(inst: Instance, m, x, y, wl, wr) -> Solution:
    """把 MILP 解翻译成项目内的 Solution，便于用评估器复核。"""
    routes: list[Route] = []
    for k in range(inst.n_trucks):
        nodes = [DEPOT]
        # 沿卡车弧走
        while True:
            cur = nodes[-1]
            nxt = None
            for j in [v for v in range(inst.base.n_nodes) if v != cur]:
                var = x.get((k, cur, j))
                if var is not None and var.X > 0.5:
                    nxt = j
                    break
            if nxt is None:
                break
            nodes.append(nxt)
            if nxt == DEPOT or len(nodes) > inst.base.n_nodes + 1:
                break
        if len(nodes) <= 2:
            continue

        pos = {nd: i for i, nd in enumerate(nodes)}
        sorties: list[Sortie] = []
        for i in nodes[1:-1]:
            if not _val(wl, (k, i)):
                continue
            # 沿无人机弧走一圈
            seq: list[int] = []
            cur = i
            while True:
                nxt = None
                for j in range(1, inst.base.n_nodes):
                    if j == cur:
                        continue
                    var = y.get((k, cur, j))
                    if var is not None and var.X > 0.5:
                        nxt = j
                        break
                if nxt is None:
                    break
                if _val(wr, (k, nxt)):
                    seq.append(("recover", nxt))
                    break
                seq.append(("cust", nxt))
                cur = nxt
            custs = tuple(v for t, v in seq if t == "cust")
            rec = [v for t, v in seq if t == "recover"]
            if not rec or not custs or rec[0] not in pos:
                continue
            sorties.append(Sortie(launch=pos[i], recover=pos[rec[0]], customers=custs))
        routes.append(Route(truck=k, nodes=nodes, sorties=sorties))
    return Solution(routes)
