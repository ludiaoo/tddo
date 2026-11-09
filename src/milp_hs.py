"""W1：能表达「同节点换电」的 MILP（与 `milp.py` 并存，不改动原模型）。

设计见 `W1_换电模型设计.md`。核心是把三类变量按「到达 / 出发」拆开：

| 到达侧 | 出发侧 | 为什么不拆不行 |
|---|---|---|
| `Ta[k,i]` | `Td[k,i]` | 原模型强制 `Td=0`（新架次计时起点），把落地时刻抹掉了，卡车就不再等机 |
| `fa_in/ga_in` | `fa_out/ga_out` | 原模型一个 `fa` 既要在回收点等于 0，又要等于新架次的装载 `ft` |
| `E_in` | `E_out` | 原模型在起飞点清零，导致上一架次最后一段的 SOC 检查被跳过 |

**等价性闸门**：`allow_hotswap=False` 时，本模型与原模型必须给出相同的最优值
（`tests/test_milp_hs.py::test_与原模型等价`）。不满足就说明拆分写错了。

用法：
    python -m scripts.run_milp_hs            # 4 个 10 客户算例，两臂对照
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from .instance import Instance
from .solution import DEPOT, Route, Solution, Sortie

M_TIME = 1.0e5
M_LOAD = 1.0e3
M_ENERGY = 10.0


@dataclass
class HsResult:
    cost: float
    bound: float
    gap: float
    cpu: float
    status: str
    solution: Solution | None
    n_vars: int
    n_cons: int


def solve_milp_hs(
    inst: Instance,
    *,
    time_limit: float = 300.0,
    mip_gap: float = 0.0,
    verbose: bool = False,
    allow_hotswap: bool = True,
    fix: Solution | None = None,
) -> HsResult:
    """求解「到达/出发拆分」的 MILP。`inst.nfz` 必须为空（与原模型同口径）。"""
    if inst.nfz:
        raise ValueError("本模型不建模禁飞区，请先用 disable_nfz(inst)")

    p = inst.params
    base = inst.base
    C = list(inst.customers)
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

    m = gp.Model("td_hotswap_milp")
    m.Params.OutputFlag = 1 if verbose else 0
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    m.Params.Threads = 0
    m.Params.MIPFocus = 1

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
    al = {(k, i): m.addVar(lb=0.0, ub=1.0, name=f"al_{k}_{i}") for k in KS for i in C}
    u = {(k, i): m.addVar(lb=0.0, ub=n + 1, name=f"u_{k}_{i}") for k in KS for i in C}

    Tt = {(k, i): m.addVar(lb=0.0, name=f"Tt_{k}_{i}") for k in KS for i in V}
    TtEnd = {k: m.addVar(lb=0.0, name=f"TtEnd_{k}") for k in KS}
    for k in KS:
        m.addConstr(Tt[k, DEPOT] == 0.0, name=f"Tt_start_{k}")

    Ta = {(k, i): m.addVar(lb=0.0, name=f"Ta_{k}_{i}") for k in KS for i in C}
    Td = {(k, i): m.addVar(lb=0.0, name=f"Td_{k}_{i}") for k in KS for i in C}
    fai = {(k, i): m.addVar(lb=0.0, name=f"fai_{k}_{i}") for k in KS for i in C}
    fao = {(k, i): m.addVar(lb=0.0, name=f"fao_{k}_{i}") for k in KS for i in C}
    gai = {(k, i): m.addVar(lb=0.0, name=f"gai_{k}_{i}") for k in KS for i in C}
    gao = {(k, i): m.addVar(lb=0.0, name=f"gao_{k}_{i}") for k in KS for i in C}
    Ein = {(k, i): m.addVar(lb=0.0, name=f"Ein_{k}_{i}") for k in KS for i in C}
    Eout = {(k, i): m.addVar(lb=0.0, name=f"Eout_{k}_{i}") for k in KS for i in C}

    F = {(k, i): m.addVar(lb=0.0, name=f"F_{k}_{i}") for k in KS for i in V}
    G = {(k, i): m.addVar(lb=0.0, name=f"G_{k}_{i}") for k in KS for i in V}
    ft = {(k, i): m.addVar(lb=0.0, name=f"ft_{k}_{i}") for k in KS for i in C}

    f = {i: float(inst.f[i]) for i in C}
    g = {i: float(inst.g[i]) for i in C}

    # ---------------- 卡车侧（与原模型逐条相同） ----------------
    for k in KS:
        m.addConstr(gp.quicksum(x[k, DEPOT, j] for j in C) <= 1, name=f"depot_out_{k}")
        m.addConstr(gp.quicksum(x[k, DEPOT, j] for j in C)
                    == gp.quicksum(x[k, i, DEPOT] for i in C), name=f"depot_bal_{k}")
        for j in C:
            m.addConstr(gp.quicksum(x[k, i, j] for i in V if i != j)
                        == gp.quicksum(x[k, j, i] for i in V if i != j), name=f"truck_flow_{k}_{j}")

    for j in C:
        m.addConstr(
            gp.quicksum(x[k, i, j] for k in KS for i in V if i != j)
            + gp.quicksum(y[k, i, j] for k in KS for i in C if i != j)
            == 1 + gp.quicksum(wr[k, j] for k in KS), name=f"serve_once_{j}")

    for k in KS:
        for i in C:
            # 无人机流守恒：起降点作为源/汇
            m.addConstr(gp.quicksum(y[k, i, j] for j in C if j != i)
                        - gp.quicksum(y[k, j, i] for j in C if j != i)
                        == wl[k, i] - wr[k, i], name=f"drone_flow_{k}_{i}")
            if not allow_hotswap:
                m.addConstr(wl[k, i] + wr[k, i] <= 1, name=f"lr_once_{k}_{i}")
                m.addConstr(wl[k, i] + wr[k, i]
                            <= gp.quicksum(x[k, j, i] for j in V if j != i),
                            name=f"lr_truck_visit_{k}_{i}")
            else:
                # 卡车必须在场——每个动作各要一次到访，但两个动作可以共用同一次到访。
                # ⚠ 不能只写 `wl+wr ≤ 1+Σx`：那会放行「wl=1 且无人到访」（1 ≤ 1+0），
                #    MILP 会凭空在没被访问的节点起飞，评估器随即判「客户未被服务」。
                truck_here = gp.quicksum(x[k, j, i] for j in V if j != i)
                m.addConstr(wl[k, i] <= truck_here, name=f"wl_truck_visit_{k}_{i}")
                m.addConstr(wr[k, i] <= truck_here, name=f"wr_truck_visit_{k}_{i}")
                m.addConstr(wl[k, i] + wr[k, i] <= 1 + truck_here,
                            name=f"lr_truck_visit_{k}_{i}")
            m.addConstr(ft[k, i] <= p.drone_capacity + M_LOAD * (1 - wl[k, i]), name=f"ft_cap_{k}_{i}")

        m.addConstr(gp.quicksum(wl[k, i] for i in C) == gp.quicksum(wr[k, i] for i in C),
                    name=f"lr_balance_{k}")

        for i in C:
            for j in C:
                if i == j:
                    continue
                m.addConstr(u[k, j] >= u[k, i] + 1 - (n + 1) * (1 - x[k, i, j]),
                            name=f"mtz_{k}_{i}_{j}")
                m.addConstr(al[k, j] >= al[k, i] + wl[k, i] - wr[k, i] - (1 - x[k, i, j]),
                            name=f"al_lb_{k}_{i}_{j}")
                m.addConstr(al[k, j] <= al[k, i] + wl[k, i] - wr[k, i] + (1 - x[k, i, j]),
                            name=f"al_ub_{k}_{i}_{j}")

    # 卡车时刻与时间窗
    for k in KS:
        for i in V:
            for j in V:
                if i == j:
                    continue
                if j == DEPOT:
                    m.addConstr(TtEnd[k] >= Tt[k, i] + sv[i]
                                + p.launch_time * (wl[k, i] if i in C else 0.0)
                                + tt_s[i][j] - M_TIME * (1 - x[k, i, j]), name=f"Tt_return_{k}_{i}")
                    continue
                m.addConstr(Tt[k, j] >= Tt[k, i] + sv[i]
                            + p.launch_time * (wl[k, i] if i in C else 0.0)
                            + tt_s[i][j]
                            + (p.recovery_time * wr[k, j] if j in C else 0.0)
                            - M_TIME * (1 - x[k, i, j]), name=f"Tt_step_{k}_{i}_{j}")
                if j in C:
                    m.addConstr(Tt[k, i] + sv[i]
                                + p.launch_time * (wl[k, i] if i in C else 0.0)
                                + tt_s[i][j] <= l_win[j] + M_TIME * (1 - x[k, i, j]),
                                name=f"Tt_window_{k}_{i}_{j}")
        for j in C:
            served = gp.quicksum(x[k, i, j] for i in V if i != j)
            m.addConstr(Tt[k, j] >= e_win[j] - M_TIME * (1 - served), name=f"Tt_wait_{k}_{j}")
            # (T5) 卡车在回收点等机落地：用**到达**时刻 Ta
            m.addConstr(Tt[k, j] >= Ta[k, j] + p.recovery_time * wr[k, j]
                        - M_TIME * (1 - wr[k, j]), name=f"Tt_wait_drone_{k}_{j}")

    # ---------------- 无人机时间：到达 / 出发 分开 ----------------
    for k in KS:
        for i in C:
            # (T3)(T4) 出发时刻：wl=1 时 = Tt+sv+launch，wl=0 时 = 0
            m.addConstr(Td[k, i] >= Tt[k, i] + sv[i] + p.launch_time - M_TIME * (1 - wl[k, i]),
                        name=f"Td_def_{k}_{i}")
            m.addConstr(Td[k, i] <= Tt[k, i] + sv[i] + p.launch_time + M_TIME * (1 - wl[k, i]),
                        name=f"Td_def_ub_{k}_{i}")
            m.addConstr(Td[k, i] <= M_TIME * wl[k, i], name=f"Td_zero_no_launch_{k}_{i}")

        for i in C:
            for j in C:
                if i == j:
                    continue
                # (T1) 续飞：从 i 的到达时刻继续
                m.addConstr(Ta[k, j] >= Ta[k, i] + sv[i] + td_s[i][j]
                            - M_TIME * (1 - y[k, i, j]) - M_TIME * wl[k, i],
                            name=f"Ta_cont_{k}_{i}_{j}")
                # (T2) 新架次：从 i 的出发时刻开始。
                # ⚠ Td 里已经含了 sv[i] + launch_time（见 Td_def），这里不能再加一次 sv[i]
                m.addConstr(Ta[k, j] >= Td[k, i] + td_s[i][j]
                            - M_TIME * (1 - y[k, i, j]) - M_TIME * (1 - wl[k, i]),
                            name=f"Ta_launch_{k}_{i}_{j}")

        for j in C:
            drone_served = gp.quicksum(y[k, i, j] for i in C if i != j) - wr[k, j]
            m.addConstr(Ta[k, j] >= e_win[j] - M_TIME * (1 - drone_served), name=f"Ta_wait_{k}_{j}")
            m.addConstr(Ta[k, j] <= l_win[j] + M_TIME * (1 - drone_served), name=f"Ta_window_{k}_{j}")

    # ---------------- 无人机载荷：到达 / 出发 分开 ----------------
    for k in KS:
        for i in C:
            m.addConstr(fai[k, i] <= M_LOAD * (1 - wr[k, i]), name=f"fai_zero_at_recover_{k}_{i}")
            # 起飞时取件量为 0（尚未开始收件）；续飞时等于到达量 + 本点取件
            m.addConstr(gao[k, i] <= M_LOAD * (1 - wl[k, i]), name=f"gao_zero_at_launch_{k}_{i}")
            # 出发载荷：起飞点取新装载 ft，否则取到达量减掉本点派件。
            # ⚠ 必须用「有没有后续无人机弧」做闸门：回收点是架次的终点，无人机不从那里继续，
            #    若对回收点也套 fao = fai - f[i]，会得到负的 fao（不可行）——
            #    这是首版报 IIS 的直接原因（fao_cont_ub 与 fai_zero_at_recover 冲突）。
            leaves = gp.quicksum(y[k, i, j] for j in C if j != i)
            m.addConstr(fao[k, i] >= ft[k, i] - M_LOAD * (1 - wl[k, i]), name=f"fao_launch_lb_{k}_{i}")
            m.addConstr(fao[k, i] <= ft[k, i] + M_LOAD * (1 - wl[k, i]), name=f"fao_launch_ub_{k}_{i}")
            m.addConstr(fao[k, i] >= fai[k, i] - f[i] - M_LOAD * wl[k, i] - M_LOAD * (1 - leaves),
                        name=f"fao_cont_lb_{k}_{i}")
            m.addConstr(fao[k, i] <= fai[k, i] - f[i] + M_LOAD * wl[k, i] + M_LOAD * (1 - leaves),
                        name=f"fao_cont_ub_{k}_{i}")
            m.addConstr(gao[k, i] >= gai[k, i] + g[i] - M_LOAD * wl[k, i] - M_LOAD * (1 - leaves),
                        name=f"gao_cont_lb_{k}_{i}")
            m.addConstr(gao[k, i] <= gai[k, i] + g[i] + M_LOAD * wl[k, i] + M_LOAD * (1 - leaves),
                        name=f"gao_cont_ub_{k}_{i}")
            # 在机容量（到达侧）
            arrives = gp.quicksum(y[k, j, i] for j in C if j != i)
            m.addConstr(fai[k, i] + gai[k, i] <= p.drone_capacity + M_LOAD * (1 - arrives),
                        name=f"drone_cap_{k}_{i}")

            for j in C:
                if i == j:
                    continue
                m.addConstr(fai[k, j] >= fao[k, i] - M_LOAD * (1 - y[k, i, j]),
                            name=f"fai_link_lb_{k}_{i}_{j}")
                m.addConstr(fai[k, j] <= fao[k, i] + M_LOAD * (1 - y[k, i, j]),
                            name=f"fai_link_ub_{k}_{i}_{j}")
                m.addConstr(gai[k, j] >= gao[k, i] - M_LOAD * (1 - y[k, i, j]),
                            name=f"gai_link_lb_{k}_{i}_{j}")
                m.addConstr(gai[k, j] <= gao[k, i] + M_LOAD * (1 - y[k, i, j]),
                            name=f"gai_link_ub_{k}_{i}_{j}")

    # ---------------- 能耗：到达 / 出发 分开 ----------------
    for k in KS:
        for i in C:
            # 起飞 = 换电 = 满电（Eout=0）；续飞时 Eout 等于到达量
            m.addConstr(Eout[k, i] <= M_ENERGY * (1 - wl[k, i]), name=f"Eout_zero_at_launch_{k}_{i}")
            leaves_e = gp.quicksum(y[k, i, j] for j in C if j != i)
            m.addConstr(Eout[k, i] >= Ein[k, i] - M_ENERGY * wl[k, i] - M_ENERGY * (1 - leaves_e),
                        name=f"Eout_cont_lb_{k}_{i}")
            m.addConstr(Eout[k, i] <= Ein[k, i] + M_ENERGY * wl[k, i] + M_ENERGY * (1 - leaves_e),
                        name=f"Eout_cont_ub_{k}_{i}")
            m.addConstr(Ein[k, i] <= p.eta_soc * p.drone_battery, name=f"soc_in_{k}_{i}")
            m.addConstr(Eout[k, i] <= p.eta_soc * p.drone_battery, name=f"soc_out_{k}_{i}")
            for j in C:
                if i == j:
                    continue
                m.addConstr(ec[k, i, j] >= dd_km[i][j] * (p.e_drone * (fai[k, j] + gai[k, j]) + p.b)
                            - M_ENERGY * (1 - y[k, i, j]), name=f"ec_lb_{k}_{i}_{j}")
                m.addConstr(ec[k, i, j] <= M_ENERGY * y[k, i, j], name=f"ec_ub_{k}_{i}_{j}")
                m.addConstr(Ein[k, j] >= Eout[k, i] + ec[k, i, j] - M_ENERGY * (1 - y[k, i, j]),
                            name=f"Ein_step_{k}_{i}_{j}")
                m.addConstr(Ein[k, j] <= Eout[k, i] + ec[k, i, j] + M_ENERGY * (1 - y[k, i, j]),
                            name=f"Ein_step_ub_{k}_{i}_{j}")

    # ---------------- 卡车载荷 ----------------
    for k in KS:
        m.addConstr(F[k, DEPOT] == gp.quicksum(
            f[j] * (gp.quicksum(x[k, i, j] for i in V if i != j)
                    + gp.quicksum(y[k, i, j] for i in C if i != j) - wr[k, j]) for j in C),
            name=f"F_init_{k}")
        m.addConstr(G[k, DEPOT] == 0.0, name=f"G_init_{k}")
        m.addConstr(F[k, DEPOT] <= p.truck_capacity, name=f"truck_cap_init_{k}")
        for i in V:
            for j in V:
                if i == j:
                    continue
                if i in C and j != DEPOT:
                    m.addConstr(F[k, j] >= F[k, i] - f[i] - ft[k, i] - M_LOAD * (1 - x[k, i, j]),
                                name=f"F_lb_{k}_{i}_{j}")
                    m.addConstr(F[k, j] <= F[k, i] - f[i] - ft[k, i] + M_LOAD * (1 - x[k, i, j]),
                                name=f"F_ub_{k}_{i}_{j}")
                    m.addConstr(G[k, j] >= G[k, i] + g[i] - M_LOAD * (1 - x[k, i, j]),
                                name=f"G_lb_{k}_{i}_{j}")
                    m.addConstr(G[k, j] <= G[k, i] + g[i] + M_LOAD * (1 - x[k, i, j]),
                                name=f"G_ub_{k}_{i}_{j}")
                m.addConstr(F[k, i] + G[k, i] <= p.truck_capacity + M_LOAD * (1 - x[k, i, j]),
                            name=f"truck_cap_{k}_{i}_{j}")

    # 起飞装载之和 = 由无人机服务客户的派件之和（原模型靠 F 的递推间接口径，这里显式钉住）
    # ⚠ 必须减掉 wr：回收点也有无人机弧进入，但那是"来接机"，不是"送这个客户的货"
    for k in KS:
        m.addConstr(
            gp.quicksum(ft[k, i] for i in C)
            == gp.quicksum(f[j] * (gp.quicksum(y[k, i, j] for i in C if i != j) - wr[k, j])
                           for j in C),
            name=f"ft_sum_{k}")

    # ---------------- 目标 ----------------
    m.setObjective(
        p.c_drone * gp.quicksum(ec.values())
        + p.c_truck * p.e_truck * gp.quicksum(dt_km[i][j] * x[k, i, j]
                                             for k in KS for i in V for j in V if i != j)
        + p.c_customer * gp.quicksum(TtEnd[k] for k in KS),
        GRB.MINIMIZE,
    )

    if fix is not None:
        _pin(m, fix, inst, x, y, wl, wr)
        m.setObjective(0.0, GRB.MINIMIZE)

    t0 = time.time()
    m.optimize()
    cpu = time.time() - t0

    if m.SolCount == 0:
        if fix is not None:
            try:
                m.computeIIS()
                bad = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
                print(f"  [IIS] 冲突约束 {len(bad)} 条：{bad[:8]}")
            except Exception as exc:                       # noqa: BLE001
                print(f"  [IIS] 计算失败：{exc}")
        return HsResult(float("nan"), float("nan"), float("nan"), cpu,
                        f"no solution(status={m.Status})", None, m.NumVars, m.NumConstrs)

    return HsResult(
        cost=float(m.ObjVal), bound=float(m.ObjBound), gap=float(m.MIPGap), cpu=cpu,
        status={GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time limit"}.get(m.Status, str(m.Status)),
        solution=_extract(inst, m, x, y, wl, wr),
        n_vars=m.NumVars, n_cons=m.NumConstrs,
    )


def _pin(m, sol: Solution, inst: Instance, x, y, wl, wr) -> None:
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


def _extract(inst: Instance, m, x, y, wl, wr) -> Solution:
    routes: list[Route] = []
    for k in range(inst.n_trucks):
        nodes = [DEPOT]
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
            if wl.get((k, i)) is None or wl[k, i].X <= 0.5:
                continue
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
                if wr.get((k, nxt)) is not None and wr[k, nxt].X > 0.5:
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
