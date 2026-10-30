"""评估器：把「解」翻译成「时刻表 + 载荷流 + 成本 + 违约清单」。

对应论文里真正需要写代码检查的 5 类约束（见 `约束逐条翻译_2-79.md` 最后一节）：

| 类别 | 约束 | 这里怎么处理 |
|---|---|---|
| 时刻递推与同步 | (22)(23)(31) | `_simulate_route` 里逐段推进；回收点取 `max(卡车到, 无人机到)` |
| 时间窗 | (35)–(38) | 卡车用 `T_i^truck + τ` 判（不等候到达），无人机用到达时刻判 |
| 能耗与 SOC | (39)–(43) | 按段累加，每段用「该段机上载荷」；单架次受 η·E_max 限制 |
| 双向载荷 | (52)–(73) | 卡车 F 递减 / G 递增；无人机每段 f、g 分别推进 |
| 禁飞 | (13)–(15)、(32)–(34) | 交给 `drone_leg` 与 `service_ok`，这里只收集违约 |

**为什么卡车时间窗要用「不等候到达」**：卡车到达时刻可能包含等无人机的时间，
若拿它去比时间窗，就等于允许卡车靠「故意晚出发」蒙过窗口（论文 (35)(36) 专门防这个）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .distance import drone_leg, service_ok, truck_leg
from .energy import driver_cost, drone_energy, truck_cost
from .instance import Instance
from .solution import DEPOT, Route, Solution

# 违约代码
V_TW_TRUCK = "TW_TRUCK"        # 卡车超出客户时间窗
V_TW_DRONE = "TW_DRONE"        # 无人机超出客户时间窗
V_SOC = "SOC"                  # 单架次电量超限
V_CAP_TRUCK = "CAP_TRUCK"      # 卡车超载
V_CAP_DRONE = "CAP_DRONE"      # 无人机超载
V_NFZ = "NFZ"                  # 禁飞区导致不可飞/不可起降
V_DRONE_BUSY = "DRONE_BUSY"    # 同一架无人机的两个架次在时间上重叠
V_LR_SAME = "LR_SAME_NODE"     # 同一节点既是回收点又是起飞点（论文的 w_i 表达不了）


@dataclass
class RouteEval:
    truck: int
    feasible: bool = True
    violations: list[str] = field(default_factory=list)
    truck_distance: float = 0.0
    drone_distance: float = 0.0
    drone_energy: float = 0.0
    finish_time: float = 0.0
    n_sorties: int = 0
    waiting_time: float = 0.0

    def cost(self, inst: Instance) -> float:
        p = inst.params
        return (
            p.c_drone * self.drone_energy
            + truck_cost(self.truck_distance, p)
            + driver_cost(self.finish_time, p)
        )


@dataclass
class EvalResult:
    routes: list[RouteEval]
    structure_errors: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        return not self.structure_errors and all(r.feasible for r in self.routes)

    def all_violations(self) -> list[str]:
        out = list(self.structure_errors)
        for r in self.routes:
            out.extend(f"车{r.truck}: {v}" for v in r.violations)
        return out


def _simulate_route(inst: Instance, route: Route) -> RouteEval:
    """沿卡车路径推进，同时推进无人机架次。"""
    p = inst.params
    base = inst.base
    ev = RouteEval(truck=route.truck)
    nodes = route.nodes

    launch_at = {s.launch: s for s in route.sorties}
    recover_at = {s.recover: s for s in route.sorties}

    # 卡车初始装载：本路线（含无人机）服务的所有客户的派件量
    served = route.served_customers()
    F = float(sum(inst.f[c] for c in served))
    G = 0.0
    if F > p.truck_capacity + 1e-9:
        ev.violations.append(f"{V_CAP_TRUCK} 出车装载 {F:.1f}kg > {p.truck_capacity:.1f}kg")
        ev.feasible = False

    t = 0.0                      # 到达当前节点的时刻
    drone_arrival: dict[int, float] = {}   # 各架次的无人机抵达回收点时刻
    busy_until: int | None = None          # 当前在飞架次的回收位置（每车只有一架无人机）

    for pos in range(len(nodes) - 1):
        i = nodes[pos]
        # --- 在 i 点：卡车服务（若 i 是客户且由卡车服务） ---
        if i != DEPOT:
            t += p.service_time
            F -= inst.f[i]
            G += inst.g[i]

        # --- 在 i 点：起飞无人机 ---
        s = launch_at.get(pos)
        if s is not None:
            # 约束 (20)(21)(49)：一辆卡车只配一架无人机，上一个架次没回收就不能再起飞。
            # 缺这条检查时，评估器会允许「架次嵌套」——两个架次在同一段时间里各飞各的，
            # 成本被显著低估，而且 ALNS 会主动去找这种非法解（实测抓到过 Set1 的嵌套解）。
            if pos in recover_at:
                # 论文只用一个 w_i 表示「起飞或回收」，同一节点同时承担两件事时，
                # 时刻递推 (23) 会拿卡车到达时刻当无人机起飞基准（此时机还没落地），
                # 模型内部就不自洽。这里直接判不可行，与模型口径保持一致。
                ev.violations.append(f"{V_LR_SAME} 位置{pos} 同时是回收点与起飞点")
                ev.feasible = False
            if busy_until is not None and pos < busy_until:
                ev.violations.append(
                    f"{V_DRONE_BUSY} 位置{pos} 起飞时，上一架次（回收位置 {busy_until}）仍在飞行"
                )
                ev.feasible = False
            busy_until = s.recover
            t += p.launch_time
            t_d, energy, ok = _fly_sortie(inst, route, s, start_node=i, t_start=t, ev=ev)
            if ok:
                drone_arrival[s.recover] = t_d
                ev.drone_energy += energy
            # 无人机带走它负责的派件，卡车账上直接扣掉（约束 56/57）
            F -= float(sum(inst.f[c] for c in s.customers))
            ev.n_sorties += 1

        # --- 前往下一节点 ---
        j = nodes[pos + 1]
        d_truck, t_truck_leg = truck_leg(base, i, j)
        ev.truck_distance += d_truck
        t_no_wait = t + t_truck_leg

        # --- 卡车时间窗 ---
        # 迟到判违约；**早到则原地等到 e_j**（标准 VRPTW 做法）。
        # 说明：论文 (35) 字面要求 T_i+τ ≥ e_j，等于禁止早到；但卡车在客户点等待
        # 与推迟上一节点出发在物理上等价，允许等待不改变可行域，只是让表示更自然。
        t = t_no_wait
        r = recover_at.get(pos + 1)
        if j != DEPOT:
            w = inst.tw[j - 1]
            if t_no_wait > w.l + 1e-6:
                ev.violations.append(
                    f"{V_TW_TRUCK} 节点{j} 到达 {t_no_wait/60:.1f}min 晚于窗口上界 {w.l/60:.1f}min"
                )
                ev.feasible = False

        # --- 回收无人机：卡车必须等无人机到达 ---
        if r is not None:
            busy_until = None                     # 无人机已回收，可以再次起飞
            ta = drone_arrival.get(pos + 1)
            if ta is not None and ta > t:
                ev.waiting_time += ta - t
                t = ta
            t += p.recovery_time
            G += float(sum(inst.g[c] for c in r.customers))

        # 早到等待到窗口开启
        if j != DEPOT:
            w = inst.tw[j - 1]
            if t < w.e:
                ev.waiting_time += w.e - t
                t = w.e

        # 卡车容量：派件 + 取件（约束 62）
        if F + G > p.truck_capacity + 1e-9:
            ev.violations.append(f"{V_CAP_TRUCK} 节点{j} 装载 {F+G:.1f}kg > {p.truck_capacity:.1f}kg")
            ev.feasible = False
        if F < -1e-6:
            ev.violations.append(f"{V_CAP_TRUCK} 节点{j} 派件量为负（{F:.2f}），说明装载或交接算错")
            ev.feasible = False

    ev.finish_time = t
    return ev


def _fly_sortie(
    inst: Instance, route: Route, s, start_node: int, t_start: float, ev: RouteEval
) -> tuple[float, float, bool]:
    """模拟一个架次：起飞点 → 客户序列 → 回收点。返回（抵达回收点时刻, 能耗, 是否可行）。"""
    p = inst.params
    base = inst.base
    ok = True

    # 起飞点的起降可行性（Fig.3 的 γ·δ）
    node_xy = (base.nodes[start_node].x, base.nodes[start_node].y)
    can, blocked = service_ok(node_xy, t_start, inst.nfz)
    if not can:
        ev.violations.append(f"{V_NFZ} 起飞点{start_node} 在禁飞期内位于区内 {blocked}")
        ev.feasible = False
        ok = False

    payload_f = float(sum(inst.f[c] for c in s.customers))
    payload_g = 0.0
    if payload_f > p.drone_capacity + 1e-9:
        ev.violations.append(f"{V_CAP_DRONE} 起飞载荷 {payload_f:.2f}kg > {p.drone_capacity:.2f}kg")
        ev.feasible = False

    t = t_start
    energy = 0.0
    cur = start_node
    seq = list(s.customers) + [route.nodes[s.recover]]

    for k, nxt in enumerate(seq):
        est = t + _straight(inst, cur, nxt) / p.drone_speed
        leg = drone_leg(base, inst.nfz, cur, nxt, t, est, allow_detour=inst.allow_detour)
        if not leg.feasible:
            ev.violations.append(f"{V_NFZ} 航段 {cur}->{nxt} 不可飞（{leg.blocked_by}）")
            ev.feasible = False
            ok = False
            return t, energy, ok

        onboard = payload_f + payload_g
        energy += drone_energy(leg.distance, onboard, p)
        ev.drone_distance += leg.distance
        t += leg.distance / p.drone_speed
        cur = nxt                     # 关键：下一段必须从当前点起飞（漏了这句会把每段都算成从起点出发）

        if k < len(seq) - 1:          # 到达客户
            c = nxt
            w = inst.tw[c - 1]
            if t > w.l + 1e-6:
                ev.violations.append(
                    f"{V_TW_DRONE} 无人机到节点{c} {t/60:.1f}min 晚于窗口上界 {w.l/60:.1f}min"
                )
                ev.feasible = False
            if t < w.e:                 # 早到则悬停等待（不计能耗，见该节注释）
                ev.waiting_time += w.e - t
                t = w.e
            payload_f -= inst.f[c]
            payload_g += inst.g[c]
            if payload_f + payload_g > p.drone_capacity + 1e-9:
                ev.violations.append(
                    f"{V_CAP_DRONE} 节点{c} 机上 {payload_f+payload_g:.2f}kg > {p.drone_capacity:.2f}kg"
                )
                ev.feasible = False
            t += p.service_time

    if energy > p.eta_soc * p.drone_battery + 1e-9:
        ev.violations.append(
            f"{V_SOC} 架次能耗 {energy:.3f}kWh > {p.eta_soc * p.drone_battery:.3f}kWh"
        )
        ev.feasible = False

    if abs(payload_f) > 1e-6:
        ev.violations.append(f"{V_CAP_DRONE} 回收时仍有派件 {payload_f:.2f}kg（约束 67）")
        ev.feasible = False

    return t, energy, ok


def _straight(inst: Instance, i: int, j: int) -> float:
    a, b = inst.base.nodes[i], inst.base.nodes[j]
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


def evaluate(
    inst: Instance, sol: Solution, check_structure: bool = True
) -> tuple[EvalResult, float]:
    """返回（逐路线结果, 总成本）。总成本 = Σ 无人机电费 + 卡车油费 + 司机工时。

    `check_structure=False` 用于**单条路线的试算**：完整性检查（每个客户都被服务）
    只有在整解层面才有意义，否则构造过程中任何一次局部试算都会被判成"客户缺失"。
    """
    from .solution import validate_structure

    evs = [_simulate_route(inst, r) for r in sol.routes]
    errs = validate_structure(sol, inst) if check_structure else []
    res = EvalResult(routes=evs, structure_errors=errs)
    total = sum(r.cost(inst) for r in evs)
    return res, total
