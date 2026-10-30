"""ALNS 的破坏与修复算子（论文 §4.2–4.3）。

四个破坏算子的**共同铁律**：一旦选中某架次的起飞点或降落点，整段 sortie 一起移除。
四个修复算子都要保证返回的解**结构完整且可行**，否则返回 None 让主循环记为失败。

实现约定：
- 算子内部一律用「按节点编号的锚点描述」（`route_to_specs`），避免位置下标平移；
- 可行性一律用 `evaluate(..., check_structure=False)` 对**受影响的单条路线**判定，
  全解层面的完整性检查留给主循环；
- 插入候选先用「距离/能耗增量」粗排，只对前若干名做完整评估（评估是瓶颈）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluator import evaluate
from .instance import Instance
from .solution import DEPOT, Route, Solution, route_from_specs, route_to_specs

TOP_K = 6          # 粗排后真正做完整评估的候选数


@dataclass
class Removal:
    sol: Solution
    removed: list[int]


# ---------------------------------------------------------------- 基础工具

def _route_eval(inst: Instance, route: Route) -> tuple[bool, float]:
    res, cost = evaluate(inst, Solution([route]), check_structure=False)
    return res.feasible, cost


def _clone(sol: Solution) -> Solution:
    return Solution([route_from_specs(r.truck, r.nodes, route_to_specs(r)) for r in sol.routes])


def remove_customers(sol: Solution, targets: set[int]) -> tuple[Solution, list[int]]:
    """从解里彻底移除若干客户，返回（新解, 实际被移除的客户列表）。

    ⚠ 返回的第二个值很关键：若被删的客户正好是某架次的锚点，**整段 sortie 失效**，
    它服务的其他客户也一起消失，必须一并进入待插入集，否则解会悄悄丢掉一批客户。
    （初版只返回新解，导致「删 8 个、丢 17 个」的假不可行。）
    """
    dropped: set[int] = set(targets)
    out = []
    for r in sol.routes:
        nodes = [n for n in r.nodes if n not in targets]
        survive = set(nodes)
        specs = []
        for p, q, cust in route_to_specs(r):
            if p not in survive or q not in survive:
                dropped.update(cust)          # 锚点没了 → 整段作废
                continue
            kept = tuple(c for c in cust if c not in targets)
            if kept:
                specs.append((p, q, kept))
        new_route = route_from_specs(r.truck, nodes, specs)
        # 兜底：架次位置冲突时 route_from_specs 会静默丢弃整段，这里把消失的客户捡回来
        lost = ({n for n in r.nodes if n != DEPOT} | {c for _, _, cs in route_to_specs(r) for c in cs}) \
            - targets - new_route.served_customers()
        dropped.update(lost)
        out.append(new_route)
    return Solution(out), sorted(dropped)


def served_map(sol: Solution) -> dict[int, tuple[int, str]]:
    """客户 → (路线序号, "truck"/"drone")。"""
    out: dict[int, tuple[int, str]] = {}
    for idx, r in enumerate(sol.routes):
        for n in r.nodes:
            if n != DEPOT:
                out[n] = (idx, "truck")
        for s in r.sorties:
            for c in s.customers:
                out[c] = (idx, "drone")
    return out


def _all_customers(sol: Solution) -> list[int]:
    return sorted(served_map(sol))


# ---------------------------------------------------------------- 破坏算子

def destroy_random(inst: Instance, sol: Solution, rng, ratio: float = 0.3) -> Removal:
    custs = _all_customers(sol)
    if not custs:
        return Removal(sol, [])
    k = max(1, int(round(ratio * len(custs))))
    picked = list(rng.choice(custs, size=min(k, len(custs)), replace=False))
    new_sol, dropped = remove_customers(sol, set(picked))
    return Removal(new_sol, dropped)


def _saving_if_removed(inst: Instance, sol: Solution, c: int) -> float:
    """粗估移除客户 c 能省多少（卡车看距离，无人机看航段距离）。"""
    base = inst.base
    where = served_map(sol)[c]
    r = sol.routes[where[0]]
    d = 0.0
    if where[1] == "truck":
        pos = r.nodes.index(c)
        if pos == 0 or pos == len(r.nodes) - 1:
            return 0.0
        a, b = r.nodes[pos - 1], r.nodes[pos + 1]
        d = float(base.truck_dist_m[a, c] + base.truck_dist_m[c, b] - base.truck_dist_m[a, b])
    else:                                  # drone
        for s in r.sorties:
            if c in s.customers:
                p, q = r.nodes[s.launch], r.nodes[s.recover]
                xs = [p] + list(s.customers) + [q]
                xs2 = [x for x in xs if x != c]
                d = sum(
                    np.hypot(base.nodes[xs[i]].x - base.nodes[xs[i + 1]].x,
                             base.nodes[xs[i]].y - base.nodes[xs[i + 1]].y)
                    for i in range(len(xs) - 1)
                ) - sum(
                    np.hypot(base.nodes[xs2[i]].x - base.nodes[xs2[i + 1]].x,
                             base.nodes[xs2[i]].y - base.nodes[xs2[i + 1]].y)
                    for i in range(len(xs2) - 1)
                )
                break
    return d


def destroy_worst(inst: Instance, sol: Solution, rng, ratio: float = 0.3) -> Removal:
    custs = _all_customers(sol)
    if not custs:
        return Removal(sol, [])
    sav = sorted(((c, _saving_if_removed(inst, sol, c)) for c in custs), key=lambda x: -x[1])
    k = max(1, int(round(ratio * len(custs))))
    # 带随机化的前 k 名，避免每轮都删同一批
    picked = [c for c, _ in sav[: min(len(sav), k * 2)]][:k]
    new_sol, dropped = remove_customers(sol, set(picked))
    return Removal(new_sol, dropped)


def destroy_related(inst: Instance, sol: Solution, rng, ratio: float = 0.3) -> Removal:
    custs = _all_customers(sol)
    if not custs:
        return Removal(sol, [])
    base = inst.base
    seed = int(rng.choice(custs))

    def score(c: int) -> float:
        d = np.hypot(base.nodes[c].x - base.nodes[seed].x, base.nodes[c].y - base.nodes[seed].y)
        # 禁飞区附近的客户算"相关"：共享同一绕行约束，成片移除才能改变绕行结构
        near_nfz = any(
            np.hypot(base.nodes[c].x - z.cx, base.nodes[c].y - z.cy) <= z.r * 2.0 for z in inst.nfz
        )
        return d * (0.6 if near_nfz else 1.0)

    k = max(1, int(round(ratio * len(custs))))
    picked = sorted(custs, key=score)[:k]
    new_sol, dropped = remove_customers(sol, set(picked))
    return Removal(new_sol, dropped)


def destroy_truck_shuffle(inst: Instance, sol: Solution, rng, ratio: float = 0.3) -> Removal:
    """打乱一条卡车路线的内部顺序；锚点失效的架次整段移除。"""
    cand = [i for i, r in enumerate(sol.routes) if len(r.nodes) >= 4]
    if not cand:
        return Removal(sol, [])
    idx = int(rng.choice(cand))
    r = sol.routes[idx]
    interior = r.nodes[1:-1]
    perm = list(rng.permutation(interior))
    new_nodes = [DEPOT] + [int(x) for x in perm] + [DEPOT]

    specs = route_to_specs(r)
    kept_specs, dropped = [], []
    pos = {n: i for i, n in enumerate(new_nodes)}
    for p, q, cust in specs:
        if p in pos and q in pos and pos[q] - pos[p] == len(cust) + 1:
            kept_specs.append((p, q, cust))
        else:
            dropped.extend(cust)

    other = [x for i, x in enumerate(sol.routes) if i != idx]
    new_sol = Solution(other + [route_from_specs(r.truck, new_nodes, kept_specs)])
    if dropped:
        new_sol, more = remove_customers(new_sol, set(dropped))
        dropped = sorted(set(dropped) | set(more))
    return Removal(new_sol, dropped)


DESTROY = {
    "random": destroy_random,
    "worst": destroy_worst,
    "related": destroy_related,
    "shuffle": destroy_truck_shuffle,
}


# ---------------------------------------------------------------- 修复算子

def _truck_candidates(inst: Instance, sol: Solution, c: int):
    """所有可行的卡车插入位置，按成本增量升序返回 [(delta, new_solution)]。"""
    out = []
    for idx, r in enumerate(sol.routes):
        specs = route_to_specs(r)
        before_ok, before_cost = _route_eval(inst, r)
        if not before_ok:
            continue
        coarse = []
        for k in range(len(r.nodes) - 1):
            a, b = r.nodes[k], r.nodes[k + 1]
            d = float(inst.base.truck_dist_m[a, c] + inst.base.truck_dist_m[c, b]
                      - inst.base.truck_dist_m[a, b])
            coarse.append((d, k))
        coarse.sort()
        for _, k in coarse[:TOP_K]:
            nodes = r.nodes[: k + 1] + [c] + r.nodes[k + 1:]
            trial = route_from_specs(r.truck, nodes, specs)
            if c not in trial.served_customers():      # 架次冲突可能把客户又丢掉
                continue
            ok, cost = _route_eval(inst, trial)
            if ok:
                new_sol = Solution([x for i, x in enumerate(sol.routes) if i != idx] + [trial])
                out.append((cost - before_cost, new_sol))
    # 允许开新路线
    if len(sol.routes) < inst.n_trucks:
        trial = Route(truck=len(sol.routes), nodes=[DEPOT, c, DEPOT], sorties=[])
        ok, cost = _route_eval(inst, trial)
        if ok:
            out.append((cost, Solution(sol.routes + [trial])))
    out.sort(key=lambda x: x[0])
    return out


def _drone_candidates(inst: Instance, sol: Solution, c: int):
    """所有可行的无人机插入候选（并入已有架次 / 新建架次）。"""
    if not inst.allow_drones:              # Truck-only 对照组：没有任何无人机候选
        return []
    out = []
    for idx, r in enumerate(sol.routes):
        specs = route_to_specs(r)
        before_ok, before_cost = _route_eval(inst, r)
        if not before_ok:
            continue

        # ① 并入已有架次：插到飞行序列的任一位置
        for i, (p, q, cust) in enumerate(specs):
            for pos in range(len(cust) + 1):
                new_cust = cust[:pos] + (c,) + cust[pos:]
                trial_specs = specs[:i] + [(p, q, new_cust)] + specs[i + 1:]
                trial = route_from_specs(r.truck, r.nodes, trial_specs)
                if c not in trial.served_customers():
                    continue
                ok, cost = _route_eval(inst, trial)
                if ok:
                    new_sol = Solution([x for j, x in enumerate(sol.routes) if j != idx] + [trial])
                    out.append((cost - before_cost, new_sol))

        # ② 新建架次：卡车路径上相邻两节点作锚点（仓库除外）
        for k in range(len(r.nodes) - 1):
            p, q = r.nodes[k], r.nodes[k + 1]
            if p == DEPOT or q == DEPOT:
                continue
            if any((sp, sq) == (p, q) for sp, sq, _ in specs):
                continue
            trial = route_from_specs(r.truck, r.nodes, specs + [(p, q, (c,))])
            if c not in trial.served_customers():
                continue
            ok, cost = _route_eval(inst, trial)
            if ok:
                new_sol = Solution([x for j, x in enumerate(sol.routes) if j != idx] + [trial])
                out.append((cost - before_cost, new_sol))

    out.sort(key=lambda x: x[0])
    return out


def _apply_repair(inst, sol, removed, rng, prefer: str) -> Solution | None:
    """通用修复流程：按顺序为每个客户挑候选插入。"""
    order = list(removed)
    cur = sol
    for c in order:
        truck = _truck_candidates(inst, cur, c)
        drone = _drone_candidates(inst, cur, c)
        pools = {"truck": truck, "drone": drone, "any": truck + drone}
        pool = pools[prefer]
        if not pool:
            pool = truck + drone
        if not pool:
            return None
        if prefer == "any":
            cur = min(pool, key=lambda x: x[0])[1]
        else:
            cur = pool[0][1]
    return cur


def repair_random(inst: Instance, rem: Removal, rng) -> Solution | None:
    order = list(rem.removed)
    rng.shuffle(order)
    cur = rem.sol
    for c in order:
        pool = _truck_candidates(inst, cur, c) + _drone_candidates(inst, cur, c)
        if not pool:
            return None
        cur = pool[int(rng.integers(len(pool)))][1]
    return cur


def repair_truck_first(inst: Instance, rem: Removal, rng) -> Solution | None:
    sol = _apply_repair(inst, rem.sol, rem.removed, rng, prefer="truck")
    if sol is None:
        return None
    # 二次转换：把值得的卡车客户改成同车架次
    improved = inst.allow_drones
    while improved:
        improved = False
        for idx, r in enumerate(sol.routes):
            for c in [n for n in r.nodes if n != DEPOT]:
                pos = r.nodes.index(c)
                if pos == 0 or pos == len(r.nodes) - 1:
                    continue
                p, q = r.nodes[pos - 1], r.nodes[pos + 1]
                if p == DEPOT or q == DEPOT:
                    continue
                specs = route_to_specs(r)
                if any((sp, sq) == (p, q) for sp, sq, _ in specs):
                    continue
                new_nodes = [n for n in r.nodes if n != c]
                trial = route_from_specs(r.truck, new_nodes, specs + [(p, q, (c,))])
                before_ok, before_cost = _route_eval(inst, r)
                ok, cost = _route_eval(inst, trial)
                if ok and cost < before_cost:
                    sol = Solution([x for i, x in enumerate(sol.routes) if i != idx] + [trial])
                    improved = True
                    break
            if improved:
                break
    return sol


def repair_drone_first(inst: Instance, rem: Removal, rng) -> Solution | None:
    return _apply_repair(inst, rem.sol, rem.removed, rng, prefer="drone")


def repair_equal(inst: Instance, rem: Removal, rng) -> Solution | None:
    return _apply_repair(inst, rem.sol, rem.removed, rng, prefer="any")


REPAIR = {
    "random": repair_random,
    "truck": repair_truck_first,
    "drone": repair_drone_first,
    "equal": repair_equal,
}
