"""初始可行解构造（论文 Algorithm 2 的两阶段）。

阶段一：**松弛优先**地把客户依次接到卡车路径末尾
    - 候选：所有未服务客户
    - 可行：追加后时间窗不破、卡车派件总量不超容量
    - 排序：slack = 时间窗上界 − 最早可行到达，取最小者（最不灵活的先安置）
    - 装不进去就收车返仓，开下一条路线

阶段二：把卡车客户改派给无人机
    - 候选：路径上 slack 最小的客户
    - 锚点：该客户在路径上的**前驱**（起飞）与**后继**（回收）
    - 四点校验：载荷 / 时间 / 序列 / 能耗 —— 这里直接调 `evaluate` 做全链路校验
    - 通过则从卡车路径删除该客户，生成一个架次

**与原算法的两处简化**（都会在报告里写明）：
1. 阶段一只检查「时间窗 + 派件总量」，取件带来的容量增量留到 `evaluate` 复核；
2. 阶段二每个客户单独成一个架次，不把同锚点对上的多个客户并进同一架次。
"""

from __future__ import annotations

from .evaluator import evaluate
from .instance import Instance
from .solution import DEPOT, Route, Solution, Sortie


def _arrival(t: float, travel: float, e: float) -> float:
    """到达时刻（早到则等到窗口开启）。"""
    return max(t + travel, e)


def slack_first_routes(inst: Instance, max_routes: int) -> list[list[int]]:
    """阶段一：松弛优先构造卡车路径，返回节点序列列表（含首尾仓库）。"""
    base = inst.base
    unserved = set(inst.customers)
    routes: list[list[int]] = []

    while unserved and len(routes) < max_routes:
        route = [DEPOT]
        t = 0.0
        f_total = 0.0

        while unserved:
            best = None
            for c in unserved:
                travel = float(base.truck_time_s[route[-1], c])
                arr = _arrival(t, travel, inst.tw[c - 1].e)
                if arr > inst.tw[c - 1].l + 1e-6:
                    continue                       # 迟到，不可行
                if f_total + float(inst.f[c]) > inst.params.truck_capacity + 1e-9:
                    continue                       # 出车派件量超容量
                sl = inst.tw[c - 1].l - arr          # 松弛 = 距窗口上界的余量
                if best is None or sl < best[0]:
                    best = (sl, c, arr)

            if best is None:
                break
            _, c, arr = best
            route.append(c)
            unserved.discard(c)
            t = arr + inst.params.service_time
            f_total += float(inst.f[c])

        route.append(DEPOT)
        if len(route) == 2:                    # 一个都没装上，避免死循环
            routes.append(route)
            break
        routes.append(route)

    return routes


def _to_sorties(nodes: list[int], specs: list[tuple[int, int, tuple[int, ...]]]):
    """把「按节点编号记录的锚点」翻译成「按位置下标记录的 Sortie」。

    返回 (sorties, ok)。若某个锚点节点已不在路径上（后续删除客户导致），ok=False。
    """
    pos = {n: i for i, n in enumerate(nodes)}
    out: list[Sortie] = []
    for p_node, q_node, cust in specs:
        if p_node not in pos or q_node not in pos:
            return [], False
        a, b = pos[p_node], pos[q_node]
        if b <= a or nodes[a] == DEPOT or nodes[b] == DEPOT:
            return [], False
        out.append(Sortie(launch=a, recover=b, customers=cust))
    # 同一位置不能被两个架次占用
    used_launch = [x.launch for x in out]
    used_recover = [x.recover for x in out]
    if len(set(used_launch)) != len(used_launch) or len(set(used_recover)) != len(used_recover):
        return [], False
    return out, True


def assign_sorties(inst: Instance, route: Route) -> Route:
    """阶段二：把该路线上的客户尽量改派给无人机（逐个尝试，可行即改）。

    锚点一律**按节点编号**记录（`specs`），只在评估/返回时才翻译成位置下标。
    原因：每删除一个客户，后续节点的位置下标都会平移，若直接存下标，之前建的架次
    会挂到错误的节点上（初版踩过：出现「无人机在仓库起降」的假违约）。
    """
    nodes = list(route.nodes)
    specs: list[tuple[int, int, tuple[int, ...]]] = []

    while True:
        # 当前路径上各客户的 slack
        t = 0.0
        cand: list[tuple[float, int, int]] = []
        for pos in range(1, len(nodes) - 1):
            c = nodes[pos]
            arr = _arrival(t, float(inst.base.truck_time_s[nodes[pos - 1], c]), inst.tw[c - 1].e)
            cand.append((inst.tw[c - 1].l - arr, pos, c))
            t = arr + inst.params.service_time
        cand.sort()

        converted = False
        for _, _, c in cand:
            idx = nodes.index(c)
            if idx == 0 or idx == len(nodes) - 1:
                continue
            p_node, q_node = nodes[idx - 1], nodes[idx + 1]
            if p_node == DEPOT or q_node == DEPOT:
                continue                                  # 仓库不能当锚点（约束 6/7）

            new_nodes = [n for n in nodes if n != c]
            # 优先并入已有同锚点架次，否则新建
            trial_specs = None
            for k, (sp, sq, cust) in enumerate(specs):
                if (sp, sq) == (p_node, q_node):
                    trial_specs = specs[:k] + [(sp, sq, cust + (c,))] + specs[k + 1:]
                    break
            if trial_specs is None:
                trial_specs = specs + [(p_node, q_node, (c,))]

            sorties, ok = _to_sorties(new_nodes, trial_specs)
            if not ok:
                continue
            trial = Route(truck=route.truck, nodes=new_nodes, sorties=sorties)
            res, _ = evaluate(inst, Solution([trial]), check_structure=False)
            if res.feasible:
                nodes, specs = new_nodes, trial_specs
                converted = True
                break

        if not converted:
            break

    sorties, ok = _to_sorties(nodes, specs)
    if not ok:                        # 理论上不会发生；保守起见退回无架次
        sorties = []
    return Route(truck=route.truck, nodes=nodes, sorties=sorties)


def routes_needed_by_windows(inst: Instance, cap: int = 99) -> int:
    """只按时间窗与容量跑一遍阶段一，估计最少需要几条路线（供生成器定 K）。"""
    return len(slack_first_routes(inst, cap))


def build_initial(inst: Instance) -> tuple[Solution, float, list[str]]:
    """构造初始解，返回（解, 成本, 提示信息）。"""
    notes: list[str] = []
    max_routes = max(inst.n_trucks, 1) + 2      # 允许略超车队规模，但记录下来
    raw = slack_first_routes(inst, max_routes)
    if len(raw) > inst.n_trucks:
        notes.append(f"用 {len(raw)} 条路线，超过车队规模 K={inst.n_trucks}（时间窗导致）")

    routes = []
    for i, nodes in enumerate(raw):
        r = Route(truck=i, nodes=nodes, sorties=[])
        routes.append(assign_sorties(inst, r) if inst.allow_drones else r)

    sol = Solution(routes)
    res, cost = evaluate(inst, sol)
    if not res.feasible:
        notes.extend(res.all_violations()[:5])
    return sol, cost, notes


def ordered_routes(inst: Instance, order, max_routes: int) -> list[list[int]] | None:
    """按**给定客户顺序**做尾部追加构造（供 PSO 解码使用）。

    与 `slack_first_routes` 的唯一差别是排序依据：这里完全听 `order`，
    不再按 slack 挑人。规则本身（追加后时间窗不破 + 派件不超容量 + 装不下就收车）
    保持一致，这样 PSO 与 ALNS 比的是**搜索能力**，而不是谁的解码器更强。

    返回 None 表示该顺序无法构造出可行解（某个客户单独一条路线也放不进去）。
    """
    base = inst.base
    routes: list[list[int]] = []
    route = [DEPOT]
    t = 0.0
    f_total = 0.0

    for c in order:
        travel = float(base.truck_time_s[route[-1], c])
        arr = _arrival(t, travel, inst.tw[c - 1].e)
        fits = (
            arr <= inst.tw[c - 1].l + 1e-6
            and f_total + float(inst.f[c]) <= inst.params.truck_capacity + 1e-9
        )
        if not fits:
            if len(route) > 1:
                routes.append(route + [DEPOT])
            route = [DEPOT]
            t = 0.0
            f_total = 0.0
            arr = _arrival(0.0, float(base.truck_time_s[DEPOT, c]), inst.tw[c - 1].e)
            if arr > inst.tw[c - 1].l + 1e-6 or float(inst.f[c]) > inst.params.truck_capacity:
                return None                     # 单独一条路线也放不进去
        route.append(int(c))
        t = arr + inst.params.service_time
        f_total += float(inst.f[c])
        if len(routes) >= max_routes:
            return None

    if len(route) > 1:
        routes.append(route + [DEPOT])
    return routes
