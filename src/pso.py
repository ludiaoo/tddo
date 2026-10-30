"""PSO 对照（Table 3 的 PSO 列）。

## 编码是自造的，必须说清楚

论文只给了 PSO 的参数（学习因子 2.05、惯性权重 0.9→0.4、与 ALNS 相同的迭代预算），
**没有给粒子编码与解码方式**。这里用最标准的 **random-key（优先级列表）编码**：

- 粒子是一个 n 维向量 `x ∈ [0,1]^n`，第 i 维就是客户 i 的优先级；
- 解码：按优先级升序得到客户插入顺序 → 交给 `ordered_routes` 做尾部追加构造
  （时间窗/容量不满足就收车返仓）→ 每条路线按 min-slack 改派无人机架次 → `evaluate`。

这样 PSO 与 ALNS 的区别就只在**搜索机制**上，解码器与 ALNS 的初始解构造是同一套规则，
不会出现「PSO 输在解码器上」的误读。代价是 PSO 的解空间被限制在「某种客户顺序 + 贪心构造」
可达的解里，这也是为什么它通常不如 ALNS——与论文的结论方向一致。

## 预算口径

论文说「与 ALNS 相同的迭代预算」。ALNS 每轮只评估 **1 个候选解**，所以这里把预算定义成
**解码次数 = ALNS 迭代数**（`iters`），再除以种群规模得到 PSO 的迭代轮数。
若沿用「轮数相同」而种群 20，PSO 会多评估 20 倍解，那不是公平比较。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .evaluator import evaluate
from .initial import assign_sorties
from .instance import Instance
from .operators import _route_eval
from .solution import DEPOT, Route, Solution

PENALTY = 1.0e6        # 解码失败（顺序构造不出可行解）时的罚值
W_START, W_END = 0.9, 0.4      # 惯性权重线性递减 [论文给定]
C1 = C2 = 2.05                 # 学习因子 [论文给定]
V_MAX = 0.25


@dataclass
class PSOResult:
    best: Solution | None
    best_cost: float
    history: list[float] = field(default_factory=list)
    iterations: int = 0
    decodes: int = 0
    failed_decodes: int = 0


TOP_K = 3        # 每条路线里真正做完整评估的插入位置数（与 ALNS 修复算子的粗排思路一致）


def _route_cost(inst: Instance, nodes: list[int]) -> tuple[bool, float]:
    return _route_eval(inst, Route(truck=0, nodes=nodes, sorties=[]))


def decode(inst: Instance, x: np.ndarray) -> tuple[Solution | None, float]:
    """random-key → 解。

    解码分三步，**每一步都是优先级的确定函数**，不带随机：
    1. 按优先级升序得到客户插入顺序；
    2. 逐个插入：在**所有现有路线的所有位置**里，先按「卡车距离增量」粗排取每路线前 `TOP_K` 个，
       再实际评估，连同「新开一条路线」一起取成本增量最小者；全不可行则本次解码失败（罚值）；
    3. 每条路线按 min-slack 改派无人机架次（与 ALNS 初始解同一套规则），整体评估。

    第 2 步只算卡车插入、不枚举无人机候选：无人机由第 3 步统一加。
    这样解码与 ALNS 的初始解构造用的是同一套规则，PSO 与 ALNS 的差别只在**搜索机制**上。
    """
    base = inst.base
    order = np.argsort(x, kind="stable") + 1        # 客户编号 1..n
    max_routes = max(inst.n_trucks, 1) + 2

    routes: list[Route] = []
    costs: list[float] = []

    for c in order:
        c = int(c)
        best: tuple[float, int, int] | None = None        # (delta, route_idx, pos)

        for idx, r in enumerate(routes):
            coarse = []
            for k in range(len(r.nodes) - 1):
                a, b = r.nodes[k], r.nodes[k + 1]
                d = float(base.truck_dist_m[a, c] + base.truck_dist_m[c, b] - base.truck_dist_m[a, b])
                coarse.append((d, k))
            coarse.sort()
            for _, k in coarse[:TOP_K]:
                trial = r.nodes[: k + 1] + [c] + r.nodes[k + 1:]
                ok, cost = _route_cost(inst, trial)
                if not ok:
                    continue
                delta = cost - costs[idx]
                if best is None or delta < best[0]:
                    best = (delta, idx, k + 1)

        if len(routes) < max_routes:
            ok, cost = _route_cost(inst, [DEPOT, c, DEPOT])
            if ok and (best is None or cost < best[0]):
                best = (cost, -1, 1)

        if best is None:
            return None, PENALTY
        _, idx, pos = best
        if idx == -1:
            routes.append(Route(truck=len(routes), nodes=[DEPOT, c, DEPOT], sorties=[]))
            costs.append(_route_cost(inst, routes[-1].nodes)[1])
        else:
            nodes = routes[idx].nodes[:pos] + [c] + routes[idx].nodes[pos:]
            routes[idx].nodes = nodes
            costs[idx] = _route_cost(inst, nodes)[1]

    if len(routes) > inst.n_trucks:
        return None, PENALTY

    if inst.allow_drones:
        routes = [assign_sorties(inst, r) for r in routes]
    for i, r in enumerate(routes):
        r.truck = i
    sol = Solution(routes)
    res, cost = evaluate(inst, sol)
    if not res.feasible:
        return None, PENALTY
    return sol, cost


def solve_pso(
    inst: Instance,
    seed: int = 0,
    *,
    iters: int | None = None,
    swarm: int = 20,
) -> PSOResult:
    iters = iters or 2000                    # 与 ALNS 同口径：解码次数
    n = inst.n_customers
    rounds = max(1, iters // swarm)
    rng = np.random.default_rng(seed)

    x = rng.random((swarm, n))
    v = rng.uniform(-V_MAX, V_MAX, size=(swarm, n))

    costs = np.empty(swarm)
    pbest = x.copy()
    pbest_cost = np.empty(swarm)
    best_sol, best_cost = None, float("inf")
    failed = 0
    history: list[float] = []

    for k in range(swarm):
        sol, c = decode(inst, x[k])
        costs[k] = c
        pbest_cost[k] = c
        if c < PENALTY:
            if sol is not None and c < best_cost:
                best_sol, best_cost = sol, c
        else:
            failed += 1
        history.append(best_cost if best_cost < float("inf") else PENALTY)

    gbest = pbest[int(np.argmin(pbest_cost))].copy()
    gbest_cost = float(pbest_cost.min())

    for it in range(rounds):
        w = W_START + (W_END - W_START) * it / max(1, rounds - 1)
        r1 = rng.random((swarm, n))
        r2 = rng.random((swarm, n))
        v = w * v + C1 * r1 * (pbest - x) + C2 * r2 * (gbest[None, :] - x)
        v = np.clip(v, -V_MAX, V_MAX)
        x = np.clip(x + v, 0.0, 1.0)

        for k in range(swarm):
            sol, c = decode(inst, x[k])
            if c >= PENALTY:
                failed += 1
                continue
            if c < pbest_cost[k]:
                pbest[k], pbest_cost[k] = x[k], c
                if c < gbest_cost - 1e-9:
                    gbest, gbest_cost = x[k].copy(), c
                    best_sol, best_cost = sol, c
        history.append(best_cost if best_cost < float("inf") else PENALTY)

    return PSOResult(
        best=best_sol,
        best_cost=best_cost,
        history=history,
        iterations=rounds,
        decodes=swarm * (rounds + 1),
        failed_decodes=failed,
    )
