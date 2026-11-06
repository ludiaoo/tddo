"""局部搜索：锚点平移（M9.13）。

## 为什么需要它

ALNS 只有破坏-修复，**没有任何直接改「架次起降点在哪」的动作**。
而 `wait_drone`（卡车在回收点等无人机）占等待的 32–61%，它完全由「起降点相对卡车行程的位置」
决定：把回收点往后挪一个节点，卡车就多一段路可走，等待可能直接消失。

破坏-修复几乎碰不到这个维度——它只能通过"删了客户重新插"间接影响锚点，
而重新插入时用的仍是同一套候选生成规则（锚点只能落在卡车相邻节点对上）。

## 做法

对每个架次的 `(launch, recover)` 做**四邻域平移**：起降点各自沿路径前后挪一个节点，
保持 `launch < recover`、都不在仓库、不被其它架次占用。只接受**成本严格下降**的移动。

复杂度：每个架次 4 次评估，一次 pass 约 `4 × 架次数`（Set13 约 80 次），
相比一次修复的上千次评估可以忽略。所以只在**刷新全局最优时**调用。

## 与 m（每车几架机）的关系

- m=1：平移只能减少"卡车早到回收点"的空等，收益有限；
- m=2：平移还能把两段架次在时间上错开，收益更大——这正是本轮要验证的假设。

"""

from __future__ import annotations

from .evaluator import evaluate
from .instance import Instance
from .operators import _route_eval
from .solution import DEPOT, Solution, route_from_specs, route_to_specs


def polish_anchors(inst: Instance, sol: Solution, max_passes: int = 3) -> tuple[Solution, float]:
    """对解里每个架次的起降锚点做一次下降式平移。返回（新解, 成本）。

    只接受严格更优且结构不变的移动；任何导致架次被丢弃的改动一律回退。
    """
    cur = sol
    res, cur_cost = evaluate(inst, cur)
    if not res.feasible:
        return sol, cur_cost

    for _ in range(max_passes):
        improved = False
        for idx, route in enumerate(cur.routes):
            specs = route_to_specs(route)
            if not specs:
                continue
            pos = {n: i for i, n in enumerate(route.nodes)}
            last = len(route.nodes) - 2          # 最后一个内部节点
            for si, (p, q, cust) in enumerate(specs):
                a, b = pos[p], pos[q]
                for da, db in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    na, nb = a + da, b + db
                    if na < 1 or nb > last or na >= nb:
                        continue
                    newp, newq = route.nodes[na], route.nodes[nb]
                    if newp == DEPOT or newq == DEPOT or (newp, newq) == (p, q):
                        continue
                    trial_specs = list(specs)
                    trial_specs[si] = (newp, newq, cust)
                    trial = route_from_specs(route.truck, route.nodes, trial_specs)
                    if len(trial.sorties) != len(specs):     # 有架次被丢弃 → 拒绝
                        continue
                    ok, cost = _route_eval(inst, trial)
                    if not ok:
                        continue
                    others = [x for j, x in enumerate(cur.routes) if j != idx]
                    cand = Solution(others + [trial])
                    cres, ccost = evaluate(inst, cand)
                    if cres.feasible and ccost < cur_cost - 1e-9:
                        cur, cur_cost = cand, ccost
                        specs = route_to_specs(trial)
                        pos = {n: i for i, n in enumerate(trial.nodes)}
                        improved = True
        if not improved:
            break
    return cur, cur_cost
