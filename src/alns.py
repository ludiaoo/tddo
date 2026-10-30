"""ALNS 主循环（论文 Algorithm 1）。

结构：初始解 → 反复「选算子 → 破坏 → 修复 → 可行性 → 退火接受 → 权重/降温/停滞」→ 返回最优解。

与论文的**一处刻意偏差**：算子权重加了下限（`WEIGHT_FLOOR`）。
论文 (75) 段用「拒绝就 ×0.9」，没有下限，一个算子早期运气差就会权重塌到接近 0、
再也选不到（算子饿死）。加下限是标准 ALNS 的常规做法，此处记录在案。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .evaluator import evaluate
from .initial import build_initial
from .instance import Instance
from .operators import DESTROY, REPAIR
from .solution import Solution

WEIGHT_FLOOR = 0.05


@dataclass
class ALNSResult:
    best: Solution
    best_cost: float
    initial_cost: float
    history: list[float] = field(default_factory=list)
    weights_destroy: dict[str, float] = field(default_factory=dict)
    weights_repair: dict[str, float] = field(default_factory=dict)
    iterations: int = 0

    @property
    def saving(self) -> float:
        if self.initial_cost <= 0:
            return 0.0
        return (self.initial_cost - self.best_cost) / self.initial_cost


def _roulette(rng, weights: dict[str, float]) -> str:
    names = list(weights)
    w = np.array([max(weights[n], WEIGHT_FLOOR) for n in names], dtype=float)
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def disable_nfz(inst: Instance) -> Instance:
    """论文 §4「退化开关」：关掉禁飞区后框架退化为标准 VRPD。

    **两套判据必须一起关**：几何绕行（Fig.4/5）与节点禁飞（Fig.3）都由 `inst.nfz` 驱动，
    清空它等于两个模块同时关闭。只关其一会得到论文没有的"第三种模型"，
    Table 3 的 Without-detour 对照就无法对应。
    """
    return replace(inst, nfz=())


def disable_drones(inst: Instance) -> Instance:
    """Table 3 的 **Truck-only** 对照组：不派任何无人机。

    实现为「开关」而不是「另一个求解器」：`allow_drones=False` 后
    ① 初始解不做第二阶段改派；② 修复算子不产生无人机候选；③ 破坏算子自然无事可做
    （解里本来就没有架次）。**车队规模 K 保持不变**——无人机是闲置的，不是减车。
    """
    return replace(inst, allow_drones=False)


def disable_detour(inst: Instance) -> Instance:
    """Table 3 的 **Without-detour** 对照：禁飞区照查，但不允许无人机绕行。

    ⚠ 与 `disable_nfz` 是两件事，别混：
    - `disable_nfz`：把禁飞区整个删掉 → 可行域**变大** → 成本更低；
    - `disable_detour`：禁飞区还生效，只是穿圆的航段直接禁飞 → 可行域**变小** → 成本更高。
    论文 Table 3 的 Without-detour 列 Increasing% 全为正，正是后者。
    """
    return replace(inst, allow_detour=False)


def solve(
    inst: Instance,
    seed: int = 0,
    *,
    use_nfz: bool = True,
    use_drone: bool = True,
    use_detour: bool = True,
    iters: int | None = None,
    t0: float | None = None,
    alpha: float | None = None,
    ratio: float | None = None,
    max_no_improve: int | None = None,
) -> ALNSResult:
    iters = iters or 2000
    t0 = t0 if t0 is not None else 300.0
    alpha = alpha if alpha is not None else 0.99
    ratio = ratio if ratio is not None else 0.30
    max_no_improve = max_no_improve or 200

    if not use_nfz:
        inst = disable_nfz(inst)
    if not use_drone:
        inst = disable_drones(inst)
    if not use_detour:
        inst = disable_detour(inst)
    rng = np.random.default_rng(seed)
    sol, cur_cost, _ = build_initial(inst)
    res, _ = evaluate(inst, sol)
    if not res.feasible:                      # 初始解不可行就不该进主循环
        raise RuntimeError(f"初始解不可行：{res.all_violations()[:3]}")

    best, best_cost, initial_cost = sol, cur_cost, cur_cost
    wd = {k: 1.0 for k in DESTROY}
    wr = {k: 1.0 for k in REPAIR}
    T = t0
    no_improve = 0
    history = [cur_cost]

    for it in range(iters):
        dn, rn = _roulette(rng, wd), _roulette(rng, wr)
        rem = DESTROY[dn](inst, sol, rng, ratio)
        cand = REPAIR[rn](inst, rem, rng)

        accepted = False
        status = "rejected"
        if cand is not None:
            cres, ccost = evaluate(inst, cand)
            if cres.feasible:
                delta = ccost - cur_cost
                if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                    sol, cur_cost = cand, ccost
                    accepted = True
                    if ccost < best_cost - 1e-9:
                        best, best_cost = cand, ccost
                        status = "best"
                        no_improve = 0
                    elif delta < 0:
                        status = "improve"
                    else:
                        status = "accepted"

        # 四档打分（论文 §5）
        if status == "best":
            wd[dn] += 2.0; wr[rn] += 2.0
        elif status == "improve":
            wd[dn] += 1.0; wr[rn] += 1.0
        elif status == "accepted":
            wd[dn] += 0.5; wr[rn] += 0.5
        else:
            wd[dn] *= 0.9; wr[rn] *= 0.9
        wd[dn] = max(wd[dn], WEIGHT_FLOOR)
        wr[rn] = max(wr[rn], WEIGHT_FLOOR)

        T *= alpha
        if not accepted and status != "best":
            no_improve += 1
        if no_improve >= max_no_improve:
            sol, cur_cost = best, best_cost      # 停滞回退到最优
            no_improve = 0
        history.append(cur_cost)

    return ALNSResult(
        best=best, best_cost=best_cost, initial_cost=initial_cost,
        history=history, weights_destroy=wd, weights_repair=wr, iterations=iters,
    )
