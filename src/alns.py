"""ALNS 主循环（论文 Algorithm 1）。

结构：初始解 → 反复「选算子 → 破坏 → 修复 → 可行性 → 退火接受 → 权重/降温/停滞」→ 返回最优解。

与论文的**一处刻意偏差**：算子权重加了下限（`WEIGHT_FLOOR`）。
论文 (75) 段用「拒绝就 ×0.9」，没有下限，一个算子早期运气差就会权重塌到接近 0、
再也选不到（算子饿死）。加下限是标准 ALNS 的常规做法，此处记录在案。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace

import numpy as np

from .evaluator import evaluate
from .initial import build_initial
from .instance import Instance
from .operators import DESTROY, REPAIR
from .solution import Solution

WEIGHT_FLOOR = float(os.environ.get("ALNS_WEIGHT_FLOOR", "0.05"))  # 可调：算子权重下限，防饿死


@dataclass
class ALNSResult:
    best: Solution
    best_cost: float
    initial_cost: float
    history: list[float] = field(default_factory=list)
    weights_destroy: dict[str, float] = field(default_factory=dict)
    weights_repair: dict[str, float] = field(default_factory=dict)
    iterations: int = 0
    t0_used: float = 300.0        # 实际使用的初温（tune=True 时是按 Δ 分布标定的，不是 300）
    restarts: int = 0             # 触发了几次「停滞扰动」
    tune: bool = True             # 是否启用三项参数修正

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


def _calibrate_t0(
    inst: Instance, sol: Solution, cost: float, rng, samples: int = 40
) -> float:
    """按「中位上升步的接受概率 ≈ 50%」标定初温。

    论文的 T0 = 300 是对它自己的成本尺度（13–285 美元）调的。我们的成本是 68–920，
    直接沿用等于初温**相对成本过大**：`exp(-Δ/300)` 对 Δ=20 也有 0.94，
    前几十轮基本是随机游走，一半预算被浪费。

    标定式：`exp(-Δ_med / T0) = 0.5  ⇒  T0 = Δ_med / ln2`。
    Δ 取初始解上若干次随机 destroy+repair 的正增量中位数（只用可行解）。
    """
    ups: list[float] = []
    for _ in range(samples):
        rem = DESTROY["random"](inst, sol, rng, 0.3)
        cand = REPAIR["random"](inst, rem, rng)
        if cand is None:
            continue
        res, c = evaluate(inst, cand)
        if res.feasible and c > cost:
            ups.append(c - cost)
    if not ups:
        return 300.0
    ups.sort()
    return max(ups[len(ups) // 2] / math.log(2.0), 1e-3)


def _kick(inst: Instance, best: Solution, best_cost: float, rng, frac: float = 0.5):
    """停滞时的**真扰动**：随机移除一半客户再随机插回。

    原实现停滞时只是 `sol = best`——回到**同一个吸引域**，等于什么都没做，
    这也是 Set13 在 60 轮就完全停滞的直接原因。真正的多样化必须扰动解本身。
    扰动后若成本超过最优的 1.3 倍，视为扰动失败，退回最优（防止把预算耗在烂解上）。
    """
    rem = DESTROY["random"](inst, best, rng, frac)
    cand = REPAIR["random"](inst, rem, rng)
    if cand is None:
        return best, best_cost
    res, c = evaluate(inst, cand)
    if res.feasible and c <= best_cost * 1.3:
        return cand, c
    return best, best_cost


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
    tune: bool = True,
    trace=None,
    op_policy: str = "adaptive",
    fixed_pair: str | None = None,
) -> ALNSResult:
    """ALNS 主循环。

    `tune=False` 复原论文的原始参数（T0=300、破坏率固定 30%、停滞回退到最优），
    用于消融对照；`tune=True`（默认）启用本项目的三处修正：

    `trace`：可选回调，每轮把「状态 + 动作 + 结果」交给它（M9.3 的诊断用，
    默认 None 时零开销）。`op_policy="uniform"` 强制 16 个算子对等概率随机选，
    用于得到**动作覆盖均衡**的样本（估 conditional 成功率时避免选择偏差）。

    | 项 | 论文 | 本项目 | 为什么 |
    |---|---|---|---|
    | 初温 | 固定 300 | 按 Δ 分布标定（中位上升步接受率 50%） | 300 是按论文成本尺度调的，我们的成本量级不同 |
    | 破坏率 | 固定 30% | 每轮 U(10%, 40%) | 固定值在 100 客户时接近「重建半张图」，单轮代价过高 |
    | 停滞 | 回退到最优 | **随机移除 50% 再插回** | 回退到最优只是回到同一吸引域，没有跳出机制 |
    """
    iters = 2000 if iters is None else iters
    if alpha is None:
        # 降温速率必须与预算挂钩。α=0.99 配 2000 轮时，温度在第 1000 轮已降到 0.013，
        # 后半个预算等于纯爬山——论文的 2000 轮 × α=0.99 本身就有这个尺度问题。
        # 这里让末温固定落到 T0×1e-3：预算越长，α 越接近 1。
        alpha = (1e-3) ** (1.0 / iters) if tune else 0.99
    max_no_improve = 200 if max_no_improve is None else max_no_improve

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

    if t0 is None:
        t0 = _calibrate_t0(inst, sol, cur_cost, rng) if tune else 300.0

    best, best_cost, initial_cost = sol, cur_cost, cur_cost
    wd = {k: 1.0 for k in DESTROY}
    wr = {k: 1.0 for k in REPAIR}
    T = t0
    no_improve = 0
    restarts = 0
    history = [cur_cost]

    names_d, names_r = tuple(DESTROY), tuple(REPAIR)
    for it in range(iters):
        if fixed_pair is not None:
            dn, rn = fixed_pair.split("|")
        elif op_policy == "uniform":
            dn = names_d[int(rng.integers(len(names_d)))]
            rn = names_r[int(rng.integers(len(names_r)))]
        else:
            dn, rn = _roulette(rng, wd), _roulette(rng, wr)
        if ratio is not None:
            r = ratio
        elif tune:
            r = float(rng.uniform(0.10, 0.40))
        else:
            r = 0.30
        rem = DESTROY[dn](inst, sol, rng, r)
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

        if trace is not None:
            n_sorties = sum(len(r.sorties) for r in sol.routes)
            drone_served = sum(len(s.customers) for r in sol.routes for s in r.sorties)
            trace({
                "it": it,
                "action": f"{dn}|{rn}",
                "destroy": dn, "repair": rn,
                "destroyed": len(rem.removed),
                "status": status,
                "delta": (ccost - cur_cost) if status != "rejected" or cand is not None else float("nan"),
                "feasible": 0 if cand is None else int(cres.feasible),
                "cur_cost": cur_cost, "best_cost": best_cost,
                "cur_over_best": cur_cost / best_cost - 1.0 if best_cost else 0.0,
                "no_improve": no_improve, "T": T, "restarts": restarts,
                "n_routes": len(sol.routes), "n_sorties": n_sorties,
                "drone_served": drone_served,
                "n_customers": inst.n_customers, "n_trucks": inst.n_trucks,
                "nfz_count": len(inst.nfz),
            })

        # 停滞计数：以「多久没刷新全局最优」为准（原实现数的是「多少轮没被接受」，
        # 与论文「连续 200 次无改进」的语义不符）
        if tune:
            no_improve = 0 if status == "best" else no_improve + 1
        elif not accepted and status != "best":
            no_improve += 1

        if no_improve >= max_no_improve:
            if tune:
                sol, cur_cost = _kick(inst, best, best_cost, rng)
                T = max(T, t0 * 0.2)          # 扰动后给回一点温度，否则新解立刻被冻结
                restarts += 1
            else:
                sol, cur_cost = best, best_cost
            no_improve = 0
        history.append(cur_cost)

    return ALNSResult(
        best=best, best_cost=best_cost, initial_cost=initial_cost,
        history=history, weights_destroy=wd, weights_repair=wr, iterations=iters,
        t0_used=float(t0), restarts=restarts, tune=tune,
    )
