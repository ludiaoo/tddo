"""LLM 提案的「禁飞区感知」破坏算子（M9.7）——候选池，**不默认启用**。

## 这是什么

前面三条路（候选预筛 / DRL 选算子 / LLM 在环）都在「固定动作集里挑」这个维度上撞了墙。
本条换维度：**造新动作**。本文件是 LLM（本项目里就是助手）针对本问题的核心结构
——时空圆形禁飞区——提出的 7 个破坏算子候选，全部落在现有接口内，不改 ALNS 主循环。

## 提案用的提示词（原文，可复现）

> 你在一个「多卡车 + 多无人机双向配送 + 时空圆形禁飞区」的 ALNS 求解器里工作。
> 现有 4 个破坏算子：random（随机移除）、worst（按移除节省额从大到小）、
> related（距离 + 禁飞区 2r 内打 6 折的相关簇）、shuffle（整条卡车路线打散重排）。
> 请提出 7 个**只利用禁飞区几何/时间结构**的破坏算子。
> 约束：
> 1. 函数签名 `(inst, sol, rng, ratio) -> Removal`，必须用 `remove_customers(sol, targets)`
>    移除（它会处理「锚点被删导致整段架次作废」的连带效应）；
> 2. 必须 O(n) 或 O(n·|Z|)，不得调用 `evaluate`；
> 3. 每个算子要有明确机制假设——为什么「按这个方式成片移除」比随机移除更容易改进；
> 4. 不得读文件、不得依赖 `ratio` 之外的外部状态。
> 输出：完整 Python 代码 + 每个算子一句机制说明。

## 候选与机制假设

| # | 算子 | 机制假设 |
|---|---|---|
| 1 | `nfz_ring` 环带移除 | 圆周外侧 0.8–1.6r 的客户，其航段在「绕行还是直飞」之间反复翻转，是最不稳定的一圈 |
| 2 | `nfz_cross` 穿圆腿移除 | 航段真的穿圆的客户，其无人机指派依赖绕行后的能耗裕度，移除后修复可改判卡车/换锚点 |
| 3 | `nfz_cross_time` 时空交叉移除 | 只有**时段重叠**的穿圆才是真约束；纯空间穿圆在时段外是免费的 |
| 4 | `nfz_anchor` 脆弱锚点整段移除 | 起降锚点贴近禁飞边的架次，稍有移动就不可行，整段移除才能重选锚点 |
| 5 | `nfz_worst_detour` 绕行最重移除 | 绕行负担最重的客户最可能被错误地派给无人机 |
| 6 | `nfz_cluster` 禁飞簇移除 | 以禁飞区中心为种子做空间相关簇，成片移除才能改变「绕哪一侧」的决策 |
| 7 | `nfz_route_reset` 受影响路线重排 | 受影响最重的那条卡车路线整段重排，一次改掉路线级的绕行结构 |

## 为什么单独一个文件

默认路径（`configs/` + `bench` 不带开关）**完全不加载**本模块，所以
`golden_check` 与 Table 3 的所有数字不受影响。启用方式见 `scripts/eval_nfz_ops.py`。
"""

from __future__ import annotations

from .geometry import segment_intersects_circle
from .operators import Removal, _all_customers, remove_customers
from .solution import DEPOT, route_to_specs


# ---------------------------------------------------------------- 共用工具

def _legs(sol) -> dict[int, tuple[int, int]]:
    """客户 → 服务航段的两端节点（前驱, 后继）。

    无人机客户取架次序列里的前后；卡车客户取路线上的前后。
    这是「这个客户的航段长什么样」的最小信息，O(n)。
    """
    out: dict[int, tuple[int, int]] = {}
    for r in sol.routes:
        nodes = r.nodes
        drone_served: set[int] = set()
        for p, q, cust in route_to_specs(r):
            seq = [p, *cust, q]
            for i in range(1, len(seq) - 1):
                out[seq[i]] = (seq[i - 1], seq[i + 1])
            drone_served.update(cust)
        for i in range(1, len(nodes) - 1):
            c = nodes[i]
            if c not in drone_served:
                out[c] = (nodes[i - 1], nodes[i + 1])
    return out


def _crossings(inst, xy, c: int, legs, *, time_only: bool) -> int:
    """客户 c 的两条相邻航段一共穿过几个（可选的：时段重叠的）禁飞圆。"""
    a, b = legs.get(c, (DEPOT, DEPOT))
    tw = inst.tw[c - 1]          # inst.tw 按客户索引（0..n-1 对应节点 1..n）
    pc = (xy[c].x, xy[c].y)
    pa = (xy[a].x, xy[a].y)
    pb = (xy[b].x, xy[b].y)
    n = 0
    for z in inst.nfz:
        if time_only and not (z.t_start < tw.l and z.t_end > tw.e):
            continue
        centre = (z.cx, z.cy)
        if (segment_intersects_circle(pa, pc, centre, z.r)
                or segment_intersects_circle(pc, pb, centre, z.r)):
            n += 1
    return n


def _pick_by_score(sol, scores: dict[int, float], ratio: float, rng) -> Removal:
    """按分数从高到低取 k 个（带一点随机化，避免每轮删同一批）。"""
    if not scores:
        return Removal(sol, [])
    k = max(1, int(round(ratio * len(scores))))
    order = sorted(scores.items(), key=lambda kv: -kv[1])
    pool = order[: min(len(order), max(k * 2, k + 2))]
    idx = rng.permutation(len(pool))[:k] if len(pool) > k else range(len(pool))
    picked = {pool[int(i)][0] for i in idx}
    new_sol, dropped = remove_customers(sol, picked)
    return Removal(new_sol, dropped)


# ---------------------------------------------------------------- 7 个候选

def nfz_ring(inst, sol, rng, ratio: float = 0.30) -> Removal:
    """提案 1：环带移除。

    取一个禁飞区，删掉距圆心 0.8r–1.6r 的客户。这一圈正是「直飞勉强可行 / 必须绕行」
    反复翻转的位置，也是无人机指派最不稳定的位置。
    """
    custs = _all_customers(sol)
    if not custs or not inst.nfz:
        return Removal(sol, [])
    xy = inst.base.nodes
    z = inst.nfz[int(rng.integers(len(inst.nfz)))]
    band = []
    for c in custs:
        d = ((xy[c].x - z.cx) ** 2 + (xy[c].y - z.cy) ** 2) ** 0.5
        if 0.8 * z.r <= d <= 1.6 * z.r:
            band.append(c)
    if len(band) < 2:                                  # 环带里没人 → 退化为随机
        k = max(1, int(round(ratio * len(custs))))
        band = [int(c) for c in rng.choice(custs, size=min(k, len(custs)), replace=False)]
    k = max(1, int(round(ratio * len(custs))))
    if len(band) > k:
        band = [band[int(i)] for i in rng.permutation(len(band))[:k]]
    new_sol, dropped = remove_customers(sol, set(band))
    return Removal(new_sol, dropped)


def nfz_cross(inst, sol, rng, ratio: float = 0.30) -> Removal:
    """提案 2：穿圆腿移除（纯空间）。

    航段真的穿圆的客户，其无人机可行性依赖绕行后的能耗裕度——换个锚点可能整段免绕。
    """
    custs = _all_customers(sol)
    if not custs or not inst.nfz:
        return Removal(sol, [])
    legs, xy = _legs(sol), inst.base.nodes
    scores = {c: float(_crossings(inst, xy, c, legs, time_only=False)) for c in custs}
    scores = {c: v for c, v in scores.items() if v > 0}
    if not scores:
        return Removal(sol, [])
    return _pick_by_score(sol, scores, ratio, rng)


def nfz_cross_time(inst, sol, rng, ratio: float = 0.30) -> Removal:
    """提案 3：时空交叉移除。

    只算**时段重叠**的穿圆——纯空间穿圆若发生在禁飞时段之外，其实不产生任何代价。
    这条是把「时间维」真正用起来的版本。
    """
    custs = _all_customers(sol)
    if not custs or not inst.nfz:
        return Removal(sol, [])
    legs, xy = _legs(sol), inst.base.nodes
    scores = {c: float(_crossings(inst, xy, c, legs, time_only=True)) for c in custs}
    scores = {c: v for c, v in scores.items() if v > 0}
    if not scores:
        return Removal(sol, [])
    return _pick_by_score(sol, scores, ratio, rng)


def nfz_anchor(inst, sol, rng, ratio: float = 0.30) -> Removal:
    """提案 4：脆弱锚点整段移除。

    起降锚点贴近禁飞边的架次，稍有移动就不可行；只删客户不删锚点，修复会被迫回到同一结构。
    这里整段移除，把「选锚点」的自由度还给修复算子。
    """
    custs = _all_customers(sol)
    if not custs or not inst.nfz:
        return Removal(sol, [])
    xy = inst.base.nodes

    def fragility(p: int, q: int) -> float:
        worst = 0.0
        for n in (p, q):
            if n == DEPOT:
                continue
            for z in inst.nfz:
                d = ((xy[n].x - z.cx) ** 2 + (xy[n].y - z.cy) ** 2) ** 0.5
                if d <= 1.5 * z.r:
                    worst = max(worst, (1.5 * z.r - d) / (1.5 * z.r))   # 越贴边越接近 1
        return worst

    scored = []
    for r in sol.routes:
        for p, q, cust in route_to_specs(r):
            f = fragility(p, q)
            if f > 0:
                scored.append((f, cust))
    target = max(1, int(round(ratio * len(custs))))
    scored.sort(key=lambda x: -x[0])
    picked: set[int] = set()
    for f, cust in scored:                              # 从最脆弱的架次开始整段收
        if len(picked) >= target:
            break
        picked.update(cust)
    if not picked:
        return Removal(sol, [])
    if len(picked) < target:                            # 脆弱架次不够 → 用「离圆最近」补足
        rest = [c for c in custs if c not in picked]
        rest.sort(key=lambda c: min(
            ((xy[c].x - z.cx) ** 2 + (xy[c].y - z.cy) ** 2) ** 0.5 for z in inst.nfz))
        picked.update(rest[: target - len(picked)])
    new_sol, dropped = remove_customers(sol, picked)
    return Removal(new_sol, dropped)


def nfz_worst_detour(inst, sol, rng, ratio: float = 0.30) -> Removal:
    """提案 5：绕行最重移除。

    航段同时贴近多个圆的客户，绕行负担最重——最可能被错误地派给无人机。
    分数 = 服务航段穿圆数 + 客户到最近圆的贴近度。
    """
    custs = _all_customers(sol)
    if not custs or not inst.nfz:
        return Removal(sol, [])
    legs, xy = _legs(sol), inst.base.nodes
    scores = {}
    for c in custs:
        n = _crossings(inst, xy, c, legs, time_only=False)
        near = min(
            (((xy[c].x - z.cx) ** 2 + (xy[c].y - z.cy) ** 2) ** 0.5) / max(z.r, 1e-6)
            for z in inst.nfz
        )
        scores[c] = n + max(0.0, 1.0 - near)            # 圆内/贴边 → 加分
    return _pick_by_score(sol, scores, ratio, rng)


def nfz_cluster(inst, sol, rng, ratio: float = 0.30) -> Removal:
    """提案 6：禁飞簇移除。

    以禁飞区中心为种子做空间相关簇。`related` 用客户做种子，这里用「约束的中心」做种子：
    成片移除才能改变「从圆的哪一侧绕」这种路线级决策。
    """
    custs = _all_customers(sol)
    if not custs or not inst.nfz:
        return Removal(sol, [])
    xy = inst.base.nodes
    z = inst.nfz[int(rng.integers(len(inst.nfz)))]
    k = max(1, int(round(ratio * len(custs))))
    scored = sorted(
        custs,
        key=lambda c: ((xy[c].x - z.cx) ** 2 + (xy[c].y - z.cy) ** 2),
    )
    pool = scored[: min(len(scored), k * 2)]
    picked = {pool[int(i)] for i in rng.permutation(len(pool))[:k]}
    new_sol, dropped = remove_customers(sol, picked)
    return Removal(new_sol, dropped)


def nfz_route_reset(inst, sol, rng, ratio: float = 0.30) -> Removal:
    """提案 7：受影响路线重排。

    找「穿圆航段最多」的那条卡车路线，把它的内部客户随机移除一部分（保留仓库端点），
    让修复重新排整条路线——相当于局部重排，但触发信号来自禁飞区而不是随机。
    """
    custs = _all_customers(sol)
    if not custs or not inst.nfz:
        return Removal(sol, [])
    legs, xy = _legs(sol), inst.base.nodes

    best_i, best_score = -1, -1.0
    for i, r in enumerate(sol.routes):
        s = sum(_crossings(inst, xy, c, legs, time_only=False)
                for c in r.nodes if c != DEPOT)
        if s > best_score and len(r.nodes) >= 4:
            best_i, best_score = i, s
    if best_i < 0:
        return Removal(sol, [])

    interior = [c for c in sol.routes[best_i].nodes if c != DEPOT]
    k = max(1, min(len(interior), int(round(ratio * len(custs)))))
    picked = {int(c) for c in rng.choice(interior, size=k, replace=False)}
    new_sol, dropped = remove_customers(sol, picked)
    return Removal(new_sol, dropped)


CANDIDATES = {
    "nfz_ring": nfz_ring,
    "nfz_cross": nfz_cross,
    "nfz_cross_time": nfz_cross_time,
    "nfz_anchor": nfz_anchor,
    "nfz_worst_detour": nfz_worst_detour,
    "nfz_cluster": nfz_cluster,
    "nfz_route_reset": nfz_route_reset,
}
