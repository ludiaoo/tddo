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

import math
import os
from dataclasses import dataclass

import numpy as np

from . import ml
from .evaluator import _simulate_route, evaluate
from .instance import Instance
from .ml import features as ml_features
from .ml.screen import LightGBMScreen
from .solution import DEPOT, Route, Solution, route_from_specs, route_to_specs

TOP_K = 6          # 粗排后真正做完整评估的候选数
# 无人机候选的全局预筛上限（M9.9）。0 = 关闭，与历史行为逐位一致。
# 背景：卡车侧一直有 TOP_K 粗排，无人机侧没有——100 客户时每个客户要构造并评估
# 约 125 条 trial 路线，占单次求解 69% 的时间（cProfile，Set13）。
DRONE_TOPK = int(os.environ.get("ALNS_DRONE_TOPK", "0"))
_PROBE = None      # 诊断用：置为 list 时，把 (代理量, 真实Δcost) 追加进去（平时零开销）


@dataclass
class Removal:
    sol: Solution
    removed: list[int]


# ---------------------------------------------------------------- 基础工具

_MEMO_CAP = 120_000          # 记忆化容量上限，超了清空（宁可再生，也不吃内存）


def _route_eval(inst: Instance, route: Route) -> tuple[bool, float]:
    """单条路线的（可行, 成本）。

    两处优化，都不改变语义：
    1. 直连 `_simulate_route(..., fast=True)`——不再构造 Solution/EvalResult，
       也不跑完整性校验（单条路线本来就不该查「客户是否都被服务」）；
       碰到第一处违约立即返回，不模拟剩余节点。
    2. **按签名记忆化**：修复算子对「未被改动的那几条路线」会反复求值
       （truck 候选算一遍、drone 候选又算一遍），这是纯重复计算。
    """
    sig = (route.truck, tuple(route.nodes),
           tuple((s.launch, s.recover, s.customers) for s in route.sorties))
    memo = getattr(inst, "_route_memo", None)
    if memo is None:
        memo = {}
        inst._route_memo = memo
    got = memo.get(sig)
    if got is not None:
        return got

    ev = _simulate_route(inst, route, fast=True)
    res = (True, ev.cost(inst)) if ev.feasible else (False, 0.0)
    if len(memo) >= _MEMO_CAP:
        memo.clear()
    memo[sig] = res
    # 顺带把完整 RouteEval 存进 ML 用的 memo（特征需要 finish/waiting 等精确量），
    # 让 ML 路径的 `before` 直接命中这里，而不是再跑一遍 fast=False 的完整仿真
    ev_memo = getattr(inst, "_route_memo_ev", None)
    if ev_memo is None:
        ev_memo = {}
        inst._route_memo_ev = ev_memo
    if len(ev_memo) >= _MEMO_CAP:
        ev_memo.clear()
    ev_memo[sig] = ev
    return res


def _route_eval_ev(inst: Instance, route: Route):
    """带 RouteEval 的记忆化版本（只给 ML 路径用，特征需要 finish/waiting 等精确量）。

    与 `_route_eval` 各自独立成 memo：off 路径完全不经过这里，行为不受影响。
    """
    sig = (route.truck, tuple(route.nodes),
           tuple((s.launch, s.recover, s.customers) for s in route.sorties))
    memo = getattr(inst, "_route_memo_ev", None)
    if memo is None:
        memo = {}
        inst._route_memo_ev = memo
    got = memo.get(sig)
    if got is not None:
        return got
    ev = _simulate_route(inst, route, fast=False)
    if len(memo) >= _MEMO_CAP:
        memo.clear()
    memo[sig] = ev
    return ev


def _keeps_all(base: Route, trial: Route, new: int) -> bool:
    """候选路线是否**既装进了新客户、又没有弄丢原有客户**。

    只查 `new in trial` 是不够的：`route_from_specs` 在锚点位置冲突时会静默丢弃整段架次，
    可能把**之前**插进去的无人机客户弄丢，而解看起来更便宜——M5 踩过，
    换禁飞区摆放后又复发（由 test_修复能还原可行解 抓出）。这条检查把整类问题一次堵死。
    """
    got = trial.served_customers()
    return new in got and base.served_customers() <= got


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
            if not _keeps_all(r, trial, c):            # 架次冲突可能把客户又丢掉
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


SCREEN_TOP_M = max(1, int(os.environ.get("ML_SCREEN_TOP_M", "5")))
SCREEN_FALLBACK = max(0.0, float(os.environ.get("ML_SCREEN_FALLBACK", "0.05")))
SCREEN_MIN_N = max(1, int(os.environ.get("ML_SCREEN_MIN_N", "100")))
_SCREEN: LightGBMScreen | None = None
_SCREEN_TRIED = False


def _get_screen() -> LightGBMScreen | None:
    """惰性加载预筛模型；缺失、lightgbm 不可用或特征不一致时返回 None（退回全量评估）。"""
    global _SCREEN, _SCREEN_TRIED
    if not _SCREEN_TRIED:
        _SCREEN_TRIED = True
        try:
            if LightGBMScreen.available():
                _SCREEN = LightGBMScreen()
        except Exception as exc:                 # 模型坏了不能拖垮求解
            print(f"[ml] 预筛模型加载失败，退回全量评估：{exc}")
            _SCREEN = None
    return _SCREEN


def _drone_candidates(inst: Instance, sol: Solution, c: int, rng=None):
    """所有可行的无人机插入候选（并入已有架次 / 新建架次），按真实 Δcost 升序。

    模式（`src/ml/state`，默认 off）：

    - **off**：全量评估，与历史行为**逐位一致**；
    - **collect**：同样全量评估（决策与随机数消耗都不变），额外把「特征 + 标签」喂给采集器；
    - **screen**：候选先打分，只评估 top-m（`ML_SCREEN_TOP_M`，默认 5）+ 5% 随机兜底
      （结果**有意**不同）。模型缺失时自动退回全量评估。

    ⚠ `_drone_candidates_full` 与 `_drone_candidates_screen` 的**候选生成规则必须一致**
    （同一个循环骨架），否则 hit-rate 指标会失真。改这里时两个函数一起改。
    """
    mode = ml.get_mode()
    if DRONE_TOPK > 0 and mode == "off":
        return _drone_candidates_pruned(inst, sol, c)
    if mode == "screen":
        # 分级启用：小算例的单次评估本来就很便宜（且大量 memo 命中），
        # 打分的固定开销反而更贵——实测 Set9（50 客户）慢 13%，Set13（100 客户）快 9%。
        # 阈值 `ML_SCREEN_MIN_N`（默认 100）以下保持原路径。
        if inst.n_customers >= SCREEN_MIN_N:
            return _drone_candidates_screen(inst, sol, c, rng)
    return _drone_candidates_full(inst, sol, c, mode)


def _drone_dist(inst: Instance, i: int, j: int) -> float:
    """直线距离（不算绕行）。预筛只要单调可比的代理量，不值得为它跑绕行几何。"""
    a, b = inst.base.nodes[i], inst.base.nodes[j]
    return math.hypot(a.x - b.x, a.y - b.y)


def _drone_candidates_pruned(inst: Instance, sol: Solution, c: int):
    """与 `_drone_candidates_full` 同构，但在**构造 trial 之前**先粗排。

    代理量：插入后该架次飞行长度的**增量**（直线距离，O(1)）。理由是卡车节点不变，
    所以成本变化几乎全部来自架次时长的变化——这与 `_restricted` 的实际目标一致。

    取法：全局前 K 名 + **每条路线各自的第一名**。后者是安全网——避免全局前 K 全落在
    同一条路线上，导致其它路线一个候选都没试（卡车侧按路线取 TOP_K，天然没这个问题）。
    """
    if not inst.allow_drones:
        return []
    desc: list[tuple[float, int, int, int, int]] = []      # (proxy, idx, kind, i/k, pos)
    for idx, r in enumerate(sol.routes):
        specs = route_to_specs(r)
        # ① 并入已有架次
        for i, (p, q, cust) in enumerate(specs):
            for pos in range(len(cust) + 1):
                prev = cust[pos - 1] if pos > 0 else p
                nxt = cust[pos] if pos < len(cust) else q
                if c in cust:                             # 已在该架次里，跳过（与全量版一致由 _keeps_all 兜底）
                    proxy = 1e18
                else:
                    proxy = (_drone_dist(inst, prev, c) + _drone_dist(inst, c, nxt)
                             - _drone_dist(inst, prev, nxt))
                desc.append((proxy, idx, 0, i, pos))
        # ② 新建架次：相邻两节点作锚点
        for k in range(len(r.nodes) - 1):
            p, q = r.nodes[k], r.nodes[k + 1]
            if p == DEPOT or q == DEPOT:
                continue
            if any((sp, sq) == (p, q) for sp, sq, _ in specs):
                continue
            desc.append((_drone_dist(inst, p, c) + _drone_dist(inst, c, q), idx, 1, k, 0))

    if not desc:
        return []
    desc.sort(key=lambda t: t[0])
    keep = {id(t) for t in desc[:DRONE_TOPK]}
    best_per_route: dict[int, tuple] = {}
    for t in desc:
        best_per_route.setdefault(t[1], t)
    keep |= {id(t) for t in best_per_route.values()}
    shortlist = [t for t in desc if id(t) in keep]

    before_cache = {i: _route_eval(inst, r) for i, r in enumerate(sol.routes)}
    out = []
    for proxy, idx, kind, pos_i, pos in shortlist:
        r = sol.routes[idx]
        specs = route_to_specs(r)
        if kind == 0:
            p, q, cust = specs[pos_i]
            new_cust = cust[:pos] + (c,) + cust[pos:]
            trial_specs = specs[:pos_i] + [(p, q, new_cust)] + specs[pos_i + 1:]
        else:
            p, q = r.nodes[pos_i], r.nodes[pos_i + 1]
            trial_specs = specs + [(p, q, (c,))]
        trial = route_from_specs(r.truck, r.nodes, trial_specs)
        if not _keeps_all(r, trial, c):
            continue
        ok, cost = _route_eval(inst, trial)
        if _PROBE is not None:                     # 诊断：连不可行的也记，才能量可行率
            _PROBE.append((proxy, ok, (cost - before_cache[idx][1]) if ok else float("nan")))
        if ok:
            before_ok, before_cost = before_cache[idx]
            if before_ok:
                other = [x for j, x in enumerate(sol.routes) if j != idx]
                out.append((cost - before_cost, Solution(other + [trial])))
    out.sort(key=lambda x: x[0])
    return out


def _drone_candidates_full(inst: Instance, sol: Solution, c: int, mode: str):
    """off / collect 共用：全量评估每个候选。collect 模式额外记录「特征 + 标签」。"""
    if not inst.allow_drones:              # Truck-only 对照组：没有任何无人机候选
        return []
    collect = mode == "collect"
    if collect:
        ml.next_step()
        ml.bump("steps")
    out = []
    for idx, r in enumerate(sol.routes):
        specs = route_to_specs(r)
        before = _route_eval_ev(inst, r)
        if not before.feasible:
            continue
        before_cost = before.cost(inst)
        ctx = ml_features.route_ctx(inst, r) if collect else None
        sinfo = ctx["specs_info"] if collect else None

        # ① 并入已有架次：插到飞行序列的任一位置
        for i, (p, q, cust) in enumerate(specs):
            for pos in range(len(cust) + 1):
                new_cust = cust[:pos] + (c,) + cust[pos:]
                trial_specs = specs[:i] + [(p, q, new_cust)] + specs[i + 1:]
                trial = route_from_specs(r.truck, r.nodes, trial_specs)
                if not _keeps_all(r, trial, c):
                    continue
                feats = None
                if collect:
                    ml.bump("attempts")
                    si = sinfo[i]
                    try:
                        feats = ml_features.drone_candidate_features(
                            inst, ctx, before,
                            kind=0, sp=si["sp"], sq=si["sq"], sortie=cust, pos=pos, c=c,
                            anchor_nfz=si["nfz"], old_len=si["old_len"],
                            anchor_truck_dist=si["truck_dist"],
                            sortie_f=si["f_sum"], sortie_g=si["g_sum"], si=si,
                        )
                    except Exception:
                        ml.bump("feat_errors")
                ok, cost = _route_eval(inst, trial)
                if collect:
                    ml.bump("evals")
                    ml.record({**(dict(zip(ml_features.FEATURE_NAMES, feats)) if feats else {}),
                               "feasible": int(ok),
                               "delta": (cost - before_cost) if ok else float("nan")})
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
            if not _keeps_all(r, trial, c):
                continue
            feats = None
            if collect:
                ml.bump("attempts")
                try:
                    feats = ml_features.drone_candidate_features(
                        inst, ctx, before,
                        kind=1, sp=k, sq=k + 1, sortie=(), pos=0, c=c,
                        anchor_nfz=ctx["leg_nfz"][k], old_len=0.0,
                        anchor_truck_dist=float(inst.base.truck_dist_m[p, q]),
                        sortie_f=0.0, sortie_g=0.0, leg_k=k,
                    )
                except Exception:
                    ml.bump("feat_errors")
            ok, cost = _route_eval(inst, trial)
            if collect:
                ml.bump("evals")
                ml.record({**(dict(zip(ml_features.FEATURE_NAMES, feats)) if feats else {}),
                           "feasible": int(ok),
                           "delta": (cost - before_cost) if ok else float("nan")})
            if ok:
                new_sol = Solution([x for j, x in enumerate(sol.routes) if j != idx] + [trial])
                out.append((cost - before_cost, new_sol))

    out.sort(key=lambda x: x[0])
    return out


def _drone_candidates_screen(inst: Instance, sol: Solution, c: int, rng=None):
    """screen 模式：候选生成规则与 `_drone_candidates_full` 完全一致，只省评估。

    打分（`LightGBMScreen.select`）：p_feas ≥ 阈值者按 Δcost_hat 升序取前 m，
    不足再用 p_feas 最高的补；另加 `ML_SCREEN_FALLBACK`（默认 5%）随机候选兜底，
    防止模型把真正的好候选永远压在评估之外。
    """
    screen_model = _get_screen()
    if screen_model is None:                 # 没模型 → 与 off 行为一致
        return _drone_candidates_full(inst, sol, c, "off")
    if not inst.allow_drones:                # Truck-only 对照组：没有任何无人机候选
        return []
    ml.next_step()
    ml.bump("steps")

    # 两阶段：先只算「特征」，线性门 + 精排之后**才**构造 trial 与评估。
    # v1 的教训：给每个候选都 route_from_specs + 跑 LightGBM，开销比省下的评估还大。
    pend: list[tuple[tuple, tuple]] = []       # (feats, gen)  gen = 重建候选所需的全部参数
    for idx, r in enumerate(sol.routes):
        specs = route_to_specs(r)
        before = _route_eval_ev(inst, r)
        if not before.feasible:
            continue
        ctx = ml_features.route_ctx(inst, r)
        f_list, g_list = ctx["f_list"], ctx["g_list"]

        for i, si in enumerate(ctx["specs_info"]):
            cust = si["cust"]
            # O(1) 硬预筛：载荷是必要条件（发射时全部派件在机 / 返航前全部取件在机），
            # 违反的候选连特征都不用算。只用于 screen 路径（collect 保持与 off 全量一致）。
            if si["f_sum"] + f_list[c] > inst.params.drone_capacity + 1e-9 \
                    or si["g_sum"] + g_list[c] > inst.params.drone_capacity + 1e-9:
                continue
            for pos in range(len(cust) + 1):
                ml.bump("attempts")
                feats = ml_features.drone_candidate_features(
                    inst, ctx, before,
                    kind=0, sp=si["sp"], sq=si["sq"], sortie=cust, pos=pos, c=c,
                    anchor_nfz=si["nfz"], old_len=si["old_len"],
                    anchor_truck_dist=si["truck_dist"],
                    sortie_f=si["f_sum"], sortie_g=si["g_sum"], si=si,
                )
                pend.append((feats, ("merge", idx, i, pos)))

        cap = inst.params.drone_capacity + 1e-9
        f_c, g_c = ctx["f_list"][c], ctx["g_list"][c]
        for k in range(len(r.nodes) - 1):
            p, q = r.nodes[k], r.nodes[k + 1]
            if p == DEPOT or q == DEPOT:
                continue
            if any((sp, sq) == (p, q) for sp, sq, _ in specs):
                continue
            if f_c + g_c > cap:            # 单客户载荷超限（派件 + 取件同时在机）
                continue
            ml.bump("attempts")
            feats = ml_features.drone_candidate_features(
                inst, ctx, before,
                kind=1, sp=k, sq=k + 1, sortie=(), pos=0, c=c,
                anchor_nfz=ctx["leg_nfz"][k], old_len=0.0,
                anchor_truck_dist=float(inst.base.truck_dist_m[p, q]),
                sortie_f=0.0, sortie_g=0.0, leg_k=k,
            )
            pend.append((feats, ("new", idx, k)))

    m = min(SCREEN_TOP_M, len(pend))
    picked = set(screen_model.select([f for f, *_ in pend], m))
    if rng is not None and SCREEN_FALLBACK > 0 and len(pend) > m:
        rest = [i for i in range(len(pend)) if i not in picked]
        n_extra = max(1, int(round(SCREEN_FALLBACK * len(pend))))
        if rest:
            for i in rng.choice(rest, size=min(n_extra, len(rest)), replace=False):
                picked.add(int(i))

    out = []
    for i in sorted(picked):
        feats, gen = pend[i]
        if gen[0] == "merge":
            _, idx, i, pos = gen
            r = sol.routes[idx]
            specs = route_to_specs(r)
            p, q, cust = specs[i]
            trial = route_from_specs(
                r.truck, r.nodes, specs[:i] + [(p, q, cust[:pos] + (c,) + cust[pos:])] + specs[i + 1:]
            )
        else:
            _, idx, k = gen
            r = sol.routes[idx]
            specs = route_to_specs(r)
            p, q = r.nodes[k], r.nodes[k + 1]
            trial = route_from_specs(r.truck, r.nodes, specs + [(p, q, (c,))])
        if not _keeps_all(r, trial, c):          # 延迟到选完之后做，代价一样
            continue
        before = _route_eval_ev(inst, r)
        ml.bump("evals")
        ok, cost = _route_eval(inst, trial)
        if ok:
            new_sol = Solution([x for j, x in enumerate(sol.routes) if j != idx] + [trial])
            out.append((cost - before.cost(inst), new_sol))
    out.sort(key=lambda x: x[0])
    return out


def _apply_repair(inst, sol, removed, rng, prefer: str) -> Solution | None:
    """通用修复流程：按顺序为每个客户挑候选插入。"""
    order = list(removed)
    cur = sol
    for c in order:
        truck = _truck_candidates(inst, cur, c)
        drone = _drone_candidates(inst, cur, c, rng)
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
        pool = _truck_candidates(inst, cur, c) + _drone_candidates(inst, cur, c, rng)
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
                # 与 _truck_candidates / _drone_candidates 同一条铁律：锚点位置冲突时
                # route_from_specs 会**静默丢弃**整段架次，客户 c 就此消失。
                # 这类「丢客户」不报错、成本还更低，会被当成更优解——M5 踩过一次，
                # 换禁飞区摆放后又复发，由 test_修复能还原可行解 抓出。
                if not _keeps_all(r, trial, c):
                    continue
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
