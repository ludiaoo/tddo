"""换电耗时（M9.21）的守门测试。

要守三条：
1. `swap_s = 0`（默认）时与历史逐位一致；
2. **同节点落地即再起飞**时，第二个架次的起飞时刻正好推迟 `swap_s`（不多不少）；
3. 换电**不在同节点**发生时（下一架次从后面的节点起飞），不额外占时间——
   这是在卡车开车间隙完成的，若连这种情况也加时间，就把换电的代价重复计了。
"""

from __future__ import annotations

import copy
import dataclasses

from src.evaluator import _simulate_route
from src.generator import build_instance
from src.io import load_config, load_sets
from src.solution import route_from_specs

SWAP = 300.0


def _inst(swap: float = 0.0):
    cfg = copy.deepcopy(load_config())
    cfg["fleet"]["allow_hotswap"] = True        # 同节点落地再起飞要求换电打开
    cfg["fleet"]["swap_s"] = swap
    by_id = {s["id"]: s for s in load_sets()}
    return build_instance(by_id[5], cfg)


def _search(inst, same_node: bool, limit: int = 40000):
    """搜一条**可行**的两架次路线（同节点换电 / 不同锚点两种形态）。

    为什么不在测试里写死节点编号：可不可行取决于载荷、时间窗、禁飞区三样，
    写死的组合会随算例调整而失效（本项目已经踩过一次）。这里直接随机搜，
    命中即返回，保证测的是"换电耗时"而不是"运气"。
    """
    import numpy as np

    rng = np.random.default_rng(20261111)
    custs = list(inst.customers)
    light = sorted(custs, key=lambda c: inst.f[c])[:8]      # 轻的客户交给无人机
    for _ in range(limit):
        if same_node:
            a, b, d = rng.choice(custs, 3, replace=False)
            nodes = [0, int(a), int(b), int(d), 0]
            pool = [c for c in light if c not in (a, b, d)]
            if len(pool) < 2:
                continue
            c1, c2 = rng.choice(pool, 2, replace=False)
            specs = [(int(a), int(b), (int(c1),)), (int(b), int(d), (int(c2),))]
        else:
            a, b, d, e = rng.choice(custs, 4, replace=False)
            nodes = [0, int(a), int(b), int(d), int(e), 0]
            pool = [c for c in light if c not in (a, b, d, e)]
            if len(pool) < 2:
                continue
            c1, c2 = rng.choice(pool, 2, replace=False)
            specs = [(int(a), int(b), (int(c1),)), (int(d), int(e), (int(c2),))]
        r = route_from_specs(0, nodes, specs)
        ev = _simulate_route(inst, r, fast=True)
        if ev.feasible and ev.n_sorties == 2:
            return r
    raise AssertionError("没搜到可行的两架次路线，测试需要换算例")


def test_default_swap_is_noop() -> None:
    """swap_s=0：与「配置里压根没有这个键」逐位一致。"""
    cfg = copy.deepcopy(load_config())
    cfg["fleet"]["allow_hotswap"] = True
    cfg["fleet"].pop("swap_s", None)
    by_id = {s["id"]: s for s in load_sets()}
    inst_old = build_instance(by_id[5], cfg)
    inst_new = _inst(0.0)

    r = _search(inst_new, same_node=True)
    a = _simulate_route(inst_new, r, fast=False)
    b = _simulate_route(inst_old, r, fast=False)
    assert a.cost(inst_new) == b.cost(inst_old)
    assert a.finish_time == b.finish_time
    assert list(a.sortie_windows) == list(b.sortie_windows)


def test_swap_delays_same_node_relaunch_exactly() -> None:
    """同节点换电：第二个架次的起飞时刻正好 +swap_s。"""
    inst0 = _inst(0.0)
    route = _search(inst0, same_node=True)
    ev0 = _simulate_route(inst0, route, fast=False)
    assert ev0.feasible and ev0.n_sorties == 2, ev0.violations

    inst1 = _inst(SWAP)
    ev1 = _simulate_route(inst1, route, fast=False)
    assert ev1.feasible and ev1.n_sorties == 2, ev1.violations

    launch0 = [w[0] for w in ev0.sortie_windows]
    launch1 = [w[0] for w in ev1.sortie_windows]
    assert abs(launch1[0] - launch0[0]) < 1e-9              # 第一个架次不受影响
    assert abs((launch1[1] - launch0[1]) - SWAP) < 1e-9     # 第二个正好推迟一份换电时间


def test_swap_not_charged_when_relaunch_is_elsewhere() -> None:
    """换电不在同节点发生（卡车要开一段路）→ 不额外占时间。"""
    inst0 = _inst(0.0)
    route = _search(inst0, same_node=False)
    ev0 = _simulate_route(inst0, route, fast=False)
    assert ev0.feasible and ev0.n_sorties == 2, ev0.violations

    inst1 = _inst(SWAP)
    ev1 = _simulate_route(inst1, route, fast=False)
    assert abs(ev1.finish_time - ev0.finish_time) < 1e-9
    assert [w[0] for w in ev1.sortie_windows] == [w[0] for w in ev0.sortie_windows]


def test_soc_and_energy_untouched_by_swap() -> None:
    """换电耗时只动时刻，不动能耗与 SOC（电池是换上的那块，不是原来那块）。"""
    inst0 = _inst(0.0)
    inst1 = _inst(SWAP)
    r = _search(inst0, same_node=True)
    assert _simulate_route(inst0, r).drone_energy == _simulate_route(inst1, r).drone_energy
    assert not any("SOC" in v for v in _simulate_route(inst1, r).violations)
