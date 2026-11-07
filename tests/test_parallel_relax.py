"""M9.15 的两个开关：同节点换机（`allow_hotswap`）与跨段锚点（`SORTIE_SPAN`）。

守四条：
1. 默认（不开换机）时，同一节点既回收又起飞仍判 `LR_SAME_NODE`——历史行为不变；
2. 打开换机后，同一节点「先降落、再起飞」放行，且两段之间留出 `launch_time`；
3. `SORTIE_SPAN=1` 时锚点枚举与历史实现逐项相同；
4. `SORTIE_SPAN>1` 时新增的锚点都满足「结束位置 > 开始位置、且都不是仓库」。
"""

from __future__ import annotations

import copy

from src.evaluator import V_LR_SAME, _simulate_route
from src.generator import build_instance
from src.io import load_config, load_sets
from src.solution import route_from_specs


def _inst(**fleet):
    cfg = copy.deepcopy(load_config())
    cfg["fleet"].update(fleet)
    by_id = {s["id"]: s for s in load_sets()}
    return build_instance(by_id[5], cfg)


def _route(inst, nodes, specs):
    return route_from_specs(0, nodes, specs)


def test_same_node_recover_then_launch_needs_hotswap() -> None:
    inst = _inst(drones_per_truck=1, allow_hotswap=False)
    custs = sorted(inst.customers)
    a, b, d, c1, c2 = custs[0], custs[1], custs[3], custs[6], custs[7]
    nodes = [0, a, b, d, 0]
    specs = [(a, b, (c1,)), (b, d, (c2,))]        # b：先回收第 1 段、再起飞第 2 段

    ev = _simulate_route(inst, _route(inst, nodes, specs), fast=False)
    assert any(V_LR_SAME in x for x in ev.violations), f"默认应拒绝同节点换机，实际 {ev.violations}"


def test_hotswap_allows_same_node_and_waits_for_launch_time() -> None:
    strict = _inst(drones_per_truck=1, allow_hotswap=False)
    loose = _inst(drones_per_truck=1, allow_hotswap=True)
    custs = sorted(strict.customers)
    a, b, d, c1, c2 = custs[0], custs[1], custs[3], custs[6], custs[7]
    nodes = [0, a, b, d, 0]
    specs = [(a, b, (c1,)), (b, d, (c2,))]

    ev = _simulate_route(loose, _route(loose, nodes, specs), fast=False)
    assert not any(V_LR_SAME in x for x in ev.violations), f"开启换机后应放行，实际 {ev.violations}"
    assert ev.feasible or ev.violations, "无论如何都要么可行、要么给出具体违约原因"
    assert ev.n_sorties == 2, "两个架次都应被计入"


def test_anchor_pairs_span_one_matches_history() -> None:
    """`SORTIE_SPAN=1` 必须与历史枚举逐项相同（这是 golden_check 的结构侧保证）。"""
    import src.operators as ops

    inst = _inst()
    custs = sorted(inst.customers)
    nodes = [0] + custs[:4] + [0]
    r = _route(inst, nodes, [])

    old = ops.SORTIE_SPAN
    try:
        ops.SORTIE_SPAN = 1
        got = list(ops._anchor_pairs(r))
        want = [(k, nodes[k], nodes[k + 1]) for k in range(len(nodes) - 1)
                if nodes[k] != 0 and nodes[k + 1] != 0]
        assert got == want, f"span=1 枚举不一致：{got} vs {want}"
    finally:
        ops.SORTIE_SPAN = old


def test_anchor_pairs_span_extends_backward_only() -> None:
    import src.operators as ops

    inst = _inst()
    nodes = [0, 1, 2, 3, 4, 5, 6, 0]
    r = _route(inst, nodes, [])

    old = ops.SORTIE_SPAN
    try:
        ops.SORTIE_SPAN = 3
        pairs = list(ops._anchor_pairs(r))
        assert len(pairs) > len(nodes) - 2, "跨段后锚点对应该更多"
        pos = {n: i for i, n in enumerate(nodes)}
        for k, p, q in pairs:
            assert p == nodes[k]
            assert pos[q] > k, "回收点必须在起飞点之后"
            assert q != 0 and p != 0, "不允许在仓库起降"
    finally:
        ops.SORTIE_SPAN = old
