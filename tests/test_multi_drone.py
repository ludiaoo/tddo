"""一卡多机（M9.12）的守门测试。

守三条：
1. `drones_per_truck=1` 与历史行为完全一致（由 `golden_check` 保证，这里做结构侧确认）；
2. 两个**时间区间重叠**的架次：m=1 判 `DRONE_BUSY`，m=2 放行；
3. 同一节点既起降又起飞：m=1 判 `LR_SAME_NODE`，m=2 放行。
"""

from __future__ import annotations

import copy

from src.evaluator import V_DRONE_BUSY, V_LR_SAME, _simulate_route
from src.generator import build_instance
from src.io import load_config, load_sets
from src.solution import route_from_specs


def _inst(m: int):
    cfg = copy.deepcopy(load_config())
    cfg["fleet"]["drones_per_truck"] = m
    by_id = {s["id"]: s for s in load_sets()}
    return build_instance(by_id[5], cfg)


def _violations(inst, nodes, specs) -> list[str]:
    r = route_from_specs(0, nodes, specs)
    ev = _simulate_route(inst, r, fast=False)
    return ev.violations


def test_overlapping_sorties_allowed_only_with_two_drones() -> None:
    inst1, inst2 = _inst(1), _inst(2)
    # 取 4 个卡车客户作锚点、2 个客户交给无人机（无人机客户不在卡车路径上）
    custs = sorted(inst1.customers)
    a, b, d, e, c1, c2 = custs[0], custs[1], custs[3], custs[4], custs[6], custs[7]
    nodes = [0, a, b, d, e, 0]
    specs = [(a, d, (c1,)), (b, e, (c2,))]      # 区间 [a,d] 与 [b,e] 重叠

    v1 = _violations(inst1, nodes, specs)
    v2 = _violations(inst2, nodes, specs)

    assert any(V_DRONE_BUSY in x for x in v1), f"m=1 应拒绝重叠架次，实际 {v1}"
    assert not any(V_DRONE_BUSY in x for x in v2), f"m=2 应放行重叠架次，实际 {v2}"


def test_same_node_launch_and_recover_allowed_only_with_two_drones() -> None:
    inst1, inst2 = _inst(1), _inst(2)
    custs = sorted(inst1.customers)
    a, b, d, c1, c2 = custs[0], custs[1], custs[3], custs[6], custs[7]
    nodes = [0, a, b, d, 0]
    # b 既是第 1 段的回收点、又是第 2 段的起飞点
    specs = [(a, b, (c1,)), (b, d, (c2,))]

    v1 = _violations(inst1, nodes, specs)
    v2 = _violations(inst2, nodes, specs)

    assert any(V_LR_SAME in x for x in v1), f"m=1 应拒绝同点起降，实际 {v1}"
    assert not any(V_LR_SAME in x for x in v2), f"m=2 应放行同点起降，实际 {v2}"


def test_default_is_still_one_drone() -> None:
    assert _inst(1).params.drones_per_truck == 1
    assert load_config()["fleet"].get("drones_per_truck", 1) == 1
