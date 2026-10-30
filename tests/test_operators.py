"""破坏/修复算子与主循环的测试。

核心是 **round-trip**：任意破坏算子拆解后，任意修复算子必须能把解还原成
「覆盖全部客户且可行」的解。这条不成立，ALNS 就会在不可行解上打转。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.alns import WEIGHT_FLOOR, solve
from src.evaluator import evaluate
from src.generator import build_instance
from src.initial import build_initial
from src.io import load_config, load_sets
from src.operators import DESTROY, REPAIR, remove_customers, served_map
from src.solution import validate_structure

CFG = load_config()
SETS = load_sets()


@pytest.fixture(scope="module")
def inst():
    return build_instance(SETS[0], CFG)          # Set1：10 客户，跑得快


@pytest.fixture(scope="module")
def base_sol(inst):
    sol, _, _ = build_initial(inst)
    return sol


def test_破坏算子保留完整性(inst, base_sol):
    """破坏后剩下的客户 + 被移除的客户 = 原解全部客户（不能悄悄丢人）。"""
    rng = np.random.default_rng(0)
    before = set(served_map(base_sol))
    for name, fn in DESTROY.items():
        rem = fn(inst, base_sol, rng, 0.3)
        left = set(served_map(rem.sol))
        assert left | set(rem.removed) == before, f"{name} 丢了客户"
        assert not (left & set(rem.removed)), f"{name} 移除集与剩余集重叠"


@pytest.mark.parametrize("dname", list(DESTROY))
@pytest.mark.parametrize("rname", list(REPAIR))
def test_修复能还原可行解(inst, base_sol, dname, rname):
    rng = np.random.default_rng(1)
    rem = DESTROY[dname](inst, base_sol, rng, 0.3)
    cand = REPAIR[rname](inst, rem, rng)
    assert cand is not None, f"{dname} → {rname} 修复失败"
    assert validate_structure(cand, inst) == []
    res, _ = evaluate(inst, cand)
    assert res.feasible, res.all_violations()[:3]


def test_移除锚点客户会连带移除整段架次():
    """铁律：删掉某架次的起飞/降落点，整段 sortie 及其客户必须一起进待插入集。

    Set1 的初始解没有架次，这里改用有架次的 Set4。
    """
    from src.generator import build_instance as _bi
    from src.initial import build_initial as _bi_sol

    inst4 = _bi(SETS[3], CFG)
    base_sol, _, _ = _bi_sol(inst4)
    route = next(r for r in base_sol.routes if r.sorties)
    s = route.sorties[0]
    anchor = route.nodes[s.launch]
    new_sol, dropped = remove_customers(base_sol, {anchor})
    assert set(s.customers) <= set(dropped)


def test_主循环能降成本(inst):
    r = solve(inst, seed=1, iters=80)
    assert r.best_cost < r.initial_cost
    assert r.saving > 0.10
    res, _ = evaluate(inst, r.best)
    assert res.feasible, res.all_violations()[:3]


def test_权重不会掉到零(inst):
    r = solve(inst, seed=2, iters=40)
    assert all(v >= WEIGHT_FLOOR - 1e-9 for v in r.weights_destroy.values())
    assert all(v >= WEIGHT_FLOOR - 1e-9 for v in r.weights_repair.values())


def test_结果可重复(inst):
    a = solve(inst, seed=3, iters=40)
    b = solve(inst, seed=3, iters=40)
    assert a.best_cost == pytest.approx(b.best_cost)


def test_退化开关_关掉禁飞区后同一解的成本不升(inst):
    """论文的 Without-detour 对照：关掉 NFZ（几何绕行 + 节点禁飞一起关）后成本不升。

    这里**不比较两次独立求解**：ALNS 是启发式，60 轮的随机轨迹不同，
    「去掉约束后搜出来的解反而略差」是正常的搜索噪声（曾出现 103.60 vs 103.58）。
    能严格断言的是：把同一份解放到放宽后的算例上重新评估，成本不会更高、且仍然可行。
    """
    from src.alns import disable_nfz

    off = disable_nfz(inst)
    assert off.nfz == ()
    assert inst.nfz != ()            # 原算例不受影响

    on_res = solve(inst, seed=1, iters=60)
    res_off, cost_off = evaluate(off, on_res.best)
    assert res_off.feasible, res_off.all_violations()[:3]
    assert cost_off <= on_res.best_cost + 1e-6


def test_truck_only_没有架次且成本更高(inst):
    """Table 3 的 Truck-only 对照组：不派无人机，成本必然不低于 ALNS。"""
    from src.alns import disable_drones

    off = disable_drones(inst)
    assert off.allow_drones is False
    assert inst.allow_drones is True          # 原算例不受影响

    with_d = solve(inst, seed=1, iters=60)
    truck_only = solve(inst, seed=1, iters=60, use_drone=False)
    assert all(not r.sorties for r in truck_only.best.routes)
    assert truck_only.best_cost >= with_d.best_cost - 1e-6
    res, _ = evaluate(inst, truck_only.best)
    assert res.feasible, res.all_violations()[:3]
