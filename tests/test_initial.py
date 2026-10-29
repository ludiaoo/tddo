"""初始解（两阶段松弛优先）的测试。"""

from __future__ import annotations

import pytest

from src.generator import build_instance
from src.initial import assign_sorties, build_initial, slack_first_routes
from src.io import load_config, load_sets
from src.solution import validate_structure

CFG = load_config()
SETS = load_sets()


@pytest.mark.parametrize("set_cfg", SETS, ids=lambda s: f"set{s['id']:02d}")
def test_全部算例产出可行初始解(set_cfg):
    inst = build_instance(set_cfg, CFG)
    sol, cost, notes = build_initial(inst)

    assert validate_structure(sol, inst) == []
    hard = [n for n in notes if not n.startswith("用 ")]
    assert not hard, f"初始解不可行：{hard}"
    assert cost > 0
    # 路线数不超过车队规模
    assert len(sol.routes) <= inst.n_trucks


def test_阶段一_松弛优先能装完所有客户():
    inst = build_instance(SETS[0], CFG)
    routes = slack_first_routes(inst, inst.n_trucks)
    served = [c for r in routes for c in r[1:-1]]
    assert sorted(served) == sorted(inst.customers)
    assert all(r[0] == 0 and r[-1] == 0 for r in routes)


def test_同锚点架次会合并():
    """同一对锚点上若已存在架次，新客户应并入而不是新建（评估器只允许一个架次占一个位置）。"""
    from src.solution import Route

    inst = build_instance(SETS[3], CFG)
    sol, _, _ = build_initial(inst)
    for r in sol.routes:
        launches = [s.launch for s in r.sorties]
        assert len(set(launches)) == len(launches)
        for s in r.sorties:
            assert r.nodes[s.launch] != 0 and r.nodes[s.recover] != 0


def test_构造可重复():
    inst_a = build_instance(SETS[2], CFG)
    inst_b = build_instance(SETS[2], CFG)
    sa, ca, _ = build_initial(inst_a)
    sb, cb, _ = build_initial(inst_b)
    assert ca == pytest.approx(cb)
    assert [r.nodes for r in sa.routes] == [r.nodes for r in sb.routes]
