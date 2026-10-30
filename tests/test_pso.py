"""PSO 对照的测试：解码是确定函数、结果可行、可重复。"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluator import evaluate
from src.generator import build_instance
from src.io import load_config, load_sets
from src.pso import PENALTY, decode, solve_pso

CFG = load_config()
SETS = load_sets()


@pytest.fixture(scope="module")
def inst():
    return build_instance(SETS[0], CFG)          # Set1：10 客户


def test_解码是确定函数(inst):
    x = np.linspace(0.05, 0.95, inst.n_customers)
    a, ca = decode(inst, x)
    b, cb = decode(inst, x)
    assert a is not None and b is not None
    assert ca == pytest.approx(cb)
    assert [r.nodes for r in a.routes] == [r.nodes for r in b.routes]


def test_pso返回可行解(inst):
    r = solve_pso(inst, seed=0, iters=60)
    assert r.best is not None
    res, cost = evaluate(inst, r.best)
    assert res.feasible, res.all_violations()[:3]
    assert cost == pytest.approx(r.best_cost)
    assert r.failed_decodes < r.decodes       # 不应该全是失败解码


def test_pso可重复(inst):
    a = solve_pso(inst, seed=1, iters=40)
    b = solve_pso(inst, seed=1, iters=40)
    assert a.best_cost == pytest.approx(b.best_cost)


def test_解码失败时返回罚值(inst):
    """把某个客户的时间窗压到不可能（l < 到仓库的最短时间）时，解码应判失败。"""
    from dataclasses import replace

    bad = replace(inst, tw=(inst.tw[0].__class__(e=0.0, l=0.0),) + inst.tw[1:])
    sol, cost = decode(bad, np.linspace(0.05, 0.95, inst.n_customers))
    assert sol is None or cost >= PENALTY
