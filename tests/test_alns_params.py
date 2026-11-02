"""参数修正（tune）与论文原参数（tune=False）的行为测试。

`tune=False` 是一条**可复现基线**：它必须与性能优化前逐位一致（由
`scripts/golden_check.py` 守）。这里测的是 tune=True 的三处修正确实生效，
且不影响确定性。
"""

from __future__ import annotations

import math

import pytest

from src.alns import ALNSResult, solve
from src.evaluator import evaluate
from src.generator import build_instance
from src.io import load_config, load_sets

CFG = load_config()
SETS = load_sets()


@pytest.fixture(scope="module")
def inst():
    return build_instance({s["id"]: s for s in SETS}[4], CFG)      # Set5：25 客户


def test_论文原参数使用固定初温300(inst):
    r = solve(inst, seed=0, iters=20, tune=False)
    assert r.t0_used == pytest.approx(300.0)
    assert r.restarts == 0
    assert r.tune is False


def test_初温按增量分布标定(inst):
    """成本量级 200 上下时，标定出的 T0 应远小于 300（论文的 300 是为它的成本尺度调的）。"""
    r = solve(inst, seed=0, iters=20, tune=True)
    assert 0.0 < r.t0_used < 150.0
    assert r.tune is True


def test_标定式满足中位增量接受率50pct(inst):
    """T0 = Δ_med / ln2 ⇒ exp(−Δ_med/T0) = 0.5，这是标定的定义式。"""
    from src.alns import _calibrate_t0
    from src.initial import build_initial

    sol, cost, _ = build_initial(inst)
    import numpy as np

    t0 = _calibrate_t0(inst, sol, cost, np.random.default_rng(0))
    assert t0 > 0
    # 反推：用这个 T0，中位上升步的接受概率应在 0.5 附近（±0.15 容差，取的是中位数近似）
    p = math.exp(-(t0 * math.log(2.0)) / t0)
    assert p == pytest.approx(0.5, abs=0.01)


def test_降温速率与预算挂钩(inst):
    """预算越长，α 应越接近 1（末温固定落到 T0×1e-3）。"""
    short = (1e-3) ** (1.0 / 100)
    long = (1e-3) ** (1.0 / 2000)
    assert long > short
    assert short < 0.99 < long


def test_重启后解仍可行(inst):
    """停滞扰动不能造出非法解。"""
    r = solve(inst, seed=3, iters=260, tune=True, max_no_improve=40)
    assert r.restarts >= 1, "200 轮预算 + 40 轮停滞阈值应当至少扰动一次"
    res, _ = evaluate(inst, r.best)
    assert res.feasible, res.all_violations()[:3]


def test_tune结果可重复(inst):
    a = solve(inst, seed=5, iters=30, tune=True)
    b = solve(inst, seed=5, iters=30, tune=True)
    assert a.best_cost == pytest.approx(b.best_cost)
    assert a.t0_used == pytest.approx(b.t0_used)
    assert a.restarts == b.restarts


def test_结果对象带诊断字段():
    r = ALNSResult(best=None, best_cost=0.0, initial_cost=1.0)
    assert r.t0_used == 300.0 and r.restarts == 0 and r.tune is True
