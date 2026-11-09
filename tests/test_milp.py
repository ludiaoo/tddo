"""MILP 的测试：与评估器逐分对账，并与 ALNS 的对照列互检。

MILP 是 M6 的「标准答案」，所以它必须满足两条：
1. **解能在评估器里复核**——目标值、里程、能耗、完工时刻逐项一致；
2. **不劣于 ALNS**——ALNS 的任何可行解都是 MILP 的可行解，所以 MILP 最优必然 ≤ ALNS。
   反过来若 ALNS 明显更优，说明两个模型对可行域的理解不一致（曾抓出架次嵌套、同点起降两个 bug）。
"""

from __future__ import annotations

import pytest

from src.alns import disable_nfz, solve
from src.evaluator import evaluate
from src.generator import build_instance
from src.io import load_config, load_sets
from src.milp import solve_milp

pytestmark = pytest.mark.slow          # 需要 Gurobi，默认不跑（见 pytest.ini）

CFG = load_config()
SETS = load_sets()


@pytest.fixture(scope="module")
def inst():
    return disable_nfz(build_instance(SETS[0], CFG))       # Set1：10 客户


@pytest.fixture(scope="module")
def milp(inst):
    return solve_milp(inst, time_limit=120.0)


def test_milp解能被评估器逐项复核(inst, milp):
    assert milp.solution is not None, milp.status
    res, cost = evaluate(inst, milp.solution)
    assert res.feasible, res.all_violations()[:3]
    assert cost == pytest.approx(milp.cost, rel=1e-6)

    b = milp.breakdown
    d_cost = sum(r.drone_energy for r in res.routes)
    d_dist = sum(r.truck_distance for r in res.routes)
    d_time = sum(r.finish_time for r in res.routes)
    assert d_cost == pytest.approx(b["drone_energy_kwh"], rel=1e-6)
    assert d_dist == pytest.approx(b["truck_distance_m"], rel=1e-6)
    assert d_time == pytest.approx(b["finish_time_s"], rel=1e-6)


def test_alns不优于milp最优(inst, milp):
    r = solve(inst, seed=0, iters=300)
    assert r.best_cost >= milp.cost - 1e-6, "ALNS 比 MILP 最优还好 → 两个模型的可行域不一致"


def test_全卡车模式与truck_only对照一致(inst):
    """MILP 关掉无人机后应与 ALNS 的 Truck-only 列重合（都是同一问题的最优/近优）。"""
    m = solve_milp(inst, time_limit=120.0, allow_drone=False)
    r = solve(inst, seed=1, iters=300, use_drone=False)
    assert m.solution is not None
    assert all(not rt.sorties for rt in m.solution.routes)
    assert r.best_cost >= m.cost - 1e-6
    assert r.best_cost <= m.cost * 1.02, f"Truck-only 的 ALNS 比最优差 {(r.best_cost/m.cost-1):.1%}"


def test_要求禁飞区时必须报错(inst):
    with pytest.raises(ValueError):
        solve_milp(build_instance(SETS[0], CFG), time_limit=10.0)


def test_钉解检验能识别换电解(inst, milp):
    """M9.16 的核心工具：把外部解钉进 MILP 判可行性。

    两条断言：
    1. MILP 自己的解钉回去必须可行（否则 `_pin` 的映射有 bug）；
    2. 换电解钉进**原模型**必须不可行——它只违反 `lr_once`，这正是「论文可行域少了那个点」的证据。
    """
    from copy import deepcopy

    from src.alns import disable_nfz as _off

    ok = solve_milp(inst, time_limit=60.0, fix=milp.solution)
    assert ok.status == "optimal", "MILP 自己的解钉回去竟然不可行 → _pin 映射有 bug"

    cfg_hot = deepcopy(CFG)
    cfg_hot["fleet"]["allow_hotswap"] = True
    hot = _off(build_instance(SETS[0], cfg_hot))
    best = min((solve(hot, seed=s, iters=800) for s in range(3)),
               key=lambda r: float(r.best_cost)).best
    bad = solve_milp(hot, time_limit=20.0, allow_hotswap=False, fix=best)
    assert bad.solution is None, "换电解在原模型里居然可行 → 说明它没用到换机，断言无意义"
