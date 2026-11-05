"""周期有界干预（M9.6）的守门测试。

三条底线：
1. 校验器必须把越界值裁回来（否则「有界监督」就是空话）；
2. random 策略必须可复现（同一 seed 两次调用完全一致）；
3. 不开干预时必须与历史结果逐位一致——这条由 `scripts/golden_check` 覆盖，
   这里只做「开了干预确实会改变轨迹」的冒烟确认。
"""

from __future__ import annotations

import pytest

from src.intervention import BIAS_LIM, Q_RANGE, TEMP_RANGE, make_policy, validate


def test_validator_clips_and_ignores_unknown_fields() -> None:
    act = validate({
        "q": 9.0,                                  # 越上界
        "temp_mult": -5.0,                         # 越下界
        "bias_d": {"random": 3.0, "不存在的算子": 1.0},
        "bias_r": {"equal": -9.0},
        "solution": "我不该出现在这里",            # 试图改解的字段
    })
    assert act is not None
    assert act["q"] == pytest.approx(Q_RANGE[1])
    assert act["temp_mult"] == pytest.approx(TEMP_RANGE[0])
    assert act["bias_d"] == {"random": pytest.approx(BIAS_LIM)}
    assert act["bias_r"]["equal"] == pytest.approx(-BIAS_LIM)
    assert "solution" not in act


def test_validate_empty_returns_none() -> None:
    assert validate(None) is None
    assert validate({}) is None


def test_random_policy_is_reproducible_and_bounded() -> None:
    state = {"it": 0, "progress": 0.0}
    a = [make_policy("random", 7)(state) for _ in range(5)]
    b = [make_policy("random", 7)(state) for _ in range(5)]
    assert a == b, "同 seed 的随机策略必须逐次一致"

    for act in a:
        assert Q_RANGE[0] <= act["q"] <= Q_RANGE[1]
        assert TEMP_RANGE[0] <= act["temp_mult"] <= TEMP_RANGE[1]
        assert all(abs(v) <= BIAS_LIM for v in act["bias_d"].values())


def test_rule_policy_switches_by_progress() -> None:
    pol = make_policy("rule")
    early, mid, late = pol({"progress": 0.1}), pol({"progress": 0.5}), pol({"progress": 0.9})
    assert early["q"] < mid["q"] < late["q"]
    assert early["temp_mult"] < late["temp_mult"]


def test_unknown_policy_raises() -> None:
    with pytest.raises(ValueError):
        make_policy("gpt")


def test_intervention_changes_trajectory_but_stays_feasible() -> None:
    """开随机干预后成本会变（轨迹确实被扰动），但解仍然可行。"""
    from src.generator import build_instance
    from src.io import load_config, load_sets
    from src.alns import solve
    from src.evaluator import evaluate

    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}
    inst = build_instance(by_id[5], cfg)

    base = solve(inst, seed=0, iters=200)
    per = solve(inst, seed=0, iters=200,
                interv_policy=make_policy("random", 0), interv_every=50)
    assert per.best_cost > 0 and base.best_cost > 0
    assert per.best_cost != base.best_cost, "干预应当改变搜索轨迹"
    res, _ = evaluate(inst, per.best)
    assert res.feasible, "干预不得产出不可行解"
