"""锚点平移（M9.13）的守门测试。

守住三条底线：
1. 平移后的解**仍可行**；
2. **服务的客户集合不变**（不能悄悄丢客户——这是本项目踩过的老坑）；
3. 成本**不上升**（只接受严格改进，累积后必然 ≤ 原成本）。
"""

from __future__ import annotations

from src.evaluator import evaluate
from src.generator import build_instance
from src.initial import build_initial
from src.io import load_config, load_sets
from src.local_search import polish_anchors


def _inst(m: int = 1):
    cfg = load_config()
    cfg["fleet"]["drones_per_truck"] = m
    by_id = {s["id"]: s for s in load_sets()}
    return build_instance(by_id[5], cfg)


def test_polish_keeps_feasible_and_cost_non_increasing() -> None:
    for m in (1, 2):
        inst = _inst(m)
        sol, cost0, _ = build_initial(inst)
        out, cost1 = polish_anchors(inst, sol)
        res, _ = evaluate(inst, out)
        assert res.feasible, f"m={m} 平移后不可行：{res.all_violations()[:3]}"
        assert cost1 <= cost0 + 1e-9, f"m={m} 平移后成本上升 {cost0} → {cost1}"
        assert out.served_customers() == sol.served_customers(), "平移丢了客户"


def test_polish_has_diminishing_returns() -> None:
    """`max_passes=3` 不一定跑到收敛，但第二次调用的额外收益必须很小（<0.05%）。"""
    inst = _inst(2)
    sol, _, _ = build_initial(inst)
    once, c1 = polish_anchors(inst, sol)
    twice, c2 = polish_anchors(inst, once)
    assert c2 <= c1 + 1e-9, "重复平移不应让成本上升"
    assert (c1 - c2) / c1 < 5e-4, f"第二次平移仍降 {(c1 - c2) / c1:.4%}，说明一次 pass 远未收敛"
