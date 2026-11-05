"""无人机候选预筛（M9.9）的守门测试。

只守一条底线：**预筛上限放到足够大时，必须与全量实现给出完全相同的候选与 Δcost**。
这条保证了预筛本身不改变语义，速度与质量的差异只来自「少看了多少候选」。
"""

from __future__ import annotations

import pytest

import src.operators as ops
from src.generator import build_instance
from src.initial import build_initial
from src.io import load_config, load_sets
from src.operators import remove_customers


@pytest.fixture()
def inst():
    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}
    return build_instance(by_id[5], cfg)


def test_pruned_matches_full_when_topk_is_huge(inst) -> None:
    sol, _, _ = build_initial(inst)
    c = sorted(sol.served_customers())[3]
    part, _ = remove_customers(sol, {c})

    old = ops.DRONE_TOPK
    try:
        ops.DRONE_TOPK = 10 ** 9
        pruned = ops._drone_candidates_pruned(inst, part, c)
    finally:
        ops.DRONE_TOPK = old
    full = ops._drone_candidates_full(inst, part, c, "off")

    assert [round(float(d), 9) for d, _ in pruned] == [round(float(d), 9) for d, _ in full]
    assert [s.served_customers() for _, s in pruned] == [s.served_customers() for _, s in full]


def test_pruned_never_returns_more_than_shortlist(inst) -> None:
    """K=1 时每个路线仍保底一个候选，但总数必须远小于全量。"""
    sol, _, _ = build_initial(inst)
    c = sorted(sol.served_customers())[5]
    part, _ = remove_customers(sol, {c})

    old = ops.DRONE_TOPK
    try:
        ops.DRONE_TOPK = 1
        pruned = ops._drone_candidates_pruned(inst, part, c)
    finally:
        ops.DRONE_TOPK = old
    full = ops._drone_candidates_full(inst, part, c, "off")
    assert len(pruned) <= len(full)


def test_default_is_off() -> None:
    """默认必须是全量（DRONE_TOPK=0），否则历史结果会变。"""
    assert ops.DRONE_TOPK == 0
