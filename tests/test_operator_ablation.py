"""算子内部策略的守门测试（M9.22）。

守三件事：
1. 两个开关的默认值与历史行为完全一致（`NFZ_DISCOUNT=0.6`、`WORST_WINDOW=1.0`），
   否则 `golden_check` 会失败——默认口径不允许被"顺手改了"。
2. `WORST_WINDOW=1.0` 时 `destroy_worst` 严格取节省额最高的 k 个；放大候选窗后
   才在窗内随机抽，且抽出的集合必须落在窗内。
3. `NFZ_DISCOUNT` 只影响禁飞区 2r 内的客户，置 1.0 时退化为纯距离排序。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import src.operators as ops
from src.generator import build_instance
from src.initial import build_initial
from src.io import load_config, load_sets
from src.operators import _all_customers, _related_score, _saving_if_removed


@pytest.fixture(scope="module")
def inst():
    return build_instance(load_sets()[0], load_config())      # Set1：10 客户


@pytest.fixture(scope="module")
def base_sol(inst):
    sol, _, _ = build_initial(inst)
    return sol


def _strict_topk(inst, sol, k: int) -> set[int]:
    """按节省额降序取前 k 个——这是 `WORST_WINDOW=1.0` 的定义。"""
    custs = _all_customers(sol)
    sav = sorted(((c, _saving_if_removed(inst, sol, c)) for c in custs), key=lambda x: -x[1])
    return {c for c, _ in sav[:k]}


def test_默认值即历史口径() -> None:
    assert ops.NFZ_DISCOUNT == 0.6
    assert ops.WORST_WINDOW == 1.0


def test_worst_默认严格取前k(inst, base_sol) -> None:
    k = max(1, int(round(0.3 * len(_all_customers(base_sol)))))
    for seed in range(5):
        rem = ops.destroy_worst(inst, base_sol, np.random.default_rng(seed), 0.3)
        assert set(rem.removed) == _strict_topk(inst, base_sol, k), f"seed={seed} 未按严格前 k 取"


def test_worst_放大候选窗后只在窗内随机(inst, base_sol, monkeypatch) -> None:
    monkeypatch.setattr(ops, "WORST_WINDOW", 2.0)
    custs = _all_customers(base_sol)
    k = max(1, int(round(0.3 * len(custs))))
    sav = sorted(((c, _saving_if_removed(inst, base_sol, c)) for c in custs), key=lambda x: -x[1])
    window = {c for c, _ in sav[: min(len(sav), k * 2)]}
    strict = _strict_topk(inst, base_sol, k)

    seen = []
    for seed in range(20):
        rem = ops.destroy_worst(inst, base_sol, np.random.default_rng(seed), 0.3)
        picked = set(rem.removed)
        assert len(picked) == k
        assert picked <= window, "抽到了候选窗之外的客户"
        seen.append(picked)
    assert any(s != strict for s in seen), "候选窗放大后仍与严格前 k 完全相同，随机化未生效"


def test_related_只给禁飞区附近的客户打折() -> None:
    """合成两个客户：一个在禁飞区 2r 内，一个在远处。折扣只应作用在前者。"""
    nodes = {
        0: SimpleNamespace(x=0.0, y=0.0),      # 种子点
        1: SimpleNamespace(x=100.0, y=0.0),    # 远处
        2: SimpleNamespace(x=0.0, y=30.0),     # 禁飞区旁（距圆心 30 m，2r = 60 m）
    }
    fake = SimpleNamespace(base=SimpleNamespace(nodes=nodes),
                           nfz=[SimpleNamespace(cx=0.0, cy=0.0, r=30.0)])

    assert _related_score(fake, 0, 1) == pytest.approx(100.0)
    assert _related_score(fake, 0, 2) == pytest.approx(30.0 * 0.6)


def test_related_关掉折扣后退化为纯距离(monkeypatch) -> None:
    nodes = {0: SimpleNamespace(x=0.0, y=0.0), 2: SimpleNamespace(x=0.0, y=30.0)}
    fake = SimpleNamespace(base=SimpleNamespace(nodes=nodes),
                           nfz=[SimpleNamespace(cx=0.0, cy=0.0, r=30.0)])
    monkeypatch.setattr(ops, "NFZ_DISCOUNT", 1.0)
    assert _related_score(fake, 0, 2) == pytest.approx(30.0)


def test_没有禁飞区时折扣不生效() -> None:
    nodes = {0: SimpleNamespace(x=0.0, y=0.0), 2: SimpleNamespace(x=0.0, y=30.0)}
    fake = SimpleNamespace(base=SimpleNamespace(nodes=nodes), nfz=[])
    assert _related_score(fake, 0, 2) == pytest.approx(30.0)
