"""几何缓存与原实现的一致性测试。

这组测试是性能优化的**守门员**：缓存把「绕行距离」预计算了，一旦预计算的语义
与原 `distance.drone_leg` 有任何偏差，成本会静默地变化——而这种错误不会报错，
只会让 Table 3 的数字悄悄偏掉。所以这里逐点比对，不复用缓存的输出。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cache import GeometryCache
from src.distance import INF, drone_leg
from src.generator import build_instance
from src.io import load_config, load_sets

CFG = load_config()
SETS = load_sets()


@pytest.fixture(scope="module")
def inst():
    return build_instance({s["id"]: s for s in SETS}[4], CFG)      # Set5：25 客户，有 3 个禁飞区


def _sample_legs(inst, n=400, seed=0):
    rng = np.random.default_rng(seed)
    N = inst.base.n_nodes
    for _ in range(n):
        i, j = (int(x) for x in rng.integers(0, N, 2))
        if i == j:
            continue
        t0 = float(rng.uniform(0.0, 5.0)) * 3600.0
        t1 = t0 + float(rng.uniform(60.0, 900.0))
        yield i, j, t0, t1


def test_缓存航段与解析实现逐点一致(inst):
    g = GeometryCache.of(inst)
    bad = []
    for i, j, t0, t1 in _sample_legs(inst):
        a = drone_leg(inst.base, inst.nfz, i, j, t0, t1)
        b = g.leg(i, j, t0, t1)
        if a.feasible != b.feasible:
            bad.append((i, j, t0, "feasible", a.feasible, b.feasible))
        elif a.feasible and (abs(a.distance - b.distance) > 1e-6 or abs(a.detour - b.detour) > 1e-6):
            bad.append((i, j, t0, "distance", a.distance, b.distance))
        elif not a.feasible and (b.distance != INF or a.blocked_by != b.blocked_by):
            bad.append((i, j, t0, "blocked", a.blocked_by, b.blocked_by))
    assert not bad, f"{len(bad)} 处不一致，例如 {bad[:3]}"


def test_关闭绕行时缓存同样收紧(inst):
    g = GeometryCache.of(inst)
    for i, j, t0, t1 in _sample_legs(inst, n=200, seed=7):
        a = drone_leg(inst.base, inst.nfz, i, j, t0, t1, allow_detour=False)
        b = g.leg(i, j, t0, t1, allow_detour=False)
        assert a.feasible == b.feasible


def test_起降判定与解析实现一致(inst):
    """service_ok 只查「节点内含的圆」，必须与原「遍历全部圆」等价。"""
    from src.distance import service_ok
    from src.geometry import point_in_circle

    g = GeometryCache.of(inst)
    for node in range(inst.base.n_nodes):
        p = (inst.base.nodes[node].x, inst.base.nodes[node].y)
        for t in np.linspace(0.0, 5 * 3600, 40):
            a, ba = service_ok(p, float(t), inst.nfz)
            b, bb = g.service_ok(node, float(t))
            assert a == b and ba == bb, f"节点{node} t={t}"


def test_禁飞区一变缓存重建(inst):
    """`replace(nfz=...)` 必须让缓存失效，否则会拿旧几何算新算例。"""
    from dataclasses import replace

    g1 = GeometryCache.of(inst)
    assert GeometryCache.of(inst) is g1          # 同一实例复用同一个缓存

    off = replace(inst, nfz=())
    g2 = GeometryCache.of(off)
    assert g2 is not g1
    assert g2.zones == ()


def test_直线距离与缓存一致(inst):
    g = GeometryCache.of(inst)
    for i, j, *_ in _sample_legs(inst, n=100):
        a = inst.base.nodes[i]
        b = inst.base.nodes[j]
        assert g.straight(i, j) == pytest.approx(
            float(np.hypot(a.x - b.x, a.y - b.y)), rel=0, abs=1e-9
        )
