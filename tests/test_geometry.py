"""几何模块的单测：用可手算的构型把切线 + 圆弧公式钉死。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.geometry import (
    detour_around_circle,
    point_in_circle,
    segment_intersects_circle,
    segment_min_distance,
    total_detour,
    zone_active_overlap,
)


def test_对称穿越圆的解析解():
    """p=(−2,0), q=(2,0), 圆(0,0,r=1)：两条切线 + 60° 圆弧。

    切线长 = √(4−1) = √3，圆弧角 = π − 2·arccos(1/2) = π/3
    → path = 2√3 + π/3 ≈ 4.5113，直线 = 4。
    """
    d = detour_around_circle((-2.0, 0.0), (2.0, 0.0), (0.0, 0.0), 1.0)
    assert d.feasible
    assert d.straight == pytest.approx(4.0)
    expected = 2 * math.sqrt(3) + math.pi / 3
    assert d.path == pytest.approx(expected, rel=1e-12)
    assert d.extra == pytest.approx(expected - 4.0, rel=1e-12)
    assert d.arc_angle == pytest.approx(math.pi / 3, rel=1e-12)


def test_同一构型旋转90度结果不变():
    d = detour_around_circle((0.0, -2.0), (0.0, 2.0), (0.0, 0.0), 1.0)
    assert d.path == pytest.approx(2 * math.sqrt(3) + math.pi / 3, rel=1e-12)


def test_不穿圆时绕行为零():
    """两端都在圆上方，线段离圆心 2 > r。"""
    d = detour_around_circle((-2.0, 2.0), (2.0, 2.0), (0.0, 0.0), 1.0)
    assert d.feasible
    assert d.extra == pytest.approx(0.0)
    assert d.path == pytest.approx(d.straight)


def test_相切退化情形连续():
    """线段恰好与圆相切（圆心到线段距离 = r）时，圆弧角应为 0，绕行距离为 0。"""
    d = detour_around_circle((-2.0, 1.0), (2.0, 1.0), (0.0, 0.0), 1.0)
    assert d.feasible
    assert d.arc_angle == pytest.approx(0.0, abs=1e-9)
    assert d.extra == pytest.approx(0.0, abs=1e-9)


def test_端点在圆内判为不可行():
    """端点在内时切线不存在 —— 必须返回 feasible=False，而不是硬算一个距离。"""
    d = detour_around_circle((0.5, 0.0), (5.0, 0.0), (0.0, 0.0), 1.0)
    assert not d.feasible
    assert math.isinf(d.extra)


def test_绕行路径不侵入圆内():
    """沿 切线 → 圆弧 → 切线 采样，所有点到圆心的距离都应 ≥ r。"""
    p, q, c, r = (-2.0, 0.3), (2.5, -0.2), (0.0, 0.0), 1.0
    d = detour_around_circle(p, q, c, r)
    assert d.feasible and d.extra > 0

    pts = [p, d.tangent_a]
    # 圆弧：从 tangent_a 转到 tangent_b
    a0 = math.atan2(d.tangent_a[1] - c[1], d.tangent_a[0] - c[0])
    a1 = math.atan2(d.tangent_b[1] - c[1], d.tangent_b[0] - c[0])
    delta = a1 - a0
    while delta <= -math.pi:
        delta += 2 * math.pi
    while delta > math.pi:
        delta -= 2 * math.pi
    for t in np.linspace(0.0, 1.0, 2000):
        ang = a0 + delta * t
        pts.append((c[0] + r * math.cos(ang), c[1] + r * math.sin(ang)))
    pts += [d.tangent_b, q]

    for x, y in pts:
        assert math.hypot(x - c[0], y - c[1]) >= r - 1e-6

    # 折线长度应与解析 path 一致
    length = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    # 折线内接于圆弧，必然略短于解析弧长，采样越密越接近
    assert length == pytest.approx(d.path, rel=1e-6)
    assert length <= d.path + 1e-9


def test_半径越大绕行越远():
    small = detour_around_circle((-3.0, 0.0), (3.0, 0.0), (0.0, 0.0), 0.5)
    big = detour_around_circle((-3.0, 0.0), (3.0, 0.0), (0.0, 0.0), 1.5)
    assert big.extra > small.extra >= 0.0


def test_绕行恒小于半圆周长加两切线():
    """绕行的圆弧角必然 < π，路径长度不可能超过「两条切线 + 半圆弧」。"""
    p, q, c, r = (-1.5, -0.2), (2.0, 0.1), (0.0, 0.0), 1.0
    d = detour_around_circle(p, q, c, r)
    dp, dq = math.dist(p, c), math.dist(q, c)
    ub = math.sqrt(dp**2 - r**2) + math.sqrt(dq**2 - r**2) + math.pi * r
    assert d.path <= ub + 1e-9
    assert d.arc_angle < math.pi


def test_点在圆内与线段相交判定():
    assert point_in_circle((0.5, 0.0), (0.0, 0.0), 1.0)
    assert not point_in_circle((1.5, 0.0), (0.0, 0.0), 1.0)
    assert segment_intersects_circle((-2, 0), (2, 0), (0, 0), 1.0)
    assert not segment_intersects_circle((-2, 2), (2, 2), (0, 0), 1.0)
    # 圆心到线段的最短距离应按「线段」算，不是按直线算
    assert segment_min_distance((1.0, 1.0), (3.0, 1.0), (0.0, 0.0)) == pytest.approx(math.sqrt(2))


@pytest.mark.parametrize(
    "case,t_from,t_to,expect",
    [
        ("1 两端都在禁飞前", 0, 100, False),
        ("2 起点在前、终点在禁飞中", 0, 300, True),
        ("3 起点在前、终点在禁飞后", 0, 900, True),
        ("4 起点在禁飞中、终点在后", 250, 900, True),
        ("5 全程在禁飞中", 250, 350, True),
        ("6 两端都在禁飞后", 800, 900, False),
    ],
)
def test_六情形时间判据(case, t_from, t_to, expect):
    """禁飞窗口 [t_start, t_end] = [200, 400]（对应论文 Fig.4 的六种相对位置）。"""
    assert zone_active_overlap(t_from, t_to, 200.0, 400.0) is expect, case


def test_多圆叠加求和():
    # 第二个圆放在航段正上方 0.4 m 处（半径 0.5）→ 确实被穿过
    zones = [((0.0, 0.0), 1.0, 0.0, 1000.0), ((2.0, 0.4), 0.5, 0.0, 1000.0)]
    total, ok = total_detour((-4.0, 0.0), (4.0, 0.0), zones)
    assert ok
    single, _ = total_detour((-4.0, 0.0), (4.0, 0.0), zones[:1])
    assert total > single


def test_多圆中任一不可行则整体不可行():
    zones = [((0.0, 0.0), 1.0, 0.0, 1000.0)]
    # 终点落在圆内
    total, ok = total_detour((-4.0, 0.0), (0.2, 0.0), zones)
    assert not ok and math.isinf(total)
