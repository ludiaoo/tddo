"""距离/航段与能耗的单测。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.distance import drone_distance, drone_leg, service_ok, truck_leg
from src.energy import (
    driver_cost,
    drone_energy,
    sortie_energy,
    sortie_energy_ok,
    truck_cost,
)
from src.instance import BaseProblem, FleetParams, NFZ, Node


def make_base(coords: list[tuple[float, float]]) -> BaseProblem:
    """构造一个玩具算例：节点按给定坐标，卡车矩阵用欧氏距离代替（仅测试用）。"""
    nodes = tuple(
        Node(idx=i, kind="depot" if i == 0 else "customer", lat=0.0, lon=0.0, x=float(x), y=float(y))
        for i, (x, y) in enumerate(coords)
    )
    n = len(coords)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = math.dist(coords[i], coords[j])
    return BaseProblem(
        name="toy", city="toy", nodes=nodes,
        truck_dist_m=d, truck_time_s=d / 10.0,
        parcel_kg=tuple([0.0] * n),
    )


@pytest.fixture()
def params() -> FleetParams:
    return FleetParams(
        drone_speed=18.0, drone_capacity=4.536, drone_battery=0.78, eta_soc=0.8,
        truck_capacity=136.08, launch_time=60.0, recovery_time=30.0, service_time=60.0,
        e_drone=5e-4, b=0.05, e_truck=0.35,
        c_drone=0.15, c_truck=1.3, c_customer=0.0063,
    )


def test_卡车距离查表而非欧氏():
    base = make_base([(0, 0), (3000, 4000)])
    d, t = truck_leg(base, 0, 1)
    assert d == pytest.approx(5000.0)
    assert t == pytest.approx(500.0)


def test_无人机直线距离():
    base = make_base([(0, 0), (3000, 4000)])
    assert drone_distance(base, 0, 1) == pytest.approx(5000.0)


def test_节点服务判定是空间与时间的合取():
    base = make_base([(0, 0), (100, 0)])
    zone = NFZ(cx=100.0, cy=0.0, r=500.0, t_start=200.0, t_end=400.0, label="h1")
    node = (base.nodes[1].x, base.nodes[1].y)

    assert service_ok(node, 100.0, [zone]) == (True, ())        # 在区内但不在禁飞期
    ok, blocked = service_ok(node, 300.0, [zone])               # 区内 + 禁飞期
    assert not ok and blocked == ("h1",)
    assert service_ok((5000.0, 0.0), 300.0, [zone]) == (True, ())  # 在禁飞期但不在区内


def test_无禁飞区时航段距离等于直线():
    base = make_base([(0, 0), (1000, 0)])
    leg = drone_leg(base, [], 0, 1, 0.0, 60.0)
    assert leg.feasible
    assert leg.distance == pytest.approx(1000.0)
    assert leg.detour == pytest.approx(0.0)


def test_禁飞期外穿过圆的航段不绕():
    """时间上不重叠（禁飞 200–400，飞行 0–60）→ 即使几何穿圆也不绕。"""
    base = make_base([(-2000, 0), (2000, 0)])
    zone = NFZ(cx=0.0, cy=0.0, r=500.0, t_start=200.0, t_end=400.0, label="h1")
    leg = drone_leg(base, [zone], 0, 1, 0.0, 60.0)
    assert leg.feasible
    assert leg.detour == pytest.approx(0.0)
    assert leg.distance == pytest.approx(4000.0)


def test_禁飞期内穿过圆的航段要绕():
    base = make_base([(-2000, 0), (2000, 0)])
    zone = NFZ(cx=0.0, cy=0.0, r=500.0, t_start=0.0, t_end=1000.0, label="h1")
    leg = drone_leg(base, [zone], 0, 1, 10.0, 60.0)
    assert leg.feasible
    assert leg.detour > 0.0
    assert leg.distance == pytest.approx(4000.0 + leg.detour)
    # 手算：r=500，d=2000 → 切线 √(2000²−500²)=1936.49，弧角 π−2·arccos(0.25)=1.8229 rad
    expected = 2 * math.sqrt(2000**2 - 500**2) + 500 * (math.pi - 2 * math.acos(0.25))
    assert leg.distance == pytest.approx(expected, rel=1e-9)


def test_端点在禁飞区内时航段不可行():
    base = make_base([(-2000, 0), (100, 0)])
    zone = NFZ(cx=0.0, cy=0.0, r=500.0, t_start=0.0, t_end=1000.0, label="h1")
    leg = drone_leg(base, [zone], 0, 1, 10.0, 60.0)
    assert not leg.feasible
    assert leg.blocked_by == ("h1",)


def test_能耗随载荷线性增长(params):
    e0 = drone_energy(1000.0, 0.0, params)
    e2 = drone_energy(1000.0, 2.0, params)
    assert e0 == pytest.approx(0.05)                     # 1 km × b
    assert e2 == pytest.approx(1 * (5e-4 * 2 + 0.05))    # 1 km × (e·L + b)
    assert e2 > e0


def test_架次能耗逐段累加(params):
    legs = [(1000.0, 4.0), (2000.0, 1.0)]
    manual = drone_energy(1000, 4.0, params) + drone_energy(2000, 1.0, params)
    assert sortie_energy(legs, params) == pytest.approx(manual)


def test_单架次电量上限(params):
    cap = params.eta_soc * params.drone_battery
    assert sortie_energy_ok(cap - 1e-6, params)
    assert not sortie_energy_ok(cap + 0.01, params)


def test_卡车与司机成本(params):
    assert truck_cost(1000.0, params) == pytest.approx(1 * 0.35 * 1.3)
    assert driver_cost(3600.0, params) == pytest.approx(3600 * 0.0063)


def test_绕行会让架次能耗上升(params):
    base = make_base([(-2000, 0), (2000, 0)])
    zone = NFZ(cx=0.0, cy=0.0, r=500.0, t_start=0.0, t_end=1000.0, label="h1")
    plain = drone_leg(base, [], 0, 1, 10.0, 60.0)
    detoured = drone_leg(base, [zone], 0, 1, 10.0, 60.0)
    e_plain = drone_energy(plain.distance, 2.0, params)
    e_detour = drone_energy(detoured.distance, 2.0, params)
    assert e_detour > e_plain
