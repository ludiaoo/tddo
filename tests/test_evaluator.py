"""评估器测试：用手算得清的玩具算例，把时刻、载荷、能耗、禁飞四类检查钉死。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluator import V_CAP_DRONE, V_NFZ, V_SOC, V_TW_DRONE, V_TW_TRUCK, evaluate
from src.instance import BaseProblem, FleetParams, Instance, NFZ, Node, TimeWindow
from src.solution import Route, Solution, Sortie, validate_structure


def params(**kw) -> FleetParams:
    base = dict(
        drone_speed=10.0,        # 36 km/h，方便手算
        drone_capacity=4.536,
        drone_battery=0.78,
        eta_soc=0.8,
        truck_capacity=136.08,
        launch_time=0.0,
        recovery_time=0.0,
        service_time=0.0,
        e_drone=5e-4,
        b=0.05,
        e_truck=0.35,
        c_drone=0.15,
        c_truck=1.3,
        c_customer=0.0063,
    )
    base.update(kw)
    return FleetParams(**base)


def make_inst(coords, f, g, tw=None, nfz=(), truck_speed=10.0, **pkw) -> Instance:
    n = len(coords)
    nodes = tuple(
        Node(idx=i, kind="depot" if i == 0 else "customer", lat=0, lon=0, x=float(x), y=float(y))
        for i, (x, y) in enumerate(coords)
    )
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d[i, j] = math.dist(coords[i], coords[j])
    base = BaseProblem(
        name="toy", city="toy", nodes=nodes,
        truck_dist_m=d, truck_time_s=d / truck_speed,
        parcel_kg=tuple(float(v) for v in f),
    )
    if tw is None:
        tw = [(0.0, 100000.0)] * (n - 1)
    return Instance(
        set_id=0, base=base,
        f=np.array([float(v) for v in f]),
        g=np.array([float(v) for v in g]),
        tw=tuple(TimeWindow(e=e, l=l) for e, l in tw),
        nfz=tuple(nfz), n_trucks=1, params=params(**pkw), seed=0,
    )


# 三点一线 + 一个偏移客户：1(1000,0) 2(2000,0) 3(1500,1000)
COORDS = [(0, 0), (1000, 0), (2000, 0), (1500, 1000)]


def test_纯卡车路线_成本可手算():
    inst = make_inst([(0, 0), (1000, 0), (2000, 0)], f=[0, 10, 10], g=[0, 0, 0])
    sol = Solution([Route(truck=0, nodes=[0, 1, 2, 0], sorties=[])])
    res, cost = evaluate(inst, sol)

    assert res.feasible, res.all_violations()
    r = res.routes[0]
    assert r.truck_distance == pytest.approx(4000.0)       # 1000 + 1000 + 2000
    assert r.finish_time == pytest.approx(400.0)           # 10 m/s
    assert r.drone_energy == 0.0

    expect = 4.0 * 0.35 * 1.3 + 400.0 * 0.0063
    assert cost == pytest.approx(expect)


def test_单架次_无人机载荷与能耗():
    """卡车 0→1→2→0，无人机在 1 起飞、服务 3、在 2 回收。"""
    inst = make_inst(COORDS, f=[0, 10, 10, 1], g=[0, 0, 0, 1])
    sol = Solution([Route(truck=0, nodes=[0, 1, 2, 0],
                          sorties=[Sortie(launch=1, recover=2, customers=(3,))])])
    res, cost = evaluate(inst, sol)
    assert res.feasible, res.all_violations()

    r = res.routes[0]
    d13 = math.hypot(500, 1000)
    # 无人机：1→3 带 1kg 派件，3→2 带 1kg 取件
    e1 = d13 / 1000 * (5e-4 * 1 + 0.05)
    e2 = d13 / 1000 * (5e-4 * 1 + 0.05)
    assert r.drone_energy == pytest.approx(e1 + e2)

    # 卡车在 2 等无人机：无人机 100 + 2*d13/10 到，卡车 200 到
    t_drone = 100 + 2 * d13 / 10
    assert t_drone > 200
    assert r.waiting_time == pytest.approx(t_drone - 200)
    assert r.finish_time == pytest.approx(t_drone + 200)


def test_载荷守恒_卡车最终清零():
    inst = make_inst(COORDS, f=[0, 10, 10, 1], g=[0, 2, 3, 1])
    sol = Solution([Route(truck=0, nodes=[0, 1, 2, 0],
                          sorties=[Sortie(launch=1, recover=2, customers=(3,))])])
    res, _ = evaluate(inst, sol)
    assert res.feasible, res.all_violations()


def test_卡车超载报违约():
    inst = make_inst(COORDS, f=[0, 100, 60, 0], g=[0, 0, 0, 0], truck_capacity=100.0)
    sol = Solution([Route(truck=0, nodes=[0, 1, 2, 0], sorties=[])])
    res, _ = evaluate(inst, sol)
    assert not res.feasible
    assert any("CAP_TRUCK" in v for v in res.all_violations())


def test_无人机超载报违约():
    inst = make_inst(COORDS, f=[0, 0, 0, 10], g=[0, 0, 0, 0], drone_capacity=4.0)
    sol = Solution([Route(truck=0, nodes=[0, 1, 2, 0],
                          sorties=[Sortie(launch=1, recover=2, customers=(3,))])])
    res, _ = evaluate(inst, sol)
    assert not res.feasible
    assert any(V_CAP_DRONE in v for v in res.all_violations())


def test_电量不足报违约():
    """把电池压到很小，长距离架次必然超 SOC。"""
    inst = make_inst(COORDS, f=[0, 0, 0, 1], g=[0, 0, 0, 0], drone_battery=0.05, eta_soc=0.8)
    sol = Solution([Route(truck=0, nodes=[0, 1, 2, 0],
                          sorties=[Sortie(launch=1, recover=2, customers=(3,))])])
    res, _ = evaluate(inst, sol)
    assert not res.feasible
    assert any(V_SOC in v for v in res.all_violations())


def test_卡车时间窗违约():
    inst = make_inst(COORDS, f=[0, 10, 10, 0], g=[0, 0, 0, 0],
                     tw=[(0, 2000), (0, 150), (0, 5000)])
    sol = Solution([Route(truck=0, nodes=[0, 1, 2, 0], sorties=[])])
    res, _ = evaluate(inst, sol)
    assert not res.feasible
    assert any(V_TW_TRUCK in v for v in res.all_violations())


def test_早到等待而非违约():
    """到达早于 e_j 时应原地等待，不计违约（标准 VRPTW 语义）。"""
    inst = make_inst([(0, 0), (1000, 0)], f=[0, 10], g=[0, 0], tw=[(500.0, 1000.0)])
    sol = Solution([Route(truck=0, nodes=[0, 1, 0], sorties=[])])
    res, _ = evaluate(inst, sol)
    assert res.feasible, res.all_violations()
    r = res.routes[0]
    assert r.waiting_time == pytest.approx(400.0)      # 100 到，等到 500
    assert r.finish_time == pytest.approx(600.0)       # 等到 500，再花 100 回仓库


def test_无人机时间窗违约():
    """客户 3 的窗口很窄，无人机到达时已过期。"""
    inst = make_inst(COORDS, f=[0, 10, 10, 1], g=[0, 0, 0, 0],
                     tw=[(0, 5000), (0, 5000), (0, 160)])
    sol = Solution([Route(truck=0, nodes=[0, 1, 2, 0],
                          sorties=[Sortie(launch=1, recover=2, customers=(3,))])])
    res, _ = evaluate(inst, sol)
    assert not res.feasible
    assert any(V_TW_DRONE in v for v in res.all_violations())


def test_禁飞区阻断起降():
    """客户 1 位于禁飞区内且起飞时刻落在禁飞期 → 不可起飞。"""
    z = NFZ(cx=1000.0, cy=0.0, r=300.0, t_start=0.0, t_end=10000.0, label="h1")
    inst = make_inst(COORDS, f=[0, 10, 10, 1], g=[0, 0, 0, 0], nfz=[z])
    sol = Solution([Route(truck=0, nodes=[0, 1, 2, 0],
                          sorties=[Sortie(launch=1, recover=2, customers=(3,))])])
    res, _ = evaluate(inst, sol)
    assert not res.feasible
    assert any(V_NFZ in v for v in res.all_violations())


def test_结构检查_客户重复与缺失():
    inst = make_inst(COORDS, f=[0, 10, 10, 1], g=[0, 0, 0, 0])
    # 客户 1 出现两次
    bad1 = Solution([Route(truck=0, nodes=[0, 1, 1, 0], sorties=[])])
    assert validate_structure(bad1, inst)
    # 客户 3 没被服务
    bad2 = Solution([Route(truck=0, nodes=[0, 1, 2, 0], sorties=[])])
    assert validate_structure(bad2, inst)
    # 客户同时被卡车和无人机服务
    bad3 = Solution([Route(truck=0, nodes=[0, 1, 2, 0],
                           sorties=[Sortie(launch=1, recover=2, customers=(2,))])])
    assert validate_structure(bad3, inst)
    # 正常
    ok = Solution([Route(truck=0, nodes=[0, 1, 2, 0],
                         sorties=[Sortie(launch=1, recover=2, customers=(3,))])])
    assert validate_structure(ok, inst) == []


def test_sortie位置非法直接报错():
    with pytest.raises(ValueError):
        Sortie(launch=2, recover=1, customers=(3,))
    with pytest.raises(ValueError):
        Sortie(launch=1, recover=2, customers=())
