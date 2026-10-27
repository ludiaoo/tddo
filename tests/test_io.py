"""原始数据接入的测试：列名清洗、编号校验、矩阵完备、坐标投影。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.instance import to_local_xy
from src.io import load_base_problem, load_config, load_sets, load_problems_index

SET1 = "20170608T121651164057"  # 10 客户 seattle


@pytest.fixture(scope="module")
def bp():
    return load_base_problem(SET1, "seattle")


def test_节点编号从0连续(bp):
    assert [nd.idx for nd in bp.nodes] == list(range(bp.n_nodes))
    assert bp.nodes[0].kind == "depot"
    assert all(nd.kind == "customer" for nd in bp.nodes[1:])


def test_客户数与Set配置一致():
    for s in load_sets():
        b = load_base_problem(s["problems"][0], s["city"])
        assert b.n_customers == s["n_customers"], f"Set{s['id']} 客户数不符"


def test_卡车矩阵完备且自环为0(bp):
    n = bp.n_nodes
    assert bp.truck_dist_m.shape == (n, n)
    assert bp.truck_time_s.shape == (n, n)
    assert not np.isnan(bp.truck_dist_m).any()
    assert not np.isnan(bp.truck_time_s).any()
    assert np.allclose(np.diag(bp.truck_dist_m), 0.0)
    assert np.allclose(np.diag(bp.truck_time_s), 0.0)


def test_卡车路网距离不短于直线距离(bp):
    """路网绕行，绝大多数 OD 对应满足 D_truck >= D_drone；允许 1% 的噪声余量。"""
    dd = bp.drone_dist_m()
    mask = (dd > 500.0) & (bp.truck_dist_m > 0)
    ratio = bp.truck_dist_m[mask] / dd[mask]
    assert np.median(ratio) > 1.0
    assert (ratio > 0.99).mean() > 0.99


def test_depot为投影原点(bp):
    depot = bp.nodes[0]
    assert depot.x == pytest.approx(0.0, abs=1e-9)
    assert depot.y == pytest.approx(0.0, abs=1e-9)


def test_局部投影的尺度合理():
    """沿纬度走 0.01°（约 1.11 km）应得到约 1110 m 的北向位移。"""
    x, y = to_local_xy(42.90 + 0.01, -78.85, 42.90, -78.85)
    assert x == pytest.approx(0.0, abs=1e-6)
    assert y == pytest.approx(1110.0, rel=0.01)


def test_投影各向同性():
    """东向与北向的 1e-4 度位移在 43°N 处分别约为 8.1 m 与 11.1 m。"""
    x, _ = to_local_xy(43.0, 0.0001, 43.0, 0.0)
    _, y = to_local_xy(43.0001, 0.0, 43.0, 0.0)
    assert y / x == pytest.approx(1.0 / math.cos(math.radians(43.0)), rel=0.01)


def test_包裹重量不含负数(bp):
    """depot 原始值为 -1，必须被归零，否则需求汇总会被负数污染。"""
    assert all(w >= 0.0 for w in bp.parcel_kg)
    assert bp.parcel_kg[0] == 0.0
    assert sum(bp.parcel_kg) > 0.0


def test_问题索引含100个算例():
    idx = load_problems_index()
    assert len(idx) == 100
    assert set(idx["numCust"]) == {8, 10, 25, 50, 100}


def test_配置可读且参数换算是SI():
    from src.instance import KMH_TO_MS, LB_TO_KG, FleetParams

    cfg = load_config()
    p = FleetParams.from_config(cfg)
    assert p.drone_speed == pytest.approx(float(cfg["fleet"]["drone_speed_kmh"]) * KMH_TO_MS)
    assert p.drone_capacity == pytest.approx(float(cfg["fleet"]["drone_payload_lb"]) * LB_TO_KG)
    assert p.truck_capacity == pytest.approx(136.08, abs=0.01)  # 300 lb
    assert p.drone_capacity == pytest.approx(4.536, abs=0.001)  # 10 lb
