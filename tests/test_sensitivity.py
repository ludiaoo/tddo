"""灵敏度实验的变体构造测试（不求解，只验参数改动是否如预期）。"""

from __future__ import annotations

import pytest

from scripts.sensitivity import (
    scale_demand,
    with_drone,
    with_period,
    with_radius,
    with_truck_capacity,
)
from src.generator import build_instance
from src.io import load_config, load_sets

CFG = load_config()
SETS = load_sets()
LB = 0.45359237


@pytest.fixture(scope="module")
def inst():
    return build_instance({s["id"]: s for s in SETS}[5], CFG)      # Set5：25 客户


def test_需求缩放同时作用于派件与取件(inst):
    d = scale_demand(inst, 1.5)
    assert d.f[1:].sum() == pytest.approx(inst.f[1:].sum() * 1.5)
    assert d.g[1:].sum() == pytest.approx(inst.g[1:].sum() * 1.5)
    assert d.n_trucks >= inst.n_trucks          # 需求变大不该减车


def test_卡车容量换算成SI且重算车队(inst):
    c = with_truck_capacity(inst, 500)
    assert c.params.truck_capacity == pytest.approx(500 * LB)
    assert c.n_trucks >= 1
    # 容量放大后，容量下界不应超过时间窗所需车数太多
    assert c.n_trucks <= max(inst.n_trucks, 1) + 6


def test_无人机参数只改指定项(inst):
    d = with_drone(inst, payload_lb=15)
    assert d.params.drone_capacity == pytest.approx(15 * LB)
    assert d.params.drone_battery == inst.params.drone_battery
    assert d.params.drone_speed == inst.params.drone_speed

    s = with_drone(inst, speed_kmh=80)
    assert s.params.drone_speed == pytest.approx(80 / 3.6)


def test_半径按倍数缩放且不改变数量(inst):
    r = with_radius(inst, 2.0)
    assert len(r.nfz) == len(inst.nfz)
    for a, b in zip(inst.nfz, r.nfz):
        assert b.r == pytest.approx(a.r * 2.0)
        assert (b.cx, b.cy) == (a.cx, a.cy)


def test_禁飞时段三种模式(inst):
    assert with_period(inst, "None").nfz == ()
    work = with_period(inst, "Work-time")
    assert work.nfz == inst.nfz
    full = with_period(inst, "Full-time")
    assert all(z.t_start == 0.0 and z.t_end > 1e5 for z in full.nfz)
    with pytest.raises(ValueError):
        with_period(inst, "Sometimes")
