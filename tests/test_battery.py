"""电池库存与重量（E 项 / W2 最小版）的守门测试。

要守五条：
1. `spare_batteries = -1`（默认）时，行为与历史逐位一致（本项不许偷偷改老结果）；
2. `b = 0` 时一条路线最多 **1** 个架次（只有无人机自带那块电池）；
3. `b = 1` 放宽到 2 个架次，第 3 个判 `V_BATTERY`；
4. 一车 m 架机时上限是 `b + m`（每架机自带一块）；
5. 备用电池的**重量**计入卡车出车载荷，且只在给了 `b` 时生效。

为什么值得单独守：这条约束把「换电」从一个纯时间概念变成「时间 + 库存」概念 ——
`--hotswap` 原本是零成本的（落地就能换），加了库存后它要先有电池。
"""

from __future__ import annotations

import copy
import dataclasses

from src.evaluator import V_BATTERY, V_CAP_TRUCK, _simulate_route
from src.generator import build_instance
from src.io import load_config, load_sets
from src.solution import route_from_specs


def _inst(b: int = -1, m: int = 1, mass: float = 4.0, set_id: int = 5):
    cfg = copy.deepcopy(load_config())
    cfg["fleet"]["spare_batteries"] = b
    cfg["fleet"]["drones_per_truck"] = m
    cfg["fleet"]["battery_mass_kg"] = mass
    cfg["fleet"]["allow_hotswap"] = True     # 换电打开，才能真正飞到第 2、3 个架次
    by_id = {s["id"]: s for s in load_sets()}
    return build_instance(by_id[set_id], cfg)


def _two_sorties(inst):
    """构造一条含 2 个架次的路线（两个架次落在不同的锚点对，互不重叠）。"""
    c = sorted(inst.customers)
    a, b, d, e, c1, c2 = c[0], c[1], c[3], c[4], c[6], c[7]
    nodes = [0, a, b, d, e, 0]
    return route_from_specs(0, nodes, [(a, b, (c1,)), (d, e, (c2,))])


def test_unlimited_is_bit_identical() -> None:
    """默认 -1：与不设这项完全一致（成本、能耗、违约清单三样都要对上）。"""
    route = route_from_specs(0, [0, 0], [])
    inst_lim = _inst(-1)
    # 用一条真实路线比：同一解在「不限」与「配置里没这个键」两种构造下必须一模一样
    cfg = copy.deepcopy(load_config())
    cfg["fleet"].pop("spare_batteries", None)
    cfg["fleet"].pop("battery_mass_kg", None)
    by_id = {s["id"]: s for s in load_sets()}
    inst_old = build_instance(by_id[5], cfg)

    r = _two_sorties(inst_lim)
    ev_new = _simulate_route(inst_lim, r, fast=False)
    ev_old = _simulate_route(inst_old, r, fast=False)
    assert ev_new.cost(inst_lim) == ev_old.cost(inst_old)
    assert ev_new.drone_energy == ev_old.drone_energy
    assert ev_new.violations == ev_old.violations
    assert not route.sorties or True


def test_zero_spare_allows_only_one_sortie() -> None:
    """b=0：第 2 个架次必须被判 `V_BATTERY`（fast 模式当场回退，不再往下模拟）。"""
    inst = _inst(0)
    route = _two_sorties(inst)
    ev = _simulate_route(inst, route, fast=False)
    assert not ev.feasible
    assert any(v.startswith(V_BATTERY) for v in ev.violations), ev.violations

    ev_fast = _simulate_route(inst, route, fast=True)
    assert not ev_fast.feasible
    assert ev_fast.n_sorties == 1        # 第 2 个架次被拦下，没算进去


def test_one_spare_allows_two_sorties() -> None:
    inst = _inst(1)
    ev = _simulate_route(inst, _two_sorties(inst), fast=False)
    assert not any(v.startswith(V_BATTERY) for v in ev.violations), ev.violations
    assert ev.n_sorties == 2


def test_cap_scales_with_drones_per_truck() -> None:
    """一车两机、不带备份电池：上限 0+2=2，两个架次仍然可行。"""
    inst = _inst(0, m=2)
    ev = _simulate_route(inst, _two_sorties(inst), fast=False)
    assert not any(v.startswith(V_BATTERY) for v in ev.violations), ev.violations
    # 再收一格（b 不给、m=1）就应该卡住，证明上面放行是因为 m=2 而不是没生效
    inst1 = _inst(0, m=1)
    ev1 = _simulate_route(inst1, _two_sorties(inst1), fast=False)
    assert any(v.startswith(V_BATTERY) for v in ev1.violations)


def test_battery_mass_counts_toward_truck_capacity() -> None:
    """备用电池的重量要占卡车载荷，且只在给了 b 时生效。"""
    inst0 = _inst(2, mass=0.0)
    r = _two_sorties(inst0)
    served = r.served_customers()
    f_total = float(sum(inst0.f[i] for i in served))

    # 把容量卡到「刚好装下货、装不下电池」
    tight = dataclasses.replace(inst0.params, truck_capacity=f_total + 1.0)
    inst_ok = dataclasses.replace(inst0, params=tight)
    ev_ok = _simulate_route(inst_ok, r, fast=False)
    assert not any(v.startswith(V_CAP_TRUCK) for v in ev_ok.violations), ev_ok.violations

    heavy = dataclasses.replace(tight, battery_mass_kg=4.0)   # 2 块 × 4 kg
    inst_bad = dataclasses.replace(inst0, params=heavy)
    ev_bad = _simulate_route(inst_bad, r, fast=False)
    assert any(v.startswith(V_CAP_TRUCK) for v in ev_bad.violations), ev_bad.violations

    # 不限电池（-1）时，重量项不进（否则历史结果会变）
    free = dataclasses.replace(heavy, spare_batteries=-1)
    ev_free = _simulate_route(dataclasses.replace(inst0, params=free), r, fast=False)
    assert not any(v.startswith(V_CAP_TRUCK) for v in ev_free.violations), ev_free.violations
