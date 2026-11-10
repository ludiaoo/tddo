"""悬停能耗（B 项）的守门测试。

要守四条：
1. `hover_power_kw = 0`（默认）时，悬停**分文不取**，与论文式(39) 的口径逐位一致；
2. 悬停计电后，多出来的能耗**恰好**是 `hover_time × e_hover`，且按 `c_drone` 进成本；
3. 悬停发生在**落地之前**，所以要并进本架次的 SOC 复核（约束 43）；
4. 只加电、不改时刻表——悬停是"等"，不是"更快到达"，`finish_time` 不变。

为什么单独一个文件：B 项是 C9（跨段锚点）能不能被引用的前提。C9 的收益有相当一部分
来自"悬停免费"这个建模漏洞，所以这条链上的每一步都要能单独钉住。
"""

from __future__ import annotations

import copy
import dataclasses

from src.evaluator import V_SOC, _simulate_route
from src.generator import build_instance
from src.io import load_config, load_sets
from src.solution import route_from_specs

HOVER_KW = 0.45          # 与 configs/config.toml 注释里的量级一致


def _inst(hover_kw: float = 0.0, set_id: int = 5):
    cfg = copy.deepcopy(load_config())
    cfg["energy"]["hover_power_kw"] = hover_kw
    by_id = {s["id"]: s for s in load_sets()}
    return build_instance(by_id[set_id], cfg)


def _find_hovering_route(inst):
    """在算例里找一条「无人机先到、卡车后到」的路线（**回收点**悬停 > 0）。

    刻意只要 `hover_recov > 0`：窗口悬停（早到等 `e_i`）几乎每条带无人机的路线都有，
    用它当判据等于没筛。人为构造容易随算例变动而失效，所以直接枚举：任取两个卡车锚点 a < b，
    让无人机 a 点起飞、b 点回收、中间捎一个客户 c。只要路网距离让卡车后到，就有悬停。
    """
    custs = sorted(inst.customers)
    nodes = [0, *custs, 0]
    for x in range(1, min(len(nodes) - 2, 8)):
        a, b = nodes[x], nodes[x + 1]
        for c in custs:
            if c in (a, b):
                continue
            r = route_from_specs(0, [0, a, b, 0], [(a, b, (c,))])
            ev = _simulate_route(inst, r, fast=False)
            if ev.feasible and ev.hover_recov > 0.0:
                return r, ev
    raise AssertionError("该算例上没找到会悬停的路线，测试需要换算例")


def test_hover_split_is_consistent() -> None:
    """诊断口径：`hover_time` 必须恒等于「回收点 + 窗口」两项之和，与计不计电无关。

    这条守的是 B 项的机制拆解（M9.18 §4.5）：如果窗口悬停只在计电时才被记录，
    那么"免费口径下的悬停"就会被系统性低估，拆解表会骗人。
    """
    inst0 = _inst(0.0)
    route, ev0 = _find_hovering_route(inst0)
    inst1 = _inst(HOVER_KW)
    ev1 = _simulate_route(inst1, route, fast=False)

    for ev in (ev0, ev1):
        assert abs(ev.hover_time - (ev.hover_recov + ev.hover_window)) < 1e-9
        assert ev.hover_recov > 0.0            # 这条路线本来就是靠"回收点悬停"找出来的
    # 计时不变、只是多了耗电
    assert abs(ev1.hover_time - ev0.hover_time) < 1e-9


def test_default_hover_is_free() -> None:
    """默认配置下悬停不耗电：这是「历史结果逐位不变」的前提。"""
    inst = _inst(0.0)
    assert inst.params.e_hover == 0.0
    _r, ev = _find_hovering_route(inst)
    assert ev.hover_time > 0.0          # 悬停确实发生了
    assert ev.hover_energy == 0.0       # 但没计电
    # 成本里也不能出现悬停项：与手工按三项相加的结果一致
    p = inst.params
    manual = p.c_drone * ev.drone_energy + ev.truck_distance / 1000.0 * p.e_truck * p.c_truck \
        + ev.finish_time * p.c_customer
    assert abs(ev.cost(inst) - manual) < 1e-9


def test_hover_energy_charged_exactly() -> None:
    """计电后多出来的电量 = hover_time × e_hover，且只影响电费。"""
    inst0 = _inst(0.0)
    route, ev0 = _find_hovering_route(inst0)

    inst1 = _inst(HOVER_KW)
    ev1 = _simulate_route(inst1, route, fast=False)

    # 时刻表不受影响：悬停是「等」，不是「更快」
    assert abs(ev1.finish_time - ev0.finish_time) < 1e-9
    assert abs(ev1.hover_time - ev0.hover_time) < 1e-9
    assert ev1.n_sorties == ev0.n_sorties

    # 电量差恰好等于 hover_time × e_hover
    assert abs(ev1.hover_energy - ev1.hover_time * inst1.params.e_hover) < 1e-12
    assert abs((ev1.drone_energy - ev0.drone_energy) - ev1.hover_energy) < 1e-12

    # 成本差恰好等于「多耗的电 × 电价」
    p = inst1.params
    assert abs((ev1.cost(inst1) - ev0.cost(inst0)) - p.c_drone * ev1.hover_energy) < 1e-12


def test_hover_counts_toward_soc() -> None:
    """悬停能耗要并进本架次复核 SOC：电池卡在「飞行够、悬停不够」的临界点上会报 V_SOC。"""
    inst0 = _inst(0.0)
    route, ev0 = _find_hovering_route(inst0)

    inst_h = _inst(HOVER_KW)
    ev_h = _simulate_route(inst_h, route, fast=False)
    e_flight, e_hover = ev0.drone_energy, ev_h.hover_energy
    assert e_hover > 0.0

    # ① 电池给到「飞行 + 1.5×悬停」：够用，不该报 SOC
    tight_ok = dataclasses.replace(
        inst_h.params, drone_battery=(e_flight + 1.5 * e_hover) / inst_h.params.eta_soc
    )
    ev_ok = _simulate_route(dataclasses.replace(inst_h, params=tight_ok), route, fast=False)
    assert not any(v.startswith(V_SOC) for v in ev_ok.violations)

    # ② 电池卡到「飞行 + 0.5×悬停」：飞得完、悬停悬不完 → 必须报 SOC
    tight_bad = dataclasses.replace(
        inst_h.params, drone_battery=(e_flight + 0.5 * e_hover) / inst_h.params.eta_soc
    )
    ev_bad = _simulate_route(dataclasses.replace(inst_h, params=tight_bad), route, fast=False)
    assert any(v.startswith(V_SOC) for v in ev_bad.violations), ev_bad.violations

    # ③ 同一块电池、悬停不计电时反而可行 —— 证明 ② 的违约确实来自悬停
    free = dataclasses.replace(tight_bad, e_hover=0.0)
    ev_free = _simulate_route(dataclasses.replace(inst_h, params=free), route, fast=False)
    assert not any(v.startswith(V_SOC) for v in ev_free.violations), ev_free.violations
