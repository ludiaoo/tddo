"""能耗与成本（论文式 39、40–43、74/79）。

无人机：`E = d · (e_drone · L + b)`，L 为该段的机上载荷（派件 + 取件）。
卡车：`e_truck · D`（升），再乘油价。
司机：`c_customer · T_返回仓`（按秒计）。

⚠ 论文 Table 2 把 `b` 的单位标成 kWh，但代入式(39) 时必须与 `e_drone·L`（kWh/km）同量纲，
所以 `b` 实际是「空载**单位距离**能耗」，单位 kWh/km。这是论文的笔误，本项目按 kWh/km 实现。

⚠ 论文未给 `e_drone / b / e_truck / 三个成本系数` 的数值，当前取量级合理值，
M6 会用 Table 3 反推校准。因此**在校准之前只做相对比较，不报绝对美元**。
"""

from __future__ import annotations

from .instance import FleetParams

KWH_PER_SEGMENT_EPS = 1e-12


def drone_energy(distance_m: float, payload_kg: float, params: FleetParams) -> float:
    """单段飞行能耗（kWh）。式(39)：E = d · (e · L + b)。"""
    d_km = distance_m / 1000.0
    return d_km * (params.e_drone * payload_kg + params.b)


def sortie_energy(legs: list[tuple[float, float]], params: FleetParams) -> float:
    """一个架次的累计能耗。

    `legs` 为 [(距离 m, 该段载荷 kg), ...]。载荷依赖能耗，所以必须逐段累加，
    不能整段平均——这是式(41)(42) 的语义。
    """
    return sum(drone_energy(d, payload, params) for d, payload in legs)


def sortie_energy_ok(energy_kwh: float, params: FleetParams) -> bool:
    """约束(43)：单架次累计 ≤ η_SoC · E_max。"""
    return energy_kwh <= params.eta_soc * params.drone_battery + KWH_PER_SEGMENT_EPS


def drone_cost(energy_kwh: float, params: FleetParams) -> float:
    """无人机能耗成本（$）。"""
    return energy_kwh * params.c_drone


def truck_cost(distance_m: float, params: FleetParams) -> float:
    """卡车燃油成本（$）：升数 = 距离(km) × e_truck。"""
    return (distance_m / 1000.0) * params.e_truck * params.c_truck


def driver_cost(finish_time_s: float, params: FleetParams) -> float:
    """司机工时成本（$）：按该车返仓时刻计。"""
    return finish_time_s * params.c_customer


def mandatory_truck_distance(inst) -> float:
    """所有客户「若全由卡车直服」的最短往返距离之和，用作 Truck-only 对照的粗下界。

    只用于 M6 的对照组做量级检查，不进求解器。
    """
    total = 0.0
    for i in inst.customers:
        total += float(inst.base.truck_dist_m[0, i]) * 2.0
    return total
