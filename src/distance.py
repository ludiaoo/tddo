"""距离与航段：把几何模块接到算例上。

两条距离来源完全不同，**不要混用**：
- **卡车**：查 mFSTSP 的路网矩阵（`tbl_truck_travel_data_PG.csv`，OSM + pgRouting 的有向矩阵），
  不要用欧氏距离推。
- **无人机**：欧氏直线 + 绕行修正；节点坐标是投影到米制平面的平面坐标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .geometry import (
    Detour,
    Point,
    detour_around_circle,
    point_in_circle,
    segment_intersects_circle,
    zone_active_overlap,
)
from .instance import BaseProblem, NFZ

INF = float("inf")


@dataclass(frozen=True)
class FlightLeg:
    """一段无人机飞行（i → j）的实际结果。"""

    distance: float          # 实际飞行距离（含绕行），米
    detour: float            # 额外绕行距离，米
    feasible: bool           # 端点是否可被无人机服务（γ·δ 全为 0）
    blocked_by: tuple[str, ...] = ()   # 导致不可行的禁飞区标签

    @property
    def straight(self) -> float:
        return self.distance - self.detour


def node_xy(base: BaseProblem, i: int) -> Point:
    nd = base.nodes[i]
    return (nd.x, nd.y)


def truck_leg(base: BaseProblem, i: int, j: int) -> tuple[float, float]:
    """卡车 i→j 的（距离 m, 时间 s），直接查路网矩阵。"""
    return float(base.truck_dist_m[i, j]), float(base.truck_time_s[i, j])


def drone_distance(base: BaseProblem, i: int, j: int) -> float:
    """无人机 i→j 的直线距离（不含绕行），米。"""
    p, q = node_xy(base, i), node_xy(base, j)
    return ((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) ** 0.5


def service_ok(node: Point, t: float, zones: Sequence[NFZ]) -> tuple[bool, tuple[str, ...]]:
    """节点在时刻 t 能否被无人机起降 —— 论文 Fig.3 的 γ·δ 判定。

    γ = 节点是否在圆内（空间），δ = 到达时刻是否落在禁飞时段（时间）。
    二者同时成立则该节点此时代不能由无人机服务，只能交给卡车。
    """
    blocked = tuple(
        z.label for z in zones if point_in_circle(node, (z.cx, z.cy), z.r) and z.active_at(t)
    )
    return (len(blocked) == 0), blocked


def drone_leg(
    base: BaseProblem,
    zones: Sequence[NFZ],
    i: int,
    j: int,
    t_depart: float,
    t_arrive: float,
) -> FlightLeg:
    """无人机 i→j 的变化距离与可行性。

    步骤：
    1. 起降可行性：i 在 t_depart、j 在 t_arrive 时若落在「禁飞期内的禁飞区」→ 不可行；
    2. 绕行：对每个**飞行区间与禁飞时段相交**的圆，按切线 + 圆弧累加 `d_ijh`；
    3. 若某圆使航段无定义（端点在该圆内），同样判为不可行。

    注意第 1 步与第 2 步是两套判据（论文 Fig.3 vs Fig.4），不要合并：
    节点在内只是「此刻不能在这一点起降」，航段穿圆才是「这一路要绕」。
    """
    p, q = node_xy(base, i), node_xy(base, j)
    straight = drone_distance(base, i, j)

    ok_i, blocked_i = service_ok(p, t_depart, zones)
    ok_j, blocked_j = service_ok(q, t_arrive, zones)
    if not (ok_i and ok_j):
        return FlightLeg(distance=INF, detour=INF, feasible=False,
                         blocked_by=tuple(dict.fromkeys(blocked_i + blocked_j)))

    total_detour = 0.0
    for z in zones:
        center = (z.cx, z.cy)
        if not zone_active_overlap(t_depart, t_arrive, z.t_start, z.t_end):
            continue                      # 时间上没交集 → 不绕（Fig.4 的情形 1、6）
        if not segment_intersects_circle(p, q, center, z.r):
            continue                      # 空间上没穿圆 → 不绕
        d: Detour = detour_around_circle(p, q, center, z.r)
        if not d.feasible:
            return FlightLeg(distance=INF, detour=INF, feasible=False, blocked_by=(z.label,))
        total_detour += d.extra

    return FlightLeg(distance=straight + total_detour, detour=total_detour, feasible=True)


def drone_flight_time(leg: FlightLeg, speed_ms: float) -> float:
    """飞行时间 = 距离 / 巡航速度（含绕行段）。"""
    return leg.distance / speed_ms
