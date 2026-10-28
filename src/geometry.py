"""圆形禁飞区的几何计算（纯数学，不依赖 Instance）。

对应论文 §2.2.1、式(1)、Fig.3–5：
- **Fig.5 / 式(1)**：绕行距离 `d_ijh = l' − l`，路径为「切线 → 圆弧 → 切线」
- **Fig.3**：节点能否被无人机服务，判据 `γ·δ`（空间 × 时间）
- **Fig.4**：航段是否需要绕行，判据 `z_end − z_start`（飞行区间与禁飞时段是否相交）

三种端点位置必须分开处理（最容易出错的地方）：
1. 两端都在圆外、且线段穿过圆 → 按切线 + 圆弧算绕行；
2. 任一端点在圆内 → **切线不存在**，返回 `feasible=False`。物理含义是「该节点在禁飞期内
   根本不能由无人机服务」，应由 `γ·δ` 判定，而不是硬算出一个绕行距离；
3. 线段不穿过圆 → 绕行距离为 0。

坐标一律米。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

EPS = 1e-9
Point = Tuple[float, float]


@dataclass(frozen=True)
class Detour:
    """单个圆对一段航段的影响。"""

    extra: float          # 额外绕行距离 = path − straight，米
    path: float           # 绕行后长度，米
    straight: float       # 直线长度，米
    arc_angle: float      # 绕行圆弧对应的圆心角，弧度
    feasible: bool        # 端点是否都在圆外（False 表示该航段在此圆下无定义）
    tangent_a: Point | None = None  # 起点侧切点
    tangent_b: Point | None = None  # 终点侧切点


def _sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def _norm(v: Point) -> float:
    return math.hypot(v[0], v[1])


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _wrap(angle: float) -> float:
    """把角度归一到 (−π, π]。"""
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def point_in_circle(p: Point, c: Point, r: float) -> bool:
    """点是否落在圆内（含边界）。对应论文的 γ_ih。"""
    return _dist(p, c) <= r + EPS


def segment_min_distance(p: Point, q: Point, c: Point) -> float:
    """圆心到线段的最短距离（不是到直线的距离）。"""
    vx, vy = q[0] - p[0], q[1] - p[1]
    wx, wy = c[0] - p[0], c[1] - p[1]
    seg_len2 = vx * vx + vy * vy
    if seg_len2 < EPS:
        return _dist(p, c)
    t = (wx * vx + wy * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    return math.hypot(p[0] + t * vx - c[0], p[1] + t * vy - c[1])


def segment_intersects_circle(p: Point, q: Point, c: Point, r: float) -> bool:
    """线段是否与圆盘有交集（端点在内也算）。"""
    return segment_min_distance(p, q, c) <= r + EPS


def detour_around_circle(p: Point, q: Point, c: Point, r: float) -> Detour:
    """无人机从 p 飞到 q、绕开圆 (c, r) 的最短路径，返回额外距离。

    设 d_p = |p−c|、d_q = |q−c|，切线与半径垂直，故
        切线长 = √(d² − r²)，切点相对圆心方向与 p 方向夹角 β = arccos(r/d)
    两条射线夹角 Δ ∈ [0, π]，则圆弧角
        θ = Δ − β_p − β_q
    若 θ ≤ 0，说明线段本来就不穿过圆（Δ 不够大），无需绕行。

    对称算例（p=(−2,0), q=(2,0), r=1）应得 path = 2√3 + π/3 ≈ 4.5113。
    """
    dp = _dist(p, c)
    dq = _dist(q, c)
    straight = _dist(p, q)

    # 端点落在圆内：切线不存在（见图 3 的判定逻辑，这种航段根本不该出现）
    if dp <= r + EPS or dq <= r + EPS:
        return Detour(
            extra=float("inf"), path=float("inf"), straight=straight,
            arc_angle=0.0, feasible=False,
        )

    beta_p = math.acos(min(1.0, r / dp))
    beta_q = math.acos(min(1.0, r / dq))

    phi_p = math.atan2(p[1] - c[1], p[0] - c[0])
    phi_q = math.atan2(q[1] - c[1], q[0] - c[0])
    delta_signed = _wrap(phi_q - phi_p)
    delta = abs(delta_signed)

    arc = delta - beta_p - beta_q
    if arc <= 0.0:
        # 线段不穿过圆
        return Detour(extra=0.0, path=straight, straight=straight,
                      arc_angle=0.0, feasible=True)

    side = 1.0 if delta_signed >= 0.0 else -1.0
    ta_angle = phi_p + side * beta_p
    tb_angle = phi_q - side * beta_q
    tangent_a = (c[0] + r * math.cos(ta_angle), c[1] + r * math.sin(ta_angle))
    tangent_b = (c[0] + r * math.cos(tb_angle), c[1] + r * math.sin(tb_angle))

    path = math.sqrt(dp * dp - r * r) + math.sqrt(dq * dq - r * r) + r * arc
    return Detour(
        extra=path - straight,
        path=path,
        straight=straight,
        arc_angle=arc,
        feasible=True,
        tangent_a=tangent_a,
        tangent_b=tangent_b,
    )


def zone_active_overlap(t_from: float, t_to: float, z_start: float, z_end: float) -> bool:
    """飞行时间区间 [t_from, t_to] 是否与禁飞时段 [z_start, z_end] 相交。

    这就是 Fig.4 的判据：`z_end_ih − z_start_jh = 1` ⇔ `T_i < t_end` 且 `T_j > t_start`。
    六种情形（起点在禁飞前/中/后 × 终点在禁飞前/中/后）都被这一条统一覆盖，
    其中情形 2–5 都要绕——注意比较的是 `t_end` 而不是 `t_start`。
    端点相接视为相交（保守处理，宁绕勿闯）。
    """
    return not (t_to < z_start - EPS or t_from > z_end + EPS)


def total_detour(
    p: Point,
    q: Point,
    zones: Sequence[tuple[Point, float, float, float]],
) -> tuple[float, bool]:
    """多圆叠加的绕行距离：Σ_h d_ijh（论文目标函数里对 h 求和）。

    `zones` 元素为 (center, radius, t_start, t_end)。
    时间窗口已由调用方筛过或在此统一判断 —— 这里要求传入的 zones 已按时间筛好。
    任一圆使航段无定义（端点在内）则整体 `feasible=False`。
    """
    total = 0.0
    for center, r, _ts, _te in zones:
        d = detour_around_circle(p, q, center, r)
        if not d.feasible:
            return float("inf"), False
        total += d.extra
    return total, True
