"""实例数据结构与单位换算。

设计原则（对应交接文档 §9 的红线）：
1. **入口处一次性转 SI**：长度 m、时间 s、质量 kg、能量 kWh；配置里可用 lb / km/h 书写，
   但一旦进入 `Instance` 就统一为 SI，任何模块不得再出现 lb。
2. 卡车距离/时间**查路网矩阵**，不按速度推算（mFSTSP 的 tbl_truck_travel_data_PG.csv 给了有向矩阵）；
   无人机用欧氏距离，坐标经局部 ENU 投影到米制平面。
3. 节点编号沿用 mFSTSP：`0` 为 depot，`1..n` 为客户。Zeng 模型中的 `U+1` 返回仓是建模技巧，
   本项目不建虚拟节点，评估器直接用弧 `(i, 0)` 表示回仓。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, sqrt
from typing import Sequence

import numpy as np

# ---- 单位换算常数 ----
LB_TO_KG = 0.45359237
KMH_TO_MS = 1.0 / 3.6

# WGS84
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563
_WGS84_E2 = _WGS84_F * (2.0 - _WGS84_F)


def to_local_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """经纬度 → 以 (lat0, lon0) 为原点的局部平面坐标（米）。

    用局部切平面（子午圈曲率半径 M + 卯酉圈曲率半径 N）而非完整 UTM 投影：
    算例区域最大约 20 km × 20 km，切平面误差 < 0.05%，远小于本项目自造算例本身的
    不确定性；同时省掉 pyproj 依赖。**若某天需要论文 Fig.8 那种标准 UTM 坐标轴，
    换 pyproj 即可，本函数是唯一需要改的地方。**
    """
    phi0 = radians(lat0)
    s = sin(phi0)
    w = sqrt(1.0 - _WGS84_E2 * s * s)
    m_radius = _WGS84_A * (1.0 - _WGS84_E2) / (w ** 3)  # 子午圈曲率半径
    n_radius = _WGS84_A / w                              # 卯酉圈曲率半径

    x = n_radius * cos(phi0) * radians(lon - lon0)
    y = m_radius * radians(lat - lat0)
    return x, y


@dataclass(frozen=True)
class Node:
    idx: int
    kind: str          # "depot" | "customer"
    lat: float
    lon: float
    x: float           # 局部平面东向，米
    y: float           # 局部平面北向，米


@dataclass(frozen=True)
class TimeWindow:
    e: float           # 最早服务时刻，秒
    l: float           # 最晚服务时刻，秒

    def width(self) -> float:
        return self.l - self.e


@dataclass(frozen=True)
class NFZ:
    """圆形时空禁飞区：空间上是固定圆，时间上是固定时段。"""

    cx: float
    cy: float
    r: float
    t_start: float     # 秒
    t_end: float       # 秒
    label: str = ""

    def contains(self, x: float, y: float) -> bool:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= self.r ** 2 + 1e-9

    def active_at(self, t: float) -> bool:
        return self.t_start <= t <= self.t_end


@dataclass(frozen=True)
class BaseProblem:
    """从 mFSTSP 原始文件读出的底图，尚未注入取件/时间窗/禁飞区。"""

    name: str
    city: str
    nodes: tuple[Node, ...]
    truck_dist_m: np.ndarray     # (n+1, n+1) 有向，路网
    truck_time_s: np.ndarray     # (n+1, n+1) 有向，路网
    parcel_kg: tuple[float, ...]  # 按节点编号索引，depot 为 0

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_customers(self) -> int:
        return self.n_nodes - 1

    def drone_dist_m(self) -> np.ndarray:
        """无人机欧氏距离矩阵（米），(n+1, n+1)。"""
        xy = np.array([[nd.x, nd.y] for nd in self.nodes], dtype=float)
        diff = xy[:, None, :] - xy[None, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))


@dataclass(frozen=True)
class FleetParams:
    drone_speed: float            # m/s
    drone_capacity: float         # kg
    drone_battery: float          # kWh
    eta_soc: float                # 单架次最大用电比例
    truck_capacity: float         # kg
    launch_time: float            # s
    recovery_time: float          # s
    service_time: float           # s
    e_drone: float                # kWh/(kg·km)
    b: float                      # kWh/km（空载单位距离能耗）
    e_truck: float                # L/km
    c_drone: float                # $/kWh
    c_truck: float                # $/L
    c_customer: float             # $/s
    drones_per_truck: int = 1     # 每车可同时在飞的架次数（论文约束 20/21 是 1，M9.12 扩展为 m）
    allow_hotswap: bool = False   # 允许「同节点先回收再起飞」（M9.15：论文用单个 w_i 把两件事
                                  # 合并成一个 0-1 变量，导致同一节点不能既降落又起飞；
                                  # 物理上无人机落地、换电池、再起飞是常规操作）
    e_hover: float = 0.0          # 悬停功率换算后的单位时间能耗，kWh/s（B 项；默认为 0，
                                  # 即「与历史结果逐位一致」——论文式(39) 只按飞行距离计电）
    # ---- E 项：电池库存与重量（W2 最小版）----
    # 每车出车时带几块**备用**电池。一台无人机自带的电池算一块，所以一条路线最多能飞
    # `spare_batteries + drones_per_truck` 个架次（每换一次电消耗一块备用）。
    # `-1` = 不限（论文口径：每飞一次就凭空换上一块满电电池），历史结果逐位不变。
    spare_batteries: int = -1
    battery_mass_kg: float = 0.0  # 单块电池质量（kg）。只在给了 spare_batteries 时计入卡车载荷
                                  # （车顶挂架放满再出发，用没用到都占重量）

    @classmethod
    def from_config(cls, cfg: dict) -> "FleetParams":
        fleet, energy, cost = cfg["fleet"], cfg["energy"], cfg["cost"]
        return cls(
            drone_speed=float(fleet["drone_speed_kmh"]) * KMH_TO_MS,
            drone_capacity=float(fleet["drone_payload_lb"]) * LB_TO_KG,
            drone_battery=float(fleet["drone_battery_kwh"]),
            eta_soc=float(fleet["eta_soc"]),
            truck_capacity=float(fleet["truck_capacity_lb"]) * LB_TO_KG,
            launch_time=float(fleet["launch_s"]),
            recovery_time=float(fleet["recovery_s"]),
            service_time=float(fleet["service_s"]),
            e_drone=float(energy["e_drone_kwh_per_kg_km"]),
            b=float(energy["b_kwh_per_km"]),
            e_truck=float(energy["e_truck_l_per_km"]),
            c_drone=float(cost["c_drone_per_kwh"]),
            c_truck=float(cost["c_truck_per_l"]),
            c_customer=float(cost["c_customer_per_s"]),
            drones_per_truck=int(fleet.get("drones_per_truck", 1)),
            allow_hotswap=bool(fleet.get("allow_hotswap", False)),
            # 悬停能耗：配置里用「千瓦」书写（人更容易判断量级），入口处换成 kWh/s
            e_hover=float(energy.get("hover_power_kw", 0.0)) / 3600.0,
            spare_batteries=int(fleet.get("spare_batteries", -1)),
            battery_mass_kg=float(fleet.get("battery_mass_kg", 0.0)),
        )


@dataclass
class Instance:
    """注入取件、时间窗、禁飞区之后、可直接喂给求解器的算例。"""

    set_id: int
    base: BaseProblem
    f: np.ndarray                 # 派件需求，形状 (n_nodes,)，depot 位置为 0
    g: np.ndarray                 # 取件需求，形状 (n_nodes,)，depot 位置为 0
    tw: tuple[TimeWindow, ...]    # 按客户索引（长度 = n_customers，对应节点 1..n）
    nfz: tuple[NFZ, ...]
    n_trucks: int
    params: FleetParams
    seed: int
    # 是否允许派无人机。False 即 Table 3 的 **Truck-only** 对照组（纯卡车基线）。
    # 放在 Instance 上而不是算子参数上：这样破坏/修复/初始解三处只需读同一个开关，
    # 不会出现「初始解没派机、修复却又派了」的不一致。
    allow_drones: bool = True
    # 是否允许无人机**绕行**禁飞区。False 即 Table 3 的 **Without-detour** 对照：
    # 禁飞区仍然生效（节点禁飞 γ·δ 照查），但穿越禁飞区的航段**直接不可飞**（不允许绕）。
    # 这是「限制」而不是「放开」——所以该列成本必然 ≥ 完整模型（论文里 Increasing% 为正）。
    allow_detour: bool = True

    # ---- 便捷访问 ----
    @property
    def n_customers(self) -> int:
        return self.base.n_customers

    @property
    def customers(self) -> Sequence[int]:
        return range(1, self.base.n_nodes)

    def drone_dist_m(self) -> np.ndarray:
        return self.base.drone_dist_m()

    def total_delivery_kg(self) -> float:
        return float(self.f.sum())

    def total_pickup_kg(self) -> float:
        return float(self.g.sum())

    def describe(self) -> str:
        p = self.params
        return (
            f"Set{self.set_id} {self.base.name} ({self.base.city}, "
            f"{self.n_customers} 客户) | K={self.n_trucks} | NFZ={len(self.nfz)} | "
            f"派件 {self.total_delivery_kg():.1f} kg / 取件 {self.total_pickup_kg():.1f} kg | "
            f"卡车容量 {p.truck_capacity:.1f} kg / 无人机 {p.drone_capacity:.2f} kg"
        )
