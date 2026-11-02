"""静态几何缓存：把「与时间无关」的几何一次性算好，评估器内循环只做查表。

## 为什么要有这一层

剖析 Set13（100 客户）15 轮 ALNS：13.4 秒里 `_fly_sortie` 占 76%，
`service_ok` 被调 83.8 万次、`point_in_circle` 670 万次、`math.hypot` 845 万次。
但这些东西**绝大多数是与时间无关的**：

| 量 | 是否依赖时间 | 现在的做法 | 应改为 |
|---|---|---|---|
| 两点的直线距离 | ❌ | 每次 `math.hypot` 重算 | 缓存 `(i,j)` |
| 节点在哪些圆内（γ） | ❌ | 每次遍历全部禁飞区 | 预处理 `node → [zone]` |
| 航段穿过哪些圆 | ❌ | 每次算点到线段距离 | 预处理 `(i,j) → [(zone, 绕行量)]` |
| 该航段要不要绕 | ✅ 看时间区间 | 每次判断 | 只查时间 |
| 端点此刻能否起降 | ✅ 看时刻 | 每次遍历全部禁飞区 | 只查该节点相关的几个圆 |

所以「绕行距离」本身是纯几何量、可以**完全预计算**，运行时只剩「哪些圆的时段与本次飞行相交」
这一个判断。这是本模块的全部思路。

## 正确性约束

缓存只能放**静态**量。`allow_detour` 不进缓存（它是运行时开关，缓存里存的是几何事实）。
缓存的失效条件是 `nfz` 变了（半径/时段/数量）或换了底图，构造函数里用签名比对。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from .distance import INF, FlightLeg
from .geometry import EPS, detour_around_circle, segment_intersects_circle

if TYPE_CHECKING:  # pragma: no cover
    from .instance import Instance


class GeometryCache:
    """按算例缓存的静态几何。用 `GeometryCache.of(inst)` 取，不要直接 new。"""

    __slots__ = (
        "zones", "sig", "xy", "node_zones", "_pairs", "hits", "misses",
        # 纯 Python 的扁平数组：numpy 标量索引每次都要建一个 np.float64 对象，
        # 在 80 万次级别的内循环里比 list 慢 3–4 倍。
        "f_list", "g_list", "tw_e", "tw_l", "truck_d", "truck_t",
    )

    def __init__(self, inst: "Instance") -> None:
        self.zones = tuple(inst.nfz)
        # 签名用于失效判断：底图身份 + 每个禁飞区的几何与时段
        self.sig = (
            id(inst.base),
            tuple((z.cx, z.cy, z.r, z.t_start, z.t_end) for z in self.zones),
        )
        self.xy = tuple((nd.x, nd.y) for nd in inst.base.nodes)

        # 节点 → 内含它的禁飞区下标（γ 是静态参数，论文 Table 2 里就是这么定义的）。
        # 用 math.hypot 而非 numpy，保证与 point_in_circle 的浮点结果逐位一致。
        node_zones: list[tuple[int, ...]] = []
        for (x, y) in self.xy:
            hits = tuple(
                h for h, z in enumerate(self.zones)
                if math.hypot(x - z.cx, y - z.cy) <= z.r + EPS
            )
            node_zones.append(hits)
        self.node_zones = tuple(node_zones)

        self._pairs: dict[tuple[int, int], tuple[float, tuple]] = {}
        self.hits = 0
        self.misses = 0

        # 需求 / 时间窗 / 卡车路网同样是静态量，转成纯 Python 对象供内循环使用
        self.f_list = tuple(map(float, inst.f))
        self.g_list = tuple(map(float, inst.g))
        self.tw_e = tuple(float(w.e) for w in inst.tw)
        self.tw_l = tuple(float(w.l) for w in inst.tw)
        self.truck_d = tuple(tuple(map(float, row)) for row in inst.base.truck_dist_m)
        self.truck_t = tuple(tuple(map(float, row)) for row in inst.base.truck_time_s)

        # 需求 / 时间窗 / 卡车路网：同样是静态量，转成纯 Python 对象供内循环使用
        self.f_list = tuple(map(float, inst.f))
        self.g_list = tuple(map(float, inst.g))
        self.tw_e = tuple(float(w.e) for w in inst.tw)
        self.tw_l = tuple(float(w.l) for w in inst.tw)
        dist, tm = inst.base.truck_dist_m, inst.base.truck_time_s
        self.truck_d = tuple(tuple(map(float, row)) for row in dist)
        self.truck_t = tuple(tuple(map(float, row)) for row in tm)

    # ------------------------------------------------------------ 构造/取用
    @classmethod
    def of(cls, inst: "Instance") -> "GeometryCache":
        """取该算例的缓存；禁飞区一变就自动重建。

        失效判断用 `is` 而不是重建签名元组：`nfz` 是 tuple，`replace(nfz=...)`
        必然产生新对象。每次调用都拼 40 个浮点的签名，在 7.7 万次调用下本身就要 0.24 s
        （剖析里它一度排进前五），属于典型的「为了安全把检查做成瓶颈」。
        """
        cur = getattr(inst, "_geom_cache", None)
        if cur is not None and cur.zones is inst.nfz:
            return cur
        cache = cls(inst)
        try:
            inst._geom_cache = cache
        except Exception:                       # 万一实例被设为不可写，退化为每次新建
            pass
        return cache

    # ------------------------------------------------------------ 静态查表
    def _pair(self, i: int, j: int) -> tuple[float, tuple]:
        """(直线距离, ((zone 下标, 绕行量, 该圆下是否可行), ...))。

        只收录**线段与圆相交**的禁飞区——不相交的根本不影响这条航段，
        这正是论文 Fig.4 里「不绕」的那几种情形在空间上的判据。
        """
        key = (i, j)
        got = self._pairs.get(key)
        if got is not None:
            self.hits += 1
            return got
        self.misses += 1

        p, q = self.xy[i], self.xy[j]
        straight = math.hypot(p[0] - q[0], p[1] - q[1])
        hits = []
        for h, z in enumerate(self.zones):
            c = (z.cx, z.cy)
            if not segment_intersects_circle(p, q, c, z.r):
                continue
            d = detour_around_circle(p, q, c, z.r)
            hits.append((h, d.extra, d.feasible))
        got = (straight, tuple(hits))
        self._pairs[key] = got
        return got

    def straight(self, i: int, j: int) -> float:
        return self._pair(i, j)[0]

    # ------------------------------------------------------------ 运行时判定
    def service_ok(self, i: int, t: float) -> tuple[bool, tuple[str, ...]]:
        """节点 i 在时刻 t 能否起降（论文 Fig.3 的 γ·δ）。

        只看「内含该节点」的那几个圆——其余圆与该点无关，遍历它们是纯浪费。
        """
        zs = self.node_zones[i]
        if not zs:
            return True, ()
        zones = self.zones
        blocked = tuple(
            zones[h].label for h in zs if zones[h].t_start <= t <= zones[h].t_end
        )
        return (not blocked), blocked

    def leg(
        self, i: int, j: int, t_depart: float, t_arrive: float, allow_detour: bool = True
    ) -> FlightLeg:
        """无人机 i→j 的实际距离与可行性（与 `distance.drone_leg` 语义完全一致）。

        顺序也保持一致：先判两端能否起降，再按禁飞区下标顺序判绕行/不可绕。
        """
        ok_i, blocked_i = self.service_ok(i, t_depart)
        ok_j, blocked_j = self.service_ok(j, t_arrive)
        if not (ok_i and ok_j):
            # 两个端点都要查：原实现无论哪端被禁都把两端的原因合并，
            # 只报一端会让违约诊断信息少一半（虽然不影响可行性与成本）。
            return FlightLeg(
                distance=INF, detour=INF, feasible=False,
                blocked_by=tuple(dict.fromkeys(blocked_i + blocked_j)),
            )

        straight, hits = self._pair(i, j)
        if not hits:
            return FlightLeg(distance=straight, detour=0.0, feasible=True)

        zones = self.zones
        total = 0.0
        for h, extra, geom_ok in hits:
            z = zones[h]
            # Fig.4：飞行区间与禁飞时段不相交 → 不绕（情形 1、6）
            if t_arrive < z.t_start - EPS or t_depart > z.t_end + EPS:
                continue
            if not allow_detour:
                # Without-detour 对照：不许绕，穿圆即不可飞（收紧，不是放开）
                return FlightLeg(distance=INF, detour=INF, feasible=False,
                                 blocked_by=(z.label,))
            if not geom_ok:
                # 端点在圆内 → 切线不存在，该航段无定义
                return FlightLeg(distance=INF, detour=INF, feasible=False,
                                 blocked_by=(z.label,))
            total += extra

        return FlightLeg(distance=straight + total, detour=total, feasible=True)
