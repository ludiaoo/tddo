"""解的表示。

设计原则：**让尽可能多的约束由结构保证**（见 `约束逐条翻译_2-79.md` 的判断列）。

```
Solution
└── Route           一条卡车路径（对应一辆车）
    ├── nodes       [0, c1, c2, ..., 0]  卡车依次经过的节点（含首尾仓库）
    └── sorties     该车无人机的架次列表
        └── Sortie  launch / recover 是 nodes 的**位置下标**，customers 是无人机服务并飞回卡车的客户序列
```

由此免费获得的约束：

- (2)(3) 每条路径首尾都是仓库        → `nodes[0] == nodes[-1] == 0`
- (16)(17) 每个客户只被服务一次      → `validate_structure()` 检查客户不重复、且不同时出现在 nodes 与 sorties
- (20)(21) 起降必须卡车在场          → sortie 的 launch/recover 是路径上的位置，天然成立
- (49) 只能在停着的客户点起飞        → 同上
- 起降点允许卡车与无人机各来一次      → 结构上就是「位置」与「客户」两套编号，不冲突

**不做的事**：不在这里判断时间窗、电量、载荷、禁飞区 —— 那些是 `evaluator` 的显式检查。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .instance import Instance

DEPOT = 0


@dataclass(frozen=True)
class Sortie:
    """一次架次：从卡车路径的某个位置起飞，服务若干客户，在另一个位置落回卡车。"""

    launch: int                    # nodes 中的位置下标（无人机在此起飞）
    recover: int                   # nodes 中的位置下标（无人机在此回收），必须 > launch
    customers: tuple[int, ...]     # 无人机服务的客户节点编号（顺序即飞行顺序）

    def __post_init__(self) -> None:
        if self.recover <= self.launch:
            raise ValueError(f"sortie 的回收位置 {self.recover} 必须晚于起飞位置 {self.launch}")
        if not self.customers:
            raise ValueError("sortie 至少要服务一个客户")
        if len(set(self.customers)) != len(self.customers):
            raise ValueError("sortie 内客户重复")


@dataclass
class Route:
    """一辆卡车的完整方案。"""

    truck: int
    nodes: list[int] = field(default_factory=lambda: [DEPOT, DEPOT])
    sorties: list[Sortie] = field(default_factory=list)

    def served_customers(self) -> set[int]:
        """本路线服务过的全部客户（卡车直服 + 无人机）。"""
        out = {n for n in self.nodes if n != DEPOT}
        for s in self.sorties:
            out.update(s.customers)
        return out

    def sortie_at_launch(self, pos: int) -> Sortie | None:
        for s in self.sorties:
            if s.launch == pos:
                return s
        return None

    def sortie_at_recover(self, pos: int) -> Sortie | None:
        for s in self.sorties:
            if s.recover == pos:
                return s
        return None


@dataclass
class Solution:
    routes: list[Route] = field(default_factory=list)

    def served_customers(self) -> set[int]:
        out: set[int] = set()
        for r in self.routes:
            out |= r.served_customers()
        return out


def validate_structure(sol: Solution, inst: Instance) -> list[str]:
    """结构合法性检查：只查「解的形状」，不查时间/电量/载荷。

    返回错误信息列表，空列表表示结构上没问题。
    """
    errs: list[str] = []
    seen: dict[int, int] = {}

    for r in sol.routes:
        if len(r.nodes) < 2 or r.nodes[0] != DEPOT or r.nodes[-1] != DEPOT:
            errs.append(f"车{r.truck}: 路径必须以仓库 0 开始并结束，实际 {r.nodes}")

        # 客户在卡车路径内不重复、且不与非仓库节点重复出现
        truck_customers = [n for n in r.nodes if n != DEPOT]
        if len(set(truck_customers)) != len(truck_customers):
            errs.append(f"车{r.truck}: 卡车路径内有重复客户 {truck_customers}")

        drone_customers: list[int] = []
        for s in r.sorties:
            drone_customers.extend(s.customers)
            if s.launch >= len(r.nodes) or s.recover >= len(r.nodes):
                errs.append(f"车{r.truck}: sortie 位置越界 ({s.launch},{s.recover})，路径长度 {len(r.nodes)}")
                continue
            if r.nodes[s.launch] == DEPOT or r.nodes[s.recover] == DEPOT:
                errs.append(f"车{r.truck}: sortie 不允许在仓库起降")
            # 同一位置不能同时是两个架次的起飞点
            same_launch = [x for x in r.sorties if x.launch == s.launch]
            same_recover = [x for x in r.sorties if x.recover == s.recover]
            if len(same_launch) > 1 or len(same_recover) > 1:
                errs.append(f"车{r.truck}: 同一位置被多个架次占用（当前规则：一位置只能起降一次）")

        if len(set(drone_customers)) != len(drone_customers):
            errs.append(f"车{r.truck}: 无人机客户重复 {drone_customers}")

        # 架次内客户不能同时出现在卡车路径上
        overlap = set(truck_customers) & set(drone_customers)
        if overlap:
            errs.append(f"车{r.truck}: 客户同时被卡车与无人机服务 {sorted(overlap)}")

        for c in r.served_customers():
            if c in seen:
                errs.append(f"客户 {c} 被车{seen[c]} 和 车{r.truck} 重复服务")
            seen[c] = r.truck

    missing = set(inst.customers) - set(seen)
    extra = set(seen) - set(inst.customers)
    if missing:
        errs.append(f"有 {len(missing)} 个客户未被服务：{sorted(missing)[:8]}")
    if extra:
        errs.append(f"出现不存在的客户编号：{sorted(extra)[:8]}")

    return errs
