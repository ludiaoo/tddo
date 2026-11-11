# TDDO · 卡车-无人机联合配送优化器

![tests](https://img.shields.io/badge/tests-255%20%28245%20fast%20%2B%2010%20Gurobi%29-brightgreen)
![python](https://img.shields.io/badge/python-3.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![full run](https://img.shields.io/badge/full%20run-1.6%20h-orange)

> **一句话**：给一张城市配送网（客户、时间窗、禁飞区），算出卡车与多无人机怎么配合最省——
> 车开哪条路、无人机从哪个点起飞、在哪追上卡车、带几块电池。
> 模型是混合整数规划，求解用自适应大邻域搜索，100 个客户 2–3 分钟出解。

![四把刀](results/figs/figB_benefit_bounds.png)

*一个结论，四把刀：同一批 30 个配对下，换电带来的降本随「算例形态 / 换电耗时 / 电池库存 /
悬停计价 / 空间分布」漂移；唯一跨零的是"每车只带 1 块备用电池"。*

---

## 问题

卡车沿路网送货，无人机挂在车顶，可以在任意停靠点起飞、在后面的某个停靠点落回车上——
期间它自己跑几个客户。难点不在"多一个运具"，而在三处耦合：

1. **时间同步**：无人机几点起飞由卡车到达决定，几点落地由飞行与客户服务决定，
   卡车在回收点要么等它、要么它在上空等卡车——两边互为约束；
2. **能量与载荷耦合**：单段能耗 = 距离 ×（单位载荷能耗 × 机上载荷 + 空载系数），
   载荷随每个客户变化，所以能耗只能逐段推，不能事后平均；
3. **时空禁飞区**：同一段航程在禁飞时段内不可飞、时段外可飞，绕行距离是**时间相关**的分段函数。

再加两条业务约束：**双向载荷**（既送也取）与**单架次电量上限**。

---

## 方法

### 模型

| 组 | 变量 / 约束 | 作用 |
|---|---|---|
| 路由 | `x[k,i,j]` | 卡车 k 是否走弧 (i,j) |
| 架次 | `y[k,i,j]` + `wl/wr[k,i]` | 无人机从 i 起飞、在 j 回收；起降成对 |
| 时间 | `Tt[k,i]` / `Td[k,i]` | 卡车到达、无人机起飞 —— 同步的核心 |
| 载荷 | `F/G[k,i]`、`ft[k,i]` | 派件剩余、取件累计、起飞装载 |
| 能量 | `E[k,i]`、`ec[i,j]` | 逐段累加、单架次 ≤ η·E_max |

约束按功能分七族（路由与流 / 架次耦合 / 时间同步 / 时间窗 / 载荷 / 能量 / 空域）。
一个关键设计：**一半约束交给解的表示**——`Route/Sortie` 三层结构让"只在停靠点起降""流守恒"
无法被违反，评估器只检查真需要检查的，单个候选解 ~30 µs。

### 算法

自适应大邻域搜索（ALNS）：4 个破坏算子（随机 / 拔贵的 / 拔相近的 / 结构重组）
× 4 个修复算子（随机 / 卡车 / 无人机 / 均衡），模拟退火接受，权重四档加分。
**Gurobi 精确解在这里的职责是定标与定级**，不是"小规模演示"——见下面的最优性证据。

工程上做了 4.6× 提速：几何缓存、路线评估记忆化、违约早退、多进程落盘续跑。

---

## 结果

**求解质量**（16 个基准算例，每例 10 个独立种子，2 000 轮）

| 指标 | 数值 |
|---|---|
| 相对纯卡车方案 | **成本低 15.9%**（同种子配对） |
| 相对初始可行解 | **再降 51.4%** |
| 100 客户单次求解 | **124–187 s**（20 核，多进程） |
| 10 客户精确解 | **4/4 证到 gap 0%**；ALNS 与最优的距离 0–1.22% |
| 禁飞区绕行的代价 | 仅 +0.5%（成本由司机工时主导，绕行距离影响有限） |

**一处模型改进 + 它的边界**

标准建模里常把"在 i 降落"和"从 i 起飞"写进同一个 0-1 变量，于是同一节点只能二选一。
可现实里换个电池再起飞是常规操作。把该变量拆成到达 / 出发两套之后：

- 若加回"一个节点只做一件事"的限制，最优值与原形式**逐位相同**（说明拆分没写错）；
- 放开后，4 个 10 客户算例的最优值**全部下降**（−0.17% ~ −4.33%），且**全部证到 gap 0%**；
- 差距精确地来自那一条约束——解只违反它。

但这个百分比不是常数。同一批 30 个配对下：

| 设定 | 降本 |
|---|---|
| 基准（电池管够、换电免费、悬停免费） | −3.37% |
| 换电要 2 min / 5 min | −1.62% / −2.26% |
| 每车 3 / 2 / 1 / 0 块备用电池 | −2.52% / −2.04% / −1.02% / **0.00%（结构上无从换电）** |
| 悬停按 1.10 kW 计电 | −2.38% |
| 无取件 / 无禁飞区 / 宽时间窗的纯派送形态 | −4.29% |

**所以对外只报区间：1.5%–3.4%，前提是每车 ≥2 块备用电池、换电不超过 5 分钟。**

---

## 快速开始

```bash
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 算例底图（83 MB 公开数据，不入库）
git clone --depth 1 https://github.com/optimatorlab/mFSTSP.git data/raw/mFSTSP
python -m src.generator                          # 生成 16 个基准算例 JSON

python -m scripts.project_check                  # 自检：产物齐不齐、默认口径有没有被改
python -m scripts.reproduce --quick              # 4 分钟冒烟（不覆盖主结果）
python -m scripts.reproduce                      # 全量约 1.6 h，20 核
pytest -q -m "not slow"                          # 245 项快速测试
```

> 精确解需要 Gurobi 学术许可证；没有时除 `scripts.run_milp` 与 10 项 `-m slow` 测试外全部可跑。

---

## 项目结构

```
configs/          模型与算例参数（TOML）
data/instances/   16 个基准算例（JSON，入库）
data/raw/         原始路网数据（不入库）
src/              模型 + 算法：26 模块 5.8k 行
scripts/          实验 / 诊断 / 出图 / 自检：35 个 5.5k 行
tests/            24 个文件 255 项测试
results/          60 份 CSV（3 402 次求解）+ figs/（figA–D 是主图）
docs/             项目书 · 成果整合 · 求职材料 · 设计笔记 · 分析记录
```

三条我觉得值得讲的技术选择：

- **表示即约束**：见上，评估器因此能做到 30 µs；
- **精确解不是装饰**：它给启发式定标（gap 0–1.22%）、给结论定级（最优值被改写）；
- **四道验证闸门**：默认口径逐位回归 → 同种子配对统计 → 最优性等价 → 双实现互查。

---

## 文档地图

| 想看什么 | 去哪 |
|---|---|
| 项目全貌：问题、建模、算法、验证 | [`docs/项目书.md`](docs/项目书.md) |
| 一页成果（四张主图 + 边界 + 负结果） | [`docs/成果整合.md`](docs/成果整合.md) |
| 工程细节：怎么跑、每个开关干什么 | [`docs/工程手册.md`](docs/工程手册.md) |
| 全部结论与证据强度分级 | [`docs/设计笔记/结论总览.md`](docs/设计笔记/结论总览.md) |
| 换电模型的变量拆分与三道闸门 | [`docs/设计笔记/W1_换电模型设计.md`](docs/设计笔记/W1_换电模型设计.md) |
| 约束逐条翻译（公式 + 实现） | [`docs/笔记/约束逐条翻译.md`](docs/笔记/约束逐条翻译.md) |
| 求职/答辩用的项目经历写法 | [`docs/求职_项目经历.md`](docs/求职_项目经历.md) |

---

## 已知局限

1. **算例是自建的**：底图坐标来自公开路网数据，需求、时间窗、禁飞区、车队规模按规则生成
   （种子固定、规则写在 `configs/`）。所以**结论可比、绝对金额不可比**。
2. **精确模型不含禁飞区与悬停**：绕行是时间相关的分段距离，线性化代价过高；悬停同理。
   最优性结论在"悬停免费"口径内成立。
3. **规模上限 100 客户**：无 200+ 外推。
4. **敏感性是单参数的**：上面四把刀各自测过，**没做联合**，叠加数值不可外推。

---

## 参考

模型形式与问题定义参考了该领域的主流建模；求解器、评估器、扩展与全部实验为本项目实现。

- Zeng J., Hu Y., Pei M. (2026). *Optimizing bidirectional delivery with multiple drones and
  trucks.* **Transportation Research Part E**, 205, 104511.
- 路网底图数据：Murray C., Raj R. (2020). *The multiple flying sidekicks traveling salesman
  problem.* **Transportation Research Part C**, 110, 368–398.

本项目为独立实现，不含第三方代码；论文 PDF 与出版方原图受版权保护，未入库。

---

## English TL;DR

**TDDO** — a truck–drone joint delivery optimizer. Given customers with time windows and
time-dependent circular no-fly zones, it decides truck routes, drone sorties (launch/recovery
points), launch times and battery logistics, minimizing drone electricity + truck fuel +
driver hours.

- **Model**: MILP with 79 constraints in seven families (routing, sortie coupling, time
  synchronization, time windows, bidirectional loads, energy, airspace).
- **Algorithm**: ALNS (4 destroy × 4 repair, simulated-annealing acceptance) with Gurobi used
  for benchmarking and for proving optimality on small instances.
- **Results** (16 benchmark instances × 10 seeds): 15.9% cheaper than a truck-only plan, 51.4%
  below the initial feasible solution, 124–187 s for 100 customers, 4/4 exact optima with 0% gap.
- **Model finding**: splitting the single "recover-or-launch" flag into separate arrival and
  departure variables lowers the proven optimum on 4/4 small instances (−0.17% to −4.33%,
  all at 0% gap). Its magnitude is conditional: 1.5%–3.4% depending on hover-energy pricing,
  swap duration, battery stock and instance form.

---

MIT License · 详见 [`LICENSE`](LICENSE)
