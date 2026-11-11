# 卡车-无人机联合配送 · 运筹优化复现与扩展

![tests](https://img.shields.io/badge/tests-255%20%28245%20fast%20%2B%2010%20Gurobi%29-brightgreen)
![python](https://img.shields.io/badge/python-3.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![reproduce](https://img.shields.io/badge/reproduce-1.6%20h-orange)

> 复现 Zeng, Hu & Pei (2026, *TR-E* 205:104511) 的「多卡-多无人机双向配送 + 时空禁飞区」模型，
> 然后用精确解把它的一条建模假设证伪，最后给出**这条结论在什么条件下成立**。
>
> **一句话**：把论文里一个 0-1 变量拆成两个语义，同节点换电可降本 **1.5%–3.4%**；
> 在 4 个 10 客户算例上，论文模型的"已证明最优"全部被打破。

![四把刀](results/figs/figB_benefit_bounds.png)

*一个结论，四把刀：同一批 30 个配对，换电收益随"算例形态 / 换电耗时 / 电池库存 / 悬停计价 / 空间分布"
漂移；唯一跨零的是"每车只带 1 块备用电池"。*

---

## 这个项目做了什么

| # | 内容 | 证据类型 |
|---|---|---|
| 1 | **复现**论文的 79 条约束模型 + ALNS，16 算例 × 10 种子 | 统计对齐（Saving 51.4% vs ≈52%） |
| 2 | 发现并放开一处**建模缺陷**：`w_i` 同时表示"降落"与"起飞" | **最优性**（4/4 算例被打破，gap 0%） |
| 3 | 给出这条结论的**边界**：悬停、换电耗时、电池库存、算例形态、空间分布 | 双口径配对 + bootstrap |
| 4 | 12 组**负结果**：预筛 / UCB / DQN / LLM 造算子 / 规则干预 / 锚点平移 | 探针 + 全量对照 |

---

## 主要结果

**复现（16 算例 × 10 种子）**

| 指标 | 论文 | 本项目 |
|---|---|---|
| Saving 均值 | ≈52% | **51.4%** |
| 100 客户 CPU | 159–254 s | **124–187 s** |
| 10 客户 MILP | 证到最优 | **4/4 gap 0%**（ALNS gap 0–1.22%） |
| PSO vs ALNS | ALNS 更优 | ALNS 更优（+13.3%） |
| Without-detour | +8.2% 量级 | +0.5%（差一个量级，已归因到成本结构） |

**换电：从"一个数"到"一个区间"**

| 设定 | 换电收益 |
|---|---|
| 全量 16×10（论文口径，悬停免费、换电免费、电池管够） | −2.26% |
| 6×5 子集（同口径） | −3.37% |
| 换电要 2 min / 5 min | −1.62% / −2.26% |
| 每车 3 / 2 / 1 / 0 块备用电池 | −2.52% / −2.04% / −1.02% / **0.00%（结构上无从换电）** |
| 悬停按 1.10 kW 计电 | −2.38% |
| 回到 mFSTSP 原形态（无取件/无禁飞区/宽窗） | −4.29% |

---

## 快速开始

```bash
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 原始算例（83 MB，第三方公开数据，不入库）
git clone --depth 1 https://github.com/optimatorlab/mFSTSP.git data/raw/mFSTSP
python -m src.generator                          # 生成 16 个金标准算例 JSON

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
data/instances/   16 个金标准算例（JSON，入库）
data/raw/         原始 mFSTSP 数据（不入库）
src/              模型 + 算法：26 模块 5.8k 行
scripts/          实验 / 诊断 / 出图 / 自检：35 个 5.5k 行
tests/            24 个文件 255 项测试
results/          60 份 CSV（3 402 次求解）+ figs/（figA–D 是主图）
docs/             项目书 · 成果整合 · 求职材料 · 里程碑 · 笔记
```

关键技术选择（详见 `docs/项目书.md`）：

- **一半约束交给解的表示**：`Route/Sortie` 三层结构让「只在停靠点起降」「流守恒」无法违反，
  评估器只检查真需要检查的，单候选 ~30 µs；
- **精确与启发式分工**：MILP 用来给 ALNS 定标、给结论定级，不是"小规模演示"；
- **四道验证闸门**：逐位回归（`golden_check`）→ 同种子配对统计 → 最优性等价 → 双实现互查。

---

## 文档地图

| 想看什么 | 去哪 |
|---|---|
| 这是什么运筹问题、怎么建模、怎么验 | [`docs/项目书.md`](docs/项目书.md) |
| 一页看完的成果（四张主图 + 边界 + 负结果） | [`docs/成果整合.md`](docs/成果整合.md) |
| 怎么跑、每个开关干什么、和论文差在哪 | [`docs/工程与复现手册.md`](docs/工程与复现手册.md) |
| 17 条结论的证据强度分级 | [`docs/milestones/M9_成果总览.md`](docs/milestones/M9_成果总览.md) |
| 换电模型的变量拆分与三道闸门 | [`docs/milestones/W1_换电模型设计.md`](docs/milestones/W1_换电模型设计.md) |
| 79 条约束逐条翻译 | [`docs/notes/约束逐条翻译_2-79.md`](docs/notes/约束逐条翻译_2-79.md) |
| 求职/答辩用的项目经历写法 | [`docs/求职_项目经历.md`](docs/求职_项目经历.md) |

---

## 已知局限（先说清）

1. **算例不是论文原算例**：底层坐标是公开数据，取件/时间窗/禁飞区/车队规模自定 → 只比趋势，不比绝对美元。
2. **MILP 不含禁飞区与悬停**：最优性结论在"悬停免费"口径内成立（与论文自洽）。
3. **规模上限 100 客户**：原始数据最大只有 100，无 200+ 外推。
4. **敏感性是单参数的**：四把刀各自测过，**没做联合**，叠加数值不可外推。

---

## 引用与致谢

- 被复现论文：Zeng J., Hu Y., Pei M. (2026). *Transportation Research Part E*, 205, 104511.
- 算例数据：Murray C., Raj R. (2020). *Transportation Research Part C*, 110, 368–398（mFSTSP）。
- 本项目为**独立复现**，不含原作者代码；论文 PDF 与 Elsevier 原图受版权保护，未入库。

---

## English TL;DR

A reproduction study of a truck–drone tandem delivery model with synchronized rendezvous,
bidirectional loads and time-dependent no-fly zones (79 MILP constraints, ALNS heuristic).

Reproduction matches the paper (saving 51.4% vs ≈52%; 100-customer CPU 124–187 s; 4/4 MILP
optima proven). Building on that, a single 0-1 flag `w_i` that conflates *recovering* and
*launching* at the same node is shown to be a modelling artefact: relaxing it yields
**1.5%–3.4% cost reduction** and provably breaks the paper's optimum on 4/4 instances.
The second half of the repo maps **where that number lives** — hover energy pricing,
battery swap duration, battery stock, instance form and spatial layout — with paired
statistics and bootstrap intervals. 12 negative results on learning-based operator
selection are reported in full.

Code: `src/` (model + ALNS), `scripts/` (experiments, diagnostics, figures). One-command
reproduction: `python -m scripts.reproduce`.

---

MIT License · 详见 [`LICENSE`](LICENSE)
