# 卡车-多无人机双向配送（时空禁飞区）复现

> 独立复现：Zeng J., Hu Y., Pei M. (2026). *Optimizing bidirectional delivery with multiple drones and trucks: a mixed-integer nonlinear model to addressing no-fly zone constraints.* **Transportation Research Part E, 205, 104511.** DOI: `10.1016/j.tre.2025.104511`
>
> 本项目为**工程复现 + 独立实现**，代码与实验均为本人依据公开论文与公开数据集编写，非原作者代码，也不代表原作者的结论可逐格复现。

## 问题

多卡车-多无人机协同配送：每辆卡车配属 1 架无人机，无人机可在任意客户点起降、单架次服务多个客户；同时处理**派件与取件**双向载荷；飞行需绕开**带时间窗的圆形禁飞区**（NFZ）。目标为最小化无人机能耗成本 + 卡车能耗成本 + 司机工时成本。

## 环境

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
```

Gurobi 学术许可证需自行配置，验证：

```bash
python -c "import gurobipy as gp; m=gp.Model(); print('ok')"
```

## 数据

算例基于 Murray & Raj (2020) mFSTSP 公开数据集（Seattle / Buffalo），需单独拉取，不入库：

```bash
git clone --depth 1 https://github.com/optimatorlab/mFSTSP.git data/raw/mFSTSP
```

原始算例为纯配送（VRPD）数据，本复现按论文 §5 注入取件需求、时间窗与禁飞区；**论文未给出这些改造的生成规则**，本项目自行定义并固化在 `configs/` 中（详见 `M0_论文要点笔记.md` §9）。

## 生成算例

```bash
python -m src.generator      # 输出 data/instances/set01.json … set16.json 与终端汇总表
```

自定规则（全部在 `configs/config.toml` 的 `[generation]`，固定种子 `20261027 + set_id`）：

| 项 | 规则 |
|---|---|
| 取件 | `g_i ~ U(0, f_i)`，向上取整到 0.5 lb |
| 时间窗 | `e_i = 0.9·τ(depot→i) + U(0,3)h`，宽 `U(30,90)` min |
| 禁飞区 | 数量 `clamp(round(n/8), 2, 8)`（10→2、50→6、100→8），半径 **`U(600,1200)` m**，圆心 = 客户点偏移 `0.25–0.55r`，时段 = 班次开始后 1–2 h 起、持续 1.2–2.2 h |
| 车队规模 | `K = max(容量下界, 时间窗所需路线数)`；只按容量会低估（Set1 容量只需 1 辆、时间窗要 3 辆） |

> 半径刻意大于论文 Fig.8 的观感（150–500 m）：按那个标度只有 3.7% 的航段需绕行、中位 12 m，
> Without-detour 对照组恒为 0，绕行机制等于没复现出来。放大到 600–1200 m 后绕行率 6.6%、中位 56 m。
> Table 3 的数字**不可能逐格复现**，因为改造规则本身未公开；本项目只要求趋势与量级一致。

## 文档

- `M0_论文要点笔记.md`：复现对象的问题、符号、约束地图、算法与参数
- `M6_对标结果.md`：Table 3 三列对照 + Gurobi 小规模验证 + 口径差异说明
- `约束逐条翻译_2-79.md`：79 条约束逐条公式 + 直译 + 为什么需要 + 启发式如何处理
- `约束逐条翻译_2-79.html`：同上，公式已渲染（MathML，离线可看）

公式用 LaTeX 写在 md 里，HTML 由 pandoc 生成：

```bash
python -m scripts.build_docs        # 需要 pandoc 3.x
```

## 可视化与测试

```bash
python -m src.visualizer      # 输出 results/figs/set01..16_network.png 与 all_sets.png
pip install -r requirements-dev.txt
python -m pytest -q           # 183 项（默认跳过需要 Gurobi 的 4 项）
python -m pytest -q -m slow   # 4 项 MILP 测试：与评估器逐项对账 + 不劣于 ALNS
python -m scripts.check_geometry   # 几何诊断：绕行率、禁飞阻断率
python -m scripts.run_alns         # 16 算例 × 三列（ALNS / Truck-only / Without-detour）→ results/table3_like.csv
python -m scripts.run_pso          # 在同一批解上补 PSO 列 → results/table3_with_pso.csv
python -m scripts.run_milp         # 4 个 10 客户算例的 MILP 最优 → results/milp_check.csv
```

### 两个容易混淆的开关

| 开关 | 作用 | 对成本的影响 | 对应 Table 3 的哪一列 |
|---|---|---|---|
| `allow_drones=False` | 不派无人机 | 变高 | **Truck-only** |
| `allow_detour=False` | 禁飞区照查，但穿圆的航段**不许绕**（直接不可飞） | 变高 | **Without-detour** |
| 清空 `nfz`（`disable_nfz`） | 禁飞区整个删掉 | 变低 | 只用于诊断，**不是** Without-detour |

Without-detour 是「收紧」不是「放开」——论文 Table 3 该列 Increasing% 全为正，就是这个原因。

### 评估器的两条硬规则（容易漏，且会显著影响成本）

| 规则 | 为什么必须有 |
|---|---|
| **架次不能重叠**（`DRONE_BUSY`） | 每车只配一架无人机，上一个架次没回收就不能再起飞。漏检时两个架次会「同时各飞各的」，成本被低估十几个百分点，ALNS 会主动去找这种非法解 |
| **同一节点不能既回收又起飞**（`LR_SAME_NODE`） | 论文只用一个 `w_i` 表示「起飞或回收」，同点连续起降会让时刻递推 (23) 拿卡车到达时刻当起飞基准，模型内部不自洽 |

`data/instances/*.json` 是**金标准**：任何人改了 `src/generator.py` 或 `configs/config.toml`，
`test_金标准算例未被改动` 会立刻失败并打印第一处差异。确认要改时重跑生成器并一起提交。

## 目录

```
configs/      参数与算例改造配置
data/raw/     原始数据（git 忽略）
src/          io / generator / geometry / energy / state / evaluator / initial_solution / operators / alns / gurobi_model / visualizer
experiments/  对标实验脚本
tests/        单元测试
results/      结果 CSV 与图
```

## 进度

- [x] M0 准备：环境、文献与数据源
- [x] M1 数据准备与算例构造
- [x] M2 基础计算：距离、绕行与能耗
- [x] M3 解的表示与成本评估
- [x] M4 初始可行解构造
- [x] M5 改进算法：ALNS 主循环
- [x] M6 实验与结果对标（Gurobi 小规模验证 + Truck-only + Without-detour + PSO）
- [ ] M7 灵敏度分析与可视化
- [ ] M8 整理与交付

## 许可与致谢

本仓库不含论文原文（Elsevier 版权）。算例数据版权归原作者，遵循其仓库条款。
