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

- [x] M0 环境与资料
- [ ] M1 数据与算例生成
- [ ] M2 几何 / NFZ / 能耗
- [ ] M3 协同状态机与评估器
- [ ] M4 初始解
- [ ] M5 ALNS
- [ ] M6 对标 Table 3

## 许可与致谢

本仓库不含论文原文（Elsevier 版权）。算例数据版权归原作者，遵循其仓库条款。
