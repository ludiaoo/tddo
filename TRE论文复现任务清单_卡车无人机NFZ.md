# TRE 论文复现任务清单（卡车-多无人机协同配送 · 时空禁飞区）

> 跨电脑执行版：只看本文件即可推进，不依赖原对话。
> 复现目标论文：Zeng J., Hu Y., Pei M. (2026). *Optimizing bidirectional delivery with multiple drones and trucks: a mixed-integer nonlinear model to addressing no-fly zone constraints.* **Transportation Research Part E, 205, 104511.** DOI: 10.1016/j.tre.2025.104511
> 复现档位：**工程复现**（ALNS 主线 + 16 算例对标 Table 3，Gurobi 仅小规模验证）。预计 5–7 周（每周 10–12 小时）。
> 核心策略：最大化复用本人 `2StageTDEVRPsolver` 仓库；非线性物理量放 Python 评估器，Gurobi 只建简化 MILP 跑小规模。

---

## M0 · 环境与资料准备（第 0 周，3–5 小时）

- [ ] 确认 Python 3.13 可用（本机即月表论文开发机，通常已装）；为新项目创建独立虚拟环境 `python -m venv .venv` 并激活
- [ ] 安装依赖：`pip install numpy pandas matplotlib gurobipy alns`（版本参考 TDEVRP 仓库 requirements.txt）
- [ ] 配置 Gurobi 学术许可证（academic license 或 WLS），命令行验证：`gurobi_cl --license`，再 `python -c "import gurobipy; print(gurobipy.gurobi.version())"`
- [ ] 骨架代码已在本机：`2StageTDEVRPsolver` 就是在这台开发机上写的，直接本地只读参考即可，无需 clone；新项目另建目录、不直接改它；若担心本地版本落后可 `git pull` 与远端同步
- [ ] 备齐论文 PDF（按 DOI 下载），重点打印：Table 2 符号表、约束 (2)–(79)、Algorithm 1–2、Section 5 参数与 Table 3
- [ ] 明确复现方式：**独立实现，不向作者索取代码或算例**；仅依据公开论文与公开数据集自行编码，实例改造逻辑严格对照论文 Section 5 与 Table 2 复现
- [ ] 检索 Murray & Raj (2020) mFSTSP 公开数据集（Seattle / Buffalo，Transportation Research Part C；作者主页或 Mendeley），下载原始坐标数据
- [ ] 新建独立 Git 仓库（私有），首个 commit 只放 README、.gitignore、requirements.txt

**验收**：本机能 import gurobipy 且许可证可用；本地 `2StageTDEVRPsolver` 骨架可正常运行、B 的新私有仓库已建；论文与数据源可访问。

---

## M1 · 数据与算例生成（第 1 周）

- [ ] 设计实例数据结构（建议 JSON/dataclass）：depot、customers（坐标、派件需求 f_i、取件需求 g_i、时间窗 [e_i,l_i]、服务时 τ_i）、trucks K、drones（每卡一机）、NFZ 列表（圆心、半径 r、禁飞起止 t_start/t_end）
- [ ] 写 mFSTSP 原始数据解析器（UTM 坐标，单位米），保留 Seattle/Buffalo 两套底图
- [ ] 按论文 Section 5 实现改造逻辑：注入 pickup 需求、服务时间窗、圆形时空 NFZ
- [ ] 生成 16 个算例：节点规模 10 / 25 / 50 / 100，每规模 4 个 Set（Set1–16 编号与 Table 3 对齐）
- [ ] 固化关键参数（论文给定，注意英制↔SI 换算，全程统一单位并写注释）：
      无人机巡航 65 km/h、最大载荷 10 lb、电池 0.78 kWh、SOC 阈值 η^SoC；卡车容量 300 lb；能耗系数 e^drone/e^truck、空载能耗 b、单位能源与司机工时成本 c
- [ ] 固定随机种子策略（论文每算例 10 个独立种子取 best），写进配置文件
- [ ] 写算例可视化（节点、卡车路线底图、NFZ 红圈），先复刻论文 Fig.8 的静态网络示意

**验收**：一条命令生成全部 16 算例并落盘；能画出带 NFZ 的网络拓扑图；需求/容量/时间窗字段与论文描述一致。

---

## M2 · 几何、物理与时空 NFZ（第 1–2 周，大量复用）

- [ ] 距离模块：卡车路网行驶距离/时间、无人机欧氏直线距离
- [ ] **迁移 TDEVRP 的圆形障碍绕行几何**（`preprocessor` 思路）：切线+相切圆弧，实现论文 Fig.5 与式(1) 的额外绕行距离 d_ijh；用 Dijkstra 在可视图上生成全局最短距离矩阵
- [ ] 实现节点可否服务判断（Fig.3 四情形）：空间 γ_i,h 与时间 δ 联合，γ·δ=1 则该节点禁飞、只能卡车服务
- [ ] 实现航段是否绕行判断（Fig.4 六情形）：用到达时刻与 t_start/t_end 比较，输出二元 z_end−z_start 决定是否 detour
- [ ] 实现无人机载荷依赖能耗式(39)：E = d·(e^drone·L + b)，区分直线段与绕段距离
- [ ] 为上述每个纯函数写单元测试，构造手算可验证的小例子（单圆相切、单航段时间窗命中）

**验收**：`pytest` 全绿；绕行距离、六种时空情形、能耗在手工样例上数值正确。此层不依赖求解，先做扎实。

---

## M3 · 解表示、协同状态机与评估器（第 2–4 周，核心新增，最难）

- [ ] 定义解结构：每条卡车路径 + 其配属无人机的 sortie 列表；sortie 记录起飞节点、无人机多客户子路径、回收节点
- [ ] 写**两层协同状态机**：沿路径递推卡车时刻 T^truck 与无人机时刻 T^drone，实现起飞/回收 rendezvous 同步（对应约束 22–31），含等待时间
- [ ] 实现**双向两套载荷流**：派件载荷 F、取件载荷 G 分别递推与守恒（约束 50–73），卡车与无人机容量约束 (62)(73)
- [ ] 实现无人机每架次能耗累计与 SOC 上限约束 (40)–(43)，及载荷在起降节点与卡车的交接
- [ ] 实现时间窗约束 (35)–(38) 与路径去环、流量守恒
- [ ] 写统一 `evaluate(solution)`：返回目标总成本（式 74/79 三项：无人机能耗成本 + 卡车能耗成本 + 司机工时成本）+ 可行性报告（各类违约标记）
- [ ] **非线性不进 Gurobi**：detour、能耗等复杂函数全部在此评估器计算（沿用 TDEVRP 的 RouteEvaluator 模式）

**验收**：对人工构造的可行/不可行解，评估器给出正确成本并能定位每一类违约；同步、载荷、SOC 逻辑各有测试覆盖。

---

## M4 · 初始解构造（第 4 周，复用为主）

- [ ] 实现 Algorithm 2 第一阶段：slack-first 插入生成卡车路径（最小时间余量优先，含载荷/时间窗可行性）
- [ ] 实现第二阶段无人机分配：对每条卡车路径按 min-slack 选候选客户，确定起降前驱/后继，校验时间、序列、载荷、能耗后生成可行 sortie
- [ ] 复用 TDEVRP `initial_solution.py` 的 Regret 插入与冗余清理思路

**验收**：16 个算例全部能生成可行初始解；记录 Initial Cost，后续用于计算 Saving%。

---

## M5 · ALNS 算子与主循环（第 4–6 周）

- [ ] 4 个 destroy 算子：random removal、worst removal、relatedness removal（空间邻近 + NFZ 邻近）、truck shuffle removal；**规则：选中起飞/回收点必须整段 sortie 移除**
- [ ] 4 个 repair 算子：random insertion、truck-first、drone-first、equal-priority insertion，全部过 M3 可行性校验
- [ ] 模拟退火接受准则：P_accept = exp(−Δ/T)，T_new = α·T，初温 300、α=0.99
- [ ] 轮盘赌自适应权重，按四档结果打分：global best +2、current improvement +1、accepted not improved +0.5、rejected/infeasible ×0.9
- [ ] 主循环参数：破坏率 30%、最大 2000 迭代、连续 200 次无改进 early-stop 并回退当前最优
- [ ] 关闭 NFZ/detour 模块时框架应退化为标准 VRPD（论文明确要求的开关，用于 Without-detour 对照组）
- [ ] 复用 TDEVRP `weighted_operators.py`、`local_search.py`、`alns_vns.py` 的骨架

**验收**：小算例上成本稳定下降、权重能自适应调整、停滞重启生效；退化开关可切换。

---

## M6 · 对标实验与 Gurobi 小规模验证（第 6–7 周）

- [ ] 实现简化 MILP（Gurobi）：仅用于 10 节点小规模求最优/近优，物理非线性用外生参数替代，不强求复刻全部 79 条约束
- [ ] 跑通 10 节点 Set1–4，确认 ALNS 能匹配 Gurobi 最优（论文：ALNS 2.8–3.4 s，Gurobi 105–180 s，gap 0%）
- [ ] 16 算例 × 10 种子取 best，产出总表，列与 Table 3 对齐：Gurobi(cost/gap/CPU)、ALNS(best/initial/saving%/CPU)、Truck-only、Without-detour
- [ ] （可选）实现 PSO 对照组，同迭代预算，局部/全局因子 2.05、惯性权重 0.9→0.4
- [ ] 校验关键结论量级：初始解改善约 45–65%；相对 truck-only 成本降 4–54%；detour 相对无绕行最多再降 21.18%；25 节点以上 Gurobi 失效而 ALNS 仍在数十至数百秒出解
- [ ] 数字不必逐格相同（数据/随机实现差异），但**趋势与量级必须一致**，偏差大则回查 M2/M3

**验收**：一张可复现的结果 CSV + 对照表，结论方向与 Table 3 吻合；所有实验脚本可一键重跑。

---

## M7 · 灵敏度分析与可视化（第 7 周，工程档可裁剪）

- [ ] 卡车容量（150–500 lb）× 需求缩放（0.5–1.5）双因素，复刻 Fig.9（成本、车队规模）
- [ ] 无人机载荷（5–15 lb）× 续航（0.60–0.90 kWh）、速度（55–80 km/h），复刻 Fig.10
- [ ] NFZ 半径（1–4 倍）× 禁飞时段（无/工作时段/全天），复刻 Fig.11
- [ ] 路线可视化复刻 Fig.8（卡车实线、无人机虚线、NFZ 红圈、UTM 坐标轴）

**验收**：图表结论与论文一致（如容量 300–400 lb 边际收益拐点、8–12 lb 载荷收益平台、半径影响强于时段）。

---

## M8 · 工程化与求职沉淀（贯穿 + 收尾）

- [ ] README：问题简介、环境、一键运行命令、目录说明、复现结果摘要
- [ ] 目录建议（镜像 TDEVRP 习惯）：`configs/ data/raw data/instances src/(io,generator,geometry,energy,state,evaluator,initial_solution,operators,alns,gurobi_model,visualizer).py experiments/ tests/ results/`
- [ ] requirements.txt 固定版本；结果输出 CSV/PNG；脚本支持断点续跑与种子配置
- [ ] **简历口径诚实**：写"复现 TR Part E 论文（DOI）并实现/扩展"，不冒充原创
- [ ] 至少加一处自己的贡献点（任选）：新 destroy/repair 算子、动态或鲁棒 NFZ、替换接受准则、引入 VNS（你 TDEVRP 的强项），并做消融对比
- [ ] 沉淀面试讲法：母车-子机 rendezvous 同步、双向载荷守恒、时空 NFZ 线性化三处难点要能白板讲清

---

## 风险与注意事项

1. **实例生成严格对照论文 Section 5 与 Table 2 符号表独立实现**，并用小规模手算样例自检几何、能耗与可行性；Table 3 只要求趋势与量级一致，不追求逐格相等（数据与随机实现差异客观存在）。
2. **单位统一是最常见 bug 源**：lb/kg、kWh、km/h 与 UTM 米混用，建议入口处一次性转 SI，内部全用 SI。
3. 非线性（能耗=距离×载荷、detour×二元）放评估器；Gurobi 只用简化 MILP 验证小规模，避免陷入 McCormick/辅助 q 变量(75)–(78) 的完整线性化泥潭。
4. 两层协同（M3）是关键路径，优先级最高；若进度紧，M7 灵敏度先砍，M0–M6 必须完整。
5. 每完成一个 M 阶段打一个 Git tag，方便跨电脑同步和回滚。
6. 数字对不上时按 M2（几何/能耗）→ M3（同步/载荷）→ M5（算子）顺序排查，不要先调参。

## 里程碑速查

| 里程碑 | 结束周 | 硬交付 |
|---|---|---|
| M0–M1 | 第 1 周 | 环境可用 + 16 算例生成 |
| M2 | 第 2 周 | 几何/NFZ/能耗单测全绿 |
| M3 | 第 4 周 | 评估器能算成本并判可行性 |
| M4–M5 | 第 6 周 | ALNS 全算例收敛 |
| M6 | 第 7 周 | 对标 Table 3 的结果表 |
| M7–M8 | 收尾 | 灵敏度图表 + README + 简历素材 |
