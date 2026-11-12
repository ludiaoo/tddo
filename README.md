# TDDO：卡车带无人机的配送路径优化

![tests](https://img.shields.io/badge/tests-255%20%28245%20fast%20%2B%2010%20Gurobi%29-brightgreen)
![python](https://img.shields.io/badge/python-3.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)
![full run](https://img.shields.io/badge/full%20run-1.6%20h-orange)

卡车沿路网送货，无人机挂在车顶。到某个停靠点无人机起飞，自己跑几个客户，再在前面某个点落回车上。
模型要定的是这四件事：卡车走哪条路、无人机从哪飞、在哪儿追上车、带几块电池。

这类问题在文献里叫 truck-drone tandem，模型形式基本定型了。我做这个项目的目的有两个：
把它跑到实际规模，以及弄清楚模型里那个「降落和起飞共用一个 0-1 变量」的写法值多少钱。

现在的结果是：16 个算例、每个 10 个种子上，比纯卡车方案便宜 15.9%；把那个变量拆开之后，
4 个小算例的最优值全部下降。但降多少取决于几个设定，区间是 1.5%–3.4%。

![四把刀](results/figs/figB_benefit_bounds.png)

*30 个配对，每次只改一个设定。唯一跨过 0 线的是「每车只带 1 块备用电池」。*

## 怎么建模

变量分五组。

| 组 | 符号 | 干什么 |
|---|---|---|
| 路由 | `x[k,i,j]` | 卡车 k 走不走弧 (i,j) |
| 架次 | `y[k,i,j]`、`wl/wr[k,i]` | 无人机从 i 起飞、在 j 回收，起降成对 |
| 时间 | `Tt[k,i]`、`Td[k,i]` | 卡车到达、无人机起飞 |
| 载荷 | `F/G[k,i]`、`ft[k,i]` | 派件剩余、取件累计、起飞装载 |
| 能量 | `E[k,i]`、`ec[i,j]` | 逐段累加，单架次不超过 η·E_max |

约束 79 条，按功能分七族（路由与流、架次耦合、时间同步、时间窗、载荷、能量、空域）。
最烦的是时间同步：无人机几点起飞看卡车什么时候到，几点落地看它跑完几个客户要多久，
而卡车在回收点还得决定等不等它。两边互为约束，绕不开。

约束没有全部写进检查器。有一半交给解的表示：`Route → Sortie` 三层结构下，「只在停靠点起降」
「流守恒」这类事根本没法违反，检查器只管剩下真正要查的，单个候选解 30 µs。
ALNS 能跑到 2000 轮，靠的就是这个数。

## 怎么求解

ALNS：4 个破坏算子（随机、拔贵的、拔相近的、结构重组）× 4 个修复算子（随机、卡车、无人机、均衡），
模拟退火接受，算子权重分四档加分。小算例另外用 Gurobi 解到最优。

精确解在这里的用途不是「小规模演示」。它有两个实际作用：给启发式定标，10 客户上 ALNS 与最优差 0–1.22%；
以及后面证明「最优值真的被改写了」时，只有它说话算数。

中间做过一轮提速，4.6 倍。主要手段是几何缓存和把路线评估结果记下来，另外加了违约早退、
多进程逐行落盘以便中断续跑。

## 结果

16 个算例（10 / 25 / 50 / 100 客户，两座城市），每个 10 个种子，2000 轮。
相对纯卡车方案成本低 15.9%（同种子配对），相对自己的初始可行解再降 51.4%。
100 客户单次 124–187 秒，20 核并行。

有个数当时觉得反常：禁飞区绕行只让成本涨 0.5%。查下来是成本主要被司机工时撑着，
绕行多出来的几公里摊到时间上不显眼。这条后来解释了实验结果里很多「不显著」。

### 那个 0-1 变量

标准写法里，「在 i 降落」和「从 i 起飞」是同一个变量，于是同一个节点只能二选一。
物理上没道理，换个电池再起飞是常规操作。拆成到达、出发两套之后：

- 把「一个节点只做一件事」的限制加回去，最优值与原形式逐位相同。这一步说明拆分本身没写错。
- 放开，4 个 10 客户算例的最优值全部下降，−0.17% 到 −4.33%，全部证到 gap 0。
- 两个解的差别只落在一条约束上，解本身只违反它。差距确实来自这里，不是别处。

但这个百分比不是常数。同一批 30 个配对，改一个设定就变：

| 设定 | 降本 |
|---|---|
| 电池管够、换电免费、悬停免费 | −3.37% |
| 换电要 2 分钟 / 5 分钟 | −1.62% / −2.26% |
| 每车 3 / 2 / 1 / 0 块备用电池 | −2.52% / −2.04% / −1.02% / **0.00%** |
| 悬停按 1.10 kW 计电 | −2.38% |
| 无取件、无禁飞区、宽时间窗（纯派送形态） | −4.29% |

0.00% 那格不是收益变小。一块电池都不带时一条路线最多一个架次，落地之后没有第二次起飞，
换电在模型里无从发生。

所以对外我只说 1.5%–3.4%，前提是每车至少 2 块备用电池、换电不超过 5 分钟。换电耗时这一条我扫了
0 / 120 / 300 秒三个点，120 和 300 之间差 0.69%，p 值 0.59，分不出高低。本来想给一条曲线，
后来只能给区间。

## 跑起来

```bash
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 底图数据 83 MB，不入库
git clone --depth 1 https://github.com/optimatorlab/mFSTSP.git data/raw/mFSTSP
python -m src.generator                        # 生成 16 个算例 JSON

python -m scripts.project_check                # 产物齐不齐、默认口径有没有被改
python -m scripts.reproduce --quick            # 4 分钟冒烟，不覆盖主结果
python -m scripts.reproduce                    # 全量约 1.6 h，20 核
pytest -q -m "not slow"                        # 245 项快速测试
```

精确解要 Gurobi 学术许可证。没有的话跑不了 `scripts.run_milp` 和 10 项 `-m slow` 的测试，其余都能跑。

## 目录

```
configs/          模型与算例参数（TOML）
data/instances/   16 个算例（JSON，入库）
data/raw/         底图数据（不入库）
src/              模型 + 算法，26 个模块 5.8k 行
scripts/          实验、诊断、出图、自检，35 个脚本 5.5k 行
tests/            24 个文件 255 项测试
results/          60 份 CSV（3 402 次求解）+ figs/
docs/             项目书、成果报告、求职材料、设计笔记
```

验证上用了四道闸：默认口径逐位回归（改性能时结果一位都不许变）、同种子配对统计（不拿各种子的最好值比）、
最优性等价、MILP 解转回评估器复算。最后一道最便宜也最有用，两边对不上就一定有一边写错了。

想细看的话：项目设计在 [`docs/项目书.md`](docs/项目书.md)，一页成果和四张图在
[`docs/成果整合.md`](docs/成果整合.md)，每个开关干什么在 [`docs/工程手册.md`](docs/工程手册.md)，
结论的证据强度分级在 [`docs/设计笔记/结论总览.md`](docs/设计笔记/结论总览.md)，
换电模型的完整形式在 [`docs/设计笔记/W1_换电模型设计.md`](docs/设计笔记/W1_换电模型设计.md)。

## 已知局限

1. 算例是自建的。底图坐标来自公开路网，需求、时间窗、禁飞区、车队规模都是按规则生成，种子固定。
   所以数字之间可以比，绝对金额不要拿去跟别人的结果对。
2. 精确模型里没有禁飞区，也没有悬停。绕行距离是时间相关的分段函数，线性化代价太高，我没做；
   悬停同理。所以「最优」这个说法只在悬停免费的前提下成立。
3. 最大只做到 100 客户，200+ 没测过，也不打算外推。
4. 上面那几个敏感性是分别测的，没做联合。叠起来是多少我不知道，拿单参数的结论去加总没有依据。

## 参考

模型形式与问题定义沿用该领域的主流建模，求解器、评估器、扩展和全部实验是本项目自己实现的。

- Zeng J., Hu Y., Pei M. (2026). Optimizing bidirectional delivery with multiple drones and
  trucks. *Transportation Research Part E*, 205, 104511.
- 底图数据：Murray C., Raj R. (2020). The multiple flying sidekicks traveling salesman problem.
  *Transportation Research Part C*, 110, 368–398.

不含第三方代码。论文 PDF 和出版方原图有版权，没有入库。

## English TL;DR

**TDDO** is a truck-drone delivery optimizer: given customers with time windows and time-dependent
circular no-fly zones, it schedules truck routes, drone sorties (where to launch and recover) and
battery logistics, minimizing electricity + fuel + driver hours.

The MILP has 79 constraints; it is solved at realistic scale by a custom ALNS, with Gurobi used to
certify small instances. On 16 instances × 10 seeds the plan is 15.9% cheaper than truck-only and
51.4% below the initial feasible solution; 100 customers take 124–187 s.

One modelling point: splitting the single recover-or-launch flag into separate arrival and departure
variables lowers the proven optimum on 4/4 small instances, by 1.5%–3.4% depending on swap duration,
battery stock and instance form. Details in `docs/`.

MIT License · 见 [`LICENSE`](LICENSE)
