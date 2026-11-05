"""DQN 式算子选择器（M9.10）：状态构造 + 前向推理 + ε-贪心。

## 设计（对应论文里的 MDP）

| 元素 | 设计 | 理由 |
|---|---|---|
| **状态** `s` | 10 维，全部是**决策时刻已知**的量：进度、相对最优、停滞计数、温度、路线数、架次数、无人机服务占比、客户数、禁飞区数、重启数 | `destroyed_frac` / `status` / `feasible` 是**动作之后**才知道的，不能进状态（否则训练时泄漏） |
| **动作** `a` | 16 = 4 破坏 × 4 修复 | 与现有轮盘赌的动作空间完全一致，只换选择规则，便于归因 |
| **奖励** `r` | `(cur_cost_prev − cur_cost)/best_cost` | 无量纲、跨算例可比；等于「这一步让当前解降了多少」。被拒绝的动作自然得 0，接受劣解得负 |
| **折扣** `γ` | 0.95（有效视界 ~20 步） | 太短退化成贪心（M9.1.5 已证伪），太长则方差爆炸 |
| **训练** | 离线 Fitted-Q Iteration（Double-DQN 式目标），MLP 10+16→64→64→1 | 环境是一整次 ALNS 求解（40–190 s），在线 DQN 要上万 episode，离线批量拟合是本项目唯一可行的形式 |

## 为什么把动作编成 one-hot 拼进输入

sklearn 的 `MLPRegressor` 做多输出时要求每个动作都有标签，而离线数据里每一步只经历了一个动作。
把动作 one-hot 拼进输入、输出单值 Q，就是标准的「拟合 Q 迭代」写法。

权重从 sklearn 导出成 JSON，推理用 numpy 手写前向（不引入 torch）。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ACTIONS: tuple[str, ...] = (
    "random|random", "random|truck", "random|drone", "random|equal",
    "worst|random", "worst|truck", "worst|drone", "worst|equal",
    "related|random", "related|truck", "related|drone", "related|equal",
    "shuffle|random", "shuffle|truck", "shuffle|drone", "shuffle|equal",
)
A_INDEX = {a: i for i, a in enumerate(ACTIONS)}

STATE_DIM = 10
T_REF = 300.0          # 与日志中 log_T_over_t0 的口径一致


def state_vector(*, it: int, iters: int, cur_over_best: float, no_improve: int,
                 T: float, n_routes: int, n_sorties: int, drone_served: int,
                 n_customers: int, nfz: int, restarts: int) -> list[float]:
    """决策时刻的状态向量。训练与推理**共用这一个函数**，避免训练/上线特征错位。"""
    return [
        it / max(iters, 1),                                   # 进度
        float(min(max(cur_over_best, 0.0), 1.0)),             # 当前解相对最优的劣化
        np.log1p(max(no_improve, 0)),                         # 停滞程度（对数压缩）
        float(np.log(max(T, 1e-9) / T_REF)),                  # 温度（相对参考尺度）
        n_routes / 10.0,
        n_sorties / 10.0,
        drone_served / max(n_customers, 1),                   # 无人机服务占比
        n_customers / 100.0,
        nfz / 5.0,
        restarts / 5.0,
    ]


class QNet:
    """从 sklearn 导出的 MLP，纯 numpy 前向。"""

    def __init__(self, coefs: list, intercepts: list):
        self.coefs = [np.asarray(c, dtype=float) for c in coefs]
        self.intercepts = [np.asarray(b, dtype=float) for b in intercepts]

    @classmethod
    def load(cls, path: str | Path) -> "QNet":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["coefs"], data["intercepts"])

    def q_values(self, state: list[float]) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        out = np.empty(len(ACTIONS), dtype=float)
        for i in range(len(ACTIONS)):
            h = np.concatenate([x, np.eye(len(ACTIONS))[i]])
            for j, (w, b) in enumerate(zip(self.coefs, self.intercepts)):
                h = h @ w + b
                if j < len(self.coefs) - 1:            # 隐藏层用 relu，输出层线性
                    h = np.maximum(h, 0.0)
            out[i] = float(h[0])
        return out

    def pick(self, state: list[float], rng=None, eps: float = 0.0) -> str:
        if eps > 0 and rng is not None and rng.random() < eps:
            return ACTIONS[int(rng.integers(len(ACTIONS)))]
        return ACTIONS[int(np.argmax(self.q_values(state)))]
