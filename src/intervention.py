"""周期性有界干预：把「LLM 当 ALNS 运行时控制器」的动作空间抽出来，换成 random / rule 来决策。

**为什么需要这个模块**

文献里 LLM 控制 ALNS 的报告增益，有一个绕不开的对照：Raoofi et al. (2026, *Systems* 14(10):1200)
自己给出的表里，「随机监督」的平均 gap 是 1.57%，已经比自适应 ALNS 的 2.93% 好 1.36 pp，
而「规则式监督」是 1.39%、LLM 是 0.18%。也就是说，**大头来自「周期性有界干预」这件事本身，
而不是来自决策者的智能**。

所以在本项目接 LLM 之前，必须先量清楚这个基线值多少：

| 臂 | 决策者 | 用途 |
|---|---|---|
| `none` | 无干预（现状） | 基线 |
| `random` | 从同一个有界动作空间均匀采样 | **本实验的核心对照** |
| `rule` | 手工三段式调度（早强化 / 中均衡 / 后逃逸） | 「聪明一点但不学习」的上限近似 |

三条臂共用同一个动作空间与同一个确定性校验器（`validate`），只有决策者换人——
这样「LLM 值多少」才可被单独归因。

**动作空间**（对齐论文的有界参数调整，不是让模型直接改解）

- `q`：本段破坏比例的目标值，裁剪到 `[0.05, 0.40]`
- `temp_mult`：温度乘子，裁剪到 `[0.75, 1.35]`
- `bias_d` / `bias_r`：每个破坏 / 修复算子的权重偏置，裁剪到 `[-0.15, +0.15]`，
  施加后按均值归一回 1.0（轮盘赌只看相对值，所以归一是无损的）

每次干预在**下一段**（`interv_every` 轮）内持续生效，与论文「每 L 轮读一次遥测、动一次参数」同构。
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .alns import DESTROY, REPAIR

Q_RANGE = (0.05, 0.40)
TEMP_RANGE = (0.75, 1.35)
BIAS_LIM = 0.15


def validate(act: dict | None) -> dict | None:
    """确定性校验器：裁剪越界值、丢弃非法字段。与论文里那道「有界监督」同构。

    任何试图直接改解的字段一律忽略——这个函数只认下面四个键。
    """
    if not act:
        return None
    out: dict = {
        "q": None,
        "temp_mult": 1.0,
        "bias_d": {},
        "bias_r": {},
    }
    if act.get("q") is not None:
        out["q"] = float(np.clip(float(act["q"]), *Q_RANGE))
    if act.get("temp_mult") is not None:
        out["temp_mult"] = float(np.clip(float(act["temp_mult"]), *TEMP_RANGE))
    for key, valid in (("bias_d", DESTROY), ("bias_r", REPAIR)):
        for k, v in (act.get(key) or {}).items():
            if k in valid:
                out[key][k] = float(np.clip(float(v), -BIAS_LIM, BIAS_LIM))
    return out


def random_policy(seed: int) -> Callable[[dict], dict]:
    """从有界动作空间**独立均匀**采样——零智能，本实验的核心对照。"""
    rng = np.random.default_rng(seed)

    def policy(state: dict) -> dict:
        return validate({
            "q": float(rng.uniform(*Q_RANGE)),
            "temp_mult": float(rng.uniform(*TEMP_RANGE)),
            "bias_d": {k: float(rng.uniform(-BIAS_LIM, BIAS_LIM)) for k in DESTROY},
            "bias_r": {k: float(rng.uniform(-BIAS_LIM, BIAS_LIM)) for k in REPAIR},
        })

    return policy


def rule_policy() -> Callable[[dict], dict]:
    """手工三段式调度：早段强化、中段均衡、后段逃逸。

    这是「不学习、但懂这个搜索」的近似上限。若 `random` 与它打平，
    说明这类干预的收益不来自对状态的理解，而来自「定期动一动」。
    """

    def policy(state: dict) -> dict:
        p = float(state.get("progress", 0.0))
        if p < 0.30:                                    # 早段：小步、降温、偏向高效修复
            act = {"q": 0.12, "temp_mult": 0.85,
                   "bias_d": {"worst": 0.05},
                   "bias_r": {"equal": 0.12}}
        elif p < 0.65:                                  # 中段：均衡，不动
            act = {"q": 0.25, "temp_mult": 1.00}
        else:                                           # 后段：大破坏 + 加热，尝试逃逸
            act = {"q": 0.38, "temp_mult": 1.25,
                   "bias_d": {"random": 0.12, "shuffle": 0.12},
                   "bias_r": {"equal": 0.05}}
        return validate(act)

    return policy


def make_policy(kind: str, seed: int = 0) -> Callable[[dict], dict] | None:
    if kind in (None, "", "none", "off"):
        return None
    if kind == "random":
        return random_policy(seed)
    if kind == "rule":
        return rule_policy()
    raise ValueError(f"未知的干预策略：{kind}（可选 none / random / rule）")
