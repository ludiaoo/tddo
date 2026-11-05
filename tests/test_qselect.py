"""DQN 选择器（M9.10）的守门测试。

守两条：
1. 状态构造函数在训练与推理之间共用，长度和取值范围必须稳定；
2. 加载的权重能给出**合法的 16 个动作之一**，且 ε=1 时完全是随机动作。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.io import PROJECT_ROOT
from src.qselect import ACTIONS, STATE_DIM, QNet, state_vector

Q_PATH = PROJECT_ROOT / "models" / "dqn" / "q.json"


def test_state_vector_shape_and_scale() -> None:
    s = state_vector(it=100, iters=2000, cur_over_best=0.12, no_improve=30, T=45.0,
                     n_routes=9, n_sorties=6, drone_served=12, n_customers=50,
                     nfz=4, restarts=2)
    assert len(s) == STATE_DIM
    assert s[0] == pytest.approx(0.05)
    assert 0.0 <= s[1] <= 1.0
    assert all(np.isfinite(s))
    # 确定性：同参数两次调用必须一致
    assert s == state_vector(it=100, iters=2000, cur_over_best=0.12, no_improve=30, T=45.0,
                             n_routes=9, n_sorties=6, drone_served=12, n_customers=50,
                             nfz=4, restarts=2)


@pytest.mark.skipif(not Q_PATH.exists(), reason="models/dqn/q.json 不存在（先跑 scripts.train_dqn）")
def test_qnet_returns_valid_action() -> None:
    net = QNet.load(Q_PATH)
    s = [0.1, 0.02, 1.0, -1.5, 0.9, 0.6, 0.25, 0.5, 0.8, 0.4]
    q = net.q_values(s)
    assert q.shape == (len(ACTIONS),)
    assert np.all(np.isfinite(q))
    assert net.pick(s) in ACTIONS


@pytest.mark.skipif(not Q_PATH.exists(), reason="models/dqn/q.json 不存在")
def test_greedy_pick_is_deterministic() -> None:
    net = QNet.load(Q_PATH)
    s = [0.3, 0.01, 0.5, -2.0, 1.2, 0.8, 0.3, 0.5, 0.6, 0.2]
    picks = {net.pick(s) for _ in range(20)}
    assert len(picks) == 1


@pytest.mark.skipif(not Q_PATH.exists(), reason="models/dqn/q.json 不存在")
def test_epsilon_one_is_uniform_random() -> None:
    net = QNet.load(Q_PATH)
    s = [0.5] * STATE_DIM
    rng = np.random.default_rng(0)
    picks = {net.pick(s, rng, eps=1.0) for _ in range(200)}
    assert len(picks) > 5, "ε=1 时应当覆盖多个动作"
    assert picks <= set(ACTIONS)
