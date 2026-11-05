"""训练 DQN 式算子选择器（M9.10）：离线 Fitted-Q Iteration。

数据来自 M9.1.5 的「均匀随机算子」日志 `results/ml/oplog.csv.gz`
（6 算例 × 4 种子 × 1500 轮 = 36 000 条决策）。用均匀策略采数据是有意的：
offline Q-learning 需要每个动作都有覆盖，否则学到的 Q 只反映被选过的动作。

⚠ 时间对齐：日志的 trace 是**每步执行完之后**记录的，所以第 t 条的 `action` 是第 t 步的动作，
但它的 `cur_cost`/`n_routes`/`T` 已经是第 t 步之后的状态。
因此转移必须是 `(s = 第 t−1 条的状态, a = 第 t 条的动作, r = 成本变化, s' = 第 t 条的状态)`。

    python -m scripts.train_dqn --in results/ml/oplog.csv.gz --rounds 5 --gamma 0.95
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from src.io import PROJECT_ROOT
from src.qselect import ACTIONS, A_INDEX, STATE_DIM, state_vector

OUT_DIR = PROJECT_ROOT / "models" / "dqn"
ITERS_IN_LOG = 1500          # 日志生成时用的轮数（M9.1.5）


def build_transitions(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """→ (states, actions, rewards, next_states)，并丢掉每条轨迹的第一个记录。"""
    S, A, R, S2 = [], [], [], []
    for _, g in df.groupby(["set", "seed"], sort=False):
        g = g.sort_values("it")
        rec = g.to_dict("records")
        for t in range(1, len(rec)):
            prev, cur = rec[t - 1], rec[t]
            if cur["action"] not in A_INDEX:
                continue
            s = state_vector(
                it=prev["it"], iters=ITERS_IN_LOG, cur_over_best=prev["cur_over_best"],
                no_improve=int(prev["no_improve"]), T=prev["T"], n_routes=int(prev["n_routes"]),
                n_sorties=int(prev["n_sorties"]), drone_served=int(prev["drone_served"]),
                n_customers=int(prev["n_customers"]), nfz=int(prev["nfz_count"]),
                restarts=int(prev["restarts"]),
            )
            s2 = state_vector(
                it=cur["it"], iters=ITERS_IN_LOG, cur_over_best=cur["cur_over_best"],
                no_improve=int(cur["no_improve"]), T=cur["T"], n_routes=int(cur["n_routes"]),
                n_sorties=int(cur["n_sorties"]), drone_served=int(cur["drone_served"]),
                n_customers=int(cur["n_customers"]), nfz=int(cur["nfz_count"]),
                restarts=int(cur["restarts"]),
            )
            r = (prev["cur_cost"] - cur["cur_cost"]) / max(cur["best_cost"], 1e-9)
            S.append(s); A.append(A_INDEX[cur["action"]]); R.append(r); S2.append(s2)
    return np.array(S), np.array(A), np.array(R), np.array(S2)


def onehot_actions(a: np.ndarray) -> np.ndarray:
    return np.eye(len(ACTIONS))[a]


def fit_q(S, A, targets, rounds_hint: int = 400):
    X = np.hstack([S, onehot_actions(A)])
    net = MLPRegressor(hidden_layer_sizes=(64, 64), activation="relu",
                       max_iter=rounds_hint, random_state=0, early_stopping=False)
    net.fit(X, targets)
    return net


def q_all(net, S: np.ndarray) -> np.ndarray:
    """对每个状态给出 16 个动作的 Q（把每个动作的 one-hot 拼进去分别前向）。"""
    n = len(S)
    out = np.empty((n, len(ACTIONS)))
    for k in range(len(ACTIONS)):
        out[:, k] = net.predict(np.hstack([S, np.tile(np.eye(len(ACTIONS))[k], (n, 1))]))
    return out


def evaluate(net, S, A, R, S2, gamma: float) -> dict:
    q = q_all(net, S)
    q2 = q_all(net, S2)
    target = R + gamma * q2.max(axis=1)
    td = target - q[np.arange(len(A)), A]
    chosen = q.argmax(axis=1)
    return {
        "td_rmse": float(np.sqrt((td ** 2).mean())),
        "policy_matches_behavior": float((chosen == A).mean()),
        "policy_action_counts": {ACTIONS[i]: int(c) for i, c in
                                 enumerate(np.bincount(chosen, minlength=len(ACTIONS)))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results/ml/oplog.csv.gz")
    ap.add_argument("--rounds", type=int, default=5, help="Fitted-Q 迭代轮数")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--holdout-seed", type=int, default=3, help="留出的种子（用于离线评估）")
    args = ap.parse_args()

    df = pd.read_csv(PROJECT_ROOT / args.inp)
    S, A, R, S2 = build_transitions(df)
    print(f"转移样本 {len(S)} 条（{df['set'].nunique()} 算例 × {df['seed'].nunique()} 种子），"
          f"状态 {S.shape[1]} 维，动作 {len(ACTIONS)} 个")

    # 留出：按种子切，避免同一条轨迹的相邻样本泄漏
    seed_of = []
    for _, g in df.groupby(["set", "seed"], sort=False):
        seed_of.extend([int(g["seed"].iloc[0])] * (len(g) - 1))
    seed_of = np.array(seed_of[: len(S)])
    tr, te = seed_of != args.holdout_seed, seed_of == args.holdout_seed
    print(f"训练 {tr.sum()} / 留出 {te.sum()}（seed={args.holdout_seed}）")

    # --- 留出模型：只用于报告离线指标 ---
    net = None
    for k in range(args.rounds):
        if net is None:
            targets = R[tr]
        else:
            q2 = q_all(net, S2[tr])
            targets = R[tr] + args.gamma * q2.max(axis=1)
        net = fit_q(S[tr], A[tr], targets)
        m = evaluate(net, S[te], A[te], R[te], S2[te], args.gamma)
        print(f"  第 {k + 1} 轮：留出 TD-RMSE {m['td_rmse']:.5f}，"
              f"与行为策略一致率 {m['policy_matches_behavior']:.1%}")
    holdout_metrics = m

    # --- 全量模型：上线用 ---
    print("全量重训（上线模型）…")
    full = None
    for k in range(args.rounds):
        if full is None:
            targets = R
        else:
            q2 = q_all(full, S2)
            targets = R + args.gamma * q2.max(axis=1)
        full = fit_q(S, A, targets)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "actions": list(ACTIONS),
        "state_dim": STATE_DIM,
        "gamma": args.gamma,
        "rounds": args.rounds,
        "n_train": int(tr.sum()),
        "n_holdout": int(te.sum()),
        "holdout": {k: v for k, v in holdout_metrics.items() if k != "policy_action_counts"},
        "holdout_policy_actions": holdout_metrics["policy_action_counts"],
        "coefs": [c.tolist() for c in full.coefs_],
        "intercepts": [b.tolist() for b in full.intercepts_],
    }
    (OUT_DIR / "q.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"写出 {OUT_DIR / 'q.json'}")
    print("\n留出集上学到的策略偏好（前 5）：")
    top = sorted(holdout_metrics["policy_action_counts"].items(), key=lambda kv: -kv[1])[:5]
    for name, cnt in top:
        print(f"  {name:<16} {cnt}")


if __name__ == "__main__":
    main()
