"""预筛模型（M9.1）：两级筛选——线性可行性门 + LightGBM 精排。

为什么两级：v1 实测 LightGBM 每次调用的固定开销 ~0.1–0.3 ms，对一个修复步里的
几百个候选**逐个**跑两模型，开销比它省下来的评估还贵。改成：

1. **线性门**（LogisticRegression，`X@w+b` 一次矩阵乘，微秒级）先砍掉
   明显不可行的候选（阈值按「可行样本召回 ≥ 99%」标定，写进 meta）；
2. 幸存者（通常 ~10–30%）再过 LightGBM 的可行性 + Δcost 精排，取 top-m。

模型缺失 / lightgbm 未安装 → `available()` 返回 False，调用方自动退回全量评估。
"""

from __future__ import annotations

import json
from pathlib import Path

from .features import FEATURE_NAMES

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "screen_v1"
FEAS_THRESHOLD = 0.5


class LightGBMScreen:
    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR):
        import lightgbm as lgb

        self.dir = Path(model_dir)
        meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
        if tuple(meta["feature_names"]) != FEATURE_NAMES:
            raise ValueError("模型特征名与当前 features.FEATURE_NAMES 不一致，请重训")
        # 用 model_str 加载而非 model_file：LightGBM 的 C 层 fopen 在 Windows 上
        # 打不开含非 ASCII（本项目路径含中文）的文件名，文本由 Python 读写绕开
        self.feas = lgb.Booster(model_str=(self.dir / "feas.txt").read_text(encoding="utf-8"))
        self.delta = lgb.Booster(model_str=(self.dir / "delta.txt").read_text(encoding="utf-8"))
        self.threshold = float(meta.get("feas_threshold", FEAS_THRESHOLD))
        # 多进程实验里每个 worker 都带一个模型；LGBM 默认每进程开满线程，
        # 10 进程 × 20 线程 = 200 线程挤 20 核，纯同步开销。预测批很小，单线程更快。
        self.feas.params["num_threads"] = 1
        self.delta.params["num_threads"] = 1
        lin = meta.get("linear") or {}
        self.lin_w = lin.get("w")
        self.lin_b = float(lin.get("b", 0.0))
        self.lin_threshold = float(lin.get("threshold", 0.0))
        self.lin_mean = lin.get("mean")
        self.lin_std = lin.get("std")

    @staticmethod
    def available(model_dir: Path | str = DEFAULT_MODEL_DIR) -> bool:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            return False
        d = Path(model_dir)
        return (d / "meta.json").exists() and (d / "feas.txt").exists() and (d / "delta.txt").exists()

    def _linear_gate(self, X) -> list[int] | None:
        """返回通过线性门的下标；模型没有线性门时返回 None（不筛）。"""
        import numpy as np

        if not self.lin_w:
            return None
        w = np.asarray(self.lin_w)
        Z = (X - np.asarray(self.lin_mean)) / np.asarray(self.lin_std)
        p = 1.0 / (1.0 + np.exp(-(Z @ w + self.lin_b)))
        return list(np.nonzero(p >= self.lin_threshold)[0])

    def score(self, X) -> tuple:
        """返回 (p_feas, delta_hat) 两个等长数组。"""
        import numpy as np

        X = np.asarray(X, dtype=float)
        p = self.feas.predict(X)
        d = self.delta.predict(X)
        return p, d

    def select(self, X, m: int) -> list[int]:
        """从全体候选里挑出要做完整评估的下标（顺序即评估顺序）。"""
        import numpy as np

        n = len(X)
        if n <= m:
            return list(range(n))
        X = np.asarray(X, dtype=float)
        kept = self._linear_gate(X)
        if kept is not None and len(kept) <= m:
            return kept
        Xs = X[kept] if kept is not None else X
        idx_map = kept if kept is not None else list(range(n))
        p, d = self.score(Xs)
        order_feas = [j for j in np.argsort(-p) if p[j] >= self.threshold]
        order_delta = [j for j in np.argsort(d) if p[j] >= self.threshold]
        picked_local: list[int] = []
        seen: set[int] = set()
        for j in order_delta + order_feas:
            if j not in seen:
                seen.add(j)
                picked_local.append(int(j))
            if len(picked_local) >= m:
                break
        if len(picked_local) < m:                     # 过了门的太少，用 p 最高的补
            for j in np.argsort(-p):
                if j not in seen:
                    seen.add(j)
                    picked_local.append(int(j))
                if len(picked_local) >= m:
                    break
        return [idx_map[j] for j in picked_local]
