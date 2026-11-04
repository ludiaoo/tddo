"""M9 的守门测试：ML 模块关掉时必须与历史结果逐位一致。

- collect 模式：全量评估 + 记录，**决策与随机数消耗都不变** → 成本应与 off 完全相等；
- screen 模式在**模型缺失**时自动退回 off 行为 → 成本也应相等；
  （screen 模式 + 真模型的结果有意不同，不在这里测）
"""

from __future__ import annotations

import pytest

from src.alns import solve
from src.generator import build_instance
from src.io import load_config, load_sets
from src.ml import state
from src.operators import _SCREEN_TRIED, _get_screen

ITERS = 40


def _solve_set1(seed: int = 0, iters: int = ITERS) -> float:
    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}
    inst = build_instance(by_id[1], cfg)
    r = solve(inst, seed=seed, iters=iters)
    return round(float(r.best_cost), 6)


def test_off_is_baseline() -> None:
    c = _solve_set1()
    assert 80.0 < c < 130.0            # 与 results/bench_main.csv 的 Set1 同量级


def test_collect_mode_identical_to_off() -> None:
    off = _solve_set1()
    rows: list[dict] = []
    state.set_mode("collect", sink=rows.append, ctx={"set_id": 1, "seed": 0})
    try:
        col = _solve_set1()
    finally:
        state.set_mode("off")
    assert col == off
    assert len(rows) > 0                                    # 确实采集到了候选
    assert all(r.get("feasible") in (0, 1) for r in rows)   # 标签干净


def test_screen_without_model_falls_back_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.operators as ops

    monkeypatch.setattr(ops.LightGBMScreen, "available", staticmethod(lambda *a, **k: False))
    ops._SCREEN = None
    ops._SCREEN_TRIED = False
    try:
        assert _get_screen() is None
        off = _solve_set1()
        state.set_mode("screen")
        scr = _solve_set1()
    finally:
        state.set_mode("off")
        ops._SCREEN = None
        ops._SCREEN_TRIED = False
    assert scr == off


def test_screen_with_model_runs_and_prunes() -> None:
    """有模型时：能跑通、评估次数显著少于候选数（真模型存在才执行）。

    用 100 客户算例：预筛按 `ML_SCREEN_MIN_N` 分级启用，50 客户以下自动走原路径。
    """
    if not _get_screen():
        pytest.skip("models/screen_v1 不存在（先跑 scripts.train_screen）")
    from src.io import PROJECT_ROOT

    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}
    inst = build_instance(by_id[13], cfg)                  # 100 客户，候选池最大
    state.set_mode("screen")
    try:
        r = solve(inst, seed=0, iters=60)
        cnt = state.counters()                             # 先取计数：set_mode 会清零
    finally:
        state.set_mode("off")
    assert r.best_cost > 0
    assert cnt.get("attempts", 0) > cnt.get("evals", 0) > 0
    # 路径存在性顺带确认（模型目录随仓库走）
    assert (PROJECT_ROOT / "models" / "screen_v1" / "meta.json").exists()
