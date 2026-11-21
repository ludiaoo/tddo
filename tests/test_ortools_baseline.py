"""外部基线守门：OR-Tools 目标口径必须与项目评估器一致。

OR-Tools 是可选依赖（见 requirements-dev.txt），未安装时整文件跳过。
本测试只跑 10 客户算例、3 秒限时，避免拖慢快速测试。
"""

from __future__ import annotations

import pytest

pytest.importorskip("ortools", reason="OR-Tools 是外部基线可选依赖")

from scripts.ortools_baseline import SCALE, _solve  # noqa: E402
from src.generator import build_instance  # noqa: E402
from src.io import load_config, load_sets  # noqa: E402


def test_ortools_objective_matches_project_cost() -> None:
    cfg = load_config()
    set_map = {s["id"]: s for s in load_sets()}
    inst = build_instance(set_map[1], cfg)
    day_h = float(cfg["generation"]["day_length_h"])

    res = _solve(inst, time_limit_s=3, day_h=day_h)

    assert res["status"] == "ok"
    assert res["objective_cost"] is not None
    assert res["recomputed_cost"] is not None
    assert res["abs_diff"] < 0.01
    # 10 客户小算例：两小时后应与同口径 ALNS 纯卡车结果同量级（~126），
    # 这里只守数量级，不守与 ALNS 的精确差（OR-Tools 限时搜索有波动）。
    assert 100.0 < res["objective_cost"] < 160.0
    assert SCALE == 1_000_000
