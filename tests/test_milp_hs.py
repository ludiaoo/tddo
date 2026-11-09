"""W1 的等价性与有效性闸门。

`src/milp_hs.py` 是"换电模型"（按到达/出发拆分变量）。它必须同时满足两件事：

1. **等价性**：加回 `wl + wr ≤ 1` 后，最优值必须与 `milp.py`（原模型）逐位相同。
   不满足就说明拆分写错了——这是本模块唯一可信的护栏。
2. **有效性**：去掉该约束后，最优值必须严格更低，且解在评估器里可行。
   不满足就说明"放松"没有真正生效（M9.16 用 `milp.py` 试过这条，失败）。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.alns import disable_nfz
from src.evaluator import evaluate
from src.generator import build_instance
from src.io import load_config, load_sets
from src.milp_hs import solve_milp_hs

pytestmark = pytest.mark.slow

CFG = load_config()
SETS = load_sets()

# 原模型（milp.py）在这两个算例上的已证明最优值，见 results/milp_check.csv
REF = {3: 67.7647, 4: 73.3771}


def _inst(sid: int, hotswap: bool = False):
    by_id = {s["id"]: s for s in load_sets()}
    inst = disable_nfz(build_instance(by_id[sid], CFG))
    if hotswap:
        inst = replace(inst, params=replace(inst.params, allow_hotswap=True))
    return inst


@pytest.mark.parametrize("sid", [3, 4])
def test_限制换电时与原模型等价(sid: int) -> None:
    r = solve_milp_hs(_inst(sid), time_limit=180.0, allow_hotswap=False)
    assert r.solution is not None, f"Set{sid} 无解"
    assert r.status == "optimal", f"Set{sid} 未证最优（{r.status}）"
    assert r.cost == pytest.approx(REF[sid], abs=1e-4), (
        f"Set{sid} 新模型 {r.cost:.4f} ≠ 原模型 {REF[sid]:.4f} → 拆分写错了"
    )


@pytest.mark.parametrize("sid", [3, 4])
def test_放开换电后最优值更低且解可行(sid: int) -> None:
    inst = _inst(sid, hotswap=True)
    r = solve_milp_hs(inst, time_limit=240.0, allow_hotswap=True)
    assert r.solution is not None, f"Set{sid} 换电模型无解"
    assert r.cost < REF[sid] - 1e-4, f"Set{sid} 换电未降低最优值（{r.cost:.4f}）"

    res, cost = evaluate(inst, r.solution)
    assert res.feasible, f"Set{sid} 换电解在评估器里不可行：{res.all_violations()[:3]}"
    assert cost == pytest.approx(r.cost, rel=1e-6), "MILP 目标值与评估器成本不一致"
