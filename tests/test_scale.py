"""规模压测的守门测试：只查合成算例构造与口径声明，不跑长实验。"""

from __future__ import annotations

import numpy as np

from scripts.scale_bench import build_scaled_base
from src.io import PROJECT_ROOT, load_config, load_sets


def _source() -> tuple[str, str]:
    sets = {int(s["id"]): s for s in load_sets()}
    ref = sets[13]
    return str(ref["problems"][0]), str(ref["city"])


def test_合成算例节点数与仓库() -> None:
    problem, city = _source()
    base = build_scaled_base(200, problem, city)
    assert base.n_customers == 200
    assert base.n_nodes == 201
    assert base.nodes[0].kind == "depot"
    assert (base.nodes[0].x, base.nodes[0].y) == (0.0, 0.0)
    assert all(n.kind == "customer" for n in base.nodes[1:])


def test_合成矩阵为对称欧氏尺度且确定性() -> None:
    problem, city = _source()
    a = build_scaled_base(200, problem, city)
    b = build_scaled_base(200, problem, city)
    np.testing.assert_array_equal(a.truck_dist_m, b.truck_dist_m)
    np.testing.assert_allclose(a.truck_dist_m, a.truck_dist_m.T, atol=1e-9)
    assert np.allclose(np.diag(a.truck_dist_m), 0.0)
    assert np.all(a.truck_time_s >= 0.0)


def test_规模结果写明工程指标边界() -> None:
    text = (PROJECT_ROOT / "results" / "scale_bench.md").read_text(encoding="utf-8")
    assert "工程指标" in text
    assert "不作" in text
    assert "合成" in text
