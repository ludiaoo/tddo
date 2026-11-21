"""天河真实路网算例（M9.23）的守门测试。

守四件事：
1. 底图能被 `load_tianhe_base` 读出来，维度、对角、连通性都对；
2. `build_instance(source="tianhe")` 走的是天河底图，不是 mFSTSP 原始算例；
3. 卡车矩阵是**有向**的（单行线），无人机距离与坐标一致；
4. 没装底图时给出明确报错，而不是静默回退到西雅图。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.generator import build_instance
from src.io import load_config, load_sets
from src.io_tianhe import list_tianhe_bases, load_tianhe_base

CFG = load_config()
TIANHE_SETS = [s for s in load_sets() if s.get("source") == "tianhe"]


def test_底图清单非空() -> None:
    assert "set01" in list_tianhe_bases()


@pytest.mark.parametrize("name", ["set01", "set05"])
def test_底图结构自洽(name: str) -> None:
    b = load_tianhe_base(name)
    n = len(b.nodes)
    assert b.city == "tianhe"
    assert b.truck_dist_m.shape == (n, n) == b.truck_time_s.shape
    assert np.allclose(np.diag(b.truck_dist_m), 0.0)
    assert np.isfinite(b.truck_dist_m).all() and np.isfinite(b.truck_time_s).all()
    assert len(b.parcel_kg) == n


def test_卡车矩阵有向() -> None:
    """天河有单行线：至少 10% 的 OD 对往返不同，否则说明矩阵被对称化了。"""
    b = load_tianhe_base("set01")
    d = b.truck_dist_m
    iu = np.triu_indices(len(d), 1)
    asym = np.abs(d[iu] - d.T[iu]) / np.maximum(d[iu], 1.0)
    assert np.median(asym) > 0.05, f"对称偏差中位 {np.median(asym):.1%}，像是无向矩阵"


def test_算例走天河底图() -> None:
    cfg_set = next(s for s in TIANHE_SETS if s["id"] == 101)
    inst = build_instance(cfg_set, CFG)
    assert inst.base.city == "tianhe"
    assert inst.base.name == "set01"
    # 坐标应落在天河区范围内（23.10–23.18 N, 113.32–113.40 E）
    lats = [nd.lat for nd in inst.base.nodes]
    lons = [nd.lon for nd in inst.base.nodes]
    assert 23.05 < min(lats) and max(lats) < 23.23
    assert 113.29 < min(lons) and max(lons) < 113.43


def test_无人机距离与坐标一致() -> None:
    """无人机用欧氏距离：抽查一对节点，必须等于平面坐标的距离。"""
    b = load_tianhe_base("set05")
    i, j = 0, 3
    xy = np.array([[nd.x, nd.y] for nd in b.nodes])
    assert b.drone_dist_m()[i, j] == pytest.approx(float(np.linalg.norm(xy[i] - xy[j])))


def test_缺底图时报错明确(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="build_tianhe"):
        load_tianhe_base("set99", data_dir=tmp_path)
