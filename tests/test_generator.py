"""算例构造的测试。

最重要的一个是 `test_金标准算例未被改动`：它把 `data/instances/*.json` 当作金标准，
任何人改了 `generator.py` 或 `config.toml`，测试会立刻失败并指出第一处差异——
这就是「算例被悄悄改动」那个坑的守门人。
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.generator import build_instance, instance_to_dict
from src.instance import LB_TO_KG
from src.io import PROJECT_ROOT, load_config, load_sets

INSTANCE_DIR = PROJECT_ROOT / "data" / "instances"
CFG = load_config()
SETS = load_sets()


def _diff(a, b, path="$", out=None, limit=4):
    """递归找出两处 JSON 结构的第一批差异，便于定位。"""
    if out is None:
        out = []
    if len(out) >= limit:
        return out
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        out.append(f"{path}: 类型不同 {type(a).__name__} vs {type(b).__name__}")
        return out
    if isinstance(a, dict):
        for k in a.keys() | b.keys():
            if k not in a or k not in b:
                out.append(f"{path}.{k}: 键缺失")
                if len(out) >= limit:
                    return out
                continue
            _diff(a[k], b[k], f"{path}.{k}", out, limit)
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: 长度 {len(a)} vs {len(b)}")
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out, limit)
            if len(out) >= limit:
                return out
        return out
    if isinstance(a, float) or isinstance(b, float):
        if abs(float(a) - float(b)) > 1e-9:
            out.append(f"{path}: {a} vs {b}")
    elif a != b:
        out.append(f"{path}: {a} vs {b}")
    return out


@pytest.mark.parametrize("set_cfg", SETS, ids=lambda s: f"set{s['id']:02d}")
def test_金标准算例未被改动(set_cfg):
    """本地重算的算例必须与仓库里的 JSON 完全一致。"""
    path = INSTANCE_DIR / f"set{set_cfg['id']:02d}.json"
    assert path.exists(), f"缺少金标准文件 {path}，请运行 python -m src.generator"

    with open(path, encoding="utf-8") as fh:
        golden = json.load(fh)
    regenerated = instance_to_dict(build_instance(set_cfg, CFG))

    diffs = _diff(golden, regenerated)
    assert not diffs, "算例与金标准不一致（改了 generator/config 就要重新生成并确认）：\n" + "\n".join(diffs)


def test_构造可重复():
    """同一配置跑两次必须完全一致（种子固定）。"""
    a = instance_to_dict(build_instance(SETS[0], CFG))
    b = instance_to_dict(build_instance(SETS[0], CFG))
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.mark.parametrize("set_cfg", SETS, ids=lambda s: f"set{s['id']:02d}")
def test_需求规则(set_cfg):
    inst = build_instance(set_cfg, CFG)
    step_kg = CFG["generation"]["demand_rounding_lb"] * LB_TO_KG

    assert inst.f[0] == 0.0 and inst.g[0] == 0.0
    assert (inst.f[1:] > 0).all(), "每个客户都应有派件需求"
    assert np.all(inst.g >= 0.0)
    assert np.all(inst.g <= inst.f + 1e-9), "取件不应超过该客户的派件量（g_i ~ U(0, f_i)）"

    ratio = inst.g[1:] / inst.f[1:]
    assert ratio.max() <= 1.0 + 1e-9
    # 需求是 0.5 lb 的整数倍
    assert np.allclose(inst.g / step_kg, np.round(inst.g / step_kg), atol=1e-6)


@pytest.mark.parametrize("set_cfg", SETS, ids=lambda s: f"set{s['id']:02d}")
def test_时间窗规则(set_cfg):
    inst = build_instance(set_cfg, CFG)
    gen = CFG["generation"]
    w_lo, w_hi = gen["time_window_width_min"]
    day = gen["day_length_h"] * 3600.0

    for i, w in enumerate(inst.tw):
        assert w.l > w.e, f"节点 {i + 1} 时间窗倒挂"
        assert w_lo * 60.0 - 1e-6 <= w.width() <= w_hi * 60.0 + 1e-6
        assert w.e >= 0.0
        assert w.l <= day, f"节点 {i + 1} 的窗口越过班次"


@pytest.mark.parametrize("set_cfg", SETS, ids=lambda s: f"set{s['id']:02d}")
def test_禁飞区规则(set_cfg):
    inst = build_instance(set_cfg, CFG)
    gen = CFG["generation"]

    # 数量对齐论文 Fig.8 的「多而小」：clamp(round(n/8), min, max)
    expect_n = round(inst.n_customers / gen["nfz_per_customers"])
    expect_n = min(max(expect_n, gen["nfz_count_min"]), gen["nfz_count_max"])
    assert len(inst.nfz) == expect_n, (
        f"禁飞区数量 {len(inst.nfz)} != 期望 {expect_n}（论文 Fig.8 里 50 客户约 6 个、100 客户约 6 个）"
    )

    # 半径必须是几百米量级，不能出现覆盖整簇的大圆
    r_lo, r_hi = gen["nfz_radius_m"]
    depot_x, depot_y = inst.base.nodes[0].x, inst.base.nodes[0].y
    for z in inst.nfz:
        assert 150.0 <= z.r <= r_hi + 1e-6, f"禁飞区半径 {z.r:.0f} m 超出量级"
        assert z.t_end > z.t_start
        d = float(np.hypot(z.cx - depot_x, z.cy - depot_y))
        assert d > z.r + 400.0, "禁飞区不应罩住仓库"

    # 回归：禁飞时段必须与时间窗有交集，否则绕行永远不触发（曾踩过这个坑）
    e_min = min(w.e for w in inst.tw)
    l_max = max(w.l for w in inst.tw)
    assert any(not (z.t_end < e_min or z.t_start > l_max) for z in inst.nfz), (
        "禁飞时段与客户时间窗完全不重叠 → 绕行逻辑不会触发"
    )

    # 至少要有客户落在禁飞区内，else γ≡0，Fig.3 的「不能起降」分支永远走不到
    xy = np.array([[nd.x, nd.y] for nd in inst.base.nodes[1:]])
    inside = sum(int((np.hypot(xy[:, 0] - z.cx, xy[:, 1] - z.cy) <= z.r).sum()) for z in inst.nfz)
    assert inside > 0, "没有任何客户位于禁飞区内，时间维度的禁飞判定无法被触发"


@pytest.mark.parametrize("set_cfg", SETS, ids=lambda s: f"set{s['id']:02d}")
def test_车队规模按容量下界(set_cfg):
    inst = build_instance(set_cfg, CFG)
    fs = CFG["fleet_size"]
    expect = int(np.ceil(inst.f.sum() * fs["capacity_slack"] / inst.params.truck_capacity))
    expect = min(max(expect, fs["min_trucks"]), fs["max_trucks"])
    assert inst.n_trucks == expect
    assert inst.f.sum() <= inst.n_trucks * inst.params.truck_capacity + 1e-6


def test_整体规模与论文对齐():
    """节点规模必须是 10/25/50/100，各 4 个，对应 Table 3 的 Set 1–16。"""
    counts = [s["n_customers"] for s in SETS]
    assert counts == [10] * 4 + [25] * 4 + [50] * 4 + [100] * 4
    assert [s["id"] for s in SETS] == list(range(1, 17))
    assert len({s["problems"][0] for s in SETS}) == 16, "16 个 Set 不应重复使用同一原始算例"
