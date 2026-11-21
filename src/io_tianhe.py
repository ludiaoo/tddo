"""广州天河真实路网算例的装载器。

与 `src.io` 的关系：那边读 mFSTSP 原始文件（西雅图/布法罗的 OSM 矩阵），这边读
`data/tianhe/base_<name>.json`——由 `scripts/build_tianhe.py` 用天河区 OSM 路网现算出来的
距离/时间矩阵。两者产出同一个 `BaseProblem`，所以下游（生成器、评估器、MILP、ALNS）
**一行都不用改**。

口径（与西雅图算例的差别都在这里）：

- 卡车距离：天河区路网最短路（Dijkstra），单位米，**有向**（单行线、掉头都按 OSM 的 oneway）；
- 卡车时间：同一张最短路，按道路等级限速折算（见 `build_tianhe.SPEED_KMH`），不含实时路况；
- 无人机：仍是节点平面坐标的欧氏距离，禁飞区绕行几何不变——只是坐标换到了天河；
- 节点坐标：从原算例**等比缩放 + 平移**得到，相对几何保持不变（见 `build_tianhe.TRANSFORM`）。

这些差别写进 `data/tianhe/base_<name>.json` 的 `meta` 字段，便于复算时核对。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .instance import BaseProblem, Node

TIANHE_DIR = Path(__file__).resolve().parent.parent / "data" / "tianhe"


def list_tianhe_bases() -> list[str]:
    if not TIANHE_DIR.is_dir():
        return []
    return sorted(p.stem.removeprefix("base_") for p in TIANHE_DIR.glob("base_*.json"))


def load_tianhe_base(name: str, data_dir: Path | str | None = None) -> BaseProblem:
    """读一个天河底图。`name` 形如 `set01`（与原算例编号对应）。"""
    root = Path(data_dir) if data_dir is not None else TIANHE_DIR
    path = root / f"base_{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"天河底图不存在：{path}\n"
            f"先运行 `python -m scripts.build_tianhe build`（需要先 fetch 路网）"
        )
    raw = json.loads(path.read_text(encoding="utf-8"))

    nodes = tuple(
        Node(idx=int(n["id"]), kind=n["kind"], lat=float(n["lat"]), lon=float(n["lon"]),
             x=float(n["x_m"]), y=float(n["y_m"]))
        for n in raw["nodes"]
    )
    dist = np.asarray(raw["truck"]["dist_m"], dtype=float)
    tim = np.asarray(raw["truck"]["time_s"], dtype=float)
    if dist.shape != (len(nodes), len(nodes)) or tim.shape != dist.shape:
        raise ValueError(f"{path.name}: 矩阵维度与节点数不符 {dist.shape} vs {len(nodes)}")
    if not np.isfinite(dist).all() or not np.isfinite(tim).all():
        raise ValueError(f"{path.name}: 矩阵含非有限值（存在不可达 OD，需先在构建阶段剔除）")

    return BaseProblem(
        name=raw.get("name", name),
        city=raw.get("city", "tianhe"),
        nodes=nodes,
        truck_dist_m=dist,
        truck_time_s=tim,
        parcel_kg=tuple(float(v) for v in raw["parcel_kg"]),
    )
