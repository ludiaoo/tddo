"""读取 mFSTSP 原始算例与项目配置。

⚠ 原始 CSV 的两个坑（实测）：
1. 表头整行以 `%` 开头，列名带前导/尾随空格，例如 `'% nodeID'`、`' parcelWtLbs '`
   → 必须统一清洗，否则按列名取值直接 KeyError。
2. 列名里带方括号和空格（`'time [sec]'`、`'distance [meters]'`）
   → 用位置索引取列更稳，本模块按位置读。

数据来源：https://github.com/optimatorlab/mFSTSP （Murray & Raj 2020）
本地路径：data/raw/mFSTSP/Problems/<ProblemName>/
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from .instance import LB_TO_KG, BaseProblem, Node, to_local_xy

# ---- 默认路径 ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "mFSTSP"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.toml"
DEFAULT_SETS = PROJECT_ROOT / "configs" / "instances.toml"


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: str(c).strip().lstrip("%").strip())


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def load_sets(path: Path | str = DEFAULT_SETS) -> list[dict]:
    with open(path, "rb") as fh:
        return tomllib.load(fh)["sets"]


def load_problems_index(data_root: Path | str = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    """problems_info.csv：100 个基础算例的索引。"""
    df = pd.read_csv(Path(data_root) / "problems_info.csv")
    return _clean_columns(df)


def load_locations(problem_dir: Path) -> tuple[list[Node], list[float]]:
    """读 tbl_locations.csv，返回节点列表（含局部平面坐标）与按原始 lb 的包裹重量。"""
    df = pd.read_csv(problem_dir / "tbl_locations.csv")
    df = _clean_columns(df)

    # 按位置取列，规避列名清洗不彻底的风险
    node_id = df.iloc[:, 0].to_numpy(dtype=int)
    node_type = df.iloc[:, 1].to_numpy(dtype=int)
    lat = df.iloc[:, 2].to_numpy(dtype=float)
    lon = df.iloc[:, 3].to_numpy(dtype=float)
    parcel_lb = df.iloc[:, 5].to_numpy(dtype=float).copy()

    # 校验：编号必须从 0 连续到 n
    if not np.array_equal(node_id, np.arange(len(node_id))):
        raise ValueError(f"{problem_dir.name}: nodeID 不是从 0 开始连续编号")
    if node_type[0] != 0:
        raise ValueError(f"{problem_dir.name}: 节点 0 必须是 depot(nodeType=0)")

    # 以 depot 为原点做局部平面投影
    lat0, lon0 = float(lat[0]), float(lon[0])
    nodes: list[Node] = []
    for k in range(len(node_id)):
        x, y = to_local_xy(float(lat[k]), float(lon[k]), lat0, lon0)
        nodes.append(
            Node(
                idx=int(node_id[k]),
                kind="depot" if node_type[k] == 0 else "customer",
                lat=float(lat[k]),
                lon=float(lon[k]),
                x=x,
                y=y,
            )
        )

    parcel_lb[0] = 0.0  # depot 原始值为 -1
    return nodes, [float(v) for v in parcel_lb]


def load_truck_matrix(problem_dir: Path, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """读 tbl_truck_travel_data_PG.csv，返回 (distance[m], time[s]) 两个 (n,n) 矩阵。

    这是 **OSM 路网** 的有向矩阵（pgRouting 生成），不是欧氏距离。
    """
    df = pd.read_csv(problem_dir / "tbl_truck_travel_data_PG.csv")
    if len(df) != n_nodes * n_nodes:
        raise ValueError(
            f"{problem_dir.name}: 卡车矩阵行数 {len(df)} != n^2 = {n_nodes ** 2}"
        )
    frm = df.iloc[:, 0].to_numpy(dtype=int)
    to = df.iloc[:, 1].to_numpy(dtype=int)
    time_s = df.iloc[:, 2].to_numpy(dtype=float)
    dist_m = df.iloc[:, 3].to_numpy(dtype=float)

    dist = np.full((n_nodes, n_nodes), np.nan)
    tim = np.full((n_nodes, n_nodes), np.nan)
    dist[frm, to] = dist_m
    tim[frm, to] = time_s

    if np.isnan(dist).any():
        missing = int(np.isnan(dist).sum())
        raise ValueError(f"{problem_dir.name}: 卡车矩阵缺 {missing} 个 OD 对")

    np.fill_diagonal(dist, 0.0)
    np.fill_diagonal(tim, 0.0)
    return dist, tim


def load_base_problem(
    problem_name: str, city: str, data_root: Path | str = DEFAULT_DATA_ROOT
) -> BaseProblem:
    """读一个 mFSTSP 算例的底图（尚未注入取件/时间窗/禁飞区）。"""
    problem_dir = Path(data_root) / "Problems" / problem_name
    if not problem_dir.is_dir():
        raise FileNotFoundError(f"算例目录不存在：{problem_dir}")

    nodes, parcel_lb = load_locations(problem_dir)
    n_nodes = len(nodes)
    truck_dist, truck_time = load_truck_matrix(problem_dir, n_nodes)

    return BaseProblem(
        name=problem_name,
        city=city,
        nodes=tuple(nodes),
        truck_dist_m=truck_dist,
        truck_time_s=truck_time,
        parcel_kg=tuple(v * LB_TO_KG for v in parcel_lb),
    )


def dump_json(obj, path: Path | str) -> None:
    """把算例写成 JSON（生成器用）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
