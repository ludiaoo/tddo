"""高德「距离测量」的最小封装：GCJ-02 转换 + 批量 OD 查询。

只做两件事，够用就好：

1. `wgs84_to_gcj02`：OSM 用的是 WGS-84，高德全线 GCJ-02。不转换的话市区偏差 300–500 m，
   查出来的是隔壁街区，但请求照样 `status=1`，不会报错——这是最容易踩的坑。
2. `distance_matrix`：n 个点 → n×n 有向（距离, 时间）矩阵。高德 `v3/distance` 一次最多
   100 个起点、1 个终点，所以一个矩阵要 n 次请求（n>100 时分批）。

**关于可复现性**：`type=1` 的驾车距离含实时路况，官方文档明确写不同时间结果会不同。
因此本模块只用于交叉校验或“某时刻快照”，不作为可复现的默认数据源；默认仍是
OSM + 本地 Dijkstra（见 `scripts/build_tianhe.py`）。

**Key 从哪来**：项目根目录的 `.env`（见 `src/env.py`），或进程环境变量 `AMAP_KEY`
（临时覆盖）。代码里不放任何 key；`.env` 在 `.gitignore` 里。
"""

from __future__ import annotations


import json
import math
import time
import urllib.parse
import urllib.request

from .env import mask, require

import numpy as np

API = "https://restapi.amap.com/v3/distance"

_A = 6378245.0
_EE = 0.00669342162296594323


def _tlat(x: float, y: float) -> float:
    r = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    r += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return r


def _tlon(x: float, y: float) -> float:
    r = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    r += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    r += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return r


def wgs84_to_gcj02(lon: float, lat: float) -> tuple[float, float]:
    dlat = _tlat(lon - 105.0, lat - 35.0)
    dlon = _tlon(lon - 105.0, lat - 35.0)
    rad = lat / 180.0 * math.pi
    magic = 1 - _EE * math.sin(rad) ** 2
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((_A * (1 - _EE)) / (magic * sqrt_magic) * math.pi)
    dlon = (dlon * 180.0) / (_A / sqrt_magic * math.cos(rad) * math.pi)
    return lon + dlon, lat + dlat


def _query(origins: list[tuple[float, float]], dest: tuple[float, float], key: str,
           retries: int = 3) -> list[tuple[float, float]]:
    o = "|".join(f"{lon:.6f},{lat:.6f}" for lon, lat in origins)
    d = f"{dest[0]:.6f},{dest[1]:.6f}"
    url = f"{API}?" + urllib.parse.urlencode({"origins": o, "destination": d, "type": 1, "key": key})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                js = json.load(r)
            if js.get("status") != "1":
                raise RuntimeError(f"高德 status={js.get('status')} info={js.get('info')}")
            res = js.get("results") or []
            if len(res) != len(origins):
                raise RuntimeError(f"返回 {len(res)} 条，应为 {len(origins)}")
            return [(float(x["distance"]), float(x["duration"])) for x in res]
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
    raise AssertionError("unreachable")


def distance_matrix(latlon: list[tuple[float, float]], key: str | None = None,
                    sleep: float = 0.12, verbose: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """WGS-84 经纬度序列 → (距离 m, 时间 s)，都是 n×n 有向矩阵。"""
    key = key or require("AMAP_KEY", "高德距离矩阵查询")
    n = len(latlon)
    if verbose:
        # 只打印首尾，日志里不出现完整 key
        print(f"  高德 Key {mask(key)} · {n}×{n} 矩阵 · 约 {n} 次请求")
    # 参数是 (经度, 纬度)。写反了高德只会回 INVALID_PARAMS，不告诉你哪里错，所以先自己查。
    for lon, lat in latlon:
        if not (70 < lon < 140 and 0 < lat < 60):
            raise ValueError(f"坐标不像中国境内：(lon={lon}, lat={lat})，注意是 (经度, 纬度)")
    gcj = [wgs84_to_gcj02(lon, lat) for lon, lat in latlon]
    dist = np.zeros((n, n))
    tim = np.zeros((n, n))
    for j in range(n):
        # 起点与终点相同会被高德判 INVALID_PARAMS，所以每次把 j 自己剔掉
        others = [i for i in range(n) if i != j]
        for s in range(0, len(others), 100):
            idx = others[s:s + 100]
            res = _query([gcj[i] for i in idx], gcj[j], key)
            for i, (dd, tt) in zip(idx, res):
                dist[j, i] = dd
                tim[j, i] = tt
            time.sleep(sleep)
        if verbose:
            print(f"    高德：终点 {j + 1}/{n}")
    return dist, tim
