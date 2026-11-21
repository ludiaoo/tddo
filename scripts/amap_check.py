"""拿高德「距离测量」核对天河算例的路网距离（需要 AMAP_KEY 环境变量）。

## 为什么单独做这一步

`build_tianhe` 的矩阵来自 OSM 路网 + 本地 Dijkstra，优点是确定、可复现、无配额；
缺点是路网版本、单行线、禁行规则未必和国内实际一致。高德是国内的商用路网，
拿它做**抽样交叉校验**比直接替换更稳：如果两者中位偏差在 5% 以内，说明 OSM 够用；
偏差很大才需要整体换数据源。

## 几个必须注意的点

- **坐标系**：高德全线 GCJ-02，OSM 是 WGS-84，市区偏差 300–500 m。请求前必须转换，
  否则「看起来差不多」，实际查的是隔壁街区。
- **路况**：`type=1` 的驾车距离**含实时路况**，官方文档明确写不同时间结果会变。
  所以每次抓取都记时间戳，且只用于校验，不进主算例。
- **批量**：`v3/distance` 的 `origins` 最多 100 个坐标对，`destination` 只有一个，
  所以一个 n×n 矩阵要 n 次请求（每次 100 起点 → 1 终点，n>100 时分批）。
- **Key**：绝不要写进代码或提交。用 `AMAP_KEY` 环境变量。

用法：
    AMAP_KEY=xxx python -m scripts.amap_check
    AMAP_KEY=xxx python -m scripts.amap_check --every 5   # 每 5 个节点取一个终点
"""

from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np
import pandas as pd

from src.io import PROJECT_ROOT
from src.io_tianhe import load_tianhe_base

OUT = PROJECT_ROOT / "results" / "amap_vs_osm.csv"

from src.amap import distance_matrix, wgs84_to_gcj02  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="高德 vs OSM 距离校验")
    ap.add_argument("--base", default="set13", help="用哪个天河底图（set01/set05/set09/set13）")
    ap.add_argument("--every", type=int, default=4, help="每 N 个节点取一个终点，控制请求数")
    ap.add_argument("--sleep", type=float, default=0.12, help="两次请求的间隔（秒）")
    a = ap.parse_args()

    key = os.environ.get("AMAP_KEY")
    if not key:
        raise SystemExit("未设置 AMAP_KEY 环境变量（不要把 key 写进代码或提交）")

    base = load_tianhe_base(a.base)
    n = len(base.nodes)
    gcj = [wgs84_to_gcj02(nd.lon, nd.lat) for nd in base.nodes]
    dest_idx = list(range(0, n, a.every))
    print(f"{a.base}：{n} 节点，取 {len(dest_idx)} 个终点 × {n} 个起点，"
          f"约 {len(dest_idx) * math.ceil(n / 100)} 次请求")

    amap_d = np.full((len(dest_idx), n), np.nan)
    amap_t = np.full((len(dest_idx), n), np.nan)
    t0 = time.time()
    for k, j in enumerate(dest_idx):
        for s in range(0, n, 100):
            idx = list(range(s, min(s + 100, n)))
            res = query([gcj[i] for i in idx], gcj[j], key)
            for i, (dd, tt) in zip(idx, res):
                amap_d[k, i] = dd
                amap_t[k, i] = tt
            time.sleep(a.sleep)
        print(f"  终点 {k + 1}/{len(dest_idx)}（节点 {j}）完成，累计 {time.time() - t0:.0f}s")

    rows = []
    for k, j in enumerate(dest_idx):
        for i in range(n):
            if i == j:
                continue
            osm = float(base.truck_dist_m[j, i])
            am = float(amap_d[k, i])
            if osm <= 0 or not np.isfinite(am):
                continue
            rows.append({
                "dest": j, "origin": i, "osm_m": osm, "amap_m": am,
                "osm_s": float(base.truck_time_s[j, i]), "amap_s": float(amap_t[k, i]),
                "ratio_d": am / osm, "ratio_t": float(amap_t[k, i]) / max(float(base.truck_time_s[j, i]), 1e-9),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    r = df["ratio_d"]
    print(f"\n配对 {len(df)} 条：距离比 中位 {r.median():.3f}  "
          f"P10 {r.quantile(.1):.3f}  P90 {r.quantile(.9):.3f}  "
          f"|偏差|>20% 占 {100 * (abs(r - 1) > 0.2).mean():.1f}%")
    rt = df["ratio_t"]
    print(f"时间比 中位 {rt.median():.3f}  P10 {rt.quantile(.1):.3f}  P90 {rt.quantile(.9):.3f}"
          f"（高德含实时路况，OSM 是自由流 ÷ 0.85）")
    print(f"距离中位：OSM {df.osm_m.median()/1000:.2f} km vs 高德 {df.amap_m.median()/1000:.2f} km")
    print(f"\n已写出 {OUT.relative_to(PROJECT_ROOT)}")
    print(f"抓取时间：{time.strftime('%Y-%m-%d %H:%M:%S')}（高德 type=1 含当时路况）")


if __name__ == "__main__":
    main()
