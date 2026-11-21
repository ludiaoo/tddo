"""把西雅图算例缩放搬到广州天河，并用天河真实路网重算卡车距离。

三步（`--step`）：

1. `fetch`：从 Overpass 拉天河区可行驶路网（`data/tianhe/tiles/*.json`，可断点续跑）；
2. `build`：建图 → 对每个源算例做「等比缩放 + 平移 + 吸附到最近路口」→
   Dijkstra 算全 OD 有向距离/时间 → 写 `data/tianhe/base_*.json` 与 `data/instances/set1xx.json`；
3. `check`：五道体检（连通性 / 三角不等式 / 对称偏差 / 卡车÷无人机距离比 / 对角为零）。

口径（写进 base 的 `meta`，复算时要对得上）：

- 距离 = 路网最短路长度；时间 = 同一路径按道路等级限速 ÷ `SPEED_FACTOR`（默认 0.85，近似平峰）；
- OSM 的 `oneway` 照做，所以矩阵**有向**；掉头、单行线都体现在里面；
- 无人机仍用欧氏距离，禁飞区绕行几何不变——只是坐标换成了天河；
- 节点由源算例**等比缩放**（保持相对几何），整体平移到天河，再吸附到最近路口（≤`SNAP_RADIUS_M`）。

用法：
    python -m scripts.build_tianhe --step fetch
    python -m scripts.build_tianhe --step build
    python -m scripts.build_tianhe --step check
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from src.generator import build_instance, instance_to_dict
from src.instance import BaseProblem, Node, to_local_xy
from src.io import dump_json, load_base_problem, load_config, load_sets

ROOT = Path(__file__).resolve().parent.parent
TIANHE = ROOT / "data" / "tianhe"
TILES = TIANHE / "tiles"
INSTANCES = ROOT / "data" / "instances"

# 天河区主体（南, 西, 北, 东）。Overpass 近期不稳定（502/超时），改用 OSM 主 API 拉小格：
# 每格约 1.4 km 见方，单次 5–10 s，25 格约 3 分钟。主 API 单次有 50k 节点上限，格子不能更大。
BBOX = (23.105, 113.325, 23.175, 113.395)      # 体育中心—珠江新城—员村一带
GRID = 5
CELL_SPLIT = tuple(
    (round(BBOX[0] + i * (BBOX[2] - BBOX[0]) / GRID, 6), round(BBOX[1] + j * (BBOX[3] - BBOX[1]) / GRID, 6),
     round(BBOX[0] + (i + 1) * (BBOX[2] - BBOX[0]) / GRID, 6), round(BBOX[1] + (j + 1) * (BBOX[3] - BBOX[1]) / GRID, 6))
    for i in range(GRID) for j in range(GRID)
)
OSM_API = "https://api.openstreetmap.org/api/0.6/map?bbox={w},{s},{e},{n}"
HIGHWAYS = ("motorway|trunk|primary|secondary|tertiary|residential|unclassified|"
            "living_street|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")

# 道路等级的默认限速（km/h）。OSM 里有 maxspeed 就用它，没有就用这张表。
SPEED_KMH = {
    "motorway": 80, "trunk": 60, "primary": 50, "secondary": 40, "tertiary": 35,
    "residential": 25, "unclassified": 25, "living_street": 15,
    "motorway_link": 45, "trunk_link": 40, "primary_link": 35,
    "secondary_link": 30, "tertiary_link": 30,
}
SPEED_FACTOR = 0.85      # 平峰折减：信号灯、拥堵、装卸占道

# 源算例 → 新算例编号。取西雅图的 4 个规模各一个（10/25/50/100 客户）。
MAPPING = {1: 101, 5: 102, 9: 103, 13: 104}
TARGET_DIAG_KM = 6.5     # 缩放后客户群的包围盒对角线目标长度（核心区约 7 km 见方）
SNAP_RADIUS_M = 600.0    # 吸附到路口的最大距离，超过就报错（说明缩放/平移不合理）
ANCHOR = (23.140, 113.360)    # 天河体育中心附近，作为客户群的中心


# ---------------------------------------------------------------- fetch

def fetch(max_attempts: int = 3) -> None:
    """按 5×5 网格从 OSM 主 API 拉原始地图（XML）。已下载的格子跳过，可断点续跑。"""
    TILES.mkdir(parents=True, exist_ok=True)
    for i, (s_, w, n, e) in enumerate(CELL_SPLIT):
        fp = TILES / f"cell{i:02d}.osm"
        if fp.exists():
            print(f"  skip cell{i:02d}（已存在 {fp.stat().st_size // 1024} KB）")
            continue
        url = OSM_API.format(w=w, s=s_, e=e, n=n)
        for attempt in range(max_attempts):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "tddo-research/1.0 (academic use)"})
                t0 = time.time()
                with urllib.request.urlopen(req, timeout=120) as r:
                    raw = r.read()
                fp.write_bytes(raw)
                print(f"  cell{i:02d}: {len(raw) // 1024} KB, {time.time() - t0:.0f}s")
                break
            except Exception as ex:
                print(f"  cell{i:02d}: {type(ex).__name__} {getattr(ex, 'code', '')}，重试")
                time.sleep(4 * (attempt + 1))
        else:
            raise SystemExit(f"cell{i:02d} 连续 {max_attempts} 次失败，稍后重跑（已成功的会跳过）")
    (TIANHE / "fetch_meta.json").write_text(json.dumps({
        "bbox": BBOX, "grid": GRID, "source": "OpenStreetMap main API (api.openstreetmap.org)",
        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- graph

def _haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(math.sqrt(h))


def _parse_speed(tags: dict, highway: str) -> float:
    raw = (tags.get("maxspeed") or "").split(";")[0].strip().lower()
    for token in raw.replace("km/h", "").replace("kmh", "").split():
        try:
            v = float(token)
        except ValueError:
            continue
        if 5 <= v <= 120:
            return v
    return float(SPEED_KMH.get(highway, 25))


def _parse_osm_xml(path: Path) -> tuple[dict[int, tuple[float, float]], list[tuple[list[int], dict]]]:
    """流式解析一个 .osm 文件，返回 (节点坐标, [(节点序列, 标签)])。"""
    import xml.etree.ElementTree as ET

    coord: dict[int, tuple[float, float]] = {}
    ways: list[tuple[list[int], dict]] = []
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "node":
            coord[int(el.get("id"))] = (float(el.get("lat")), float(el.get("lon")))
            el.clear()
        elif el.tag == "way":
            refs = [int(nd.get("ref")) for nd in el.findall("nd")]
            tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
            ways.append((refs, tags))
            el.clear()
    return coord, ways


def load_graph() -> tuple[dict[int, tuple[float, float]], list[tuple[int, int, float, float]]]:
    """合并各格，返回 (osmid → (lat, lon), [(u, v, 长度m, 速度km/h)])，边已含单行方向。"""
    coord: dict[int, tuple[float, float]] = {}
    edges: list[tuple[int, int, float, float]] = []
    ways_kept = 0
    for fp in sorted(TILES.glob("cell*.osm")):
        c, ws = _parse_osm_xml(fp)
        coord.update(c)
        for refs, tags in ws:
            hw = tags.get("highway", "")
            if hw not in SPEED_KMH or len(refs) < 2:
                continue
            ways_kept += 1
            spd = _parse_speed(tags, hw)
            one = str(tags.get("oneway", "")).lower()
            jun = str(tags.get("junction", "")).lower()
            forward = one not in ("-1", "reverse")
            backward = one not in ("yes", "true", "1") and jun != "roundabout"
            for a, b in zip(refs, refs[1:]):
                pa, pb = coord.get(a), coord.get(b)
                if pa is None or pb is None:
                    continue
                d = _haversine(pa, pb)
                if d <= 0:
                    continue
                if forward:
                    edges.append((a, b, d, spd))
                if backward:
                    edges.append((b, a, d, spd))
    print(f"  路网：{ways_kept} 条 way，{len(coord)} 个节点，{len(edges)} 条有向边")
    return coord, edges


def _largest_scc(n_nodes: int, edges: list[tuple[int, int, float, float]]) -> tuple[np.ndarray, dict]:
    """只保留强连通分量最大的那块，否则最短路会出 inf。"""
    idx = {n: i for i, n in enumerate(sorted({e[0] for e in edges} | {e[1] for e in edges}))}
    row = np.array([idx[u] for u, _, _, _ in edges])
    col = np.array([idx[v] for _, v, _, _ in edges])
    g = csr_matrix((np.ones(len(row)), (row, col)), shape=(len(idx), len(idx)))
    n_comp, labels = connected_components(g, directed=True, connection="strong")
    sizes = np.bincount(labels)
    keep = int(sizes.argmax())
    print(f"  强连通分量 {n_comp} 个，最大 {sizes[keep]} / {len(idx)} 个节点")
    return labels == keep, idx


# ---------------------------------------------------------------- transform

def _fit_scale(src_xy: np.ndarray, diag_m: float) -> float:
    span = float(np.hypot(*(src_xy.max(0) - src_xy.min(0))))
    return diag_m / span if span > 0 else 1.0


def _place_in_tianhe(src_latlon: np.ndarray, scale: float,
                     anchor: tuple[float, float] = ANCHOR) -> np.ndarray:
    """源经纬度 → 天河经纬度：等比缩放 + 平移（不旋转，保持原朝向）。"""
    lat0, lon0 = src_latlon[:, 0].mean(), src_latlon[:, 1].mean()
    xy = np.array([to_local_xy(la, lo, lat0, lon0) for la, lo in src_latlon]) * scale
    return np.array([(anchor[0] + y / 110_540.0, anchor[1] + x / (111_320.0 * math.cos(math.radians(anchor[0]))))
                     for x, y in xy])


def _snap(points_latlon: np.ndarray, coord: dict[int, tuple[float, float]],
          keep: np.ndarray, idx: dict[int, int]) -> tuple[list[int], list[float]]:
    """把每个点吸附到最近的**保留节点**上；同一节点被占用则退而取次近。"""
    # 只考虑真正进了图（有边）的节点：OSM 原始数据里还有大量不属于任何道路的节点
    osmid = [n for n in coord if n in idx]
    latlon = np.array([coord[n] for n in osmid])
    lat0, lon0 = latlon[:, 0].mean(), latlon[:, 1].mean()
    xy = np.array([to_local_xy(la, lo, lat0, lon0) for la, lo in latlon])
    ok = np.array([bool(keep[idx[n]]) for n in osmid])
    tree = cKDTree(xy[ok])
    ok_ids = [n for n, m in zip(osmid, ok) if m]

    picked, offsets = [], []
    used: set[int] = set()
    for la, lo in points_latlon:
        p = np.array(to_local_xy(la, lo, lat0, lon0))
        for k in range(8):                       # 最近的 8 个里挑一个没被占的
            dist, j = tree.query(p, k=min(8, len(ok_ids)))
            dist = np.atleast_1d(dist); j = np.atleast_1d(j)
            got = None
            for d, jj in zip(dist, j):
                if int(jj) not in used:
                    got = (ok_ids[int(jj)], float(d))
                    break
            if got is not None:
                break
        if got is None:
            raise SystemExit("吸附失败：候选节点都被占用")
        if got[1] > SNAP_RADIUS_M:
            raise SystemExit(f"吸附距离 {got[1]:.0f} m 超过 {SNAP_RADIUS_M:.0f} m，"
                             f"说明缩放后落到了没有路的地方；调小 TARGET_DIAG_KM 重试")
        used.add(int(got[0]))
        picked.append(int(got[0]))
        offsets.append(got[1])
    return picked, offsets


def _matrices(snapped: list[int], coord, edges, keep, idx) -> tuple[np.ndarray, np.ndarray]:
    """Dijkstra 算 |snapped| × N，再取出目标列。距离与时间各跑一遍（同一拓扑，权重不同）。"""
    # OSM 里同一对节点常有多条边（并行路/分段），直接堆进 csr 会被求和成两倍长度，
    # 所以先按 (u,v) 取最短的那条。
    best: dict[tuple[int, int], list[float]] = {}
    for u, v, d, spd in edges:
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None or not (keep[iu] and keep[iv]):
            continue
        t = d / (spd * SPEED_FACTOR / 3.6)
        slot = best.setdefault((iu, iv), [d, t])
        if d < slot[0]:
            slot[0], slot[1] = d, t

    n = keep.size
    row = np.fromiter((k[0] for k in best), dtype=int)
    col = np.fromiter((k[1] for k in best), dtype=int)
    vals = np.array(list(best.values()), dtype=float)
    gl = csr_matrix((vals[:, 0], (row, col)), shape=(n, n))
    gt = csr_matrix((vals[:, 1], (row, col)), shape=(n, n))
    targets = np.array([idx[s] for s in snapped])

    dist = np.empty((len(targets), len(targets)))
    tim = np.empty_like(dist)
    for a in range(0, len(targets), 20):        # 分块，避免一次性吃满内存
        block = targets[a:a + 20]
        dl = dijkstra(gl, directed=True, indices=block)
        tt = dijkstra(gt, directed=True, indices=block)
        dist[a:a + 20] = dl[:, targets]
        tim[a:a + 20] = tt[:, targets]
    return dist, tim


# ---------------------------------------------------------------- build

def build(only: set[int] | None = None) -> None:
    coord, edges = load_graph()
    keep, idx = _largest_scc(len({e[0] for e in edges} | {e[1] for e in edges}), edges)
    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}

    for src_id, new_id in MAPPING.items():
        if only and new_id not in only:
            continue
        src_cfg = by_id[src_id]
        base0 = load_base_problem(src_cfg["problems"][0], src_cfg["city"])
        latlon = np.array([[n.lat, n.lon] for n in base0.nodes])
        xy = np.array([[n.x, n.y] for n in base0.nodes])

        # 缩放：先按目标对角线定比例，若有点落到吸附半径外就逐次缩小
        scale = _fit_scale(xy, TARGET_DIAG_KM * 1000.0)
        for attempt in range(6):
            placed = _place_in_tianhe(latlon, scale)
            try:
                snapped, offs = _snap(placed, coord, keep, idx)
                break
            except SystemExit as ex:
                if attempt == 5:
                    raise
                print(f"  Set{src_id}: {ex}，scale × 0.85 重试")
                scale *= 0.85

        dist, tim = _matrices(snapped, coord, edges, keep, idx)
        if not np.isfinite(dist).all() or not np.isfinite(tim).all():
            bad = int((~np.isfinite(dist)).sum() + (~np.isfinite(tim)).sum())
            raise SystemExit(f"Set{src_id}: 矩阵有 {bad} 个不可达 OD")

        lat0, lon0 = placed[0]
        nodes = tuple(
            Node(idx=i, kind=base0.nodes[i].kind, lat=float(placed[i, 0]), lon=float(placed[i, 1]),
                 x=to_local_xy(float(placed[i, 0]), float(placed[i, 1]), lat0, lon0)[0],
                 y=to_local_xy(float(placed[i, 0]), float(placed[i, 1]), lat0, lon0)[1])
            for i in range(len(placed))
        )
        name = f"set{src_id:02d}"
        base = BaseProblem(name=name, city="tianhe", nodes=nodes, truck_dist_m=dist,
                           truck_time_s=tim, parcel_kg=base0.parcel_kg)

        dump_json({
            "name": name, "city": "tianhe",
            "nodes": [{"id": n.idx, "kind": n.kind, "lat": n.lat, "lon": n.lon,
                       "x_m": n.x, "y_m": n.y} for n in nodes],
            "truck": {"dist_m": dist.tolist(), "time_s": tim.tolist()},
            "parcel_kg": list(base.parcel_kg),
            "meta": {
                "source_set": src_id, "new_set": new_id,
                "map": "OpenStreetMap via Overpass API", "bbox": list(BBOX),
                "speed_factor": SPEED_FACTOR, "speed_kmh": SPEED_KMH,
                "transform": {"scale": round(float(scale), 5), "rotate_deg": 0.0,
                              "target_diag_km": TARGET_DIAG_KM, "anchor": list(ANCHOR),
                              "snap_radius_m": SNAP_RADIUS_M},
                "snap_offset_m": {"mean": round(float(np.mean(offs)), 1),
                                  "max": round(float(np.max(offs)), 1)},
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }, TIANHE / f"base_{name}.json")

        # 用同一套生成规则造算例（需求/取件/时间窗/禁飞区/车队规模），只是 base 换成天河
        set_cfg = {"id": new_id, "city": "tianhe", "problems": [src_cfg["problems"][0]],
                   "n_customers": base0.n_customers, "depot": src_cfg.get("depot", "center"),
                   "source": "tianhe", "base": name}
        inst = build_instance(set_cfg, cfg, base=base)
        dump_json(instance_to_dict(inst), INSTANCES / f"set{new_id}.json")
        span = float(np.hypot(*(np.array([[n.x, n.y] for n in nodes]).max(0)
                                - np.array([[n.x, n.y] for n in nodes]).min(0)))) / 1000
        print(f"  Set{src_id} → Set{new_id}: {base0.n_customers} 客户, 跨度 {span:.1f} km, "
              f"truck 距离中位 {np.median(dist[dist > 0]) / 1000:.2f} km, "
              f"吸附 均值 {np.mean(offs):.0f} m / 最大 {np.max(offs):.0f} m")


# ---------------------------------------------------------------- check

def check() -> None:
    from src.io_tianhe import load_tianhe_base

    print(f"{'算例':<8}{'节点':>5}{'对角':>7}{'距离kN':>9}{'三角':>7}{'对称':>7}"
          f"{'卡车÷无人机':>12}{'时间kN':>9}")
    for src_id, new_id in MAPPING.items():
        b = load_tianhe_base(f"set{src_id:02d}")
        n = len(b.nodes)
        d = b.truck_dist_m
        xy = np.array([[nd.x, nd.y] for nd in b.nodes])
        drone = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        iu = np.triu_indices(n, 1)
        ratio = d[iu] / np.maximum(drone[iu], 1.0)

        # 三角不等式：抽样 3000 组 (i,j,k) 检查
        rng = np.random.default_rng(0)
        trip = [(d[i, k] <= d[i, j] + d[j, k] + 1e-6)
                for i, j, k in rng.integers(0, n, size=(3000, 3)) if len({i, j, k}) == 3]
        sym = np.abs(d[iu] - d.T[iu]) / np.maximum(d[iu], 1.0)
        # ratio = 卡车路网距离 / 无人机直线距离（同一 OD 对），即绕行倍数
        print(f"set{new_id:<4}{n:>5}{np.hypot(*(xy.max(0) - xy.min(0))) / 1000:>7.1f}"
              f"{np.median(d[d > 0]) / 1000:>9.2f}"
              f"{100 * np.mean(trip):>6.0f}%{np.median(sym):>6.0%}"
              f"{np.median(ratio):>12.2f}"
              f"{np.median(b.truck_time_s[b.truck_time_s > 0]) / 60:>8.1f}")

        assert np.allclose(np.diag(d), 0), "对角线必须为 0"
        assert np.isfinite(d).all() and np.isfinite(b.truck_time_s).all(), "有不可达 OD"
        assert np.mean(trip) > 0.995, f"三角不等式通过率只有 {np.mean(trip):.3f}"
        assert 1.0 <= np.median(ratio) <= 3.0, "卡车/无人机绕行倍数异常"


def main() -> None:
    ap = argparse.ArgumentParser(description="天河真实路网算例构建")
    ap.add_argument("--step", choices=["fetch", "build", "check", "all"], default="all")
    ap.add_argument("--only", default=None, help="只重建这些新编号，逗号分隔（如 101,102）")
    a = ap.parse_args()
    only = {int(x) for x in a.only.split(",")} if a.only else None
    if a.step in ("fetch", "all"):
        print("[1/3] 拉取路网")
        fetch()
    if a.step in ("build", "all"):
        print("[2/3] 建图并生成算例")
        build(only)
    if a.step in ("check", "all"):
        print("[3/3] 体检")
        check()


if __name__ == "__main__":
    main()
