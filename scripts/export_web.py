"""把天河算例的解导出成「能贴到高德地图上」的网页与静态图。

三件事：

1. 求解（或用缓存）得到卡车路线与无人机架次；
2. 卡车段**沿真实路网**取最短路（Dijkstra + 前驱回溯），不是两点直线——
   否则地图上会出现穿过珠江的线；
3. 全部坐标做 WGS-84 → GCJ-02（高德底图用火星坐标），输出：
   - `docs/web/solution_<set>.json`：前端可直接吃的数据
   - `docs/web/tddo_map.html`：交互页面（高德 JS API，需要 Web端(JS API) Key）
   - `results/figs/amap_static_<set>.png`：静态地图（用 Web服务 Key，立即可看）

用法：
    python -m scripts.export_web --set 112 --seed 0
    AMAP_KEY=xxx python -m scripts.export_web --set 112 --seed 0 --static
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.alns import solve
from src.amap import wgs84_to_gcj02
from src.evaluator import evaluate
from src.generator import build_instance
from src.instance import to_local_xy
from src.io import PROJECT_ROOT, load_config, load_sets
from src.solution import solution_to_dict

WEB = PROJECT_ROOT / "docs" / "web"
FIGS = PROJECT_ROOT / "results" / "figs"


# ---------------------------------------------------------------- 路网

def load_graph():
    """复用 build_tianhe 的建图逻辑，另建邻接表与 KD 树，供最短路回溯。"""
    from scripts.build_tianhe import load_graph as _lg, _largest_scc

    coord, edges = _lg()
    keep, idx = _largest_scc(len({e[0] for e in edges} | {e[1] for e in edges}), edges)
    adj: dict[int, list[tuple[int, float]]] = {}
    best: dict[tuple[int, int], float] = {}
    for u, v, d, _ in edges:
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None or not (keep[iu] and keep[iv]):
            continue
        if d < best.get((iu, iv), float("inf")):
            best[(iu, iv)] = d
    for (iu, iv), d in best.items():
        adj.setdefault(iu, []).append((iv, d))

    ids = [n for n in coord if n in idx and keep[idx[n]]]
    latlon = np.array([coord[n] for n in ids])
    lat0, lon0 = latlon[:, 0].mean(), latlon[:, 1].mean()
    xy = np.array([to_local_xy(la, lo, lat0, lon0) for la, lo in latlon])
    return coord, adj, idx, ids, cKDTree(xy), (lat0, lon0), xy


def shortest_path(adj, src: int, dst: int, limit: int = 400_000) -> list[int]:
    """Dijkstra + 前驱回溯。图规模 1.5 万节点、30 来段，够用。"""
    if src == dst:
        return [src]
    dist = {src: 0.0}
    prev: dict[int, int] = {}
    pq = [(0.0, src)]
    seen = 0
    while pq and seen < limit:
        d, u = heapq.heappop(pq)
        if u == dst:
            break
        if d > dist.get(u, float("inf")):
            continue
        seen += 1
        for v, w in adj.get(u, ()):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if dst not in dist:
        return [src, dst]
    path = [dst]
    while path[-1] != src:
        path.append(prev[path[-1]])
    return path[::-1]


# ---------------------------------------------------------------- 解 → 折线

def _gcj(points: list[tuple[float, float]]) -> list[list[float]]:
    """(lat, lon) 列表 → GCJ-02 的 [lon, lat]（高德用这个顺序）。"""
    out = []
    for lat, lon in points:
        g_lon, g_lat = wgs84_to_gcj02(lon, lat)
        out.append([round(g_lon, 6), round(g_lat, 6)])
    return out


def build_payload(set_id: int, seed: int, iters: int) -> dict:
    cfg_set = next(s for s in load_sets() if s["id"] == set_id)
    inst = build_instance(cfg_set, load_config())
    res = solve(inst, seed=seed, iters=iters)
    if res.best is None:
        raise SystemExit("求解失败")
    ev, cost = evaluate(inst, res.best)
    if not ev.feasible:
        raise SystemExit(f"解不可行：{ev.all_violations()[:3]}")

    coord, adj, idx, ids, tree, (lat0, lon0), xy = load_graph()
    inv = {v: k for k, v in idx.items()}          # 紧凑编号 → osmid
    node_osm: list[int] = []
    for nd in inst.base.nodes:
        _, j = tree.query(to_local_xy(nd.lat, nd.lon, lat0, lon0))
        node_osm.append(ids[int(np.atleast_1d(j).ravel()[0])])

    routes = []
    for ri, rt in enumerate(res.best.routes):
        legs = []
        nodes = list(rt.nodes)
        for a, b in zip(nodes, nodes[1:]):
            pa = shortest_path(adj, idx[node_osm[a]], idx[node_osm[b]])
            legs.append([coord[inv[i]] for i in pa])     # 紧凑编号换回 osmid 再取坐标
        # 拼接时去掉相邻段重复的接点
        poly: list[tuple[float, float]] = []
        for leg in legs:
            poly.extend(leg[1:] if poly and leg and poly[-1] == leg[0] else leg)
        sorties = []
        for s in rt.sorties:
            seq = [s.launch, *s.customers, s.recover]
            sorties.append({
                "launch": int(s.launch), "recover": int(s.recover),
                "customers": [int(c) for c in s.customers],
                "path": _gcj([(inst.base.nodes[i].lat, inst.base.nodes[i].lon) for i in seq]),
            })
        if not rt.served_customers():   # 空车（[0,0]）不画，否则地图上多个孤立点
            continue
        routes.append({
            "truck": int(rt.truck), "nodes": [int(x) for x in nodes],
            "path": _gcj(poly), "sorties": sorties,
        })

    nfz = []
    for z in inst.nfz:
        lat = lat0 + z.cy / 110_540.0
        lon = lon0 + z.cx / (111_320.0 * math.cos(math.radians(lat0)))
        nfz.append({"center": _gcj([(lat, lon)])[0], "r": float(z.r),
                    "t0": float(z.t_start), "t1": float(z.t_end), "label": z.label})

    pts = _gcj([(nd.lat, nd.lon) for nd in inst.base.nodes])
    return {
        "set": set_id, "seed": seed, "iters": iters,
        "cost": round(float(cost), 4),
        "city": inst.base.city,
        "n_customers": inst.base.n_customers,
        "n_trucks": inst.n_trucks,
        "n_sorties": sum(len(r["sorties"]) for r in routes),
        "drone_served": sum(len(s["customers"]) for r in routes for s in r["sorties"]),
        "depot": pts[0],
        "customers": pts[1:],
        "nfz": nfz,
        "routes": routes,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------- 静态图

def static_map(payload: dict, key: str) -> Path | None:
    """用「Web服务」Key 出一张静态图（不需要 JS Key）。一次最多 4 条 path。"""
    paths = []
    for r in payload["routes"][:4]:
        pts = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in r["path"][::max(1, len(r["path"]) // 90)])
        paths.append(f"6,0x1F4E79,0.9,,0:{pts}")
    size = "1024*1024"
    # 以路线包围盒的中心取景，而不是 depot：depot 常在边角，按它会裁掉一半路线
    pts_all = [p for r in payload["routes"] for p in r["path"]]
    lons = [p[0] for p in pts_all]; lats = [p[1] for p in pts_all]
    cx, cy = (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2
    span_km = max((max(lons) - min(lons)) * 96, (max(lats) - min(lats)) * 111)
    zoom = 14 if span_km < 3 else 13 if span_km < 7 else 12
    loc = f"{cx:.6f},{cy:.6f}"
    url = "https://restapi.amap.com/v3/staticmap?" + urllib.parse.urlencode(
        {"location": loc, "zoom": zoom, "size": size, "scale": 2,
         "paths": "|".join(paths), "key": key})
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            raw = r.read()
        if raw[:4] != b"\x89PNG":
            print("  静态图失败：", raw[:200].decode("utf-8", "ignore"))
            return None
        out = FIGS / f"amap_static_set{payload['set']}.png"
        out.write_bytes(raw)
        print(f"  {out.relative_to(PROJECT_ROOT)}")
        return out
    except Exception as ex:
        print("  静态图失败：", type(ex).__name__, ex)
        return None


# ---------------------------------------------------------------- 页面

HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TDDO · 天河算例路线</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
  html,body{margin:0;height:100%;font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif;color:#222}
  #map{position:absolute;inset:0}
  .panel{position:absolute;top:14px;left:14px;z-index:999;background:#fff;border:1px solid #e3e3e3;
         border-radius:8px;padding:12px 14px;min-width:268px;box-shadow:0 2px 12px rgba(0,0,0,.1)}
  .panel h1{font-size:15px;margin:0 0 8px}
  .panel table{border-collapse:collapse;font-size:12.5px}
  .panel td{padding:1px 10px 1px 0;color:#555;white-space:nowrap}
  .panel b{color:#111}
  .legend{position:absolute;right:14px;bottom:14px;z-index:999;background:#fff;border:1px solid #e3e3e3;
          border-radius:8px;padding:10px 12px;font-size:12.5px;box-shadow:0 2px 12px rgba(0,0,0,.1)}
  .legend div{margin:4px 0;display:flex;align-items:center;gap:7px}
  .lg-truck{width:24px;border-top:4px solid #1F4E79;border-radius:2px}
  .lg-drone{width:24px;border-top:3px dashed #E67E22}
  .lg-l{width:10px;height:10px;border-radius:50%;border:2px solid #2E7D5A;background:#fff}
  .lg-r{width:9px;height:9px;background:#E67E22;border:1px solid #fff;box-sizing:border-box}
  .lg-nfz{width:12px;height:12px;border-radius:50%;border:1px solid #C0392B;background:rgba(192,57,43,.15)}
  .note{font-size:11.5px;color:#999;margin-top:8px;line-height:1.5}
  .tnum{background:#1F4E79;color:#fff;border-radius:9px;font-size:11px;line-height:16px;
        text-align:center;width:18px;height:18px;border:1.5px solid #fff;font-weight:600}
</style></head>
<body>
<div id="map"></div>
<div class="panel">
  <h1>天河算例 · 卡车与无人机路线</h1>
  <table id="stats"></table>
  <div class="note">卡车为实线并按车编号；无人机为橙色虚线，绿圈起飞、橙方块回收。
  右下角图例可对照，右上角可开关图层与切换底图。</div>
</div>
<div class="legend">
  <div><span class="lg-truck"></span>卡车路径（实线，编号 1–n）</div>
  <div><span class="lg-drone"></span>无人机架次（虚线）</div>
  <div><span class="lg-l"></span>架次起飞点</div>
  <div><span class="lg-r"></span>架次回收点</div>
  <div><span class="lg-nfz"></span>禁飞区</div>
</div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = __DATA__;
const rows = [
  ['算例', DATA.set + ' · ' + DATA.city],
  ['客户 / 卡车', DATA.n_customers + ' / ' + DATA.n_trucks],
  ['架次 / 无人机服务', DATA.n_sorties + ' / ' + DATA.drone_served],
  ['目标成本', DATA.cost],
  ['种子 / 迭代', DATA.seed + ' / ' + DATA.iters],
  ['生成时间', DATA.generated_at],
];
document.getElementById('stats').innerHTML =
  rows.map(([k, v]) => `<tr><td>${k}</td><td><b>${v}</b></td></tr>`).join('');

const amap = L.tileLayer('https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
  { subdomains: '1234', maxZoom: 18, attribution: '&copy; 高德地图' });
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  { maxZoom: 19, attribution: '&copy; OpenStreetMap' });

const map = L.map('map', { layers: [amap], zoomControl: true });

const gTruck = L.layerGroup().addTo(map);
const gDrone = L.layerGroup().addTo(map);
const gNfz   = L.layerGroup().addTo(map);
const gPts   = L.layerGroup().addTo(map);
L.control.layers(
  { '高德底图': amap, 'OSM 底图（坐标参考）': osm },
  { '卡车路径': gTruck, '无人机架次': gDrone, '禁飞区': gNfz, '客户点': gPts },
  { position: 'topright', collapsed: false }).addTo(map);

const palette = ['#1F4E79','#2E7D5A','#8E5A9E','#B26A00','#2A6F7F','#7A4E2D'];
const bounds = [];
let k = 0;

DATA.routes.forEach((r, i) => {
  k += 1;
  const c = palette[i % palette.length];
  const ll = r.path.map(([lon, lat]) => [lat, lon]);
  // 白色描边打底：两条路线重合时也能分出层次
  L.polyline(ll, { color: '#fff', weight: 8, opacity: .9 }).addTo(gTruck);
  const line = L.polyline(ll, { color: c, weight: 4.5, opacity: .95, lineJoin: 'round' })
    .addTo(gTruck)
    .bindTooltip(`卡车 ${r.truck} · ${r.nodes.length - 2} 个停靠点 · 架次 ${r.sorties.length}`,
                 { sticky: true });
  // 路线中点放编号，避免只靠颜色区分
  const mid = ll[Math.floor(ll.length / 2)];
  L.marker(mid, {
    icon: L.divIcon({ className: '', html: `<div class="tnum">${k}</div>`, iconSize: [18, 18], iconAnchor: [9, 9] }),
    interactive: false,
  }).addTo(gTruck);

  r.sorties.forEach((s, si) => {
    const sp = s.path.map(([lon, lat]) => [lat, lon]);
    L.polyline(sp, { color: '#fff', weight: 5, opacity: .85 }).addTo(gDrone);
    L.polyline(sp, { color: '#E67E22', weight: 3, opacity: 1, dashArray: '7 6' })
      .addTo(gDrone)
      .bindTooltip(`卡车 ${r.truck} · 架次 ${si + 1} · ${s.customers.length} 个客户`, { sticky: true });
    L.circleMarker(sp[0], { radius: 5, color: '#2E7D5A', weight: 2, fillColor: '#fff', fillOpacity: 1 })
      .addTo(gDrone).bindTooltip('起飞', { direction: 'top' });
    L.circleMarker(sp[sp.length - 1], { radius: 5, color: '#fff', weight: 1.5, fillColor: '#E67E22', fillOpacity: 1 })
      .addTo(gDrone).bindTooltip('回收', { direction: 'top' });
  });

  ll.forEach(p => bounds.push(p));
});

DATA.nfz.forEach(z => {
  L.circle([z.center[1], z.center[0]], { radius: z.r, color: '#C0392B', weight: 1,
    fillColor: '#C0392B', fillOpacity: .1 }).addTo(gNfz);
});

L.circleMarker([DATA.depot[1], DATA.depot[0]], { radius: 8, color: '#fff', weight: 2.5,
  fillColor: '#111', fillOpacity: 1 }).addTo(gPts).bindPopup('depot（配送中心）');
DATA.customers.forEach(p => {
  L.circleMarker([p[1], p[0]], { radius: 3.6, color: '#fff', weight: 1,
    fillColor: '#C0392B', fillOpacity: .95 }).addTo(gPts);
});

if (bounds.length) map.fitBounds(bounds, { padding: [44, 44] });
else map.setView([DATA.depot[1], DATA.depot[0]], 13);
</script>
</body></html>
"""


def write_html(payload: dict) -> Path:
    WEB.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    out = WEB / f"tddo_map_set{payload['set']}.html"
    out.write_text(html, encoding="utf-8")
    (WEB / f"solution_set{payload['set']}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="导出可贴到高德地图的解")
    ap.add_argument("--set", type=int, default=112)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--static", action="store_true", help="同时用 Web服务 Key 出静态图")
    a = ap.parse_args()

    print("求解并导出…")
    payload = build_payload(a.set, a.seed, a.iters)
    html = write_html(payload)
    print(f"  {html.relative_to(PROJECT_ROOT)}")
    print(f"  {WEB.joinpath(f'solution_set{a.set}.json').relative_to(PROJECT_ROOT)}")
    if a.static:
        key = os.environ.get("AMAP_KEY")
        if not key:
            print("  未设置 AMAP_KEY，跳过静态图")
        else:
            static_map(payload, key)


if __name__ == "__main__":
    main()
