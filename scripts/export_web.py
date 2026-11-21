"""把天河算例的解导出成「能贴到高德地图上」的网页与静态图。

四件事：

1. 求解并保留**初始解**（`build_initial`）与**最终解**，两者都能画、能对比；
2. 卡车段沿真实路网取最短路（Dijkstra + 前驱回溯），不是两点直线；
3. 全部坐标 WGS-84 → GCJ-02；
4. 输出：
   - `docs/web/tddo_map_<set>.html`：交互页面（按车/按架次选择、全选/反选、初始 vs 最终）
   - `docs/web/solution_<set>.json`：前端数据
   - `results/figs/amap_static_<set>.png`：静态图（Web服务 Key）

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
from src.evaluator import _simulate_route, evaluate
from src.generator import build_instance
from src.initial import build_initial
from src.instance import to_local_xy
from src.io import PROJECT_ROOT, load_config, load_sets
from src.solution import Solution

WEB = PROJECT_ROOT / "docs" / "web"
FIGS = PROJECT_ROOT / "results" / "figs"


# ---------------------------------------------------------------- 路网

def load_graph():
    """复用 build_tianhe 的建图逻辑，另建邻接表与 KD 树，供最短路回溯。"""
    from scripts.build_tianhe import _largest_scc, load_graph as _lg

    coord, edges = _lg()
    keep, idx = _largest_scc(len({e[0] for e in edges} | {e[1] for e in edges}), edges)
    best: dict[tuple[int, int], float] = {}
    for u, v, d, _ in edges:
        iu, iv = idx.get(u), idx.get(v)
        if iu is None or iv is None or not (keep[iu] and keep[iv]):
            continue
        if d < best.get((iu, iv), float("inf")):
            best[(iu, iv)] = d
    adj: dict[int, list[tuple[int, float]]] = {}
    for (iu, iv), d in best.items():
        adj.setdefault(iu, []).append((iv, d))

    ids = [n for n in coord if n in idx and keep[idx[n]]]
    latlon = np.array([coord[n] for n in ids])
    lat0, lon0 = latlon[:, 0].mean(), latlon[:, 1].mean()
    xy = np.array([to_local_xy(la, lo, lat0, lon0) for la, lo in latlon])
    return coord, adj, idx, ids, cKDTree(xy), (lat0, lon0)


def shortest_path(adj, src: int, dst: int, limit: int = 400_000) -> list[int]:
    """Dijkstra + 前驱回溯；图规模 1.5 万节点、几十段，够用。"""
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


def _gcj(points) -> list[list[float]]:
    """(lat, lon) → GCJ-02 的 [lon, lat]（高德用这个顺序）。"""
    out = []
    for lat, lon in points:
        g_lon, g_lat = wgs84_to_gcj02(lon, lat)
        out.append([round(g_lon, 6), round(g_lat, 6)])
    return out


# ---------------------------------------------------------------- 解 → 折线

def sortie_node_seq(nodes: list[int], s) -> list[int]:
    """一个架次的节点序列：起飞站 → 各客户 → 归队站。

    `s.launch` / `s.recover` 是 ``nodes`` 里的**位置下标**，不是节点编号，
    必须先换成 ``nodes[pos]`` 再用（曾经直接拿去做节点索引，端点画到 1–4 km 外）。
    """
    return [int(nodes[s.launch]), *[int(c) for c in s.customers], int(nodes[s.recover])]


def encode_solution(inst, sol: Solution, graph) -> dict:
    """把一个解编码成前端数据：每条卡车路径 + 每个架次的折线，以及汇总指标。"""
    coord, adj, idx, ids, tree, (lat0, lon0) = graph
    inv = {v: k for k, v in idx.items()}          # 紧凑编号 → osmid
    node_osm = []
    for nd in inst.base.nodes:
        _, j = tree.query(to_local_xy(nd.lat, nd.lon, lat0, lon0))
        node_osm.append(ids[int(np.atleast_1d(j).ravel()[0])])

    # 先逐路线模拟：既拿每个架次的起降时刻，也用来核对几何。
    evs = [_simulate_route(inst, rt, fast=False) for rt in sol.routes]

    routes = []
    for k, rt in enumerate(sol.routes):
        if not rt.served_customers():              # 空车（[0,0]）不画
            continue
        nodes = list(rt.nodes)
        poly: list[tuple[float, float]] = []
        for a, b in zip(nodes, nodes[1:]):
            pa = shortest_path(adj, idx[node_osm[a]], idx[node_osm[b]])
            leg = [coord[inv[i]] for i in pa]
            poly.extend(leg[1:] if poly and leg and poly[-1] == leg[0] else leg)

        ev = evs[k]
        wins = list(ev.sortie_windows or [])
        specs = sorted(rt.sorties, key=lambda s: s.launch)   # 按卡车行程顺序编号
        # 只有全部架次都成功时，sortie_windows 才能与 specs 一一对应
        timed = len(wins) == len(specs)
        sorties = []
        for j, s in enumerate(specs):
            # 注意：s.launch / s.recover 是 **nodes 里的位置下标**，不是节点编号。
            # 之前直接拿它去索引 inst.base.nodes，导致架次起降点画到了 1–4 km 之外
            # （看起来就像"无人机飞到一半没回来"）。
            seq = sortie_node_seq(nodes, s)
            lnode, rnode = seq[0], seq[-1]
            item = {
                "launch": int(s.launch), "recover": int(s.recover),
                "launch_node": lnode, "recover_node": rnode,
                "customers": [int(c) for c in s.customers],
                "path": _gcj([(inst.base.nodes[i].lat, inst.base.nodes[i].lon) for i in seq]),
            }
            if timed:
                item["t_launch_min"] = round(wins[j][0] / 60.0, 1)
                item["t_land_min"] = round(wins[j][1] / 60.0, 1)
            sorties.append(item)

        stops = [
            {"node": int(n), "pos": pos, "ll": _gcj([(inst.base.nodes[n].lat, inst.base.nodes[n].lon)])[0]}
            for pos, n in enumerate(nodes)
            if not (pos == len(nodes) - 1 and int(n) == 0)     # 末尾回仓不重复画
        ]
        # 守门：架次的起降点必须落在卡车某个停靠点上。端点若飞到几百米外，
        # 说明又拿位置下标当节点编号用了（这一处真的错过一次）。
        for s_ in sorties:
            for tag, pt in (("起飞", s_["path"][0]), ("归队", s_["path"][-1])):
                near = min(
                    math.hypot((pt[0] - st["ll"][0]) * 96_000, (pt[1] - st["ll"][1]) * 111_000)
                    for st in stops
                ) if stops else 0.0
                if near > 50.0:
                    print(f"  警告：卡车 {rt.truck} 的{tag}点离最近停靠 {near:.0f} m，端点可能不在卡车上")
        routes.append({"truck": int(rt.truck), "nodes": [int(x) for x in nodes],
                       "stops": stops, "path": _gcj(poly), "sorties": sorties})

    _, cost = evaluate(inst, sol)
    return {
        "cost": round(float(cost), 4),
        "n_trucks_used": len(routes),
        "n_sorties": sum(len(r["sorties"]) for r in routes),
        "drone_served": sum(len(s["customers"]) for r in routes for s in r["sorties"]),
        "makespan_s": round(max((e.finish_time for e in evs), default=0.0), 1),
        "routes": routes,
    }


def build_payload(set_id: int, seed: int, iters: int) -> dict:
    cfg_set = next(s for s in load_sets() if s["id"] == set_id)
    inst = build_instance(cfg_set, load_config())

    init, init_cost, _ = build_initial(inst)
    res = solve(inst, seed=seed, iters=iters)
    if res.best is None:
        raise SystemExit("求解失败")

    graph = load_graph()
    initial = encode_solution(inst, init, graph)
    final = encode_solution(inst, res.best, graph)
    if not np.isclose(initial["cost"], init_cost, rtol=1e-6):
        print(f"  注意：初始解成本复算 {initial['cost']} ≠ build_initial 返回 {init_cost}")

    nfz = []
    lat0, lon0 = inst.base.nodes[0].lat, inst.base.nodes[0].lon
    for z in inst.nfz:
        lat = lat0 + z.cy / 110_540.0
        lon = lon0 + z.cx / (111_320.0 * math.cos(math.radians(lat0)))
        nfz.append({"center": _gcj([(lat, lon)])[0], "r": float(z.r),
                    "t0": float(z.t_start), "t1": float(z.t_end), "label": z.label})

    pts = _gcj([(nd.lat, nd.lon) for nd in inst.base.nodes])
    return {
        "set": set_id, "seed": seed, "iters": iters, "city": inst.base.city,
        "n_customers": inst.base.n_customers, "n_trucks": inst.n_trucks,
        "depot": pts[0], "customers": pts[1:], "nfz": nfz,
        "initial": initial, "final": final,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# ---------------------------------------------------------------- 静态图

def static_map(payload: dict, key: str) -> Path | None:
    """用「Web服务」Key 出一张静态图（不需要 JS Key）。一次最多 4 条 path。"""
    paths = []
    for r in payload["final"]["routes"][:4]:
        pts = ";".join(f"{lon:.6f},{lat:.6f}" for lon, lat in r["path"][::max(1, len(r["path"]) // 90)])
        paths.append(f"6,0x1F4E79,0.9,,0:{pts}")
    pts_all = [p for r in payload["final"]["routes"] for p in r["path"]]
    lons = [p[0] for p in pts_all]; lats = [p[1] for p in pts_all]
    cx, cy = (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2
    span_km = max((max(lons) - min(lons)) * 96, (max(lats) - min(lats)) * 111)
    zoom = 14 if span_km < 3 else 13 if span_km < 7 else 12
    url = "https://restapi.amap.com/v3/staticmap?" + urllib.parse.urlencode(
        {"location": f"{cx:.6f},{cy:.6f}", "zoom": zoom, "size": "1024*1024", "scale": 2,
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
  .panel{position:absolute;top:12px;left:12px;z-index:999;background:#fff;border:1px solid #e3e3e3;
         border-radius:10px;padding:12px 14px;width:326px;box-shadow:0 2px 14px rgba(0,0,0,.1);
         max-height:calc(100% - 24px);display:flex;flex-direction:column}
  .panel h1{font-size:15px;margin:0 0 8px}
  .modes{display:flex;gap:6px;margin:2px 0 10px}
  .modes button{flex:1;padding:5px 0;border:1px solid #d8d8d8;background:#fafafa;border-radius:6px;
                cursor:pointer;font-size:12.5px;color:#444}
  .modes button.on{background:#1F4E79;border-color:#1F4E79;color:#fff}
  table.cmp{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:8px}
  table.cmp th,table.cmp td{padding:2px 3px;text-align:right;border-bottom:1px solid #f2f2f2}
  table.cmp th:first-child,table.cmp td:first-child{text-align:left;color:#666}
  table.cmp td.dn{color:#1E7B45}
  table.cmp td.up{color:#B03A2E}
  .bar{display:flex;gap:6px;margin:2px 0 8px}
  .bar button{padding:3px 9px;font-size:12px;border:1px solid #d8d8d8;background:#fff;
              border-radius:5px;cursor:pointer;color:#444}
  .bar button:hover{background:#f3f6fa}
  .list{overflow:auto;flex:1;border-top:1px solid #f0f0f0;padding-top:6px;min-height:80px}
  .row{display:flex;align-items:center;gap:7px;padding:2px 0;font-size:12.5px}
  .row.sub{margin-left:22px;color:#555;font-size:12px}
  .sw{width:14px;border-top:4px solid #999;border-radius:2px;flex:none}
  .sw.d{border-top-style:dashed}
  .row label{display:flex;align-items:center;gap:6px;cursor:pointer;flex:1}
  .solo{margin-left:auto;font-size:11px;color:#9aa;cursor:pointer;border:0;background:none;padding:0 2px}
  .solo:hover{color:#1F4E79}
  .legend{position:absolute;right:12px;bottom:12px;z-index:999;background:#fff;border:1px solid #e3e3e3;
          border-radius:8px;padding:8px 11px;font-size:12px;box-shadow:0 2px 12px rgba(0,0,0,.1)}
  .legend div{margin:3px 0;display:flex;align-items:center;gap:6px}
  .note{font-size:11px;color:#9a9a9a;margin-top:6px;line-height:1.5}
</style></head>
<body>
<div id="map"></div>

<div class="panel">
  <h1>天河算例 · 卡车与无人机路线</h1>
  <div class="modes">
    <button data-mode="final" class="on">最终解</button>
    <button data-mode="initial">初始解</button>
    <button data-mode="overlay">叠加对比</button>
  </div>
  <table class="cmp" id="cmp"></table>
  <div class="bar">
    <button id="all">全选</button><button id="none">全不选</button><button id="inv">反选</button>
  </div>
  <div class="list" id="list"></div>
  <div class="note" id="note"></div>
</div>

<div class="legend">
  <div><span class="sw" style="border-top-color:#1F4E79"></span>卡车路径（实线，按车分色）</div>
  <div><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#1F4E79;border:1.5px solid #fff"></span>卡车停靠点（悬停看第几站）</div>
  <div><span class="sw d" style="border-top-color:#E67E22"></span>无人机架次（虚线，同色属同一辆车）</div>
  <div><span style="display:inline-block;width:9px;height:9px;border-radius:50%;border:2px solid #E67E22;background:#fff"></span>架次起飞点</div>
  <div><span style="display:inline-block;width:9px;height:9px;background:#E67E22;border:2px solid #1b1b1b;border-radius:2px"></span><b style="font-weight:600">归队点</b>（方块，落回卡车）</div>
  <div><span class="sw d" style="border-top-color:#9AA0A6"></span>初始解（灰色虚线）</div>
</div>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const DATA = __DATA__;
const PAL = ['#1F4E79','#2E7D5A','#8E5A9E','#B26A00','#2A6F7F','#7A4E2D','#4A5FA5','#7D6B3F'];
const DASH = ['7 6', '3 5', '12 5 3 5', '2 6'];
let mode = 'final';
let selT = new Set(), selS = new Set();

const amap = L.tileLayer('https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
  { subdomains: '1234', maxZoom: 18, attribution: '&copy; 高德地图' });
const osm = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
  { maxZoom: 19, attribution: '&copy; OpenStreetMap' });
const map = L.map('map', { layers: [amap], zoomControl: true });
const gRoute = L.layerGroup().addTo(map);
const gNfz = L.layerGroup().addTo(map);
const gPts = L.layerGroup().addTo(map);
L.control.layers({ '高德底图': amap, 'OSM 底图（坐标参考）': osm },
                 { '路线': gRoute, '禁飞区': gNfz, '客户点': gPts },
                 { position: 'topright', collapsed: false }).addTo(map);

// ---------- 初始 vs 最终
const I = DATA.initial, F = DATA.final;
const num = v => (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(1));
const rows = [
  ['成本', I.cost, F.cost, true],
  ['使用卡车', I.n_trucks_used, F.n_trucks_used, false],
  ['架次数', I.n_sorties, F.n_sorties, false],
  ['无人机服务', I.drone_served, F.drone_served, false],
  ['完工 (h)', I.makespan_s / 3600, F.makespan_s / 3600, true],
];
document.getElementById('cmp').innerHTML =
  '<tr><th>指标</th><th>初始</th><th>最终</th><th>变化</th></tr>' +
  rows.map(([k, a, b, pct]) => {
    const d = b - a, rel = a ? d / Math.abs(a) * 100 : 0;
    const cls = d < 0 ? 'dn' : (d > 0 ? 'up' : '');
    const txt = d === 0 ? '—' : (pct ? `${rel > 0 ? '+' : ''}${rel.toFixed(1)}%` : `${d > 0 ? '+' : ''}${d}`);
    return `<tr><td>${k}</td><td>${num(a)}</td><td><b>${num(b)}</b></td><td class="${cls}">${txt}</td></tr>`;
  }).join('');

// ---------- 列表与选择
const current = () => (mode === 'initial' ? I.routes : F.routes);

function selectAll() {
  const rs = current();
  selT = new Set(rs.map((_, i) => 't' + i));
  selS = new Set();
  rs.forEach((r, i) => r.sorties.forEach((_, j) => selS.add(`t${i}s${j}`)));
}

function renderList() {
  const rs = current();
  const box = document.getElementById('list');
  box.innerHTML = rs.map((r, i) => {
    const col = PAL[i % PAL.length];
    const sub = r.sorties.map((s, j) => {
      const key = `t${i}s${j}`;
      const t = (s.t_launch_min !== undefined) ? ` · ${s.t_launch_min}→${s.t_land_min}min` : '';
      return `<div class="row sub"><span class="sw d" style="border-top-color:${col}"></span>
        <label><input type="checkbox" data-k="${key}" ${selS.has(key) ? 'checked' : ''}>
        架次 ${j + 1} · ${s.customers.length} 客户${t}</label></div>`;
    }).join('');
    return `<div class="row"><span class="sw" style="border-top-color:${col}"></span>
      <label><input type="checkbox" data-t="${i}" ${selT.has('t' + i) ? 'checked' : ''}>
      卡车 ${r.truck} · ${r.nodes.length - 2} 停靠 · ${r.sorties.length} 架次</label>
      <button class="solo" data-solo="${i}">只看</button></div>${sub}`;
  }).join('') || '<div class="note">该解没有可用路线</div>';

  box.querySelectorAll('input[data-t]').forEach(el => el.onchange = e => {
    const i = +e.target.dataset.t, on = e.target.checked;
    if (on) selT.add('t' + i); else selT.delete('t' + i);
    current()[i].sorties.forEach((_, j) => {
      const k = `t${i}s${j}`;
      if (on) selS.add(k); else selS.delete(k);
    });
    renderList(); draw();
  });
  box.querySelectorAll('input[data-k]').forEach(el => el.onchange = e => {
    const k = e.target.dataset.k;
    if (e.target.checked) selS.add(k); else selS.delete(k);
    draw();
  });
  box.querySelectorAll('button[data-solo]').forEach(el => el.onclick = e => {
    const i = +e.currentTarget.dataset.solo;
    selT = new Set(['t' + i]);
    selS = new Set(current()[i].sorties.map((_, j) => `t${i}s${j}`));
    renderList(); draw();
  });
}

// ---------- 绘制
function draw() {
  gRoute.clearLayers();
  if (mode === 'overlay') { drawSol(I, true, true); drawSol(F, false); }
  else drawSol(mode === 'initial' ? I : F, false);
}

function drawSol(sol, ghost, ignoreSel) {
  sol.routes.forEach((r, i) => {
    // 叠加模式下初始解整体作为背景，不跟随勾选（两套解的卡车编号不能一一对应）
    if (!ignoreSel && !selT.has('t' + i)) return;
    const col = ghost ? '#9AA0A6' : PAL[i % PAL.length];
    const ll = r.path.map(([lon, lat]) => [lat, lon]);
    L.polyline(ll, { color: '#fff', weight: ghost ? 5 : 8, opacity: .85 }).addTo(gRoute);
    L.polyline(ll, {
      color: col, weight: ghost ? 3 : 4.5, opacity: ghost ? .85 : .95,
      dashArray: ghost ? '4 4' : null, lineJoin: 'round',
    }).addTo(gRoute)
      .bindTooltip(`${ghost ? '初始解 · ' : ''}卡车 ${r.truck} · ${r.nodes.length - 2} 停靠 · ${r.sorties.length} 架次`,
                   { sticky: true });

    if (!ghost) {
      const mid = ll[Math.floor(ll.length / 2)];
      L.marker(mid, { interactive: false, icon: L.divIcon({ className: '',
        html: `<div style="background:${col};color:#fff;border-radius:9px;font-size:11px;line-height:16px;
          text-align:center;width:18px;height:18px;border:1.5px solid #fff;font-weight:600">${i + 1}</div>`,
        iconSize: [18, 18], iconAnchor: [9, 9] }) }).addTo(gRoute);

      // 卡车停靠点：无人机就是在这里起降的，画出来才看得出"归队"落在哪
      (r.stops || []).forEach(st => {
        L.circleMarker([st.ll[1], st.ll[0]], { radius: 3.2, color: '#fff', weight: 1.2,
          fillColor: col, fillOpacity: .95 }).addTo(gRoute)
          .bindTooltip(`卡车 ${r.truck} · 第 ${st.pos} 站 · 节点 ${st.node}`, { direction: 'top' });
      });
    }

    r.sorties.forEach((s, j) => {
      if (!ignoreSel && !selS.has(`t${i}s${j}`)) return;
      const sp = s.path.map(([lon, lat]) => [lat, lon]);
      const tt = (s.t_launch_min !== undefined)
        ? ` · ${s.t_launch_min}→${s.t_land_min} min` : '';
      L.polyline(sp, { color: '#fff', weight: 5, opacity: .8 }).addTo(gRoute);
      L.polyline(sp, { color: ghost ? '#9AA0A6' : col, weight: 3, opacity: 1,
        dashArray: ghost ? '3 4' : DASH[j % DASH.length] }).addTo(gRoute)
        .bindTooltip(`${ghost ? '初始解 · ' : ''}卡车 ${r.truck} · 架次 ${j + 1} · ${s.customers.length} 客户${tt}`,
                     { sticky: true });
      if (!ghost) {
        // 起飞：空心圆环（从卡车第 launch 站出发）
        L.circleMarker(sp[0], { radius: 4.5, color: col, weight: 2, fillColor: '#fff', fillOpacity: 1 })
          .addTo(gRoute).bindTooltip(`起飞 · 卡车 ${r.truck} 第 ${s.launch} 站（节点 ${s.launch_node}）`,
                                     { direction: 'top' });
        // 归队：实心方块 + 深色描边，明显区别于"停在半路"
        L.marker(sp[sp.length - 1], { interactive: true, icon: L.divIcon({ className: '',
          html: `<div style="width:11px;height:11px;background:${col};border:2px solid #1b1b1b;
            border-radius:2px"></div>`, iconSize: [11, 11], iconAnchor: [5.5, 5.5] }) })
          .addTo(gRoute)
          .bindTooltip(`归队 · 落回卡车 ${r.truck} 第 ${s.recover} 站（节点 ${s.recover_node}）`,
                       { direction: 'top' });
      }
    });
  });
}

// ---------- 固定图层
DATA.nfz.forEach(z => L.circle([z.center[1], z.center[0]], {
  radius: z.r, color: '#C0392B', weight: 1, fillColor: '#C0392B', fillOpacity: .1 }).addTo(gNfz));
L.circleMarker([DATA.depot[1], DATA.depot[0]], { radius: 8, color: '#fff', weight: 2.5,
  fillColor: '#111', fillOpacity: 1 }).addTo(gPts).bindPopup('depot（配送中心）');
DATA.customers.forEach(p => L.circleMarker([p[1], p[0]], { radius: 3.4, color: '#fff',
  weight: 1, fillColor: '#C0392B', fillOpacity: .95 }).addTo(gPts));

const allPts = [];
F.routes.forEach(r => r.path.forEach(p => allPts.push([p[1], p[0]])));
if (allPts.length) map.fitBounds(allPts, { padding: [44, 44] });
else map.setView([DATA.depot[1], DATA.depot[0]], 13);

// ---------- 事件
const noteEl = document.getElementById('note');
const RENDEZVOUS = '说明：无人机不在原地等卡车，它飞到客户点投递后，落回卡车<b>后面某一站</b>（方块＝归队点）；'
  + '那段时间卡车一直在往前开，所以归队点通常不在起飞点旁边。';
const setNote = () => {
  const head = mode === 'overlay'
    ? '叠加：灰色虚线是初始解，彩色实线是最终解；勾选作用于「最终解」的卡车与架次。'
    : mode === 'initial' ? '当前显示初始可行解（未优化）。勾选可单独查看某辆车。'
    : '当前显示 ALNS 最终解。勾选左侧可只保留部分卡车/架次。';
  noteEl.innerHTML = `${head}<br>${RENDEZVOUS}`;
};

document.querySelectorAll('.modes button').forEach(b => b.onclick = () => {
  mode = b.dataset.mode;
  document.querySelectorAll('.modes button').forEach(x => x.classList.toggle('on', x === b));
  selectAll(); renderList(); draw(); setNote();
});
document.getElementById('all').onclick = () => { selectAll(); renderList(); draw(); };
document.getElementById('none').onclick = () => { selT.clear(); selS.clear(); renderList(); draw(); };
document.getElementById('inv').onclick = () => {
  const rs = current();
  const nextT = new Set(rs.map((_, i) => 't' + i).filter(k => !selT.has(k)));
  rs.forEach((r, i) => r.sorties.forEach((_, j) => {
    const k = `t${i}s${j}`;
    if (selS.has(k)) selS.delete(k); else selS.add(k);
  }));
  selT = nextT;
  renderList(); draw();
};

selectAll(); renderList(); draw(); setNote();
</script>
</body></html>
"""


AMAP_HTML = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TDDO · 天河算例（高德 JS API）</title>
<style>
  html,body{margin:0;height:100%;font:14px/1.5 system-ui,"Microsoft YaHei",sans-serif;color:#222}
  #map{position:absolute;inset:0}
  .panel{position:absolute;top:12px;left:12px;z-index:999;background:#fff;border:1px solid #e3e3e3;
         border-radius:10px;padding:12px 14px;width:330px;box-shadow:0 2px 14px rgba(0,0,0,.12);
         max-height:calc(100% - 24px);display:flex;flex-direction:column}
  .panel h1{font-size:15px;margin:0 0 8px}
  .modes{display:flex;gap:6px;margin:2px 0 10px}
  .modes button{flex:1;padding:5px 0;border:1px solid #d8d8d8;background:#fafafa;border-radius:6px;
                cursor:pointer;font-size:12.5px;color:#444}
  .modes button.on{background:#1F4E79;border-color:#1F4E79;color:#fff}
  table.cmp{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:8px}
  table.cmp th,table.cmp td{padding:2px 3px;text-align:right;border-bottom:1px solid #f2f2f2}
  table.cmp th:first-child,table.cmp td:first-child{text-align:left;color:#666}
  table.cmp td.dn{color:#1E7B45}
  table.cmp td.up{color:#B03A2E}
  .bar{display:flex;gap:6px;margin:2px 0 8px}
  .bar button{padding:3px 9px;font-size:12px;border:1px solid #d8d8d8;background:#fff;
              border-radius:5px;cursor:pointer;color:#444}
  .list{overflow:auto;flex:1;border-top:1px solid #f0f0f0;padding-top:6px;min-height:80px}
  .row{display:flex;align-items:center;gap:7px;padding:2px 0;font-size:12.5px}
  .row.sub{margin-left:22px;color:#555;font-size:12px}
  .sw{width:14px;border-top:4px solid #999;border-radius:2px;flex:none}
  .sw.d{border-top-style:dashed}
  .row label{display:flex;align-items:center;gap:6px;cursor:pointer;flex:1}
  .solo{margin-left:auto;font-size:11px;color:#9aa;cursor:pointer;border:0;background:none}
  .note{font-size:11px;color:#9a9a9a;margin-top:6px;line-height:1.5}
  .legend{position:absolute;right:12px;bottom:12px;z-index:999;background:#fff;border:1px solid #e3e3e3;
          border-radius:8px;padding:8px 11px;font-size:12px;box-shadow:0 2px 12px rgba(0,0,0,.1)}
  .legend div{margin:3px 0;display:flex;align-items:center;gap:6px}
</style></head>
<body><div id="map"></div>
<div class="panel">
  <h1>天河算例 · 高德 JS API</h1>
  <div class="modes">
    <button data-mode="final" class="on">最终解</button>
    <button data-mode="initial">初始解</button>
    <button data-mode="overlay">叠加对比</button>
  </div>
  <table class="cmp" id="cmp"></table>
  <div class="bar"><button id="all">全选</button><button id="none">全不选</button><button id="inv">反选</button></div>
  <div class="list" id="list"></div>
  <div class="note" id="note"></div>
</div>
<div class="legend">
  <div><span class="sw" style="border-top-color:#1F4E79"></span>卡车（实线，带方向）</div>
  <div><span class="sw d" style="border-top-color:#E67E22"></span>无人机架次（虚线）</div>
  <div><span class="sw d" style="border-top-color:#9AA0A6"></span>初始解（灰色）</div>
</div>
<script>
const DATA = __DATA__, KEY = "__KEY__", SECRET = "__SECRET__";
window._AMapSecurityConfig = { securityJsCode: SECRET };
const s = document.createElement('script');
s.src = 'https://webapi.amap.com/maps?v=2.0&key=' + encodeURIComponent(KEY) + '&plugin=AMap.Scale,AMap.ToolBar';
s.onload = () => init();
s.onerror = () => { document.getElementById('map').innerHTML =
  '<div style="padding:24px;color:#B03A2E">高德 JS API 加载失败：检查 Key 类型（Web端 JS API）、安全密钥与域名白名单。</div>'; };
document.head.appendChild(s);

const PAL = ['#1F4E79','#2E7D5A','#8E5A9E','#B26A00','#2A6F7F','#7A4E2D','#4A5FA5','#7D6B3F'];
const DASH = [[7,6],[3,5],[12,5,3,5],[2,6]];
let mode = 'final', selT = new Set(), selS = new Set();
let map, info, overlays = [];

function init() {
  map = new AMap.Map('map', { zoom: 13, center: DATA.depot, viewMode: '2D',
    mapStyle: 'amap://styles/normal' });
  map.addControl(new AMap.Scale());
  map.addControl(new AMap.ToolBar({ position: { right: '18px', top: '120px' } }));
  info = new AMap.InfoWindow({ offset: new AMap.Pixel(0, -14), isCustom: false });

  buildCmp(); selectAll(); renderListChildren(); render(); bindUI();
  document.getElementById('note').innerHTML =
    '当前显示 ALNS 最终解。勾选左侧可只保留部分卡车/架次；卡车线带方向箭头。'
    + '<br>无人机飞到客户点投递后，落回卡车<b>后面某一站</b>（方块＝归队点），'
    + '那段时间卡车一直在往前开，所以归队点通常不在起飞点旁边。';
}

function buildCmp() {
  const I = DATA.initial, F = DATA.final;
  const num = v => (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(1));
  const rows = [
    ['成本', I.cost, F.cost, true],
    ['使用卡车', I.n_trucks_used, F.n_trucks_used, false],
    ['架次数', I.n_sorties, F.n_sorties, false],
    ['无人机服务', I.drone_served, F.drone_served, false],
    ['完工 (h)', I.makespan_s / 3600, F.makespan_s / 3600, true],
  ];
  document.getElementById('cmp').innerHTML = '<tr><th>指标</th><th>初始</th><th>最终</th><th>变化</th></tr>' +
    rows.map(([k, a, b, pct]) => {
      const d = b - a, rel = a ? d / Math.abs(a) * 100 : 0;
      const cls = d < 0 ? 'dn' : (d > 0 ? 'up' : '');
      const txt = d === 0 ? '—' : (pct ? `${rel > 0 ? '+' : ''}${rel.toFixed(1)}%` : `${d > 0 ? '+' : ''}${d}`);
      return `<tr><td>${k}</td><td>${num(a)}</td><td><b>${num(b)}</b></td><td class="${cls}">${txt}</td></tr>`;
    }).join('');
}

const current = () => (mode === 'initial' ? DATA.initial.routes : DATA.final.routes);

function selectAll() {
  const rs = current();
  selT = new Set(rs.map((_, i) => 't' + i));
  selS = new Set();
  rs.forEach((r, i) => r.sorties.forEach((_, j) => selS.add(`t${i}s${j}`)));
}

function renderListChildren() {
  renderList();
}

function renderList() {
  const rs = current(), box = document.getElementById('list');
  box.innerHTML = rs.map((r, i) => {
    const col = PAL[i % PAL.length];
    const sub = r.sorties.map((s, j) => {
      const key = `t${i}s${j}`;
      const t = (s.t_launch_min !== undefined) ? ` · ${s.t_launch_min}→${s.t_land_min}min` : '';
      return `<div class="row sub"><span class="sw d" style="border-top-color:${col}"></span>
        <label><input type="checkbox" data-k="${key}" ${selS.has(key) ? 'checked' : ''}>
        架次 ${j + 1} · ${s.customers.length} 客户${t}</label></div>`;
    }).join('');
    return `<div class="row"><span class="sw" style="border-top-color:${col}"></span>
      <label><input type="checkbox" data-t="${i}" ${selT.has('t' + i) ? 'checked' : ''}>
      卡车 ${r.truck} · ${r.nodes.length - 2} 停靠 · ${r.sorties.length} 架次</label>
      <button class="solo" data-solo="${i}">只看</button></div>${sub}`;
  }).join('') || '<div class="note">该解没有可用路线</div>';

  box.querySelectorAll('input[data-t]').forEach(el => el.onchange = e => {
    const i = +e.target.dataset.t, on = e.target.checked;
    if (on) selT.add('t' + i); else selT.delete('t' + i);
    current()[i].sorties.forEach((_, j) => {
      const k = `t${i}s${j}`;
      if (on) selS.add(k); else selS.delete(k);
    });
    renderList(); render();
  });
  box.querySelectorAll('input[data-k]').forEach(el => el.onchange = e => {
    const k = e.target.dataset.k;
    if (e.target.checked) selS.add(k); else selS.delete(k);
    render();
  });
  box.querySelectorAll('button[data-solo]').forEach(el => el.onclick = e => {
    const i = +e.currentTarget.dataset.solo;
    selT = new Set(['t' + i]);
    selS = new Set(current()[i].sorties.map((_, j) => `t${i}s${j}`));
    renderList(); render();
  });
}

function clearOverlays() {
  if (overlays.length) { map.remove(overlays); overlays = []; }
}

function add(o) { overlays.push(o); return o; }

function render() {
  clearOverlays();
  if (mode === 'overlay') { drawSol(DATA.initial, true, true); drawSol(DATA.final, false, false); }
  else drawSol(mode === 'initial' ? DATA.initial : DATA.final, false, false);
  drawFixed();
  if (overlays.length) map.setFitView(overlays, false, [50, 50, 50, 50]);
}

function drawSol(sol, ghost, ignoreSel) {
  sol.routes.forEach((r, i) => {
    if (!ignoreSel && !selT.has('t' + i)) return;
    const col = ghost ? '#9AA0A6' : PAL[i % PAL.length];
    const line = add(new AMap.Polyline({
      path: r.path, strokeColor: col, strokeWeight: ghost ? 3 : 5,
      strokeOpacity: ghost ? .85 : .95, strokeStyle: ghost ? 'dashed' : 'solid',
      strokeDasharray: ghost ? [6, 6] : undefined, lineJoin: 'round',
      showDir: !ghost, zIndex: ghost ? 40 : 55,
    }));
    line.on('mouseover', e => info.open(map, e.lnglat,
      `${ghost ? '初始解 · ' : ''}卡车 ${r.truck} · ${r.nodes.length - 2} 停靠 · ${r.sorties.length} 架次`));
    line.setMap(map);

    if (!ghost) {
      const mid = r.path[Math.floor(r.path.length / 2)];
      add(new AMap.Text({ text: String(i + 1), position: mid, offset: new AMap.Pixel(-9, -9),
        style: { background: col, color: '#fff', border: '1.5px solid #fff', borderRadius: '9px',
                 padding: '1px 4px', fontSize: '11px', fontWeight: '600' }, zIndex: 90 }));

      // 卡车停靠点：架次的起降都落在这些点上
      (r.stops || []).forEach(st => {
        const m = add(new AMap.CircleMarker({ center: st.ll, radius: 3, strokeColor: '#fff',
          strokeWeight: 1.2, fillColor: col, fillOpacity: .95, zIndex: 82 }));
        m.on('mouseover', e => info.open(map, e.lnglat,
          `卡车 ${r.truck} · 第 ${st.pos} 站 · 节点 ${st.node}`));
      });
    }

    r.sorties.forEach((s, j) => {
      if (!ignoreSel && !selS.has(`t${i}s${j}`)) return;
      const tt = (s.t_launch_min !== undefined) ? ` · ${s.t_launch_min}→${s.t_land_min} min` : '';
      const pl = add(new AMap.Polyline({
        path: s.path, strokeColor: ghost ? '#9AA0A6' : col, strokeWeight: 2.5,
        strokeOpacity: 1, strokeStyle: 'dashed',
        strokeDasharray: ghost ? [3, 4] : DASH[j % DASH.length], zIndex: 70,
      }));
      pl.on('mouseover', e => info.open(map, e.lnglat,
        `${ghost ? '初始解 · ' : ''}卡车 ${r.truck} · 架次 ${j + 1} · ${s.customers.length} 客户${tt}`));
      pl.setMap(map);
      if (!ghost) {
        const lc = add(new AMap.CircleMarker({ center: s.path[0], radius: 4.5, strokeColor: col,
          strokeWeight: 2, fillColor: '#fff', fillOpacity: 1, zIndex: 84 }));
        lc.on('mouseover', e => info.open(map, e.lnglat,
          `起飞 · 卡车 ${r.truck} 第 ${s.launch} 站（节点 ${s.launch_node}）`));
        // 归队：方块标记，与"起飞"的圆环区分开
        const rc = add(new AMap.Marker({
          position: s.path[s.path.length - 1], offset: new AMap.Pixel(-5.5, -5.5), zIndex: 86,
          content: `<div style="width:11px;height:11px;background:${col};border:2px solid #1b1b1b;border-radius:2px"></div>`,
        }));
        rc.on('mouseover', e => info.open(map, e.lnglat,
          `归队 · 落回卡车 ${r.truck} 第 ${s.recover} 站（节点 ${s.recover_node}）`));
      }
    });
  });
}

function drawFixed() {
  DATA.nfz.forEach(z => add(new AMap.Circle({ center: z.center, radius: z.r,
    strokeColor: '#C0392B', strokeWeight: 1, fillColor: '#C0392B', fillOpacity: .12, zIndex: 30 })));
  add(new AMap.CircleMarker({ center: DATA.depot, radius: 7, strokeColor: '#fff',
    strokeWeight: 2.5, fillColor: '#111', fillOpacity: 1, zIndex: 95 }));
  DATA.customers.forEach(p => add(new AMap.CircleMarker({ center: p, radius: 3.4,
    strokeColor: '#fff', strokeWeight: 1, fillColor: '#C0392B', fillOpacity: .95, zIndex: 85 })));
  overlays.forEach(o => o.setMap ? o.setMap(map) : map.add(o));
}

function bindUI() {
  const note = document.getElementById('note');
  document.querySelectorAll('.modes button').forEach(b => b.onclick = () => {
    mode = b.dataset.mode;
    document.querySelectorAll('.modes button').forEach(x => x.classList.toggle('on', x === b));
    selectAll(); renderList(); render();
    note.innerHTML = (mode === 'overlay'
      ? '叠加：灰色虚线是初始解，彩色实线是最终解；勾选只作用于最终解。'
      : mode === 'initial' ? '当前显示初始可行解（未优化）。' : '当前显示 ALNS 最终解。')
      + '<br>无人机飞到客户点投递后，落回卡车<b>后面某一站</b>（方块＝归队点）。';
  });
  document.getElementById('all').onclick = () => { selectAll(); renderList(); render(); };
  document.getElementById('none').onclick = () => { selT.clear(); selS.clear(); renderList(); render(); };
  document.getElementById('inv').onclick = () => {
    const rs = current();
    selT = new Set(rs.map((_, i) => 't' + i).filter(k => !selT.has(k)));
    rs.forEach((r, i) => r.sorties.forEach((_, j) => {
      const k = `t${i}s${j}`;
      if (selS.has(k)) selS.delete(k); else selS.add(k);
    }));
    renderList(); render();
  };
}
</script></body></html>
"""


def write_amap_html(payload: dict, key: str, secret: str) -> Path:
    """生成调用高德 JS API 的页面。**含 Key，必须 gitignore，不要提交。**"""
    WEB.mkdir(parents=True, exist_ok=True)
    html = (AMAP_HTML
            .replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
            .replace("__KEY__", key).replace("__SECRET__", secret))
    out = WEB / f"tddo_map_set{payload['set']}_amap.html"
    out.write_text(html, encoding="utf-8")
    return out


def write_html(payload: dict) -> Path:
    WEB.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    out = WEB / f"tddo_map_set{payload['set']}.html"
    out.write_text(html, encoding="utf-8")
    (WEB / f"solution_set{payload['set']}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="导出可贴到高德地图的解（含初始/最终对比）")
    ap.add_argument("--set", type=int, default=112)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--static", action="store_true", help="同时用 Web服务 Key 出静态图")
    ap.add_argument("--amap-js", action="store_true",
                    help="额外生成高德 JS API 版本（读 AMAP_JS_KEY / AMAP_JS_SECRET，产物不入库）")
    a = ap.parse_args()

    print("求解并导出…")
    payload = build_payload(a.set, a.seed, a.iters)
    print(f"  初始解 {payload['initial']['cost']} → 最终解 {payload['final']['cost']}")
    html = write_html(payload)
    print(f"  {html.relative_to(PROJECT_ROOT)}")
    if a.amap_js:
        jk, js = os.environ.get("AMAP_JS_KEY"), os.environ.get("AMAP_JS_SECRET")
        if not (jk and js):
            print("  未设置 AMAP_JS_KEY / AMAP_JS_SECRET，跳过 JS API 版本")
        else:
            out = write_amap_html(payload, jk, js)
            print(f"  {out.relative_to(PROJECT_ROOT)}（含 Key，已 gitignore）")
    if a.static:
        key = os.environ.get("AMAP_KEY")
        if not key:
            print("  未设置 AMAP_KEY，跳过静态图")
        else:
            static_map(payload, key)


if __name__ == "__main__":
    main()
