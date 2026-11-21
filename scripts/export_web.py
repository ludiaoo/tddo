"""把天河算例的解导出成数据，再交给 `or-research` skill 内置的 amap 模块出图。

本项目只负责两件项目特有的事：

1. 求解并保留**初始解**（`build_initial`）与**最终解**；
2. 卡车段沿真实路网取最短路（Dijkstra + 前驱回溯，不是两点直线），
   坐标 WGS-84 → GCJ-02，编码成 `docs/web/solution_set<N>.json`。

渲染（解 → payload → 高德底图 → 单文件页面）全部复用 skill，不再在本项目里
维护第二份 HTML 模板：

    solution_set<N>.json
      → skill/amap/scripts/adapt_tddo.py    解 → payload（含架次端点守门）
      → skill/amap/scripts/build_map.py     高德静态底图 + 页面

用法：

    python -m scripts.export_web --set 112 --seed 0
    python -m scripts.export_web --set 112 --amap-js   # 额外出可拖拽版（含 JS key，不入库）
    python -m scripts.export_web --set 112 --static    # 额外出带线路的静态 PNG

skill 默认在 `~/.workbuddy/skills/or-research`，地图模块是它的 `amap/` 子目录；
换位置可用环境变量 `OR_RESEARCH_MAP_DIR` 直接指向该模块目录。
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.alns import solve
from src.amap import wgs84_to_gcj02
from src.env import get, mask
from src.evaluator import _simulate_route, evaluate
from src.generator import build_instance
from src.initial import build_initial
from src.instance import to_local_xy
from src.io import PROJECT_ROOT, load_config, load_sets
from src.solution import Solution

WEB = PROJECT_ROOT / "docs" / "web"
FIGS = PROJECT_ROOT / "results" / "figs"
SKILL_DIR = Path(os.environ.get("OR_RESEARCH_MAP_DIR") or
                 Path.home() / ".workbuddy" / "skills" / "or-research" / "amap")


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


# ---------------------------------------------------------------- 交给 skill 出图

def _run_skill(script: str, *args) -> None:
    path = SKILL_DIR / "scripts" / script
    if not path.is_file():
        raise SystemExit(
            f"找不到 skill 脚本：{path}\n"
            "  安装：把 or-research 放到 ~/.workbuddy/skills/（地图模块在它的 amap/ 子目录），"
            "或设置 OR_RESEARCH_MAP_DIR 指向该模块目录。")
    subprocess.run([sys.executable, str(path), *[str(a) for a in args]],
                   cwd=PROJECT_ROOT, check=True)


def write_html(payload: dict) -> Path:
    """出可提交的无凭据页面 + 数据文件，返回 HTML 路径。

    页面里不写任何 JS key：底图已 base64 内嵌，断网也能开；
    想要可拖拽缩放版用 `--amap-js` 另出一份（不入库）。
    """
    WEB.mkdir(parents=True, exist_ok=True)
    set_id = payload["set"]
    src = WEB / f"solution_set{set_id}.json"
    src.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    pld = WEB / f"solution_set{set_id}_payload.json"
    _run_skill("adapt_tddo.py", src, "-o", pld)

    html = WEB / f"tddo_map_set{set_id}.html"
    _run_skill("build_map.py", pld, "-o", html, "--no-js",
               "--cache-dir", WEB / "basemap")
    return html


def write_amap_html(payload: dict, key: str, secret: str) -> Path:
    """兼容旧调用名；JS 凭据由 skill 从 .env 读，不再由本项目拼进 HTML。"""
    if key != get("AMAP_JS_KEY") or secret != get("AMAP_JS_SECRET"):
        print("  传入的 JS 凭据与 .env 不一致，页面实际使用 .env 中的值")
    set_id = payload["set"]
    out = WEB / f"tddo_map_set{set_id}_amap.html"
    _run_skill("build_map.py", WEB / f"solution_set{set_id}_payload.json",
               "-o", out, "--cache-dir", WEB / "basemap")
    return out


def static_map(payload: dict, key: str) -> Path | None:
    """Web 服务 key 出带线路的静态 PNG；HTTP 客户端复用 skill，不写第二份。"""
    sys.path.insert(0, str(SKILL_DIR / "assets"))
    import amap_client as amap                    # noqa: E402

    paths = []
    for r in payload["final"]["routes"][:4]:      # 高德一次最多 4 条 path
        pts = ";".join(f"{lon:.6f},{lat:.6f}"
                       for lon, lat in r["path"][::max(1, len(r["path"]) // 90)])
        paths.append(f"6,0x1F4E79,0.9,,0:{pts}")
    all_pts = [p for r in payload["final"]["routes"] for p in r["path"]]
    lons, lats = [p[0] for p in all_pts], [p[1] for p in all_pts]
    center = ((min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2)
    span_km = max((max(lons) - min(lons)) * 96, (max(lats) - min(lats)) * 111)
    zoom = 14 if span_km < 3 else 13 if span_km < 7 else 12
    try:
        raw, _ = amap.staticmap(amap.Client(key), center, zoom, 1024, 1024,
                                scale=2, cache_dir=FIGS / "basemap", paths=paths)
    except (RuntimeError, SystemExit) as ex:
        print("  静态图失败：", ex)
        return None
    out = FIGS / f"amap_static_set{payload['set']}.png"
    out.write_bytes(raw)
    print(f"  {out.relative_to(PROJECT_ROOT)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="导出解数据，并用 or-research 的 amap 模块出高德底图")
    ap.add_argument("--set", type=int, default=112)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--static", action="store_true", help="同时用 Web服务 Key 出静态 PNG")
    ap.add_argument("--amap-js", action="store_true",
                    help="额外生成可拖拽版（JS Key 读 .env，产物含凭据、已 gitignore）")
    a = ap.parse_args()

    print("求解并导出…")
    payload = build_payload(a.set, a.seed, a.iters)
    print(f"  初始解 {payload['initial']['cost']} → 最终解 {payload['final']['cost']}")
    html = write_html(payload)
    print(f"  {html.relative_to(PROJECT_ROOT)}（无凭据，可提交）")
    if a.amap_js:
        jk, js = get("AMAP_JS_KEY"), get("AMAP_JS_SECRET")
        if not (jk and js):
            print("  .env 里没有 AMAP_JS_KEY / AMAP_JS_SECRET，跳过可拖拽版"
                  "（模板见 .env.example）")
        else:
            out = write_amap_html(payload, jk, js)
            print(f"  {out.relative_to(PROJECT_ROOT)}（含 Key {mask(jk)}，已 gitignore）")
    if a.static:
        key = get("AMAP_KEY")
        if not key:
            print("  .env 里没有 AMAP_KEY，跳过静态图（模板见 .env.example）")
        else:
            static_map(payload, key)


if __name__ == "__main__":
    main()
