"""TDDO 命令行入口。

设计原则：**薄封装**。这里不放任何新算法，只把已有的 `src.*` 与 `scripts.*`
能力收拢成一条命令，方便安装、演示与容器化。原始调用方式
`python -m scripts.bench ...` 继续有效。

    tddo doctor                      环境体检
    tddo solve --set 104 --iters 300 单算例求解并写 JSON
    tddo bench --sets 104 --iters 300 批量实验（参数透传给 scripts.bench）
    tddo baseline                    OR-Tools 外部基线
    tddo export-web --set 112        高德底图可视化
    tddo check                       项目自检 + golden_check
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .energy import driver_cost, truck_cost

FALLBACK_VERSION = "0.1.0"


def _json_default(obj):
    """JSON 序列化兜底：numpy 标量转 Python 标量，其他类型照常报错。"""
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("tddo")
    except Exception:
        return FALLBACK_VERSION


def _delegate(module: str, argv: Sequence[str]) -> int:
    """临时改写 sys.argv 后调用既有脚本的 main()，保持参数语义不变。"""
    old_argv = sys.argv[:]
    sys.argv = [module, *argv]
    try:
        imported = importlib.import_module(module)
        imported.main()
        return 0
    except SystemExit as exc:  # argparse / project_check 用 SystemExit 表示退出码
        return int(exc.code or 0)
    finally:
        sys.argv = old_argv


def _cmd_doctor(args: argparse.Namespace) -> int:
    from .env import env_file_status
    from .io import DEFAULT_CONFIG, DEFAULT_DATA_ROOT, DEFAULT_SETS, PROJECT_ROOT

    print(f"TDDO {_version()} | Python {sys.version.split()[0]}")
    print(f"项目根目录: {PROJECT_ROOT}")
    missing: list[str] = []

    print("\n[依赖]")
    for name, optional in (
        ("numpy", False), ("pandas", False), ("matplotlib", False),
        ("alns", False), ("gurobipy", False), ("ortools", True),
    ):
        found = importlib.util.find_spec(name) is not None
        if not found and not optional:
            missing.append(name)
        mark = "✓" if found else ("·" if optional else "✗")
        suffix = "（可选：外部基线）" if optional and not found else ""
        print(f"  {mark} {name:<12}{suffix}")

    print("\n[配置与数据]")
    for label, path in (
        ("config.toml", DEFAULT_CONFIG),
        ("instances.toml", DEFAULT_SETS),
        ("mFSTSP 数据", DEFAULT_DATA_ROOT),
    ):
        hit = path.exists()
        if not hit:
            missing.append(str(path.relative_to(PROJECT_ROOT)))
        mark = "✓" if hit else "·"
        note = "（运行时挂载；求解需要）" if "mFSTSP" in label and not hit else ""
        print(f"  {mark} {label:<14}{path.relative_to(PROJECT_ROOT)}{note}")

    print("\n[密钥与许可证]")
    status = env_file_status()
    print(f"  {'✓' if status['exists'] else '·'} .env "
          f"{'已配置 ' + ', '.join(status['keys']) if status['exists'] and status['keys'] else '未配置（仅高德功能需要）'}")
    lic = os.environ.get("GRB_LICENSE_FILE")
    print(f"  {'✓' if lic else '·'} Gurobi 许可证 "
          f"{'由 GRB_LICENSE_FILE 指定' if lic else '未设置（仅精确求解需要）'}")

    if missing and args.strict:
        print(f"\n缺少必要项: {', '.join(missing)}")
        return 1
    print("\n体检完成。" + ("（--strict：有缺失项）" if missing and args.strict else ""))
    return 0


def _cmd_solve(args: argparse.Namespace) -> int:
    from .alns import solve
    from .evaluator import evaluate
    from .generator import build_instance
    from .io import PROJECT_ROOT, load_config, load_sets

    cfg = load_config()
    set_map = {s["id"]: s for s in load_sets()}
    if args.set_id not in set_map:
        raise SystemExit(f"未知算例 set {args.set_id}；可选 {sorted(set_map)}")

    inst = build_instance(set_map[args.set_id], cfg)
    if args.no_drone:
        inst.allow_drones = False
    if args.no_detour:
        inst.allow_detour = False

    result = solve(inst, seed=args.seed, iters=args.iters)
    evs, total = evaluate(inst, result.best)
    p = inst.params

    routes = []
    for route, ev in zip(result.best.routes, evs.routes):
        sorties = []
        for s in route.sorties:
            launch_node = int(route.nodes[s.launch])
            recover_node = int(route.nodes[s.recover])
            sorties.append({
                "launch_pos": int(s.launch),
                "recover_pos": int(s.recover),
                "launch_node": launch_node,
                "recover_node": recover_node,
                "customers": [int(c) for c in s.customers],
                "sequence": [launch_node, *[int(c) for c in s.customers], recover_node],
            })
        routes.append({
            "truck": int(route.truck),
            "truck_nodes": [int(n) for n in route.nodes],
            "sorties": sorties,
            "truck_distance_m": round(ev.truck_distance, 2),
            "drone_distance_m": round(ev.drone_distance, 2),
            "drone_energy_kwh": round(ev.drone_energy, 6),
            "finish_time_s": round(ev.finish_time, 2),
            "n_sorties": ev.n_sorties,
            "feasible": ev.feasible,
        })

    payload = {
        "set": inst.set_id,
        "instance": inst.base.name,
        "city": inst.base.city,
        "seed": args.seed,
        "iters": args.iters,
        "n_customers": inst.n_customers,
        "n_trucks": inst.n_trucks,
        "initial_cost": round(result.initial_cost, 4),
        "best_cost": round(result.best_cost, 4),
        "recheck_cost": round(total, 4),
        "saving_pct": round(result.saving * 100, 2),
        "feasible": evs.feasible,
        "violations": evs.all_violations(),
        "cost_breakdown": {
            "drone_energy": round(sum(e.drone_energy for e in evs.routes) * p.c_drone, 4),
            "truck_fuel": round(truck_cost(sum(e.truck_distance for e in evs.routes), p), 4),
            "driver": round(driver_cost(sum(e.finish_time for e in evs.routes), p), 4),
        },
        "routes": routes,
    }

    out = Path(args.out) if args.out else PROJECT_ROOT / "results" / f"cli_solution_set{args.set_id}.json"
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    print(inst.describe())
    print(f"初始成本 {result.initial_cost:.2f} → 最优成本 {result.best_cost:.2f}"
          f"（节省 {result.saving * 100:.1f}%）")
    print(f"路线 {len(routes)} 条，架次 {sum(r['n_sorties'] for r in routes)} 个，"
          f"可行性 {'通过' if evs.feasible else '未通过'}")
    print(f"已写出 {out.relative_to(PROJECT_ROOT) if out.is_relative_to(PROJECT_ROOT) else out}")
    if not evs.feasible:
        for v in evs.all_violations()[:5]:
            print(f"  ! {v}")
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tddo",
        description="卡车-无人机联合配送优化器：安装、求解、实验、可视化与自检的统一入口",
    )
    parser.add_argument("--version", action="version", version=_version())
    sub = parser.add_subparsers(dest="command", required=True)

    p_doc = sub.add_parser("doctor", help="检查依赖、配置、数据与密钥状态")
    p_doc.add_argument("--strict", action="store_true", help="缺少配置/数据时返回非零退出码")
    p_doc.set_defaults(func=_cmd_doctor)

    p_solve = sub.add_parser("solve", help="单算例求解，写出结构化 JSON")
    p_solve.add_argument("--set", dest="set_id", type=int, default=104, help="算例编号，默认 104")
    p_solve.add_argument("--seed", type=int, default=0)
    p_solve.add_argument("--iters", type=int, default=300)
    p_solve.add_argument("--out", default=None, help="输出路径（默认 results/cli_solution_set<id>.json）")
    p_solve.add_argument("--no-drone", action="store_true", help="纯卡车对照")
    p_solve.add_argument("--no-detour", action="store_true", help="禁飞区不可绕行对照")
    p_solve.set_defaults(func=_cmd_solve)

    for name, module, help_text in (
        ("bench", "scripts.bench", "批量实验，参数原样透传给 scripts.bench"),
        ("baseline", "scripts.ortools_baseline", "OR-Tools 纯卡车 VRPTW 外部基线"),
        ("export-web", "scripts.export_web", "导出解数据并生成高德底图页面"),
        ("check", "scripts.project_check", "项目自检（含 golden_check，可加 --with-tests）"),
    ):
        # 关闭子解析器自带的 -h，让 --help 透传给被委托的脚本
        p = sub.add_parser(name, help=help_text, add_help=False)
        p.add_argument("argv", nargs=argparse.REMAINDER, help=help_text)
        p.set_defaults(func=lambda a, m=module: _delegate(m, a.argv))

    return parser


DELEGATED = {
    "bench": "scripts.bench",
    "baseline": "scripts.ortools_baseline",
    "export-web": "scripts.export_web",
    "check": "scripts.project_check",
}


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # 委托类命令把命令名之后的全部参数原样交给目标脚本（含 --help）。
    # argparse 的 REMAINDER 无法可靠吞掉选项，所以在解析之前先分流。
    if argv and argv[0] in DELEGATED:
        return _delegate(DELEGATED[argv[0]], argv[1:])
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
