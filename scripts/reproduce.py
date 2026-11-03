"""一键复现：按顺序把 M6–M7 的全部实验跑一遍。

    python -m scripts.reproduce              # 全量（约 1.6 小时，20 核机器上）
    python -m scripts.reproduce --quick      # 冒烟：小轮数 + 少种子，约 4 分钟，只验证链路
    python -m scripts.reproduce --only bench,summarize
    python -m scripts.reproduce --skip sensitivity,milp

每一步都是**独立的 `python -m scripts.xxx`**，本脚本只负责排序、计时与日志，
不复制任何算法逻辑——单步失败时可以照着打印出的命令重跑那一步。

`--quick` 只用于验证链路，**不会覆盖任何主结果**：bench / pso 写到 `*_quick.csv`，
sensitivity 与 routes 只读缓存重画图，原始 CSV 一个字节都不动。

产物与耗时（20 核 / Windows，实测）：

| 步骤 | 命令 | 产物 | 全量耗时 |
|---|---|---|---|
| bench | `scripts.bench --exp main/nodetour/truckonly/ablate` | `results/bench_*.csv` | ≈ 55 min |
| pso | `scripts.bench_pso` | `results/bench_pso.csv` | ≈ 8 min |
| summarize | `scripts.summarize_bench` | `results/table3_v2.csv/.md` | 秒级 |
| sensitivity | `scripts.sensitivity --which fig9/fig10/fig11/fig11b`（四进程并行） | `results/sens_*.csv` + `figs/fig9–12` | ≈ 15 min |
| routes | `scripts.plot_routes --refresh` | `results/figs/routes_fig8.png` | ≈ 4 min |
| milp | `scripts.run_milp` | `results/milp_check.csv` | ≈ 9 min |

`sensitivity` 的四个子实验写各自的 CSV，互不依赖，所以并行起四个进程。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from src.io import PROJECT_ROOT

PY = sys.executable

# name -> (并发组, 预计分钟)：组内并行，组间串行
STEPS: dict[str, tuple[list[list[str]], float]] = {
    "bench": (
        [
            ["-m", "scripts.bench", "--exp", "main", "--seeds", "10", "--iters", "2000",
             "--refresh", "--workers", "10"],
            ["-m", "scripts.bench", "--exp", "nodetour", "--seeds", "10", "--iters", "2000",
             "--no-detour", "--refresh", "--workers", "10"],
            ["-m", "scripts.bench", "--exp", "truckonly", "--seeds", "10", "--iters", "2000",
             "--no-drone", "--refresh", "--workers", "10"],
            ["-m", "scripts.bench", "--exp", "ablate", "--seeds", "5", "--iters", "2000",
             "--no-tune", "--refresh", "--workers", "10"],
        ],
        55.0,
    ),
    "pso": ([["-m", "scripts.bench_pso", "--seeds", "10", "--iters", "2000", "--refresh"]], 8.0),
    "summarize": ([["-m", "scripts.summarize_bench"]], 0.1),
    "sensitivity": (
        [
            ["-m", "scripts.sensitivity", "--which", "fig9", "--refresh"],
            ["-m", "scripts.sensitivity", "--which", "fig10", "--refresh"],
            ["-m", "scripts.sensitivity", "--which", "fig11", "--refresh"],
            ["-m", "scripts.sensitivity", "--which", "fig11b", "--refresh"],
        ],
        15.0,
    ),
    "routes": ([["-m", "scripts.plot_routes", "--refresh"]], 4.0),
    "milp": ([["-m", "scripts.run_milp"]], 9.0),
}

ORDER = ["bench", "pso", "summarize", "sensitivity", "routes", "milp"]
PARALLEL = {"sensitivity"}          # 组内并行的步骤

# 冒烟模式：只跑链路，数字不可用于报告。
# ⚠ 所有 quick 步骤都**不得覆盖主结果的 CSV**：bench 用 --exp quick、pso 用 --exp quick，
#   sensitivity 与 routes 只读缓存重画图（不加 --refresh）。
QUICK: dict[str, list[list[str]]] = {
    "bench": [["-m", "scripts.bench", "--exp", "quick", "--seeds", "1", "--iters", "60",
               "--sets", "1,5", "--refresh", "--workers", "2"]],
    "pso": [["-m", "scripts.bench_pso", "--seeds", "1", "--iters", "200", "--sets", "1,5",
             "--exp", "quick", "--refresh"]],
    "sensitivity": [["-m", "scripts.sensitivity", "--which", "fig9"]],
    "routes": [["-m", "scripts.plot_routes"]],
}


def _spawn(argv: list[str]):
    cmd = [PY, *argv]
    print(f"\n$ {' '.join(cmd[1:])}", flush=True)
    # 子进程的 stdout 是管道，默认块缓冲 → 进度要等退出才出现；这里强制无缓冲
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    return cmd, subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                 errors="replace", env=env)


def _collect(cmd, proc, log) -> bool:
    t0 = time.time()
    for line in proc.stdout:
        print("  " + line.rstrip(), flush=True)
        log.write(line)
    log.flush()
    ok = proc.wait() == 0
    print(f"  → {'完成' if ok else '失败'}（{time.time() - t0:.0f}s）", flush=True)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="冒烟模式：小轮数、少种子、少算例")
    ap.add_argument("--only", default="", help="只跑这些步骤，逗号分隔")
    ap.add_argument("--skip", default="", help="跳过这些步骤，逗号分隔")
    args = ap.parse_args()

    if args.quick:
        plan = [(k, 1.0, QUICK[k]) for k in ORDER if k in QUICK]
    else:
        plan = [(k, STEPS[k][1], STEPS[k][0]) for k in ORDER]

    only = {x.strip() for x in args.only.split(",") if x.strip()}
    skip = {x.strip() for x in args.skip.split(",") if x.strip()}
    plan = [(k, est, cmds) for k, est, cmds in plan
            if (not only or k in only) and k not in skip]

    total = sum(est for _, est, _ in plan)
    print(f"计划：{', '.join(k for k, _, _ in plan)}（预计 {total:.0f} min）")

    log_path = PROJECT_ROOT / "results" / "reproduce.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    t_all = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        for name, _est, cmds in plan:
            print(f"\n=== {name} ===", flush=True)
            if name in PARALLEL:
                procs = [_spawn(argv) for argv in cmds]
                oks = [_collect(cmd, proc, log) for cmd, proc in procs]
                if not all(oks):
                    failed.append(name)
                continue
            for argv in cmds:
                cmd, proc = _spawn(argv)
                if not _collect(cmd, proc, log):
                    failed.append(name)      # 同组前一条失败就不继续
                    break
    print(f"\n总耗时 {(time.time() - t_all) / 60:.1f} min，日志 {log_path}")
    if failed:
        print(f"失败步骤：{', '.join(failed)}")
        raise SystemExit(1)
    print("全部完成。表在 results/table3_v2.csv，图在 results/figs/。")


if __name__ == "__main__":
    main()
