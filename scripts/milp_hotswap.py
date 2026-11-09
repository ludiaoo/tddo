"""M9.16：换电放松下的 MILP 精确解——「论文模型的最优值被高估」是否成立。

## 为什么这个实验是决定性的

M9.15 用 ALNS 测出「允许同节点换机」在 16 算例上降 2.26%，但那是启发式结果：
审稿人可以说「你的启发式只是找到了更好的近似」。而在**已证明最优**的 MILP 上，
如果能构造出一个在放松模型里可行、且成本低于原模型最优值的解，这个说法就无法回避：

    原模型最优 = Z（Gurobi 证明）
    放松模型里有可行解 X，cost(X) < Z
    ⇒ 原模型的最优值在真实运行里被高估了（至少 cost(X) 与 Z 之间的差距）

## 放松了什么（对应 src/milp.py 的 allow_hotswap）

| 原约束 | 放松后 |
|---|---|
| `wl[k,i] + wr[k,i] <= 1` | 去掉（同一节点可先回收再起飞） |
| `wl + wr <= Σx`（卡车到访） | `wl + wr <= 1 + Σx`（一个节点承担两件事） |
| `Td[k,i] <= M(1-wl)` | `Td <= M(1-wl+wr)`（热换点保留落地时刻，Tt 才能等到机） |

换电的时间成本仍然存在：`Tt >= Td + recovery`（等机落地）+ `launch_time`（重新起飞），
与原模型一致，没有白拿时间。

## 口径

与 `scripts/run_milp.py` 完全一致：**关闭禁飞区**（MILP 不建模 NFZ），同一个算例、
同一组参数、同一 Gurobi 设置。两边唯一的差别就是那三条约束。

用法：
    python -m scripts.milp_hotswap               # 4 个 10 客户算例
    python -m scripts.milp_hotswap --tl 600      # 放宽时间上限
"""

from __future__ import annotations

import argparse
import copy
import csv

from src.alns import disable_nfz
from src.evaluator import evaluate
from src.generator import build_instance
from src.io import PROJECT_ROOT, load_config, load_sets
from src.milp import solve_milp

OUT = PROJECT_ROOT / "results" / "milp_hotswap.csv"
SMALL = [1, 2, 3, 4]


def _inst(sid: int, hotswap: bool):
    cfg = copy.deepcopy(load_config())
    cfg["fleet"]["allow_hotswap"] = hotswap
    by_id = {s["id"]: s for s in load_sets()}
    return disable_nfz(build_instance(by_id[sid], cfg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tl", type=float, default=300.0, help="MILP 时间上限（秒/算例/臂）")
    args = ap.parse_args()

    rows = []
    print(f"{'Set':>3} | {'原模型最优':>11} {'gap':>5} {'cpu':>5} | "
          f"{'换电解':>11} {'gap':>5} {'cpu':>5} | {'降幅':>7} | 复核")
    print("-" * 78)

    for sid in SMALL:
        base_inst = _inst(sid, False)
        r0 = solve_milp(base_inst, time_limit=args.tl, mip_gap=0.0)
        hs_inst = _inst(sid, True)
        r1 = solve_milp(hs_inst, time_limit=args.tl, mip_gap=0.0)

        note = ""
        if r1.solution is not None:
            # 用独立评估器复核：MILP 说可行不够，评估器也要认可（两者是两套实现）
            res, c_eval = evaluate(hs_inst, r1.solution)
            ok = abs(res.feasible) and abs(c_eval - r1.cost) < 1e-4 * max(1.0, r1.cost)
            note = "✓ 可行" if ok else f"✗ {res.all_violations()[:1]}"
            if not ok:
                print(f"   ⚠ Set{sid} 换电解在评估器里没通过：{res.all_violations()[:3]}  "
                      f"MILP={r1.cost:.4f} eval={c_eval:.4f}")
        else:
            note = f"无解({r1.status})"

        delta = (r1.cost / r0.cost - 1) * 100 if r0.cost == r0.cost and r1.cost == r1.cost else float("nan")
        rows.append({
            "set": sid,
            "n_customers": base_inst.n_customers,
            "n_trucks": base_inst.n_trucks,
            "base_cost": round(r0.cost, 4),
            "base_bound": round(r0.bound, 4),
            "base_gap_pct": round(r0.gap * 100, 4),
            "base_status": r0.status,
            "hotswap_cost": round(r1.cost, 4),
            "hotswap_bound": round(r1.bound, 4),
            "hotswap_gap_pct": round(r1.gap * 100, 4),
            "hotswap_status": r1.status,
            "delta_pct": round(delta, 4),
            "eval_note": note,
        })
        print(f"{sid:>3} | {r0.cost:>11.4f} {r0.gap*100:>4.2f}% {r0.cpu:>4.0f}s | "
              f"{r1.cost:>11.4f} {r1.gap*100:>4.2f}% {r1.cpu:>4.0f}s | {delta:>+6.2f}% | {note}",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["hotswap_cost"] < r["base_cost"] - 1e-6 and r["eval_note"].startswith("✓")]
    print(f"\n换电解严格优于原模型最优的算例：{len(ok)}/{len(rows)}")
    if ok:
        best = max(ok, key=lambda r: -r["delta_pct"])
        print(f"  最大降幅 Set{best['set']}：{best['delta_pct']:+.2f}%"
              f"（{best['base_cost']:.4f} → {best['hotswap_cost']:.4f}）")
    print(f"已写出 {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
