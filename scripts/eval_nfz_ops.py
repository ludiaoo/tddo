"""LLM 提案算子的三关评测（M9.7）：能不能用 → 同状态下好不好 → 全程序是否更优。

为什么要三关，而不是直接跑 bench
--------------------------------
直接跑 16 算例 × 10 种子，一个候选 ~16 分钟、噪声 1.3%。7 个候选就是两小时，
而且很可能被一个「看起来对、实际丢客户」的算子污染结果（`78bb3d5` 踩过）。

所以按成本从低到高排：

| 关 | 花多少 | 查什么 | 不过就淘汰的理由 |
|---|---|---|---|
| ① 契约校验 | 秒级 | 客户守恒、不崩、修复后必须还原为可行解 | 破坏算子把客户弄丢了，后面所有数字都无意义 |
| ② 同状态配对 | 分钟级 | 同一搜索状态下，候选 vs `random` 各跑 50 轮的配对差 | 连「同一个状态换个破坏方式」都赢不了，全程序更难赢 |
| ③ 全程序 | 十几分钟/个 | 只把 ① ② 的头部放进 2000 轮 × 10 种子配对 | 局部好≠长程好（`worst|equal` 的反例） |

     python -m scripts.eval_nfz_ops --stage validate
     python -m scripts.eval_nfz_ops --stage screen --sets 5,7,9 --seeds 2
     python -m scripts.eval_nfz_ops --stage show

⚠ 第二关的相对基准是 `random|equal`（现状里表现最好的固定组合），不是自适应池——
先保证「同等条件下更好」，再谈并进池子。
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import pandas as pd

from src.alns import DESTROY, REPAIR, solve
from src.evaluator import evaluate
from src.generator import build_instance
from src.initial import build_initial
from src.io import PROJECT_ROOT, load_config, load_sets
from src.nfz_ops import CANDIDATES
from src.operators import _all_customers

OUT = PROJECT_ROOT / "results" / "ml" / "nfz_ops_screen.csv"
MAIN_ITERS = 1500
ALPHA = (1e-3) ** (1.0 / MAIN_ITERS)

BASELINE = "random"          # 第二关的相对基准（固定用 equal 修复）


def _ctx():
    cfg = load_config()
    return cfg, {s["id"]: s for s in load_sets()}


# ---------------------------------------------------------------- ① 契约校验

def validate(reps: int = 5) -> bool:
    """契约校验：**客户守恒是硬约束，修复成功与否是软指标**。

    为什么软：主循环里修复失败（返回 None 或不可行）本来就会被拒绝，不算错误；
    真正致命的是「悄悄丢客户」——那会让解看起来更便宜，而成本统计是错的（`78bb3d5`）。
    所以这里只把守恒/崩溃当红线，修复成功率拿来和 `random` 基准比。
    """
    cfg, by_id = _ctx()
    print(f"=== 第 ① 关：契约校验（16 算例 × {len(CANDIDATES) + 1} 算子 × {reps} 次）===")
    rows = []
    hard_fail = 0
    ops = {"random(基准)": DESTROY["random"], **CANDIDATES}

    for sid in sorted(by_id):
        inst = build_instance(by_id[sid], cfg)
        base, _, _ = build_initial(inst)
        served_before = set(base.served_customers())
        for name, op in ops.items():
            for rep in range(reps):
                rng = np.random.default_rng(1000 * sid + rep)
                note, cons_ok, repaired = "", True, False
                try:
                    r = op(inst, base, rng, 0.30)
                    kept = set(r.sol.served_customers())
                    removed = set(r.removed)
                    if not removed <= served_before:
                        cons_ok, note = False, "删了不在解里的客户"
                    elif kept != served_before - removed:
                        cons_ok, note = False, f"丢失 {len(served_before - removed - kept)} 个客户"
                    else:
                        cand = REPAIR["equal"](inst, r, rng)
                        if cand is not None:
                            res, _ = evaluate(inst, cand)
                            repaired = bool(res.feasible)
                            if not repaired:
                                note = res.all_violations()[0][:60]
                        else:
                            note = "修复返回 None"
                except Exception as exc:                 # noqa: BLE001
                    cons_ok, note = False, f"崩溃 {type(exc).__name__}: {exc}"
                if not cons_ok:
                    hard_fail += 1
                    print(f"  ✗ set{sid:>2} {name:<16} {note}")
                rows.append(dict(set=sid, op=name, rep=rep, cons_ok=cons_ok,
                                 repaired=repaired, n_removed=len(r.removed) if cons_ok else 0,
                                 note=note))
        print(f"  set{sid:>2} 完成")

    df = pd.DataFrame(rows)
    df.to_csv(OUT.with_name("nfz_ops_validate.csv"), index=False)
    g = df.groupby("op").agg(守恒=("cons_ok", "mean"), 修复成功=("repaired", "mean"),
                             平均删客户=("n_removed", "mean"))
    g[["守恒", "修复成功"]] = (g[["守恒", "修复成功"]] * 100).round(1)
    print("\n", g.round(2).sort_values("修复成功", ascending=False).to_string())
    base_rate = g.loc["random(基准)", "修复成功"]
    weak = g[(g.index != "random(基准)") & (g.修复成功 < base_rate - 15)]
    if len(weak):
        print(f"\n⚠ 修复成功率低于基准 {base_rate:.0f}% 超过 15pp：{list(weak.index)}")
    print(f"\n结论: 客户守恒 {'全部通过' if hard_fail == 0 else f'{hard_fail} 次失败'}")
    return hard_fail == 0


# ---------------------------------------------------------------- ② 同状态配对

def _states(inst, seed: int, marks, iters: int = MAIN_ITERS) -> dict:
    got = {}

    def tr(rec):
        if rec["it"] in marks:
            got[rec["it"]] = {"sol": copy.deepcopy(rec["sol"]), "cur": float(rec["cur_cost"]),
                              "best": float(rec["best_cost"]), "T": float(rec["T"])}

    solve(inst, seed=seed, iters=iters, trace=tr)
    return got


def screen(sets: list[int], seeds: int, marks=(30, 100, 300, 800, 1300), K: int = 50) -> None:
    cfg, by_id = _ctx()
    DESTROY.update(CANDIDATES)                    # 让 fixed_pair 能查到候选
    arms = [BASELINE, *CANDIDATES]
    rows = []
    t0 = time.time()
    for sid in sets:
        inst = build_instance(by_id[sid], cfg)
        for seed in range(seeds):
            sts = _states(inst, seed, marks)
            for it, st in sorted(sts.items()):
                for arm in arms:
                    r = solve(inst, seed=seed, iters=K, init_sol=copy.deepcopy(st["sol"]),
                              t0=st["T"], alpha=ALPHA, fixed_pair=f"{arm}|equal")
                    rows.append(dict(set=sid, seed=seed, it=it, arm=arm,
                                     cost=float(r.best_cost), cur=st["cur"]))
                print(f"  set{sid} seed{seed} it{it}: {len(arms)} 臂", flush=True)
        print(f"[set{sid}] 累计 {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n写出 {OUT}（{len(df)} 行）")
    show()


def show() -> None:
    df = pd.read_csv(OUT)
    piv = df.pivot_table(index=["set", "seed", "it"], columns="arm", values="cost")
    base = piv[BASELINE]
    print(f"=== 第 ② 关：同状态下相对 `{BASELINE}|equal` 的配对差（负=更好）===")
    rows = []
    for arm in piv.columns:
        if arm == BASELINE:
            continue
        d = (piv[arm] / base - 1) * 100
        rows.append(dict(op=arm, 配对均值=d.mean(), 中位=d.median(),
                         更好=int((d < -1e-9).sum()), 更差=int((d > 1e-9).sum()),
                         最好=d.min(), 最差=d.max()))
    t = pd.DataFrame(rows).sort_values("配对均值")
    print(t.round(2).to_string(index=False))
    try:
        from scipy.stats import wilcoxon
        print("\nWilcoxon p（vs 基准）:")
        for arm in t.op:
            d = (piv[arm] / base - 1) * 100
            print(f"  {arm:<16} p={wilcoxon(d).pvalue:.4f}")
    except Exception:                              # noqa: BLE001
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["validate", "screen", "show", "all"])
    ap.add_argument("--sets", default="5,7,9")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--marks", default="30,100,300,800,1300")
    ap.add_argument("--K", type=int, default=50)
    args = ap.parse_args()

    if args.stage in ("validate", "all"):
        validate()
    if args.stage in ("screen", "all"):
        screen([int(x) for x in args.sets.split(",")], args.seeds,
               tuple(int(x) for x in args.marks.split(",")), args.K)
    elif args.stage == "show":
        show()


if __name__ == "__main__":
    main()
