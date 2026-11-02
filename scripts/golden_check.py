"""黄金结果：固定 (算例, 种子, 轮数) 的 ALNS 成本。

注意：基线用 `tune=False` 跑（论文原始参数）。这样它只守「性能优化不改结果」，
参数修正属于另一件事，由 benchmark 单独对照。


用途：性能优化**不允许改变结果**。改之前跑一次存盘，改之后重跑比对，
必须逐位相同（ALNS 是确定性的：同种子 → 同随机流 → 同解）。
若有差异，说明优化动了语义（通常是禁飞判定或早退逻辑），必须查清再继续。

用法：
    python -m scripts.golden_check --save     # 存基线（改代码前）
    python -m scripts.golden_check            # 比对（改代码后）
"""

from __future__ import annotations

import argparse
import json
import time

from src.alns import solve
from src.generator import build_instance
from src.io import PROJECT_ROOT, load_config, load_sets

OUT = PROJECT_ROOT / "results" / "golden.json"

# (set_id, seed, iters)：覆盖四个规模，且 13 号是「60 轮就早熟」的那类
CASES = [(1, 0, 80), (4, 0, 80), (5, 0, 60), (9, 0, 40), (13, 0, 25)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="存为基线")
    args = ap.parse_args()

    cfg = load_config()
    by_id = {s["id"]: s for s in load_sets()}

    got: dict[str, float] = {}
    t_all = time.time()
    for sid, seed, iters in CASES:
        inst = build_instance(by_id[sid], cfg)
        t0 = time.time()
        r = solve(inst, seed=seed, iters=iters, tune=False)
        got[f"set{sid:02d}_seed{seed}_it{iters}"] = round(float(r.best_cost), 9)
        print(
            f"Set{sid:2d} seed={seed} iters={iters:3d}  cost={r.best_cost:12.6f}  "
            f"saving={r.saving:5.1%}  {time.time()-t0:5.1f}s",
            flush=True,
        )
    print(f"总耗时 {time.time()-t_all:.1f}s")

    if args.save:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(got, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已存基线 {OUT.relative_to(PROJECT_ROOT)}")
        return

    if not OUT.exists():
        raise SystemExit("没有基线，先跑 --save")
    base = json.loads(OUT.read_text(encoding="utf-8"))
    bad = []
    for k, v in base.items():
        new = got.get(k)
        if new is None or abs(new - v) > 1e-6:
            bad.append((k, v, new))
    if bad:
        print("\n❌ 结果变了（优化改了语义）：")
        for k, old, new in bad:
            print(f"  {k}: {old} → {new}")
        raise SystemExit(1)
    print("\n✅ 与基线逐位一致")


if __name__ == "__main__":
    main()
