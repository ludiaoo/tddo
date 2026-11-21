"""W1 遗留项：`ft` 的逐架次口径，到底松在哪里、值不值得补。

## 问题

`milp_hs.py` 里 `ft[k,i]` 是"卡车在节点 i 交给无人机的货重"，只受两条约束：

- 每次起飞 ≤ `drone_capacity`（换电时一块电池能带的量）；
- 全局 `Σ_i ft[k,i] = Σ_j f_j·(y − wr)`。

于是模型可以在**更早的起飞点多卸一点、后面的架次少卸**：总量守恒、每架次不超容，
但卡车在某一段提前"变轻"了。这是 W1 §8 记的那条"够用但偏弱"。

## 这次做了什么

1. 加一条**有效**的首跳切平面（`tight_ft=True`，默认关闭）：
   无人机从 i 起飞，第一段弧 i→j 上带的就是客户 j 的货，所以 `ft[k,i] ≥ f_j`。
2. 在 4 个 10 客户算例上跑「关 / 开」两组，看最优值变不变。
   - 变了 → 说明旧松弛被利用过，切平面有实质作用；
   - 没变 → 说明这批算例的原最优本来就满足逐架次口径，松弛只是理论上的口子。

全量版本（把 ft 与"该架次服务哪些客户"逐架次绑定）需要引入架次—客户指派变量，
属于 W3 的模型重构，不在这里做。

用法：
    python -m scripts.milp_ft_check
    python -m scripts.milp_ft_check --sets 3,4 --time-limit 240
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from src.alns import disable_nfz
from src.evaluator import evaluate
from src.generator import build_instance
from src.io import PROJECT_ROOT, load_config, load_sets
from src.milp_hs import solve_milp_hs

RESULTS = PROJECT_ROOT / "results"


def _inst(sid: int):
    by_id = {s["id"]: s for s in load_sets()}
    inst = disable_nfz(build_instance(by_id[sid], load_config()))
    # 放开换电：只有在放松后的模型上，ft 的口子才有机会被利用
    return replace(inst, params=replace(inst.params, allow_hotswap=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="1,2,3,4")
    ap.add_argument("--time-limit", type=float, default=300.0)
    args = ap.parse_args()
    ids = [int(x) for x in args.sets.split(",")]

    rows = []
    for sid in ids:
        inst = _inst(sid)
        row = {"set": sid, "n_customers": inst.n_customers}
        for tag, tight in (("loose", False), ("tight", True)):
            r = solve_milp_hs(inst, time_limit=args.time_limit, allow_hotswap=True, tight_ft=tight)
            ok, cost = (False, float("nan"))
            if r.solution is not None:
                res, cost = evaluate(inst, r.solution)
                ok = res.feasible
            row[f"{tag}_cost"] = round(float(r.cost), 4) if r.solution is not None else None
            row[f"{tag}_status"] = r.status
            row[f"{tag}_feasible"] = ok
            row[f"{tag}_n_sorties"] = (sum(len(rt.sorties) for rt in r.solution.routes)
                                       if r.solution is not None else None)
        a, b = row.get("loose_cost"), row.get("tight_cost")
        row["delta"] = (round(b - a, 4) if (a is not None and b is not None) else None)
        rows.append(row)
        print(f"Set{sid:<3} 松 {a} ({row['loose_status']}, 可行={row['loose_feasible']})  "
              f"紧 {b} ({row['tight_status']}, 可行={row['tight_feasible']})  "
              f"差 {row['delta']}")

    df = pd.DataFrame(rows)
    out = RESULTS / "milp_ft_check.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    same = sum(1 for r in rows if r["delta"] == 0.0)
    print(f"\n{len(rows)} 个算例里 {same} 个最优值完全不变")
    print(f"写出 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
