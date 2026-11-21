"""项目自检：一条命令看清"这个项目现在是什么状态"。

为什么需要它：这个仓库的产物分散在 `src/`、`scripts/`、`results/`、24 份文档里，
每次改完东西、要交出去之前，第一件事是确认**该在的都在、该对的都对**。
手工 `ls` 一遍容易漏，所以固化成脚本。

检查四类：
1. **规模**：代码/测试/脚本行数与文件数（不追求精确到个位，量级对就行）。
2. **关键产物**：主结果 CSV、四张主图、复现图、MILP 结果是否齐。
3. **一致性**：`golden_check`（默认口径逐位回归）是否通过。
4. **测试**（可选，`--with-tests`）：跑一遍快速测试。

用法：
    python -m scripts.project_check              # 快检（几秒）
    python -m scripts.project_check --with-tests # 含全部快速测试（约 20 s）
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from src.io import PROJECT_ROOT

# 必须存在的关键产物（缺了就说明某一步没跑或被覆盖）
REQUIRED = {
    "主结果": ["table3_v2.csv", "table3_v2.md", "milp_check.csv", "milp_hotswap.csv"],
    "扩展结果": ["bench_main.csv", "bench_hotswap.csv", "bench_m2.csv",
                 "native_check.csv", "hover_effect.csv", "battery_effect.csv",
                 "robustness.csv", "spatial_check.csv"],
    "主图": ["figA_hotswap_timeline.png", "figB_benefit_bounds.png",
             "figC_battery_stock.png", "figD_structure.png"],
    "复现对标图": ["routes_fig8.png", "fig9_capacity_demand.png",
                   "fig10_drone_payload.png", "fig11_nfz_sensitivity.png",
                   "fig12_mission_count.png"],
    "外部基线": ["ortools_baseline.csv", "ortools_baseline.md"],
    "规模压测": ["scale_bench.csv", "scale_bench.md"],
    "规模图": ["figF_scale_time.png"],
}
DOCS = ["README.md", "LICENSE", "CITATION.cff", ".github/workflows/ci.yml",
        ".env.example", "pyproject.toml", "Dockerfile", ".dockerignore",
        "src/env.py", "src/cli.py", "tests/test_cli.py",
        "docs/项目书.md", "docs/成果整合.md", "docs/求职_项目经历.md", "docs/求职_能力评估.md",
        "docs/求职_外部基线.md", "docs/求职_业务场景.md", "docs/求职_文献定位.md",
        "docs/工程手册.md",
        "docs/设计笔记/README.md", "docs/设计笔记/结论总览.md",
        "docs/设计笔记/W1_换电模型设计.md",
        "docs/设计笔记/M9.23_天河真实路网算例.md",
        "docs/设计笔记/M9.24_天河重跑.md",
        "docs/论文_大纲.md", "docs/论文_正文_引言.md",
        "docs/web/tddo_map_set112.html"]


def _count(pattern: str) -> tuple[int, int]:
    files = sorted(PROJECT_ROOT.glob(pattern))
    return len(files), sum(len(f.read_text(encoding="utf-8", errors="replace").splitlines())
                           for f in files)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-tests", action="store_true", help="跑一遍快速测试")
    args = ap.parse_args()

    ok = True
    print("=" * 62)
    print("无人机-卡车联合配送 · 项目自检")
    print("=" * 62)

    print("\n[规模]")
    for label, pat in (("src", "src/**/*.py"), ("scripts", "scripts/*.py"), ("tests", "tests/*.py")):
        n, lines = _count(pat)
        print(f"  {label:<8} {n:3d} 个文件  {lines:5d} 行")

    print("\n[产物]")
    for group, names in REQUIRED.items():
        for name in names:
            path = (PROJECT_ROOT / "results" / "figs" / name) if name.endswith(".png") \
                else (PROJECT_ROOT / "results" / name)
            hit = path.exists()
            ok = ok and hit
            print(f"  {'✓' if hit else '✗'} {group:<6} {name}")
    for name in DOCS:
        hit = (PROJECT_ROOT / name).exists()
        ok = ok and hit
        print(f"  {'✓' if hit else '✗'} 文档    {name}")

    print("\n[密钥配置]")
    from src.env import env_file_status
    st = env_file_status()
    if st["exists"]:
        print(f"  ✓ .env 存在，已填：{', '.join(st['keys']) or '（空）'}（值不打印）")
    else:
        print("  · 没有 .env（只有高德相关命令需要；模板见 .env.example）")

    print("\n[一致性] golden_check …", end=" ", flush=True)
    r = subprocess.run([sys.executable, "-m", "scripts.golden_check"],
                       cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8")
    passed = "逐位一致" in (r.stdout or "")
    ok = ok and passed
    print("逐位一致" if passed else f"未通过（exit {r.returncode}）")
    if not passed:
        print((r.stdout or "")[-400:])

    if args.with_tests:
        print("\n[测试] pytest -m 'not slow' …", flush=True)
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not slow"],
                           cwd=PROJECT_ROOT, text=True, encoding="utf-8")
        ok = ok and r.returncode == 0

    print("\n" + ("全部通过。" if ok else "有项目未就绪，见上面的 ✗。"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
