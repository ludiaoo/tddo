"""CLI 守门测试：安装后的一条命令必须可用，且参数透传不能被 argparse 吃掉。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from src.io import PROJECT_ROOT


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "src.cli", *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def test_顶层帮助列出全部子命令() -> None:
    r = _run("--help")
    assert r.returncode == 0
    for name in ("doctor", "solve", "bench", "baseline", "export-web", "check"):
        assert name in r.stdout


def test_doctor不修改任何状态且返回零() -> None:
    r = _run("doctor")
    assert r.returncode == 0
    assert "体检完成" in r.stdout
    assert "项目根目录" in r.stdout


def test_solve写出结构化JSON(tmp_path: Path) -> None:
    out = tmp_path / "solution.json"
    r = _run("solve", "--set", "1", "--iters", "1", "--seed", "0", "--out", str(out))
    assert r.returncode == 0, r.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))

    assert payload["set"] == 1
    assert payload["feasible"] is True
    assert payload["best_cost"] > 0
    assert abs(payload["best_cost"] - payload["recheck_cost"]) < 1e-6
    assert payload["routes"]

    for route in payload["routes"]:
        assert route["truck_nodes"][0] == 0 and route["truck_nodes"][-1] == 0
        for s in route["sorties"]:
            # launch/recover 是位置下标，必须同时给出节点编号，防止历史 bug 回归
            assert {"launch_pos", "recover_pos", "launch_node", "recover_node"} <= s.keys()
            assert s["sequence"][0] == s["launch_node"]
            assert s["sequence"][-1] == s["recover_node"]

    assert not re.search(r"\b[0-9a-f]{32}\b", out.read_text(encoding="utf-8"))


def test_委托命令的help透传给原脚本() -> None:
    r = _run("bench", "--help")
    assert r.returncode == 0
    assert "scripts.bench" in r.stdout
    assert "--sets" in r.stdout
