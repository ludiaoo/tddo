"""`.env` 读取（`src/env.py`）的守门测试。

重点不是"能不能读"，而是三件容易出事的事：

1. **优先级**：进程环境变量必须压过 `.env`，否则临时覆盖会失效（CI 里靠这个）；
2. **报错信息**：缺 key 时要告诉人去哪个文件填，而不是抛一个 KeyError；
3. **不硬编码**：`src/` 与 `scripts/` 里不允许出现 32 位十六进制串（高德 key 的形态）。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.env import ENV_EXAMPLE, ENV_FILE, PROJECT_ROOT, get, load_env, mask, parse_env_text, require


def test_解析注释_引号_export_行尾() -> None:
    text = """
# 这是注释
AMAP_KEY=abc123
export AMAP_JS_KEY = "quoted-value"
AMAP_JS_SECRET='single'
URL=https://example.com/path#fragment
TRAILING=value   # 行尾注释
NO_EQUALS_SIGN
"""
    d = parse_env_text(text)
    assert d["AMAP_KEY"] == "abc123"
    assert d["AMAP_JS_KEY"] == "quoted-value"
    assert d["AMAP_JS_SECRET"] == "single"
    # 不带空白的 # 不当注释，否则 URL 会被切坏
    assert d["URL"] == "https://example.com/path#fragment"
    assert d["TRAILING"] == "value"
    assert "NO_EQUALS_SIGN" not in d


def test_空值与BOM不影响解析() -> None:
    d = parse_env_text("\ufeffAMAP_KEY=\nOTHER=1\n")
    assert d["AMAP_KEY"] == "" and d["OTHER"] == "1"


def test_文件不存在时静默(tmp_path: Path) -> None:
    assert load_env(tmp_path / "nope.env") == {}


def test_环境变量优先于文件(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / ".env"
    f.write_text("AMAP_KEY=from_file\n", encoding="utf-8")
    monkeypatch.setenv("AMAP_KEY", "from_shell")
    load_env(f)
    assert get("AMAP_KEY") == "from_shell"
    load_env(f, override=True)
    assert get("AMAP_KEY") == "from_file"


def test_缺失时给出可执行的报错() -> None:
    with pytest.raises(RuntimeError) as ei:
        require("AMAP_KEY_DOES_NOT_EXIST", "单元测试")
    msg = str(ei.value)
    assert "AMAP_KEY_DOES_NOT_EXIST" in msg
    assert ".env" in msg and ENV_EXAMPLE.name in msg


def test_脱敏只留首尾() -> None:
    assert mask("abcd1234efgh5678ijkl9012") == "abcd…9012"
    assert mask("short") == "*****"
    assert mask(None) == "(空)"


def test_模板存在且字段齐全() -> None:
    assert ENV_EXAMPLE.exists(), "缺少 .env.example，别人 clone 后不知道该填什么"
    text = ENV_EXAMPLE.read_text(encoding="utf-8")
    for key in ("AMAP_KEY", "AMAP_JS_KEY", "AMAP_JS_SECRET"):
        assert re.search(rf"^{key}=", text, re.M), f".env.example 缺 {key}"
    # 模板里不允许出现真实值
    assert not re.search(r"^AMAP_[A-Z_]*=\S+", text, re.M), ".env.example 里带了具体值"


def test_密钥文件被git忽略() -> None:
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^\.env$", ignore, re.M), ".gitignore 没有忽略 .env"
    assert re.search(r"^!\.env\.example$", ignore, re.M), ".env.example 应被显式放行"


@pytest.mark.parametrize("folder", ["src", "scripts", "tests"])
def test_源码里没有硬编码的32位密钥(folder: str) -> None:
    """高德 key 是 32 位十六进制；出现在源码（含测试）里就是硬编码。

    `tests/` 也要扫：这条测试本身曾经把真实 key 当样例写进去过，属于同一类疏漏。
    """
    hits = []
    for p in (PROJECT_ROOT / folder).rglob("*.py"):
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"[0-9a-f]{32}", line):
                hits.append(f"{p.relative_to(PROJECT_ROOT)}:{i}")
    assert not hits, f"{folder}/ 里出现疑似硬编码 key：{hits}"


def test_本地env存在时不进git() -> None:
    """有 .env 的机器上，它必须是 untracked（这里只查规则，不查 git 状态）。"""
    if not ENV_FILE.exists():
        pytest.skip("本机没有 .env")
    ignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "\n.env\n" in ignore
