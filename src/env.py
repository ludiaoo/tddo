"""项目级密钥/配置读取：从项目根目录的 `.env` 读，不把 key 写进代码。

**为什么自己写**：只有一个需求（读 .env 到环境变量），引入 ``python-dotenv``
等于给一个纯研究项目加依赖；这里 60 行够用，且行为可测。

**优先级**（前者覆盖后者）：

1. 进程环境变量（``AMAP_KEY=... python -m ...``）——适合 CI 或临时覆盖；
2. 项目根目录 ``.env``——日常使用；
3. 调用处给的默认值。

**文件格式**：一行一个 ``KEY=value``，支持 ``export KEY=value``、``#`` 注释、
单/双引号包裹、行尾 ``  # 注释``、UTF-8 BOM。

用法::

    from src.env import get, require, mask

    key = require("AMAP_KEY")          # 缺失时抛错，信息里写清去哪个文件填
    key = get("AMAP_KEY")              # 缺失返回 None
    print(mask(key))                   # 只打印首尾，日志里不出现完整 key
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
ENV_EXAMPLE = PROJECT_ROOT / ".env.example"

# 进程内只解析一次；测试可传 force=True 重读
_loaded: Path | None = None


def parse_env_text(text: str) -> dict[str, str]:
    """把 .env 文本解析成字典。解析不了的行直接跳过（宁缺不炸）。"""
    out: dict[str, str] = {}
    for raw in text.lstrip("\ufeff").splitlines():     # 有人用记事本存成带 BOM 的
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        val = val.strip()
        if val[:1] in ("'", '"'):
            quote = val[0]
            end = val.find(quote, 1)
            val = val[1:end] if end > 0 else val[1:]
        else:
            # 行尾注释：只有前面带空白才算注释，避免把 URL 里的 # 当注释切掉
            cut = val.find(" #")
            if cut > 0:
                val = val[:cut].rstrip()
        out[key] = val
    return out


def load_env(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """读 .env 并写入 ``os.environ``。

    ``override=False``（默认）时，已存在的进程环境变量优先，不会被文件覆盖。
    文件不存在时什么也不做，返回空字典。
    """
    global _loaded
    target = Path(path) if path is not None else ENV_FILE
    if not target.exists():
        return {}
    parsed = parse_env_text(target.read_text(encoding="utf-8-sig"))
    for k, v in parsed.items():
        if override or k not in os.environ:
            os.environ[k] = v
    _loaded = target
    return parsed


def get(name: str, default: str | None = None) -> str | None:
    """取一个配置项；第一次调用时顺带加载项目根目录的 .env。"""
    if _loaded is None:
        load_env()
    val = os.environ.get(name)
    return val if val not in ("", None) else default


def require(name: str, purpose: str = "") -> str:
    """取一个**必须存在**的配置项，缺失时给可执行的提示。"""
    val = get(name)
    if val:
        return val
    hint = f"（用于{purpose}）" if purpose else ""
    raise RuntimeError(
        f"缺少配置项 {name}{hint}。\n"
        f"  做法：把 {ENV_EXAMPLE.name} 复制成 {ENV_FILE.name}，填入该项；\n"
        f"  或临时用环境变量覆盖：{name}=xxx python -m ...\n"
        f"  说明：{ENV_FILE.name} 已在 .gitignore 里，不会被提交。"
    )


def mask(value: str | None, keep: int = 4) -> str:
    """日志用：只留首尾各 ``keep`` 位。"""
    if not value:
        return "(空)"
    if len(value) <= keep * 2:
        return "*" * len(value)
    return f"{value[:keep]}…{value[-keep:]}"


def env_file_status() -> dict:
    """给自检/日志用：文件在不在、写了哪几项（**只报键名，不报值**）。"""
    exists = ENV_FILE.exists()
    keys = sorted(parse_env_text(ENV_FILE.read_text(encoding="utf-8-sig")).keys()) if exists else []
    return {"path": str(ENV_FILE), "exists": exists, "keys": keys}
