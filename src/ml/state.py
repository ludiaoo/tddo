"""ML 运行时状态（M9）。

三种模式（默认 `off`，保证与历史结果**逐位一致**）：

| 模式 | 行为 |
|---|---|
| `off` | 一切照旧，不碰任何 ML 代码路径 |
| `collect` | 照常**全量**评估每个候选（结果与 off 完全一致），把「特征 + 标签」喂给 sink |
| `screen` | 候选先打分，只评估 top-m + 5% 随机兜底（结果**有意**不同） |

模式是进程级全局状态：多进程实验里由每个 worker 的 initializer 设置。
"""

from __future__ import annotations

MODE_OFF = "off"
MODE_COLLECT = "collect"
MODE_SCREEN = "screen"
MODES = (MODE_OFF, MODE_COLLECT, MODE_SCREEN)

_mode: str = MODE_OFF
_sink = None                       # collect 模式的落盘回调：row: dict -> None
_ctx: dict = {}                    # (set_id, seed) 等上下文，随行写入每行
_step: int = 0                     # 第几次进入候选生成（一个修复步 = 一次）
_counters: dict[str, int] = {}


def set_mode(mode: str, *, sink=None, ctx: dict | None = None) -> None:
    global _mode, _sink, _ctx, _step, _counters
    if mode not in MODES:
        raise ValueError(f"未知模式 {mode!r}，可选 {MODES}")
    _mode = mode
    _sink = sink
    _ctx = dict(ctx or {})
    _step = 0
    _counters = {}


def get_mode() -> str:
    return _mode


def next_step() -> int:
    """候选生成器每进入一次自增；用于把「同一个修复步里的候选」分组。"""
    global _step
    _step += 1
    return _step


def current_step() -> int:
    return _step


def record(row: dict) -> None:
    """collect 模式下把一行候选记录交给 sink（由采集脚本注入）。"""
    if _sink is not None:
        _sink({"**set": _ctx.get("set_id"), "**seed": _ctx.get("seed"),
               "**step": current_step(), **row})


def bump(key: str, n: int = 1) -> None:
    _counters[key] = _counters.get(key, 0) + n


def counters() -> dict[str, int]:
    return dict(_counters)


def reset_counters() -> None:
    global _counters
    _counters = {}
