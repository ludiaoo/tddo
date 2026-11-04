"""机器学习扩展包（M9）。

- `state`：运行时开关（off / collect / screen），默认 off，保证与历史结果逐位一致
- `features`：无人机插入候选的廉价特征（不跑路线仿真）
- `screen`：LightGBM 预筛模型（缺失时自动退化为全量评估）

用法（多进程实验里在 worker 的 initializer 里设置）::

    from src.ml import state
    state.set_mode("screen")          # 或 "collect"
"""

from .state import (
    MODE_COLLECT,
    MODE_OFF,
    MODE_SCREEN,
    MODES,
    bump,
    counters,
    current_step,
    get_mode,
    next_step,
    record,
    reset_counters,
    set_mode,
)

__all__ = [
    "MODE_OFF", "MODE_COLLECT", "MODE_SCREEN", "MODES",
    "get_mode", "set_mode", "next_step", "current_step",
    "record", "bump", "counters", "reset_counters",
]
