"""配色的测试：调色板可导入、图层属性可套用、中文在小字号下能渲染。

最后一条不是洁癖：SimSun 内嵌点阵字模，6 pt 时 matplotlib 会渲染成**空白**，
只有真的画一张再量文字宽度才测得出来。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from src import style


def test_调色板长度与去重():
    assert len(style.PALETTE) == 5
    assert len(set(style.PALETTE)) == 5
    assert all(c.startswith("#") and len(c) == 7 for c in style.PALETTE)
    assert len(set(style.ROUTES)) >= 12          # 路线图要能分开十几条线


def test_apply_设置字体族为列表():
    """matplotlib 只在 font.family 是**列表**时才做逐字回退（中文靠这个）。"""
    style.apply()
    fam = plt.rcParams["font.family"]
    assert isinstance(fam, list) and len(fam) >= 2
    assert "SimSun" not in fam, "SimSun 内嵌点阵，小字号会渲染成空白"


def test_小字号中文能渲染出非零宽度():
    style.apply(font_size=7)
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.scatter([1], [1], label="客户")
    leg = ax.legend(fontsize=6)
    fig.canvas.draw()
    bb = leg.get_texts()[0].get_window_extent(fig.canvas.get_renderer())
    plt.close(fig)
    assert bb.width > 5 and bb.height > 3


def test_line_helper_用调色板():
    style.apply()
    fig, ax = plt.subplots()
    style.line(ax, [0, 1], [0, 1], 2, label="x")
    color = matplotlib.colors.to_hex(ax.lines[0].get_color())
    plt.close(fig)
    assert color.lower() == style.PALETTE[2].lower()
