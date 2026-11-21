"""网页导出的几何正确性（`M9.23` §10）。

这里只测最容易错的那一步：`Sortie.launch / recover` 是 **nodes 里的位置下标**，
不是节点编号。曾经直接把位置当编号去索引 `inst.base.nodes`，结果每个架次的
起降点都画到了 1–4 km 之外——地图上看起来就像"无人机飞到一半没回来"。
"""

from __future__ import annotations

from src.solution import Sortie
from scripts.export_web import sortie_node_seq


def test_起降点取的是节点编号而不是下标() -> None:
    nodes = [0, 21, 9, 8, 16, 20, 22, 2, 0]
    s = Sortie(launch=3, recover=4, customers=(7, 1, 13))
    seq = sortie_node_seq(nodes, s)
    assert seq == [8, 7, 1, 13, 16], seq


def test_首尾必须落在卡车停靠点上() -> None:
    nodes = [0, 5, 14, 3, 11, 0]
    for launch, recover in ((1, 2), (2, 3), (3, 4)):
        s = Sortie(launch=launch, recover=recover, customers=(9,))
        seq = sortie_node_seq(nodes, s)
        assert seq[0] == nodes[launch]
        assert seq[-1] == nodes[recover]
        assert seq[0] in nodes and seq[-1] in nodes


def test_客户序列保持原顺序() -> None:
    nodes = [0, 4, 6, 0]
    s = Sortie(launch=1, recover=2, customers=(12, 10, 18))
    assert sortie_node_seq(nodes, s)[1:-1] == [12, 10, 18]
