from __future__ import annotations

from ranked_enumeration.decomposition import TDNode, TreeDecomposition
from ranked_enumeration.iterator import RankedEnumerator
from ranked_enumeration.model import Atom, CQ, Relation
from ranked_enumeration.ranking import AdditiveRankModel, ConstantRankModel


def test_empty_relation_produces_no_results() -> None:
    cq = CQ(
        atoms=(Atom("R", ("x", "y")), Atom("S", ("y", "z"))),
        output_vars=("x", "y", "z"),
    )
    td = TreeDecomposition(
        nodes={
            "n0": TDNode(id="n0", bag_vars=("x", "y"), parent=None, children=("n1",)),
            "n1": TDNode(id="n1", bag_vars=("y", "z"), parent="n0", children=tuple()),
        },
        root="n0",
    )
    relations = {
        "R": Relation(name="R", vars=("x", "y"), rows=[]),
        "S": Relation(name="S", vars=("y", "z"), rows=[(1, 2)]),
    }

    enum = RankedEnumerator(cq, relations, td, ConstantRankModel())
    assert list(enum) == []
    assert enum.top_k(5) == []


def test_single_node_query_with_dedup_and_topk() -> None:
    cq = CQ(
        atoms=(Atom("R", ("x", "y")),),
        output_vars=("x", "y"),
    )
    td = TreeDecomposition(
        nodes={
            "n0": TDNode(id="n0", bag_vars=("x", "y"), parent=None, children=tuple()),
        },
        root="n0",
    )
    relations = {
        "R": Relation(name="R", vars=("x", "y"), rows=[(2, 1), (1, 1), (2, 1)]),
    }
    rank = AdditiveRankModel(lambda _n, a: float(a["x"] + a["y"]))

    enum = RankedEnumerator(cq, relations, td, rank)

    assert list(enum) == [(1, 1), (2, 1)]
    assert enum.top_k(1) == [(1, 1)]
    assert enum.top_k(0) == []
