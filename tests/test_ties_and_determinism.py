from __future__ import annotations

from ranked_enumeration.decomposition import TDNode, TreeDecomposition
from ranked_enumeration.iterator import RankedEnumerator
from ranked_enumeration.model import Atom, CQ, Relation
from ranked_enumeration.ranking import ConstantRankModel


def test_equal_scores_use_lexicographic_tie_break() -> None:
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
        "R": Relation(name="R", vars=("x", "y"), rows=[(2, 9), (1, 9)]),
        "S": Relation(name="S", vars=("y", "z"), rows=[(9, 8), (9, 7)]),
    }

    enum = RankedEnumerator(cq, relations, td, ConstantRankModel())

    expected = [
        (1, 9, 7),
        (1, 9, 8),
        (2, 9, 7),
        (2, 9, 8),
    ]
    assert list(enum) == expected
    assert list(enum) == expected
