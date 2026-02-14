from __future__ import annotations

from ranked_enumeration.baseline import baseline_ranked
from ranked_enumeration.decomposition import TDNode, TreeDecomposition
from ranked_enumeration.iterator import RankedEnumerator
from ranked_enumeration.model import Atom, CQ, Relation
from ranked_enumeration.ranking import AdditiveRankModel


def test_iterator_matches_bruteforce_oracle() -> None:
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
        "R": Relation(name="R", vars=("x", "y"), rows=[(1, 10), (2, 10), (3, 11)]),
        "S": Relation(name="S", vars=("y", "z"), rows=[(10, 100), (10, 101), (11, 5)]),
    }

    rank_model = AdditiveRankModel(lambda _node, a: float(sum(a.values())))

    oracle = baseline_ranked(cq, relations, td, rank_model)
    enum = RankedEnumerator(cq, relations, td, rank_model)

    assert list(enum) == oracle
    assert enum.top_k(2) == oracle[:2]
