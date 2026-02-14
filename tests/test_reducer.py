from __future__ import annotations

from ranked_enumeration.bag_relations import build_bag_relations
from ranked_enumeration.decomposition import TDNode, TreeDecomposition
from ranked_enumeration.model import Atom, CQ, Relation
from ranked_enumeration.reducer import run_full_reducer


def _query_and_td() -> tuple[CQ, TreeDecomposition]:
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
    return cq, td


def test_reducer_prunes_unsupported_parent_rows() -> None:
    cq, td = _query_and_td()
    relations = {
        "R": Relation(name="R", vars=("x", "y"), rows=[(1, 10), (2, 20)]),
        "S": Relation(name="S", vars=("y", "z"), rows=[(10, 100)]),
    }

    bags = build_bag_relations(cq, relations, td)
    reduced = run_full_reducer(td, bags)

    assert set(reduced["n0"].tuples) == {(1, 10)}
    assert set(reduced["n1"].tuples) == {(10, 100)}


def test_reducer_empties_all_when_no_support() -> None:
    cq, td = _query_and_td()
    relations = {
        "R": Relation(name="R", vars=("x", "y"), rows=[(1, 10), (2, 20)]),
        "S": Relation(name="S", vars=("y", "z"), rows=[]),
    }

    bags = build_bag_relations(cq, relations, td)
    reduced = run_full_reducer(td, bags)

    assert reduced["n0"].tuples == []
    assert reduced["n1"].tuples == []
