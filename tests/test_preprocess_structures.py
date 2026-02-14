from __future__ import annotations

from ranked_enumeration.decomposition import TDNode, TreeDecomposition
from ranked_enumeration.iterator import RankedEnumerator
from ranked_enumeration.model import Atom, CQ, Relation
from ranked_enumeration.ranking import AdditiveRankModel


def test_preprocess_root_stream_and_successor_cache() -> None:
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
        "R": Relation(name="R", vars=("x", "y"), rows=[(1, 10), (2, 10)]),
        "S": Relation(name="S", vars=("y", "z"), rows=[(10, 100), (10, 101)]),
    }
    rank_model = AdditiveRankModel(
        lambda node_id, assignment: float(assignment["x"]) if node_id == "n0" else float(assignment["z"])
    )

    enumerator = RankedEnumerator(cq, relations, td, rank_model)
    root_state = enumerator._preprocessed.root_state()  # noqa: SLF001
    root_stream = root_state.get_stream(tuple())

    assert root_stream is not None

    first = root_stream.get(0)
    second = root_stream.get(1)
    assert first is not None
    assert second is not None
    assert first.successor is second
    assert first.score <= second.score
