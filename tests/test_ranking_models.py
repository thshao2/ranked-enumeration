from __future__ import annotations

import warnings

import pytest

from ranked_enumeration.baseline import baseline_ranked
from ranked_enumeration.decomposition import TDNode, TreeDecomposition
from ranked_enumeration.iterator import RankedEnumerator
from ranked_enumeration.model import Atom, CQ, Relation
from ranked_enumeration.ranking import (
    CBoundedAdditiveRankModel,
    LexicographicRankModel,
    TupleBasedRankModel,
    VertexBasedRankModel,
)


def _three_way_query_fixture() -> tuple[CQ, TreeDecomposition, dict[str, Relation]]:
    cq = CQ(
        atoms=(
            Atom("R", ("x", "y")),
            Atom("S", ("y", "z")),
            Atom("T", ("z", "w")),
        ),
        output_vars=("x", "y", "z", "w"),
    )
    td = TreeDecomposition(
        nodes={
            "n0": TDNode(id="n0", bag_vars=("y", "z"), parent=None, children=("n1", "n2")),
            "n1": TDNode(id="n1", bag_vars=("x", "y"), parent="n0", children=tuple()),
            "n2": TDNode(id="n2", bag_vars=("z", "w"), parent="n0", children=tuple()),
        },
        root="n0",
    )
    relations = {
        "R": Relation(name="R", vars=("x", "y"), rows=[(1, 5), (2, 5), (3, 6)]),
        "S": Relation(name="S", vars=("y", "z"), rows=[(5, 7), (5, 8), (6, 7)]),
        "T": Relation(name="T", vars=("z", "w"), rows=[(7, "A"), (7, "B"), (8, "C")]),
    }
    return cq, td, relations


def test_tuple_based_ranking_matches_expected_and_baseline() -> None:
    cq, td, relations = _three_way_query_fixture()
    rank_model = TupleBasedRankModel(
        cq=cq,
        td=td,
        tuple_weights={
            "R": {(1, 5): 3.0, (2, 5): 1.0, (3, 6): 2.0},
            "S": {(5, 7): 2.0, (5, 8): 5.0, (6, 7): 1.0},
            "T": {(7, "A"): 4.0, (7, "B"): 0.0, (8, "C"): 2.0},
        },
    )

    expected = [
        (2, 5, 7, "B"),
        (3, 6, 7, "B"),
        (1, 5, 7, "B"),
        (2, 5, 7, "A"),
        (3, 6, 7, "A"),
        (2, 5, 8, "C"),
        (1, 5, 7, "A"),
        (1, 5, 8, "C"),
    ]

    enum = RankedEnumerator(cq, relations, td, rank_model)
    assert list(enum) == expected
    assert baseline_ranked(cq, relations, td, rank_model) == expected


def test_vertex_based_ranking_matches_expected_and_baseline() -> None:
    cq, td, relations = _three_way_query_fixture()
    rank_model = VertexBasedRankModel(
        td=td,
        vertex_weights={
            "x": {1: 4.0, 2: 1.0, 3: 2.0},
            "y": {5: 2.0, 6: 0.0},
            "z": {7: 3.0, 8: 1.0},
            "w": {"A": 5.0, "B": 2.0, "C": 4.0},
        },
    )

    expected = [
        (3, 6, 7, "B"),
        (2, 5, 7, "B"),
        (2, 5, 8, "C"),
        (3, 6, 7, "A"),
        (1, 5, 7, "B"),
        (1, 5, 8, "C"),
        (2, 5, 7, "A"),
        (1, 5, 7, "A"),
    ]

    enum = RankedEnumerator(cq, relations, td, rank_model)
    assert list(enum) == expected
    assert baseline_ranked(cq, relations, td, rank_model) == expected


def test_lexicographic_ranking_matches_expected_and_baseline() -> None:
    cq, td, relations = _three_way_query_fixture()
    rank_model = LexicographicRankModel(td=td, lex_order=("z", "x", "w", "y"))

    expected = [
        (1, 5, 7, "A"),
        (1, 5, 7, "B"),
        (2, 5, 7, "A"),
        (2, 5, 7, "B"),
        (3, 6, 7, "A"),
        (3, 6, 7, "B"),
        (1, 5, 8, "C"),
        (2, 5, 8, "C"),
    ]

    enum = RankedEnumerator(cq, relations, td, rank_model)
    assert list(enum) == expected
    assert baseline_ranked(cq, relations, td, rank_model) == expected


def test_c_bounded_ranking_positive_case() -> None:
    cq = CQ(
        atoms=(Atom("R", ("x", "y")), Atom("S", ("y", "z"))),
        output_vars=("x", "y", "z"),
    )
    td = TreeDecomposition(
        nodes={
            "n0": TDNode(id="n0", bag_vars=("y", "z"), parent=None, children=("n1",)),
            "n1": TDNode(id="n1", bag_vars=("x", "y"), parent="n0", children=tuple()),
        },
        root="n0",
    )
    relations = {
        "R": Relation(name="R", vars=("x", "y"), rows=[(1, 1), (2, 1), (3, 2)]),
        "S": Relation(name="S", vars=("y", "z"), rows=[(1, 10), (2, 20)]),
    }
    rank_model = CBoundedAdditiveRankModel(
        local_fn=lambda node_id, a: float(a["x"]) if node_id == "n1" else 0.0,
        c=2,
    )

    enum = RankedEnumerator(cq, relations, td, rank_model)
    assert list(enum) == [(1, 1, 10), (2, 1, 10), (3, 2, 20)]


def test_c_bounded_ranking_violation_raises() -> None:
    cq = CQ(
        atoms=(Atom("R", ("x", "y")), Atom("S", ("y", "z"))),
        output_vars=("x", "y", "z"),
    )
    td = TreeDecomposition(
        nodes={
            "n0": TDNode(id="n0", bag_vars=("y", "z"), parent=None, children=("n1",)),
            "n1": TDNode(id="n1", bag_vars=("x", "y"), parent="n0", children=tuple()),
        },
        root="n0",
    )
    relations = {
        "R": Relation(name="R", vars=("x", "y"), rows=[(1, 1), (2, 1), (3, 2)]),
        "S": Relation(name="S", vars=("y", "z"), rows=[(1, 10), (2, 20)]),
    }
    rank_model = CBoundedAdditiveRankModel(
        local_fn=lambda node_id, a: float(a["x"]) if node_id == "n1" else 0.0,
        c=1,
    )

    with pytest.raises(ValueError, match="c-bounded constraint violated"):
        RankedEnumerator(cq, relations, td, rank_model)


def test_legacy_combine_signature_still_works_with_warning() -> None:
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

    class LegacyModel:
        def local_weight(self, node_id: str, bag_assignment: dict[str, object]) -> float:
            _ = node_id
            return float(sum(v for v in bag_assignment.values() if isinstance(v, int)))

        def combine(self, local: float, child_scores: list[float]) -> float:
            return float(local + sum(child_scores))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = list(RankedEnumerator(cq, relations, td, LegacyModel()))

    assert out == [(1, 10, 100), (1, 10, 101), (2, 10, 100), (2, 10, 101)]
    assert any(item.category is DeprecationWarning for item in caught)
