from __future__ import annotations

import pytest

from ranked_enumeration.decomposition import TDNode, TreeDecomposition
from ranked_enumeration.iterator import RankedEnumerator
from ranked_enumeration.model import Atom, CQ, Relation
from ranked_enumeration.ranking import (
    AdditiveRankModel,
    CBoundedAdditiveRankModel,
    LexicographicRankModel,
    TupleBasedRankModel,
    VertexBasedRankModel,
)


def _weighted_four_way_query() -> tuple[CQ, TreeDecomposition]:
    cq = CQ(
        atoms=(
            Atom("R1", ("x", "y", "w1")),
            Atom("R2", ("y", "z", "w2")),
            Atom("R3", ("z", "w", "w3")),
            Atom("R4", ("z", "u", "w4")),
        ),
        output_vars=("x", "y", "w1", "z", "w2", "w", "w3", "u", "w4"),
    )

    td = TreeDecomposition(
        nodes={
            "n0": TDNode(
                id="n0",
                bag_vars=("y", "z", "w2"),
                parent=None,
                children=("n1", "n2", "n3"),
            ),
            "n1": TDNode(
                id="n1",
                bag_vars=("x", "y", "w1"),
                parent="n0",
                children=tuple(),
            ),
            "n2": TDNode(
                id="n2",
                bag_vars=("z", "w", "w3"),
                parent="n0",
                children=tuple(),
            ),
            "n3": TDNode(
                id="n3",
                bag_vars=("z", "u", "w4"),
                parent="n0",
                children=tuple(),
            ),
        },
        root="n0",
    )
    return cq, td


def _weighted_four_way_relations() -> dict[str, Relation]:
    return {
        "R1": Relation(
            name="R1",
            vars=("x", "y", "w1"),
            rows=[(1, 5, 1), (2, 5, 6), (3, 8, 2)],
        ),
        "R2": Relation(
            name="R2",
            vars=("y", "z", "w2"),
            rows=[(5, 10, 1), (5, 11, 4), (8, 10, 2), (9, 12, 1)],
        ),
        "R3": Relation(
            name="R3",
            vars=("z", "w", "w3"),
            rows=[(10, "A", 1), (10, "B", 3), (11, "C", 2)],
        ),
        "R4": Relation(
            name="R4",
            vars=("z", "u", "w4"),
            rows=[(10, "X", 2), (10, "Y", 5), (11, "Z", 1)],
        ),
    }


def _weighted_expected_additive_order() -> list[tuple[object, ...]]:
    return [
        (1, 5, 1, 10, 1, "A", 1, "X", 2),
        (1, 5, 1, 10, 1, "B", 3, "X", 2),
        (3, 8, 2, 10, 2, "A", 1, "X", 2),
        (1, 5, 1, 10, 1, "A", 1, "Y", 5),
        (1, 5, 1, 11, 4, "C", 2, "Z", 1),
        (3, 8, 2, 10, 2, "B", 3, "X", 2),
        (1, 5, 1, 10, 1, "B", 3, "Y", 5),
        (2, 5, 6, 10, 1, "A", 1, "X", 2),
        (3, 8, 2, 10, 2, "A", 1, "Y", 5),
        (2, 5, 6, 10, 1, "B", 3, "X", 2),
        (3, 8, 2, 10, 2, "B", 3, "Y", 5),
        (2, 5, 6, 10, 1, "A", 1, "Y", 5),
        (2, 5, 6, 11, 4, "C", 2, "Z", 1),
        (2, 5, 6, 10, 1, "B", 3, "Y", 5),
    ]


def _weighted_lexicographic_order() -> list[tuple[object, ...]]:
    return [
        (1, 5, 1, 10, 1, "A", 1, "X", 2),
        (1, 5, 1, 10, 1, "A", 1, "Y", 5),
        (1, 5, 1, 10, 1, "B", 3, "X", 2),
        (1, 5, 1, 10, 1, "B", 3, "Y", 5),
        (1, 5, 1, 11, 4, "C", 2, "Z", 1),
        (2, 5, 6, 10, 1, "A", 1, "X", 2),
        (2, 5, 6, 10, 1, "A", 1, "Y", 5),
        (2, 5, 6, 10, 1, "B", 3, "X", 2),
        (2, 5, 6, 10, 1, "B", 3, "Y", 5),
        (2, 5, 6, 11, 4, "C", 2, "Z", 1),
        (3, 8, 2, 10, 2, "A", 1, "X", 2),
        (3, 8, 2, 10, 2, "A", 1, "Y", 5),
        (3, 8, 2, 10, 2, "B", 3, "X", 2),
        (3, 8, 2, 10, 2, "B", 3, "Y", 5),
    ]


def _weighted_rank_model() -> AdditiveRankModel:
    # total weight = w1 + w2 + w3 + w4
    return AdditiveRankModel(
        lambda node_id, a: float(
            a["w1"]
            if node_id == "n1"
            else a["w2"]
            if node_id == "n0"
            else a["w3"]
            if node_id == "n2"
            else a["w4"]
        )
    )


def _weighted_tuple_rank_model(cq: CQ, td: TreeDecomposition) -> TupleBasedRankModel:
    return TupleBasedRankModel(
        cq=cq,
        td=td,
        tuple_weights={
            "R1": {(1, 5, 1): 1.0, (2, 5, 6): 6.0, (3, 8, 2): 2.0},
            "R2": {(5, 10, 1): 1.0, (5, 11, 4): 4.0, (8, 10, 2): 2.0, (9, 12, 1): 1.0},
            "R3": {(10, "A", 1): 1.0, (10, "B", 3): 3.0, (11, "C", 2): 2.0},
            "R4": {(10, "X", 2): 2.0, (10, "Y", 5): 5.0, (11, "Z", 1): 1.0},
        },
    )


def _weighted_vertex_rank_model(td: TreeDecomposition) -> VertexBasedRankModel:
    return VertexBasedRankModel(
        td=td,
        vertex_weights={
            "x": {1: 0.0, 2: 0.0, 3: 0.0},
            "y": {5: 0.0, 8: 0.0},
            "w1": {1: 1.0, 2: 2.0, 6: 6.0},
            "z": {10: 0.0, 11: 0.0, 12: 0.0},
            "w2": {1: 1.0, 2: 2.0, 4: 4.0},
            "w": {"A": 0.0, "B": 0.0, "C": 0.0},
            "w3": {1: 1.0, 2: 2.0, 3: 3.0},
            "u": {"X": 0.0, "Y": 0.0, "Z": 0.0},
            "w4": {1: 1.0, 2: 2.0, 5: 5.0},
        },
    )


def test_weighted_four_way_ranked_output_matches_expected() -> None:
    cq, td = _weighted_four_way_query()
    relations = _weighted_four_way_relations()
    expected = _weighted_expected_additive_order()

    enum = RankedEnumerator(cq, relations, td, _weighted_rank_model())

    assert list(enum) == expected
    assert enum.top_k(5) == expected[:5]


def test_weighted_four_way_with_tuple_based_rank_model_matches_expected() -> None:
    cq, td = _weighted_four_way_query()
    relations = _weighted_four_way_relations()
    expected = _weighted_expected_additive_order()

    enum = RankedEnumerator(cq, relations, td, _weighted_tuple_rank_model(cq, td))
    assert list(enum) == expected


def test_weighted_four_way_with_vertex_based_rank_model_matches_expected() -> None:
    cq, td = _weighted_four_way_query()
    relations = _weighted_four_way_relations()
    expected = _weighted_expected_additive_order()

    enum = RankedEnumerator(cq, relations, td, _weighted_vertex_rank_model(td))
    assert list(enum) == expected


def test_weighted_four_way_with_lexicographic_rank_model_matches_expected() -> None:
    cq, td = _weighted_four_way_query()
    relations = _weighted_four_way_relations()
    expected = _weighted_lexicographic_order()

    rank_model = LexicographicRankModel(td=td, lex_order=cq.output_vars)
    enum = RankedEnumerator(cq, relations, td, rank_model)
    assert list(enum) == expected


def test_weighted_four_way_with_c_bounded_rank_model_matches_expected() -> None:
    cq, td = _weighted_four_way_query()
    relations = _weighted_four_way_relations()
    expected = _weighted_expected_additive_order()

    rank_model = CBoundedAdditiveRankModel(local_fn=_weighted_rank_model().local_fn, c=3)
    enum = RankedEnumerator(cq, relations, td, rank_model)
    assert list(enum) == expected


def test_weighted_four_way_with_c_bounded_rank_model_violates_bound() -> None:
    cq, td = _weighted_four_way_query()
    relations = _weighted_four_way_relations()

    rank_model = CBoundedAdditiveRankModel(local_fn=_weighted_rank_model().local_fn, c=2)
    with pytest.raises(ValueError, match="c-bounded constraint violated"):
        RankedEnumerator(cq, relations, td, rank_model)


def test_weighted_four_way_duplicate_rows_do_not_change_output() -> None:
    cq, td = _weighted_four_way_query()
    relations = {
        "R1": Relation(
            name="R1",
            vars=("x", "y", "w1"),
            rows=[(2, 5, 6), (1, 5, 1), (3, 8, 2), (1, 5, 1), (2, 5, 6)],
        ),
        "R2": Relation(
            name="R2",
            vars=("y", "z", "w2"),
            rows=[(5, 11, 4), (5, 10, 1), (8, 10, 2), (5, 10, 1), (9, 12, 1)],
        ),
        "R3": Relation(
            name="R3",
            vars=("z", "w", "w3"),
            rows=[(10, "B", 3), (11, "C", 2), (10, "A", 1), (10, "A", 1)],
        ),
        "R4": Relation(
            name="R4",
            vars=("z", "u", "w4"),
            rows=[(10, "Y", 5), (10, "X", 2), (11, "Z", 1), (10, "X", 2)],
        ),
    }

    expected = _weighted_expected_additive_order()

    enum = RankedEnumerator(cq, relations, td, _weighted_rank_model())
    assert list(enum) == expected


def test_branching_four_way_query_matches_expected_order() -> None:
    cq = CQ(
        atoms=(
            Atom("A", ("a", "b")),
            Atom("B", ("b", "c")),
            Atom("C", ("c", "d")),
            Atom("D", ("c", "e")),
        ),
        output_vars=("a", "b", "c", "d", "e"),
    )
    td = TreeDecomposition(
        nodes={
            "n0": TDNode(
                id="n0", bag_vars=("b", "c"), parent=None, children=("n1", "n2", "n3")
            ),
            "n1": TDNode(id="n1", bag_vars=("a", "b"), parent="n0", children=tuple()),
            "n2": TDNode(id="n2", bag_vars=("c", "d"), parent="n0", children=tuple()),
            "n3": TDNode(id="n3", bag_vars=("c", "e"), parent="n0", children=tuple()),
        },
        root="n0",
    )

    relations = {
        "A": Relation(name="A", vars=("a", "b"), rows=[(1, 1), (2, 1), (3, 2)]),
        "B": Relation(name="B", vars=("b", "c"), rows=[(1, 5), (1, 6), (2, 5)]),
        "C": Relation(name="C", vars=("c", "d"), rows=[(5, 7), (6, 8)]),
        "D": Relation(name="D", vars=("c", "e"), rows=[(5, 9), (6, 4)]),
    }

    # total score = a + c + d + e
    rank_model = AdditiveRankModel(
        lambda node_id, a: float(
            a["c"] if node_id == "n0" else a["a"] if node_id == "n1" else a["d"] if node_id == "n2" else a["e"]
        )
    )

    expected = [
        (1, 1, 6, 8, 4),
        (2, 1, 6, 8, 4),
        (1, 1, 5, 7, 9),
        (2, 1, 5, 7, 9),
        (3, 2, 5, 7, 9),
    ]

    enum = RankedEnumerator(cq, relations, td, rank_model)
    assert list(enum) == expected
    assert enum.top_k(3) == expected[:3]
