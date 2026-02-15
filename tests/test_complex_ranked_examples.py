from __future__ import annotations

from ranked_enumeration.decomposition import TDNode, TreeDecomposition
from ranked_enumeration.iterator import RankedEnumerator
from ranked_enumeration.model import Atom, CQ, Relation
from ranked_enumeration.ranking import AdditiveRankModel


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


def test_weighted_four_way_ranked_output_matches_expected() -> None:
    cq, td = _weighted_four_way_query()
    relations = {
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

    expected = [
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

    enum = RankedEnumerator(cq, relations, td, _weighted_rank_model())

    assert list(enum) == expected
    assert enum.top_k(5) == expected[:5]


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

    expected = [
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
