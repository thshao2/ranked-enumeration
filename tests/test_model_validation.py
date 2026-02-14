from __future__ import annotations

import pytest

from ranked_enumeration.decomposition import TDNode, TreeDecomposition, validate_tree_decomposition
from ranked_enumeration.iterator import RankedEnumerator
from ranked_enumeration.model import Atom, CQ, Relation
from ranked_enumeration.ranking import ConstantRankModel


def _base_relations() -> dict[str, Relation]:
    return {
        "R": Relation(name="R", vars=("x", "y"), rows=[(1, 2)]),
        "S": Relation(name="S", vars=("y", "z"), rows=[(2, 3)]),
    }


def _base_cq() -> CQ:
    return CQ(
        atoms=(Atom("R", ("x", "y")), Atom("S", ("y", "z"))),
        output_vars=("x", "y", "z"),
    )


def _base_td() -> TreeDecomposition:
    nodes = {
        "n0": TDNode(id="n0", bag_vars=("x", "y"), parent=None, children=("n1",)),
        "n1": TDNode(id="n1", bag_vars=("y", "z"), parent="n0", children=tuple()),
    }
    return TreeDecomposition(nodes=nodes, root="n0")


def test_invalid_output_vars_rejected() -> None:
    cq = CQ(
        atoms=(Atom("R", ("x", "y")), Atom("S", ("y", "z"))),
        output_vars=("x", "y"),
    )
    with pytest.raises(ValueError, match="output_vars"):
        RankedEnumerator(cq, _base_relations(), _base_td(), ConstantRankModel())


def test_running_intersection_violation_rejected() -> None:
    cq = _base_cq()
    td = TreeDecomposition(
        nodes={
            "n0": TDNode(id="n0", bag_vars=("x", "y"), parent=None, children=("n1",)),
            "n1": TDNode(id="n1", bag_vars=("x",), parent="n0", children=("n2",)),
            "n2": TDNode(id="n2", bag_vars=("y", "z"), parent="n1", children=tuple()),
        },
        root="n0",
    )
    with pytest.raises(ValueError, match="Running intersection"):
        validate_tree_decomposition(cq, td)


def test_atom_coverage_violation_rejected() -> None:
    cq = _base_cq()
    td = TreeDecomposition(
        nodes={
            "n0": TDNode(id="n0", bag_vars=("x",), parent=None, children=("n1",)),
            "n1": TDNode(id="n1", bag_vars=("y", "z"), parent="n0", children=tuple()),
        },
        root="n0",
    )
    with pytest.raises(ValueError, match="covers atom"):
        validate_tree_decomposition(cq, td)
