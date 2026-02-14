from __future__ import annotations

import itertools
import random
from typing import Any

from ranked_enumeration.decomposition import TDNode, TreeDecomposition
from ranked_enumeration.model import Atom, CQ, Relation


def make_path_query(length: int) -> tuple[CQ, TreeDecomposition]:
    if length < 1:
        raise ValueError("Path length must be >= 1")

    atoms: list[Atom] = []
    nodes: dict[str, TDNode] = {}

    for i in range(length):
        atom_name = f"R{i}"
        vars_ = (f"x{i}", f"x{i+1}")
        atoms.append(Atom(name=atom_name, vars=vars_))

    for i in range(length):
        node_id = f"n{i}"
        parent = None if i == 0 else f"n{i-1}"
        children = tuple([f"n{i+1}"] if i + 1 < length else [])
        nodes[node_id] = TDNode(
            id=node_id,
            bag_vars=(f"x{i}", f"x{i+1}"),
            parent=parent,
            children=children,
        )

    output_vars = tuple(f"x{i}" for i in range(length + 1))
    cq = CQ(atoms=tuple(atoms), output_vars=output_vars)
    td = TreeDecomposition(nodes=nodes, root="n0")
    return cq, td


def make_star_query(arms: int) -> tuple[CQ, TreeDecomposition]:
    if arms < 1:
        raise ValueError("Number of arms must be >= 1")

    atoms: list[Atom] = []
    nodes: dict[str, TDNode] = {}

    for i in range(arms):
        atom_name = f"S{i}"
        vars_ = ("c", f"y{i}")
        atoms.append(Atom(name=atom_name, vars=vars_))

    nodes["n0"] = TDNode(
        id="n0",
        bag_vars=("c", "y0"),
        parent=None,
        children=tuple(f"n{i}" for i in range(1, arms)),
    )
    for i in range(1, arms):
        nodes[f"n{i}"] = TDNode(
            id=f"n{i}",
            bag_vars=("c", f"y{i}"),
            parent="n0",
            children=tuple(),
        )

    output_vars = ("c",) + tuple(f"y{i}" for i in range(arms))
    cq = CQ(atoms=tuple(atoms), output_vars=output_vars)
    td = TreeDecomposition(nodes=nodes, root="n0")
    return cq, td


def instantiate_relations(
    cq: CQ,
    domain_size: int,
    tuple_probability: float,
    seed: int,
) -> dict[str, Relation]:
    rng = random.Random(seed)
    domain = list(range(domain_size))

    relations: dict[str, Relation] = {}
    for atom in cq.atoms:
        rows: list[tuple[Any, ...]] = []
        for row in itertools.product(domain, repeat=len(atom.vars)):
            if rng.random() <= tuple_probability:
                rows.append(tuple(row))
        relations[atom.name] = Relation(name=atom.name, vars=atom.vars, rows=rows)
    return relations
