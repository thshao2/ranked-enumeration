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


def make_binary_tree_query(depth: int) -> tuple[CQ, TreeDecomposition]:
    if depth < 1:
        raise ValueError("Binary tree depth must be >= 1")

    atoms: list[Atom] = []
    nodes: dict[str, TDNode] = {}
    decomp_parent: dict[str, str | None] = {}

    def edge_name(parent_idx: int, child_idx: int) -> str:
        return f"B{parent_idx}_{child_idx}"

    num_tree_nodes = 2 ** (depth + 1) - 1
    for parent_idx in range(2**depth - 1):
        for child_idx in (2 * parent_idx + 1, 2 * parent_idx + 2):
            atom_name = edge_name(parent_idx, child_idx)
            vars_ = (f"v{parent_idx}", f"v{child_idx}")
            atoms.append(Atom(name=atom_name, vars=vars_))

            node_id = f"n_{atom_name}"
            nodes[node_id] = TDNode(
                id=node_id,
                bag_vars=vars_,
                parent=None,
                children=tuple(),
            )

            if parent_idx == 0 and child_idx == 1:
                decomp_parent[node_id] = None
            elif parent_idx == 0:
                decomp_parent[node_id] = f"n_{edge_name(0, 1)}"
            else:
                grandparent = (parent_idx - 1) // 2
                decomp_parent[node_id] = f"n_{edge_name(grandparent, parent_idx)}"

    children_by_parent: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    root = f"n_{edge_name(0, 1)}"
    for node_id, parent in decomp_parent.items():
        if parent is not None:
            children_by_parent[parent].append(node_id)

    for node_id, node in list(nodes.items()):
        nodes[node_id] = TDNode(
            id=node.id,
            bag_vars=node.bag_vars,
            parent=decomp_parent[node_id],
            children=tuple(children_by_parent[node_id]),
        )

    output_vars = tuple(f"v{i}" for i in range(num_tree_nodes))
    cq = CQ(atoms=tuple(atoms), output_vars=output_vars)
    td = TreeDecomposition(nodes=nodes, root=root)
    return cq, td


def make_caterpillar_query(spine_length: int) -> tuple[CQ, TreeDecomposition]:
    if spine_length < 2:
        raise ValueError("Caterpillar spine length must be >= 2")

    atoms: list[Atom] = []
    nodes: dict[str, TDNode] = {}

    for i in range(spine_length):
        atom_name = f"P{i}"
        vars_ = (f"x{i}", f"x{i+1}")
        atoms.append(Atom(name=atom_name, vars=vars_))
        node_id = f"n_{atom_name}"
        parent = None if i == 0 else f"n_P{i-1}"
        nodes[node_id] = TDNode(
            id=node_id,
            bag_vars=vars_,
            parent=parent,
            children=tuple(),
        )

    for i in range(spine_length):
        atom_name = f"L{i}"
        vars_ = (f"x{i+1}", f"y{i}")
        atoms.append(Atom(name=atom_name, vars=vars_))
        node_id = f"n_{atom_name}"
        parent = f"n_P{i}"
        nodes[node_id] = TDNode(
            id=node_id,
            bag_vars=vars_,
            parent=parent,
            children=tuple(),
        )

    children_by_parent: dict[str, list[str]] = {node_id: [] for node_id in nodes}
    for node_id, node in nodes.items():
        if node.parent is not None:
            children_by_parent[node.parent].append(node_id)

    for node_id, node in list(nodes.items()):
        nodes[node_id] = TDNode(
            id=node.id,
            bag_vars=node.bag_vars,
            parent=node.parent,
            children=tuple(children_by_parent[node_id]),
        )

    output_vars = tuple(f"x{i}" for i in range(spine_length + 1)) + tuple(
        f"y{i}" for i in range(spine_length)
    )
    cq = CQ(atoms=tuple(atoms), output_vars=output_vars)
    td = TreeDecomposition(nodes=nodes, root="n_P0")
    return cq, td


def make_benchmark_query(shape: str, size: int) -> tuple[CQ, TreeDecomposition]:
    if shape == "path":
        return make_path_query(size)
    if shape == "star":
        return make_star_query(size)
    if shape == "binary_tree":
        return make_binary_tree_query(size)
    if shape == "caterpillar":
        return make_caterpillar_query(size)
    raise ValueError(f"Unsupported benchmark shape: {shape}")


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
