from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ranked_enumeration.decomposition import TreeDecomposition, atoms_per_node
from ranked_enumeration.model import CQ, Relation, row_to_assignment


@dataclass
class BagRelation:
    vars: tuple[str, ...]
    tuples: list[tuple[Any, ...]]


def _is_consistent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    overlap = set(left).intersection(right)
    return all(left[var] == right[var] for var in overlap)


def _natural_join_assignments(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for l_item in left:
        for r_item in right:
            if _is_consistent(l_item, r_item):
                merged = dict(l_item)
                merged.update(r_item)
                out.append(merged)
    return out


def _dedup_tuple_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[tuple[Any, ...]] = []
    for row in rows:
        if row in seen:
            continue
        seen.add(row)
        deduped.append(row)
    return deduped


def build_bag_relations(
    cq: CQ,
    relations: dict[str, Relation],
    td: TreeDecomposition,
) -> dict[str, BagRelation]:
    owned_atoms = atoms_per_node(cq, td)
    bag_relations: dict[str, BagRelation] = {}

    for node_id, node in td.nodes.items():
        local_atoms = owned_atoms[node_id]
        if not local_atoms:
            raise ValueError(
                f"Node {node_id} has no owned atoms. For v1, provide a decomposition where each node owns at least one atom."
            )

        partials: list[dict[str, Any]] = [dict()]
        for atom in local_atoms:
            relation = relations[atom.name]
            atom_assignments = [row_to_assignment(atom, row) for row in relation.rows]
            partials = _natural_join_assignments(partials, atom_assignments)
            if not partials:
                break

        local_var_cover: set[str] = set()
        for atom in local_atoms:
            local_var_cover.update(atom.vars)

        missing = [var for var in node.bag_vars if var not in local_var_cover]
        if missing:
            raise ValueError(
                f"Node {node_id} bag has variables not covered by owned atoms: {missing}. "
                "Use a tighter decomposition for v1."
            )

        tuple_rows: list[tuple[Any, ...]] = []
        for assignment in partials:
            tuple_rows.append(tuple(assignment[var] for var in node.bag_vars))

        bag_relations[node_id] = BagRelation(
            vars=node.bag_vars,
            tuples=_dedup_tuple_rows(tuple_rows),
        )

    return bag_relations
