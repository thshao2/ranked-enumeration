from __future__ import annotations

from copy import deepcopy
from typing import Any

from ranked_enumeration.bag_relations import BagRelation
from ranked_enumeration.decomposition import TreeDecomposition, postorder_nodes, preorder_nodes


def _separator_vars(parent_vars: tuple[str, ...], child_vars: tuple[str, ...]) -> tuple[str, ...]:
    child_set = set(child_vars)
    return tuple(var for var in parent_vars if var in child_set)


def _indices(vars_order: tuple[str, ...], selected: tuple[str, ...]) -> tuple[int, ...]:
    pos = {var: i for i, var in enumerate(vars_order)}
    return tuple(pos[var] for var in selected)


def _key_set(
    tuples_data: list[tuple[Any, ...]],
    key_idx: tuple[int, ...],
) -> set[tuple[Any, ...]]:
    if not tuples_data:
        return set()
    if not key_idx:
        return {()}
    return {tuple(row[i] for i in key_idx) for row in tuples_data}


def _filter_by_keys(
    tuples_data: list[tuple[Any, ...]],
    key_idx: tuple[int, ...],
    allowed: set[tuple[Any, ...]],
) -> list[tuple[Any, ...]]:
    if not tuples_data:
        return []
    if not key_idx:
        return list(tuples_data) if () in allowed else []

    out: list[tuple[Any, ...]] = []
    for row in tuples_data:
        key = tuple(row[i] for i in key_idx)
        if key in allowed:
            out.append(row)
    return out


def run_full_reducer(
    td: TreeDecomposition,
    bag_relations: dict[str, BagRelation],
) -> dict[str, BagRelation]:
    reduced = deepcopy(bag_relations)

    # Bottom-up: each child constrains parent via separator support.
    post = postorder_nodes(td)
    for node_id in post:
        node = td.nodes[node_id]
        if node.parent is None:
            continue
        parent_id = node.parent
        child_rel = reduced[node_id]
        parent_rel = reduced[parent_id]

        sep = _separator_vars(parent_rel.vars, child_rel.vars)
        parent_idx = _indices(parent_rel.vars, sep)
        child_idx = _indices(child_rel.vars, sep)

        allowed_parent_keys = _key_set(child_rel.tuples, child_idx)
        parent_rel.tuples = _filter_by_keys(parent_rel.tuples, parent_idx, allowed_parent_keys)

    # Top-down: each parent constrains children after parent pruning settled.
    pre = preorder_nodes(td)
    for parent_id in pre:
        parent_rel = reduced[parent_id]
        parent_node = td.nodes[parent_id]
        for child_id in parent_node.children:
            child_rel = reduced[child_id]
            sep = _separator_vars(parent_rel.vars, child_rel.vars)
            parent_idx = _indices(parent_rel.vars, sep)
            child_idx = _indices(child_rel.vars, sep)
            allowed_child_keys = _key_set(parent_rel.tuples, parent_idx)
            child_rel.tuples = _filter_by_keys(child_rel.tuples, child_idx, allowed_child_keys)

    return reduced
