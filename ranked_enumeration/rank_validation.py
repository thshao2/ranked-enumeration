from __future__ import annotations

from typing import Any, Callable, Mapping

from ranked_enumeration.decomposition import TreeDecomposition


def _key_vars_for_node(td: TreeDecomposition, node_id: str) -> tuple[str, ...]:
    node = td.nodes[node_id]
    if node.parent is None:
        return tuple()
    parent_vars = set(td.nodes[node.parent].bag_vars)
    return tuple(var for var in node.bag_vars if var in parent_vars)


def validate_c_bounded_local_scores(
    td: TreeDecomposition,
    reduced_bag_relations: Mapping[str, Any],
    local_weight_fn: Callable[[str, Mapping[str, Any]], float],
    c: int,
) -> None:
    if c <= 0:
        raise ValueError("c must be a positive integer")

    for node_id, node in td.nodes.items():
        rel = reduced_bag_relations[node_id]
        key_vars = _key_vars_for_node(td, node_id)
        key_pos = [node.bag_vars.index(var) for var in key_vars]

        score_sets_by_key: dict[tuple[Any, ...], set[float]] = {}
        for row in rel.tuples:
            bag_assignment = {var: row[i] for i, var in enumerate(node.bag_vars)}
            key = tuple(row[i] for i in key_pos) if key_pos else tuple()
            score = float(local_weight_fn(node_id, bag_assignment))

            bucket = score_sets_by_key.setdefault(key, set())
            bucket.add(score)
            if len(bucket) > c:
                raise ValueError(
                    (
                        f"c-bounded constraint violated at node '{node_id}' key={key}: "
                        f"found {len(bucket)} distinct local scores (c={c})"
                    )
                )
