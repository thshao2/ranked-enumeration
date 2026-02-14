from __future__ import annotations

from typing import Any

from ranked_enumeration.decomposition import TreeDecomposition
from ranked_enumeration.model import Atom, CQ, Relation, row_to_assignment
from ranked_enumeration.ranking import RankModel


def _consistent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    overlap = set(left).intersection(right)
    return all(left[var] == right[var] for var in overlap)


def _join_with_atom(
    partials: list[dict[str, Any]], atom: Atom, relation: Relation
) -> list[dict[str, Any]]:
    atom_rows = [row_to_assignment(atom, row) for row in relation.rows]
    out: list[dict[str, Any]] = []
    for p in partials:
        for a in atom_rows:
            if _consistent(p, a):
                merged = dict(p)
                merged.update(a)
                out.append(merged)
    return out


def materialize_full_join(
    cq: CQ,
    relations: dict[str, Relation],
) -> list[dict[str, Any]]:
    partials: list[dict[str, Any]] = [dict()]
    for atom in cq.atoms:
        partials = _join_with_atom(partials, atom, relations[atom.name])
        if not partials:
            return []

    # Set semantics on full output variables.
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for assignment in partials:
        out = tuple(assignment[var] for var in cq.output_vars)
        if out in seen:
            continue
        seen.add(out)
        deduped.append(assignment)
    return deduped


def score_assignment(
    td: TreeDecomposition,
    rank_model: RankModel,
    assignment: dict[str, Any],
) -> float:
    def dfs(node_id: str) -> float:
        node = td.nodes[node_id]
        local_assignment = {var: assignment[var] for var in node.bag_vars}
        child_scores = [dfs(child_id) for child_id in node.children]
        local = rank_model.local_weight(node_id, local_assignment)
        return float(rank_model.combine(local, child_scores))

    return dfs(td.root)


def baseline_ranked(
    cq: CQ,
    relations: dict[str, Relation],
    td: TreeDecomposition,
    rank_model: RankModel,
) -> list[tuple[Any, ...]]:
    assignments = materialize_full_join(cq, relations)

    scored: list[tuple[float, tuple[Any, ...]]] = []
    for assignment in assignments:
        out = tuple(assignment[var] for var in cq.output_vars)
        score = score_assignment(td, rank_model, assignment)
        scored.append((score, out))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [out for _, out in scored]
