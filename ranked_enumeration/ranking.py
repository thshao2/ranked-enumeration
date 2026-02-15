from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

from ranked_enumeration.decomposition import TreeDecomposition, assign_atom_owners, assign_variable_owners
from ranked_enumeration.model import Atom, CQ


RankValue = Any

_WARNED_LEGACY_COMBINE_CLASSES: set[type[Any]] = set()
_COMBINE_ARITY_CACHE: dict[type[Any], int] = {}


class RankModel(Protocol):
    def local_weight(self, node_id: str, bag_assignment: Mapping[str, Any]) -> RankValue:
        ...

    def combine(
        self,
        node_id: str,
        local: RankValue,
        child_scores: Sequence[RankValue],
    ) -> RankValue:
        ...


class RankValidationModel(Protocol):
    def validate(self, td: TreeDecomposition, reduced_bag_relations: Mapping[str, Any]) -> None:
        ...


def _combine_arity(rank_model: RankModel) -> int:
    cls = type(rank_model)
    cached = _COMBINE_ARITY_CACHE.get(cls)
    if cached is not None:
        return cached

    signature = inspect.signature(rank_model.combine)
    positional_params = [
        p
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    arity = len(positional_params)
    _COMBINE_ARITY_CACHE[cls] = arity
    return arity


def combine_rank_values(
    rank_model: RankModel,
    node_id: str,
    local: RankValue,
    child_scores: Sequence[RankValue],
) -> RankValue:
    arity = _combine_arity(rank_model)

    if arity == 3:
        return rank_model.combine(node_id, local, child_scores)

    if arity == 2:
        cls = type(rank_model)
        if cls not in _WARNED_LEGACY_COMBINE_CLASSES:
            warnings.warn(
                (
                    f"{cls.__name__}.combine(local, child_scores) is deprecated; "
                    "use combine(node_id, local, child_scores)."
                ),
                DeprecationWarning,
                stacklevel=2,
            )
            _WARNED_LEGACY_COMBINE_CLASSES.add(cls)
        # mypy/typing: legacy models are intentionally supported at runtime.
        return rank_model.combine(local, child_scores)  # type: ignore[misc,call-arg]

    raise TypeError(
        f"Unsupported combine signature for {type(rank_model).__name__}; expected 2 or 3 positional args, got {arity}."
    )


@dataclass
class AdditiveRankModel:
    local_fn: Callable[[str, Mapping[str, Any]], float]

    def local_weight(self, node_id: str, bag_assignment: Mapping[str, Any]) -> RankValue:
        return float(self.local_fn(node_id, bag_assignment))

    def combine(
        self,
        node_id: str,
        local: RankValue,
        child_scores: Sequence[RankValue],
    ) -> RankValue:
        _ = node_id
        return float(local + sum(child_scores))


@dataclass
class ConstantRankModel:
    value: float = 0.0

    def local_weight(self, node_id: str, bag_assignment: Mapping[str, Any]) -> RankValue:
        _ = node_id
        _ = bag_assignment
        return float(self.value)

    def combine(
        self,
        node_id: str,
        local: RankValue,
        child_scores: Sequence[RankValue],
    ) -> RankValue:
        _ = node_id
        return float(local + sum(child_scores))


@dataclass
class TupleBasedRankModel:
    cq: CQ
    td: TreeDecomposition
    tuple_weights: dict[str, dict[tuple[Any, ...], float]]
    _owned_atoms_by_node: dict[str, list[Atom]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        owners = assign_atom_owners(self.cq, self.td)
        self._owned_atoms_by_node = {node_id: [] for node_id in self.td.nodes}
        for atom_idx, node_id in owners.items():
            atom = self.cq.atoms[atom_idx]
            self._owned_atoms_by_node[node_id].append(atom)

        for atom in self.cq.atoms:
            if atom.name not in self.tuple_weights:
                raise ValueError(f"Missing tuple weight map for relation {atom.name}")

    def local_weight(self, node_id: str, bag_assignment: Mapping[str, Any]) -> RankValue:
        total = 0.0
        for atom in self._owned_atoms_by_node[node_id]:
            key = tuple(bag_assignment[var] for var in atom.vars)
            weight_map = self.tuple_weights[atom.name]
            if key not in weight_map:
                raise ValueError(
                    f"Missing tuple weight for {atom.name}{key} during local score evaluation"
                )
            total += float(weight_map[key])
        return total

    def combine(
        self,
        node_id: str,
        local: RankValue,
        child_scores: Sequence[RankValue],
    ) -> RankValue:
        _ = node_id
        return float(local + sum(child_scores))

    def validate(self, td: TreeDecomposition, reduced_bag_relations: Mapping[str, Any]) -> None:
        for node_id, node in td.nodes.items():
            rel = reduced_bag_relations[node_id]
            bag_pos = {var: i for i, var in enumerate(node.bag_vars)}
            for row in rel.tuples:
                for atom in self._owned_atoms_by_node[node_id]:
                    key = tuple(row[bag_pos[var]] for var in atom.vars)
                    if key not in self.tuple_weights[atom.name]:
                        raise ValueError(
                            f"Missing tuple weight for {atom.name}{key} in reduced bag {node_id}"
                        )


@dataclass
class VertexBasedRankModel:
    td: TreeDecomposition
    vertex_weights: dict[str, dict[Any, float]]
    _owned_vars_by_node: dict[str, tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        all_vars: list[str] = []
        seen: set[str] = set()
        for node in self.td.nodes.values():
            for var in node.bag_vars:
                if var not in seen:
                    seen.add(var)
                    all_vars.append(var)

        owners = assign_variable_owners(self.td, all_vars)
        grouped: dict[str, list[str]] = {node_id: [] for node_id in self.td.nodes}
        for var in all_vars:
            grouped[owners[var]].append(var)
        self._owned_vars_by_node = {node_id: tuple(vars_) for node_id, vars_ in grouped.items()}

        for var in all_vars:
            if var not in self.vertex_weights:
                raise ValueError(f"Missing vertex weight map for variable '{var}'")

    def local_weight(self, node_id: str, bag_assignment: Mapping[str, Any]) -> RankValue:
        total = 0.0
        for var in self._owned_vars_by_node[node_id]:
            value = bag_assignment[var]
            if value not in self.vertex_weights[var]:
                raise ValueError(
                    f"Missing vertex weight for ({var}={value}) during local score evaluation"
                )
            total += float(self.vertex_weights[var][value])
        return total

    def combine(
        self,
        node_id: str,
        local: RankValue,
        child_scores: Sequence[RankValue],
    ) -> RankValue:
        _ = node_id
        return float(local + sum(child_scores))

    def validate(self, td: TreeDecomposition, reduced_bag_relations: Mapping[str, Any]) -> None:
        for node_id, node in td.nodes.items():
            rel = reduced_bag_relations[node_id]
            bag_pos = {var: i for i, var in enumerate(node.bag_vars)}
            for row in rel.tuples:
                for var in self._owned_vars_by_node[node_id]:
                    value = row[bag_pos[var]]
                    if value not in self.vertex_weights[var]:
                        raise ValueError(
                            f"Missing vertex weight for ({var}={value}) in reduced bag {node_id}"
                        )


@dataclass
class LexicographicRankModel:
    td: TreeDecomposition
    lex_order: tuple[str, ...]
    _var_owner: dict[str, str] = field(init=False, repr=False)
    _owned_lex_by_node: dict[str, tuple[str, ...]] = field(init=False, repr=False)
    _owned_pos_by_node: dict[str, dict[str, int]] = field(init=False, repr=False)
    _subtree_nodes_by_node: dict[str, set[str]] = field(init=False, repr=False)
    _subtree_vars_by_node: dict[str, set[str]] = field(init=False, repr=False)
    _subtree_lex_by_node: dict[str, tuple[str, ...]] = field(init=False, repr=False)
    _var_child_source: dict[str, dict[str, str]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(set(self.lex_order)) != len(self.lex_order):
            raise ValueError("lex_order contains duplicate variables")

        all_td_vars: set[str] = set()
        for node in self.td.nodes.values():
            all_td_vars.update(node.bag_vars)

        if set(self.lex_order) != all_td_vars:
            raise ValueError(
                f"lex_order must cover exactly TD variables; expected {sorted(all_td_vars)}, got {list(self.lex_order)}"
            )

        self._var_owner = assign_variable_owners(self.td, self.lex_order)

        grouped: dict[str, list[str]] = {node_id: [] for node_id in self.td.nodes}
        for var in self.lex_order:
            grouped[self._var_owner[var]].append(var)
        self._owned_lex_by_node = {
            node_id: tuple(vars_) for node_id, vars_ in grouped.items()
        }
        self._owned_pos_by_node = {
            node_id: {var: i for i, var in enumerate(vars_)}
            for node_id, vars_ in self._owned_lex_by_node.items()
        }

        self._subtree_nodes_by_node = {}

        def visit(node_id: str) -> set[str]:
            acc = {node_id}
            for child_id in self.td.nodes[node_id].children:
                acc.update(visit(child_id))
            self._subtree_nodes_by_node[node_id] = acc
            return acc

        visit(self.td.root)

        self._subtree_vars_by_node = {
            node_id: {
                var for var, owner in self._var_owner.items() if owner in subtree_nodes
            }
            for node_id, subtree_nodes in self._subtree_nodes_by_node.items()
        }

        self._subtree_lex_by_node = {
            node_id: tuple(var for var in self.lex_order if var in vars_set)
            for node_id, vars_set in self._subtree_vars_by_node.items()
        }

        self._var_child_source = {node_id: {} for node_id in self.td.nodes}
        for node_id, node in self.td.nodes.items():
            for child_id in node.children:
                for var in self._subtree_vars_by_node[child_id]:
                    self._var_child_source[node_id][var] = child_id

    def local_weight(self, node_id: str, bag_assignment: Mapping[str, Any]) -> RankValue:
        return tuple(bag_assignment[var] for var in self._owned_lex_by_node[node_id])

    def combine(
        self,
        node_id: str,
        local: RankValue,
        child_scores: Sequence[RankValue],
    ) -> RankValue:
        local_tuple = tuple(local)
        child_score_by_id = {
            child_id: tuple(child_scores[i])
            for i, child_id in enumerate(self.td.nodes[node_id].children)
        }

        merged: list[Any] = []
        for var in self._subtree_lex_by_node[node_id]:
            owner = self._var_owner[var]
            if owner == node_id:
                local_pos = self._owned_pos_by_node[node_id][var]
                merged.append(local_tuple[local_pos])
                continue

            source_child = self._var_child_source[node_id].get(var)
            if source_child is None:
                raise ValueError(
                    f"Lexicographic combine failed for node {node_id}: no child source for variable '{var}'"
                )

            child_tuple = child_score_by_id[source_child]
            child_var_pos = self._subtree_lex_by_node[source_child].index(var)
            merged.append(child_tuple[child_var_pos])

        return tuple(merged)


@dataclass
class CBoundedAdditiveRankModel:
    local_fn: Callable[[str, Mapping[str, Any]], float]
    c: int

    def __post_init__(self) -> None:
        if self.c <= 0:
            raise ValueError("c must be a positive integer")

    def local_weight(self, node_id: str, bag_assignment: Mapping[str, Any]) -> RankValue:
        return float(self.local_fn(node_id, bag_assignment))

    def combine(
        self,
        node_id: str,
        local: RankValue,
        child_scores: Sequence[RankValue],
    ) -> RankValue:
        _ = node_id
        return float(local + sum(child_scores))

    def validate(self, td: TreeDecomposition, reduced_bag_relations: Mapping[str, Any]) -> None:
        from ranked_enumeration.rank_validation import validate_c_bounded_local_scores

        validate_c_bounded_local_scores(
            td=td,
            reduced_bag_relations=reduced_bag_relations,
            local_weight_fn=self.local_weight,
            c=self.c,
        )
