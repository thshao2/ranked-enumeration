from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

from ranked_enumeration.bag_relations import BagRelation
from ranked_enumeration.decomposition import TreeDecomposition, postorder_nodes
from ranked_enumeration.ranking import RankModel


@dataclass
class Cell:
    node_id: str
    key: tuple[Any, ...]
    local_tuple: tuple[Any, ...]
    child_indices: tuple[int, ...]
    child_keys: tuple[tuple[Any, ...], ...]
    child_cells: tuple[Cell, ...]
    score: float
    assignment: dict[str, Any]
    output_tuple: tuple[Any, ...]
    successor: Cell | None = None


@dataclass
class NodeState:
    node_id: str
    bag_vars: tuple[str, ...]
    key_vars: tuple[str, ...]
    child_ids: tuple[str, ...]
    child_states: dict[str, "NodeState"]
    bag_tuples_by_key: dict[tuple[Any, ...], list[tuple[Any, ...]]]
    subtree_vars: tuple[str, ...]
    rank_model: RankModel
    streams: dict[tuple[Any, ...], "KeyStream"] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._bag_pos = {var: i for i, var in enumerate(self.bag_vars)}

    def key_for_child(self, child_id: str, theta: tuple[Any, ...]) -> tuple[Any, ...]:
        child_state = self.child_states[child_id]
        return tuple(theta[self._bag_pos[var]] for var in child_state.key_vars)

    def local_assignment(self, theta: tuple[Any, ...]) -> dict[str, Any]:
        return {var: theta[i] for i, var in enumerate(self.bag_vars)}

    def get_stream(self, key: tuple[Any, ...]) -> "KeyStream | None":
        if key not in self.bag_tuples_by_key:
            return None
        stream = self.streams.get(key)
        if stream is None:
            stream = KeyStream(node_state=self, key=key, thetas=self.bag_tuples_by_key[key])
            self.streams[key] = stream
        return stream

    def get_cell(self, key: tuple[Any, ...], index: int) -> Cell | None:
        stream = self.get_stream(key)
        if stream is None:
            return None
        return stream.get(index)


class KeyStream:
    def __init__(
        self,
        node_state: NodeState,
        key: tuple[Any, ...],
        thetas: list[tuple[Any, ...]],
    ) -> None:
        self.node_state = node_state
        self.key = key
        self._thetas = thetas
        self._frontier: list[tuple[Any, ...]] = []
        self._seen_states: set[tuple[int, tuple[int, ...]]] = set()
        self._emitted: list[Cell] = []
        self._initialized = False
        self._serial = 0

    def _evaluate_state(
        self, theta_idx: int, child_indices: tuple[int, ...]
    ) -> dict[str, Any] | None:
        theta = self._thetas[theta_idx]
        bag_assignment = self.node_state.local_assignment(theta)

        child_cells: list[Cell] = []
        child_keys: list[tuple[Any, ...]] = []

        for j, child_id in enumerate(self.node_state.child_ids):
            child_key = self.node_state.key_for_child(child_id, theta)
            child_keys.append(child_key)
            child_cell = self.node_state.child_states[child_id].get_cell(child_key, child_indices[j])
            if child_cell is None:
                return None
            child_cells.append(child_cell)

        merged_assignment = dict(bag_assignment)
        for child_cell in child_cells:
            for var, value in child_cell.assignment.items():
                if var in merged_assignment and merged_assignment[var] != value:
                    return None
                merged_assignment[var] = value

        local = self.node_state.rank_model.local_weight(self.node_state.node_id, bag_assignment)
        score = self.node_state.rank_model.combine(local, [c.score for c in child_cells])

        output_tuple = tuple(merged_assignment[var] for var in self.node_state.subtree_vars)
        return {
            "theta_idx": theta_idx,
            "theta": theta,
            "child_indices": child_indices,
            "child_keys": tuple(child_keys),
            "child_cells": tuple(child_cells),
            "assignment": merged_assignment,
            "score": float(score),
            "output_tuple": output_tuple,
        }

    def _push_state(self, theta_idx: int, child_indices: tuple[int, ...]) -> None:
        state_id = (theta_idx, child_indices)
        if state_id in self._seen_states:
            return
        evaluated = self._evaluate_state(theta_idx, child_indices)
        if evaluated is None:
            return

        self._seen_states.add(state_id)
        theta = evaluated["theta"]
        output_tuple = evaluated["output_tuple"]
        score = evaluated["score"]

        entry = (
            score,
            output_tuple,
            theta,
            child_indices,
            self._serial,
            evaluated,
        )
        self._serial += 1
        heapq.heappush(self._frontier, entry)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        num_children = len(self.node_state.child_ids)
        base_child_indices = tuple(0 for _ in range(num_children))
        for theta_idx in range(len(self._thetas)):
            self._push_state(theta_idx, base_child_indices)

    def get(self, index: int) -> Cell | None:
        self._ensure_initialized()
        if index < 0:
            return None

        while len(self._emitted) <= index:
            if not self._frontier:
                return None

            _, _, _, _, _, payload = heapq.heappop(self._frontier)
            cell = Cell(
                node_id=self.node_state.node_id,
                key=self.key,
                local_tuple=payload["theta"],
                child_indices=payload["child_indices"],
                child_keys=payload["child_keys"],
                child_cells=payload["child_cells"],
                score=payload["score"],
                assignment=payload["assignment"],
                output_tuple=payload["output_tuple"],
            )
            if self._emitted:
                self._emitted[-1].successor = cell
            self._emitted.append(cell)

            if self.node_state.child_ids:
                theta_idx = payload["theta_idx"]
                current = payload["child_indices"]
                for j in range(len(current)):
                    next_indices = list(current)
                    next_indices[j] += 1
                    self._push_state(theta_idx, tuple(next_indices))

        return self._emitted[index]


@dataclass
class PreprocessedInstance:
    states: dict[str, NodeState]
    root: str

    def root_state(self) -> NodeState:
        return self.states[self.root]


def _key_vars_for_node(td: TreeDecomposition, node_id: str) -> tuple[str, ...]:
    node = td.nodes[node_id]
    if node.parent is None:
        return tuple()
    parent_vars = set(td.nodes[node.parent].bag_vars)
    return tuple(var for var in node.bag_vars if var in parent_vars)


def build_preprocessing(
    td: TreeDecomposition,
    bag_relations: dict[str, BagRelation],
    rank_model: RankModel,
    output_vars: tuple[str, ...],
) -> PreprocessedInstance:
    post = postorder_nodes(td)

    subtree_var_sets: dict[str, set[str]] = {}
    for node_id in post:
        node = td.nodes[node_id]
        acc = set(node.bag_vars)
        for child_id in node.children:
            acc.update(subtree_var_sets[child_id])
        subtree_var_sets[node_id] = acc

    states: dict[str, NodeState] = {}

    for node_id in post:
        node = td.nodes[node_id]
        rel = bag_relations[node_id]
        key_vars = _key_vars_for_node(td, node_id)
        key_pos = [rel.vars.index(var) for var in key_vars]

        grouped: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
        for row in rel.tuples:
            key = tuple(row[i] for i in key_pos) if key_pos else tuple()
            grouped.setdefault(key, []).append(row)

        subtree_vars = tuple(var for var in output_vars if var in subtree_var_sets[node_id])

        child_states = {child_id: states[child_id] for child_id in node.children}
        states[node_id] = NodeState(
            node_id=node_id,
            bag_vars=node.bag_vars,
            key_vars=key_vars,
            child_ids=node.children,
            child_states=child_states,
            bag_tuples_by_key=grouped,
            subtree_vars=subtree_vars,
            rank_model=rank_model,
        )

    return PreprocessedInstance(states=states, root=td.root)
