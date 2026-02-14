from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from ranked_enumeration.model import Atom, CQ


@dataclass
class TDNode:
    id: str
    bag_vars: tuple[str, ...]
    parent: str | None
    children: tuple[str, ...]


@dataclass
class TreeDecomposition:
    nodes: dict[str, TDNode]
    root: str


def preorder_nodes(td: TreeDecomposition) -> list[str]:
    order: list[str] = []
    stack = [td.root]
    while stack:
        node_id = stack.pop()
        order.append(node_id)
        children = list(td.nodes[node_id].children)
        children.reverse()
        stack.extend(children)
    return order


def postorder_nodes(td: TreeDecomposition) -> list[str]:
    order: list[str] = []

    def visit(node_id: str) -> None:
        for child in td.nodes[node_id].children:
            visit(child)
        order.append(node_id)

    visit(td.root)
    return order


def compute_depths(td: TreeDecomposition) -> dict[str, int]:
    depths: dict[str, int] = {td.root: 0}
    queue = deque([td.root])
    while queue:
        cur = queue.popleft()
        for child in td.nodes[cur].children:
            depths[child] = depths[cur] + 1
            queue.append(child)
    return depths


def validate_tree_decomposition(cq: CQ, td: TreeDecomposition) -> None:
    if td.root not in td.nodes:
        raise ValueError("Tree decomposition root is missing")

    # Node id consistency and bag cleanliness.
    for node_id, node in td.nodes.items():
        if node_id != node.id:
            raise ValueError(f"Node dictionary key {node_id} does not match node.id {node.id}")
        if len(set(node.bag_vars)) != len(node.bag_vars):
            raise ValueError(f"Node {node_id} bag_vars contains duplicates")

    # Parent/child coherence.
    edge_count = 0
    for node_id, node in td.nodes.items():
        if node.parent is None:
            if node_id != td.root:
                raise ValueError(f"Only root can have parent=None; got {node_id}")
        else:
            if node.parent not in td.nodes:
                raise ValueError(f"Node {node_id} has unknown parent {node.parent}")
            if node_id not in td.nodes[node.parent].children:
                raise ValueError(
                    f"Node {node_id} parent {node.parent} does not list it as a child"
                )
        for child in node.children:
            edge_count += 1
            if child not in td.nodes:
                raise ValueError(f"Node {node_id} references unknown child {child}")
            if td.nodes[child].parent != node_id:
                raise ValueError(
                    f"Child {child} parent mismatch: expected {node_id}, got {td.nodes[child].parent}"
                )

    if edge_count != len(td.nodes) - 1:
        raise ValueError("Tree decomposition must have exactly |V|-1 edges")

    # Connectedness.
    visited: set[str] = set()
    queue = deque([td.root])
    while queue:
        cur = queue.popleft()
        if cur in visited:
            continue
        visited.add(cur)
        node = td.nodes[cur]
        if node.parent is not None:
            queue.append(node.parent)
        for child in node.children:
            queue.append(child)

    if visited != set(td.nodes):
        raise ValueError("Tree decomposition graph is not connected")

    # Running intersection for each variable in bags.
    vars_to_nodes: dict[str, set[str]] = {}
    for node_id, node in td.nodes.items():
        for var in node.bag_vars:
            vars_to_nodes.setdefault(var, set()).add(node_id)

    for var, nodes_with_var in vars_to_nodes.items():
        start = next(iter(nodes_with_var))
        seen_in_induced: set[str] = set()
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur in seen_in_induced:
                continue
            if cur not in nodes_with_var:
                continue
            seen_in_induced.add(cur)
            node = td.nodes[cur]
            if node.parent is not None:
                queue.append(node.parent)
            for child in node.children:
                queue.append(child)

        if seen_in_induced != nodes_with_var:
            raise ValueError(
                f"Running intersection violated for variable '{var}': {sorted(nodes_with_var)}"
            )

    # Every atom must be covered by at least one bag.
    for atom in cq.atoms:
        atom_vars = set(atom.vars)
        covered = any(atom_vars.issubset(set(node.bag_vars)) for node in td.nodes.values())
        if not covered:
            raise ValueError(f"No bag covers atom {atom.name}{atom.vars}")


def assign_atom_owners(cq: CQ, td: TreeDecomposition) -> dict[int, str]:
    depths = compute_depths(td)
    owners: dict[int, str] = {}

    for i, atom in enumerate(cq.atoms):
        atom_vars = set(atom.vars)
        candidates = [
            node_id
            for node_id, node in td.nodes.items()
            if atom_vars.issubset(set(node.bag_vars))
        ]
        if not candidates:
            raise ValueError(f"No valid owner bag found for atom {atom.name}{atom.vars}")
        owners[i] = min(candidates, key=lambda node_id: depths[node_id])

    return owners


def atoms_per_node(cq: CQ, td: TreeDecomposition) -> dict[str, list[Atom]]:
    owners = assign_atom_owners(cq, td)
    grouped: dict[str, list[Atom]] = {node_id: [] for node_id in td.nodes}
    for atom_idx, node_id in owners.items():
        grouped[node_id].append(cq.atoms[atom_idx])
    return grouped
