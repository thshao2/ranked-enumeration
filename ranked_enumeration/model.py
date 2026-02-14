from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Atom:
    name: str
    vars: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Atom name must be non-empty")
        if not self.vars:
            raise ValueError(f"Atom {self.name} must have at least one variable")
        if len(set(self.vars)) != len(self.vars):
            raise ValueError(f"Atom {self.name} contains duplicate variables")


@dataclass(frozen=True)
class CQ:
    atoms: tuple[Atom, ...]
    output_vars: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.atoms:
            raise ValueError("CQ must contain at least one atom")
        if len(set(self.output_vars)) != len(self.output_vars):
            raise ValueError("output_vars contains duplicates")


@dataclass
class Relation:
    name: str
    vars: tuple[str, ...]
    rows: list[tuple[Any, ...]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Relation name must be non-empty")
        if len(set(self.vars)) != len(self.vars):
            raise ValueError(f"Relation {self.name} has duplicate variable names")

        normalized: set[tuple[Any, ...]] = set()
        for row in self.rows:
            if len(row) != len(self.vars):
                raise ValueError(
                    f"Relation {self.name} row arity {len(row)} does not match schema arity {len(self.vars)}"
                )
            normalized.add(tuple(row))
        self.rows = list(normalized)


def cq_variables(cq: CQ) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for atom in cq.atoms:
        for var in atom.vars:
            if var not in seen:
                seen.add(var)
                ordered.append(var)
    return tuple(ordered)


def validate_query_and_relations(cq: CQ, relations: dict[str, Relation]) -> None:
    all_vars = set(cq_variables(cq))
    if set(cq.output_vars) != all_vars:
        raise ValueError(
            "For v1, output_vars must be exactly the full set of query variables"
        )

    for atom in cq.atoms:
        if atom.name not in relations:
            raise ValueError(f"Missing relation for atom {atom.name}")
        relation = relations[atom.name]
        if len(relation.vars) != len(atom.vars):
            raise ValueError(
                f"Arity mismatch for atom {atom.name}: atom has {len(atom.vars)} vars, relation has {len(relation.vars)} vars"
            )
        if relation.vars != atom.vars:
            raise ValueError(
                f"Relation schema vars for {atom.name} must exactly match atom vars for v1: "
                f"expected {atom.vars}, got {relation.vars}"
            )


def row_to_assignment(atom: Atom, row: tuple[Any, ...]) -> dict[str, Any]:
    if len(atom.vars) != len(row):
        raise ValueError("row_to_assignment arity mismatch")
    return {var: row[i] for i, var in enumerate(atom.vars)}
