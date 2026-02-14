from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence


class RankModel(Protocol):
    def local_weight(self, node_id: str, bag_assignment: Mapping[str, Any]) -> float:
        ...

    def combine(self, local: float, child_scores: Sequence[float]) -> float:
        ...


@dataclass
class AdditiveRankModel:
    local_fn: Callable[[str, Mapping[str, Any]], float]

    def local_weight(self, node_id: str, bag_assignment: Mapping[str, Any]) -> float:
        return float(self.local_fn(node_id, bag_assignment))

    def combine(self, local: float, child_scores: Sequence[float]) -> float:
        return float(local + sum(child_scores))


@dataclass
class ConstantRankModel:
    value: float = 0.0

    def local_weight(self, node_id: str, bag_assignment: Mapping[str, Any]) -> float:
        return float(self.value)

    def combine(self, local: float, child_scores: Sequence[float]) -> float:
        return float(local + sum(child_scores))
