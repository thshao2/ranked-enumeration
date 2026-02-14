from __future__ import annotations

from collections.abc import Iterator
from itertools import islice
from typing import Any

from ranked_enumeration.bag_relations import BagRelation, build_bag_relations
from ranked_enumeration.decomposition import TreeDecomposition, validate_tree_decomposition
from ranked_enumeration.model import CQ, Relation, validate_query_and_relations
from ranked_enumeration.preprocess import Cell, PreprocessedInstance, build_preprocessing
from ranked_enumeration.ranking import RankModel
from ranked_enumeration.reducer import run_full_reducer


class RankedEnumerator:
    def __init__(
        self,
        cq: CQ,
        relations: dict[str, Relation],
        td: TreeDecomposition,
        rank_model: RankModel,
    ) -> None:
        validate_query_and_relations(cq, relations)
        validate_tree_decomposition(cq, td)

        self.cq = cq
        self.relations = relations
        self.td = td
        self.rank_model = rank_model

        bag_relations = build_bag_relations(cq, relations, td)
        reduced = run_full_reducer(td, bag_relations)

        self._bag_relations: dict[str, BagRelation] = bag_relations
        self._reduced_bag_relations: dict[str, BagRelation] = reduced
        self._preprocessed: PreprocessedInstance = build_preprocessing(
            td=td,
            bag_relations=reduced,
            rank_model=rank_model,
            output_vars=cq.output_vars,
        )

    @property
    def reduced_bag_relations(self) -> dict[str, BagRelation]:
        return self._reduced_bag_relations

    def _root_cell(self, index: int) -> Cell | None:
        root_state = self._preprocessed.root_state()
        root_stream = root_state.get_stream(tuple())
        if root_stream is None:
            return None
        return root_stream.get(index)

    def __iter__(self) -> Iterator[tuple[Any, ...]]:
        return _RankedEnumeratorIterator(self)

    def top_k(self, k: int) -> list[tuple[Any, ...]]:
        if k <= 0:
            return []
        return list(islice(iter(self), k))


class _RankedEnumeratorIterator:
    def __init__(self, engine: RankedEnumerator) -> None:
        self._engine = engine
        self._index = 0
        self._seen: set[tuple[Any, ...]] = set()

    def __iter__(self) -> "_RankedEnumeratorIterator":
        return self

    def __next__(self) -> tuple[Any, ...]:
        while True:
            cell = self._engine._root_cell(self._index)
            self._index += 1
            if cell is None:
                raise StopIteration

            out = cell.output_tuple
            if out in self._seen:
                continue
            self._seen.add(out)
            return out
