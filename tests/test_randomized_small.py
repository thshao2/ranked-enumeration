from __future__ import annotations

from ranked_enumeration.baseline import baseline_ranked
from ranked_enumeration.generators import instantiate_relations, make_path_query, make_star_query
from ranked_enumeration.iterator import RankedEnumerator
from ranked_enumeration.ranking import AdditiveRankModel


def _rank_model() -> AdditiveRankModel:
    return AdditiveRankModel(lambda _node, assignment: float(sum(assignment.values())))


def test_randomized_small_path_and_star_match_oracle() -> None:
    for seed in range(15):
        for make_query in (lambda: make_path_query(3), lambda: make_star_query(3)):
            cq, td = make_query()
            relations = instantiate_relations(
                cq=cq,
                domain_size=4,
                tuple_probability=0.35,
                seed=seed,
            )
            rank_model = _rank_model()
            oracle = baseline_ranked(cq, relations, td, rank_model)
            enum = RankedEnumerator(cq, relations, td, rank_model)
            assert list(enum) == oracle
