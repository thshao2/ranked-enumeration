from __future__ import annotations

import pytest

from ranked_enumeration.ranking import AdditiveRankModel, ConstantRankModel


@pytest.fixture
def additive_rank_model() -> AdditiveRankModel:
    return AdditiveRankModel(lambda _node, assignment: float(sum(assignment.values())))


@pytest.fixture
def constant_rank_model() -> ConstantRankModel:
    return ConstantRankModel(0.0)
