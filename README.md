# Ranked Enumeration for Acyclic CQs (Python)

Implementation of a ranked enumeration algorithm for acyclic conjunctive queries, following:

- Shaleen Deep and Paraschos Koutris, *Ranked Enumeration of Conjunctive Query Results* (ICDT 2021)
- DOI: https://doi.org/10.4230/LIPIcs.ICDT.2021.5

## What this project supports (v1)

- Acyclic conjunctive queries
- User-supplied tree decompositions
- Full-output enumeration (no projection)
- Deterministic tie-breaking by `(score, output_tuple_lex)`
- Ranked iterator + `top_k(k)` helper
- Ranking models:
  - `AdditiveRankModel`
  - `ConstantRankModel`
  - `TupleBasedRankModel`
  - `VertexBasedRankModel`
  - `LexicographicRankModel`
  - `CBoundedAdditiveRankModel`

## Ranking Model Selection Guide

| Model | When to use | Expected inputs |
|---|---|---|
| `AdditiveRankModel` | You want a custom decomposable numeric score and can define local bag scoring logic directly. | `local_fn(node_id, bag_assignment) -> float` |
| `ConstantRankModel` | You want no scoring preference (all answers same score) and rely on deterministic tie-breaking. | `value` (single constant float) |
| `TupleBasedRankModel` | Your score is a sum of base-tuple weights (relation-tuple contributions). | `cq`, `td`, `tuple_weights: {relation_name: {(tuple_values): weight}}` |
| `VertexBasedRankModel` | Your score is a sum of variable-value weights (attribute/domain-value contributions). | `td`, `vertex_weights: {var: {value: weight}}` |
| `LexicographicRankModel` | You need lexicographic ranking instead of scalar additive ranking. | `td`, `lex_order` (ordered variables, ascending) |
| `CBoundedAdditiveRankModel` | You want additive ranking plus runtime enforcement of practical acyclic c-bounded local-score constraints. | `local_fn(node_id, bag_assignment) -> float`, `c` (positive int) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pytest
```

Run tests:

```bash
python3 -m pytest -q
```

## End-to-end Example 1: Manual CQ + Tree Decomposition

```python
from ranked_enumeration import (
    Atom,
    CQ,
    Relation,
    TDNode,
    TreeDecomposition,
    AdditiveRankModel,
    RankedEnumerator,
)

# Query: Q(x,y,z) :- R(x,y), S(y,z)
cq = CQ(
    atoms=(
        Atom("R", ("x", "y")),
        Atom("S", ("y", "z")),
    ),
    output_vars=("x", "y", "z"),
)

# Tree decomposition with two bags
# n0: {x,y} (root), n1: {y,z} (child)
td = TreeDecomposition(
    nodes={
        "n0": TDNode(id="n0", bag_vars=("x", "y"), parent=None, children=("n1",)),
        "n1": TDNode(id="n1", bag_vars=("y", "z"), parent="n0", children=tuple()),
    },
    root="n0",
)

# Base relations (set semantics are enforced by Relation)
relations = {
    "R": Relation(name="R", vars=("x", "y"), rows=[(1, 10), (2, 10), (3, 11)]),
    "S": Relation(name="S", vars=("y", "z"), rows=[(10, 100), (10, 101), (11, 5)]),
}

# Simple additive score over bag assignments
rank_model = AdditiveRankModel(lambda _node, a: float(sum(a.values())))

engine = RankedEnumerator(cq, relations, td, rank_model)

print("Top-3:", engine.top_k(3))
print("All ranked results:")
for ans in engine:
    print(ans)
```

## End-to-end Example 2: Compare Against Brute-force Oracle

```python
from ranked_enumeration import AdditiveRankModel, RankedEnumerator
from ranked_enumeration.baseline import baseline_ranked
from ranked_enumeration.generators import make_path_query, instantiate_relations

cq, td = make_path_query(3)  # atoms over (x0,x1), (x1,x2), (x2,x3)
relations = instantiate_relations(cq, domain_size=5, tuple_probability=0.3, seed=42)
rank_model = AdditiveRankModel(lambda _n, a: float(sum(a.values())))

enum_out = list(RankedEnumerator(cq, relations, td, rank_model))
oracle_out = baseline_ranked(cq, relations, td, rank_model)

assert enum_out == oracle_out
print("Iterator output matches oracle.")
```

## End-to-end Example 3: Alternative Ranking Models

```python
from ranked_enumeration import (
    CBoundedAdditiveRankModel,
    LexicographicRankModel,
    TupleBasedRankModel,
    VertexBasedRankModel,
)

# Tuple-based (weight per base relation tuple)
tuple_rank = TupleBasedRankModel(
    cq=cq,
    td=td,
    tuple_weights={
        "R": {(1, 10): 2.0, (2, 10): 1.0, (3, 11): 4.0},
        "S": {(10, 100): 1.0, (10, 101): 3.0, (11, 5): 2.0},
    },
)

# Vertex-based (weight per variable assignment)
vertex_rank = VertexBasedRankModel(
    td=td,
    vertex_weights={
        "x": {1: 0.0, 2: 1.0, 3: 2.0},
        "y": {10: 0.5, 11: 1.5},
        "z": {5: 0.0, 100: 2.0, 101: 3.0},
    },
)

# Lexicographic (ascending by z, then x, then y)
lex_rank = LexicographicRankModel(td=td, lex_order=("z", "x", "y"))

# c-bounded additive (raises if the c-bound is violated after reduction)
c_rank = CBoundedAdditiveRankModel(
    local_fn=lambda node_id, a: float(sum(v for v in a.values() if isinstance(v, (int, float)))),
    c=2,
)
```

## End-to-end Example 4: Run Benchmark Scripts

```bash
python3 bench/bench_topk.py --shape path --size 4 --domain 20 --p 0.08 --k 50 --seed 0
python3 bench/bench_delay.py --shape path --size 4 --domain 20 --p 0.08 --limit 200 --seed 0
python3 bench/bench_compare_materialize_sort.py --shape star --size 4 --domain 20 --p 0.08 --seed 0
```

## Project Layout

```text
ranked_enumeration/   # core algorithm + models + iterator + baseline
tests/                # correctness and randomized tests
bench/                # simple benchmark scripts
docs/                 # design and experiment protocol notes
```

## Notes

- `pytest` is required only for running tests.
- The implementation expects relation schemas to match atom variable order exactly in v1.
