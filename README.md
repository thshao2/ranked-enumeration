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

## End-to-end Example 3: Run Benchmark Scripts

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
