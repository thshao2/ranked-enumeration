# Experiment Protocol

## Purpose
The benchmark scripts in this repository serve two goals:

1. Validate that the ranked enumerator returns the same ordered output as the brute-force oracle.
2. Measure where ranked enumeration helps in practice, especially for early output and `top_k` queries.

All benchmark scripts generate a synthetic acyclic conjunctive query, instantiate random in-memory relations, run the ranked enumerator with an additive score, and report timing metrics. The brute-force baseline fully materializes the join result and globally sorts it.

## Benchmark Scripts

### `bench/bench_topk.py`
Measures how long it takes to construct the ranked enumerator and return the first `k` tuples.

Use this script when you want to show:
- preprocessing overhead,
- `top_k` latency,
- that ranked enumeration can avoid full materialization when only a small prefix is needed.

Example:

```bash
.venv/bin/python bench/bench_topk.py --shape path --size 20 --domain 50 --p 0.24 --k 50 --seed 0
```

### `bench/bench_delay.py`
Measures delay-oriented metrics from the ranked enumerator.

Use this script when you want to show:
- first-result latency,
- average delay between outputs,
- p95 delay,
- streaming throughput.

Example:

```bash
.venv/bin/python bench/bench_delay.py --shape binary_tree --size 3 --domain 10 --p 0.12 --limit 100 --seed 0
```

### `bench/bench_compare_materialize_sort.py`
Compares full ranked enumeration against the brute-force materialize-and-sort baseline.

Use this script when you want to show:
- exact output agreement,
- full enumeration time for the ranked iterator,
- full baseline time,
- whether the ranked method is faster or slower end-to-end on a given workload.

Example:

```bash
.venv/bin/python bench/bench_compare_materialize_sort.py --shape star --size 6 --domain 20 --p 0.08 --seed 0
```

### `bench/bench_sweep_summary.py`
Runs many benchmark cases across multiple shapes, sizes, and random seeds, then prints both per-run results and an aggregated summary.

Example:

```bash
.venv/bin/python bench/bench_sweep_summary.py \
  --shapes path star binary_tree caterpillar \
  --sizes 4 6 \
  --domain 20 \
  --p 0.08 \
  --k 50 \
  --limit 200 \
  --seeds 0 1 2
```

## Quick Start

### 1. Run tests first
This confirms the implementation is correct before timing anything.

```bash
.venv/bin/python -m pytest -q
```

### 2. Run one benchmark case per script

```bash
.venv/bin/python bench/bench_topk.py --shape path --size 4 --domain 20 --p 0.08 --k 50 --seed 0
.venv/bin/python bench/bench_delay.py --shape path --size 4 --domain 20 --p 0.08 --limit 200 --seed 0
.venv/bin/python bench/bench_compare_materialize_sort.py --shape path --size 4 --domain 20 --p 0.08 --seed 0
```

### 3. Run a multi-seed sweep for a presentation summary

```bash
.venv/bin/python bench/bench_sweep_summary.py \
  --shapes path star binary_tree caterpillar \
  --sizes 4 \
  --domain 20 \
  --p 0.08 \
  --k 50 \
  --limit 200 \
  --seeds 0 1 2 3 4
```

## Parameters
All benchmark scripts use the same core workload parameters.

### `--shape`
Selects the query structure.

Allowed values:
- `path`
- `star`
- `binary_tree`
- `caterpillar`

This is usually the first parameter to choose, because it controls the join pattern and branching behavior.

### `--size`
Controls query size, but the exact meaning depends on `--shape`.

- For `path`: number of edges/atoms in the chain.
- For `star`: number of arms/atoms attached to the center variable.
- For `binary_tree`: tree depth.
- For `caterpillar`: spine length.

Because the semantics differ by shape, do not compare two cases with the same `size` unless the shape is also the same.

### `--domain`
The active domain is the set of integer values `{0, 1, ..., domain - 1}` used for each attribute.

Larger domains:
- create larger candidate relation spaces,
- usually reduce accidental join matches if `p` is fixed,
- can increase relation generation cost.

### `--p`
The Bernoulli sampling probability used when generating each possible relation tuple.

For every atom relation, the generator enumerates all tuples over the active domain and includes each one independently with probability `p`.

Implications:
- smaller `p` gives sparser relations,
- larger `p` gives denser relations,
- very large `p` can make the join output explode.

For a binary relation over domain size `d`, each relation has approximately `p * d^2` tuples on average.

### `--k`
Only used by `bench_topk.py` and `bench_sweep_summary.py`.

This is the number of ranked tuples requested from the enumerator. Use small `k` if your presentation goal is to demonstrate the advantage of incremental ranked access over full sorting.

### `--limit`
Only used by `bench_delay.py` and `bench_sweep_summary.py`.

This is the maximum number of outputs consumed when measuring delay metrics. It exists so that delay measurement stays practical even when the full result contains many tuples.

### `--seed`
Sets the pseudorandom seed for relation generation.

Changing `seed` changes the concrete relation contents while keeping the same query shape and parameter regime. Use multiple seeds when you want a fairer summary and not a single lucky or unlucky instance.

### `--shapes`, `--sizes`, `--seeds`
These list-valued arguments are only used by `bench_sweep_summary.py`.

They allow batch execution over many configurations in one command.

## Query Shapes

### Path
A path query is a chain of binary relations:

```text
R0(x0, x1) ⋈ R1(x1, x2) ⋈ ... ⋈ R(L-1)(x(L-1), xL)
```

If `--size L` is used, there are:
- `L` atoms,
- `L + 1` variables,
- one shared variable between consecutive atoms.

This is the simplest acyclic shape and is useful as a baseline workload.

### Star
A star query has one center variable shared by all arms:

```text
S0(c, y0) ⋈ S1(c, y1) ⋈ ... ⋈ S(A-1)(c, y(A-1))
```

If `--size A` is used, there are:
- `A` atoms,
- one central variable `c`,
- `A` leaf variables.

This shape tends to create high fan-out around the center and is useful for showing the effect of branching.

### Binary Tree
A binary-tree query organizes binary relations along the edges of a full binary tree:

```text
B0_1(v0, v1) ⋈ B0_2(v0, v2) ⋈ B1_3(v1, v3) ⋈ B1_4(v1, v4) ⋈ ...
```

If `--size D` is used, then `D` is the tree depth. The generated query has:
- `2^(D+1) - 1` variables,
- one atom for each tree edge,
- branching at every internal node.

This is a more demanding acyclic structure than a path because ranking decisions propagate through multiple subtrees.

### Caterpillar
A caterpillar query has a path-like spine with one leaf attached at each spine step:

```text
P0(x0, x1) ⋈ P1(x1, x2) ⋈ ... ⋈ P(L-1)(x(L-1), xL)
⋈ L0(x1, y0) ⋈ L1(x2, y1) ⋈ ... ⋈ L(L-1)(xL, y(L-1))
```

If `--size L` is used, then `L` is the spine length. The generated query has:
- `L` spine atoms,
- `L` leaf atoms,
- a chain of shared `x` variables,
- one extra `y` leaf variable attached to each spine edge.

This shape mixes path behavior with repeated local branching and is often useful for showing more realistic intermediate complexity.

## How Relation Generation Works
For each relation in the chosen query:

1. The script enumerates all tuples over the active domain.
2. Each tuple is included independently with probability `p`.
3. The resulting relation is loaded into the benchmark instance.

This means:
- larger `domain` and larger `p` both make relations denser,
- larger `size` increases the number of atoms and variables,
- the final join output size can vary significantly across seeds.

Because of this variability, one-off timings can be misleading. For reporting, use multiple seeds and summarize the results.

## Output Interpretation

### `bench_topk.py` output
Example:

```text
shape=path size=20 domain=50 p=0.24 k=50 prep_s=0.034767 topk_s=0.068668 returned=50
```

Field meanings:
- `shape=path`: the workload family.
- `size=20`: here, a 20-edge path query.
- `domain=50`: attribute values come from `{0, ..., 49}`.
- `p=0.24`: each candidate relation tuple is included with probability `0.24`.
- `k=50`: the benchmark requested the first 50 answers.
- `prep_s=0.034767`: time spent constructing the ranked enumerator and preprocessing its internal structures.
- `topk_s=0.068668`: time to extract the first 50 ranked answers after preprocessing.
- `returned=50`: the enumerator actually returned 50 answers.

How to read it:
- Total time to get the first 50 outputs is approximately `prep_s + topk_s`.
- If `k` is much smaller than the full result size, this is often the most relevant ranked-enumeration metric.
- If `returned < k`, then the query result contained fewer than `k` answers.

### `bench_delay.py` output
Example:

```text
preprocessing_s=0.002814
count=100
first_result_latency_s=0.000119
mean_delay_s=0.000043
p95_delay_s=0.000081
throughput_rows_per_s=23254.67
```

Field meanings:
- `preprocessing_s`: same preprocessing concept as above.
- `count`: number of outputs actually consumed for the measurement, up to `--limit`.
- `first_result_latency_s`: time from the first `next()` call until the first answer is produced.
- `mean_delay_s`: average time gap between consecutive outputs over the measured prefix.
- `p95_delay_s`: 95th percentile delay; useful because averages can hide occasional expensive successor generation steps.
- `throughput_rows_per_s`: outputs per second over the measured prefix.

How to read it:
- `first_result_latency_s` matters when the user wants an answer quickly.
- `mean_delay_s` and `p95_delay_s` matter when the system is used as a streaming ranked iterator.
- Lower is better for latency metrics; higher is better for throughput.

### `bench_compare_materialize_sort.py` output
Example:

```text
preprocessing_s=0.001920
enumeration_s=12.263695 rows=1005859
baseline_materialize_sort_s=21.717178 rows=1005859
outputs_match=True
```

Field meanings:
- `preprocessing_s`: ranked enumerator construction time.
- `enumeration_s`: time to enumerate the entire ranked output after preprocessing.
- `rows=1005859`: total number of answers produced.
- `baseline_materialize_sort_s`: time for the brute-force baseline that materializes all answers and sorts them globally.
- `outputs_match=True`: correctness check that both methods produced exactly the same ordered result.

How to read it:
- Full ranked time is approximately `preprocessing_s + enumeration_s`.
- Compare that against `baseline_materialize_sort_s`.
- If `outputs_match=False`, the benchmark result should not be trusted; fix correctness first.
- Even if ranked enumeration is not faster end-to-end, it may still be much better for first-answer and `top_k` access.

### `bench_sweep_summary.py` output
This script prints two kinds of lines.

Per-run line:

```text
run shape=path size=4 seed=0 rows=387 prep_s=0.001146 topk_s=0.000593 first_s=0.000029 mean_delay_s=0.000007 p95_delay_s=0.000011 throughput_rows_per_s=145231.42 enum_s=0.003876 baseline_s=0.006944 outputs_match=True k=50 limit=200
```

Summary line:

```text
summary shape=path size=4 runs=3 rows_mean=401.33 rows_std=18.77 prep_mean_s=0.001098 prep_std_s=0.000041 topk_mean_s=0.000611 topk_std_s=0.000052 first_mean_s=0.000031 first_std_s=0.000004 mean_delay_s=0.000007 p95_delay_s=0.000011 throughput_mean_rows_per_s=142508.13 enum_mean_s=0.003945 enum_std_s=0.000210 baseline_mean_s=0.007311 baseline_std_s=0.000344 baseline_over_ranked_full=1.85 baseline_over_ranked_topk=11.96 outputs_match_all=True k=50 limit=200
```

Field meanings:
- `run`: one exact `(shape, size, seed)` configuration.
- `summary`: aggregate over all requested seeds for a given `(shape, size)`.
- `rows_mean`, `rows_std`: mean and standard deviation of output cardinality.
- `prep_mean_s`, `topk_mean_s`, `enum_mean_s`, `baseline_mean_s`: average timings across seeds.
- `baseline_over_ranked_full`: ratio `baseline_mean_s / enum_mean_s`.
- `baseline_over_ranked_topk`: ratio `baseline_mean_s / topk_mean_s`.
- `outputs_match_all=True`: all per-seed correctness checks succeeded.

How to read it:
- This is the best script for your presentation because it reduces seed noise.
- `baseline_over_ranked_topk` is often the most persuasive number when ranked access is the main story.
- `baseline_over_ranked_full` is the end-to-end full-output comparison.
- Check `rows_std` before drawing strong conclusions; very high variance means the seeds are producing substantially different instances.

## Recommended Workflow

### 1. Start with correctness
Run `bench_compare_materialize_sort.py` on a few representative cases and confirm `outputs_match=True`.

### 2. Show early-output benefits
Run `bench_topk.py` and `bench_delay.py` on cases where the total output is much larger than `k`.

This shows that:

- ranked enumeration can be useful even when full-output time is not dramatically better,
- because the user often does not need all answers.

### 3. Summarize across seeds
Use `bench_sweep_summary.py`.

For example:

```bash
.venv/bin/python bench/bench_sweep_summary.py \
  --shapes path star binary_tree caterpillar \
  --sizes 4 6 \
  --domain 20 \
  --p 0.08 \
  --k 50 \
  --limit 200 \
  --seeds 0 1 2 3 4
```

### 4. Compare small-`k` and full-output stories separately
Do not collapse them into one number.

Instead report:
- preprocessing cost,
- first-result latency,
- `top_k` time,
- full enumeration time,
- brute-force materialize-and-sort time.

Those measure different use cases.

## Choosing Input Values

### Good starting values
These are practical settings that usually run quickly:

```text
path:         --size 8  --domain 20 --p 0.08
star:         --size 6  --domain 20 --p 0.08
binary_tree:  --size 3  --domain 10 --p 0.12
caterpillar:  --size 5  --domain 10 --p 0.12
```

### If results are too small
Increase one of:
- `--p`
- `--domain`
- `--size`

Usually increasing `p` is the fastest way to get more outputs.

### If results become too large
Decrease one of:
- `--p`
- `--domain`
- `--size`

Reducing `p` is usually the safest first adjustment because join output size can grow very quickly.

### Practical caution
Large `domain` combined with large `p` can make relation generation and baseline sorting very expensive. The brute-force baseline is the first component likely to become impractical.

## Reporting Template
For each reported workload, record:

- `shape`
- `size`
- `domain`
- `p`
- seed set used
- mean and standard deviation of output size
- mean preprocessing time
- mean first-result latency
- mean `top_k` time
- mean full ranked enumeration time
- mean brute-force materialize-and-sort time
- correctness status (`outputs_match`)
