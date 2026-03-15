from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranked_enumeration.baseline import baseline_ranked
from ranked_enumeration.generators import instantiate_relations, make_benchmark_query
from ranked_enumeration.iterator import RankedEnumerator
from ranked_enumeration.metrics import measure_iterator_delays
from ranked_enumeration.ranking import AdditiveRankModel


@dataclass
class RunResult:
    shape: str
    size: int
    seed: int
    rows: int
    prep_s: float
    topk_s: float
    first_result_latency_s: float
    mean_delay_s: float
    p95_delay_s: float
    throughput_rows_per_s: float
    enumeration_s: float
    baseline_materialize_sort_s: float
    outputs_match: bool


def _safe_mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("inf")


def run_case(
    shape: str,
    size: int,
    domain: int,
    p: float,
    k: int,
    limit: int,
    seed: int,
) -> RunResult:
    cq, td = make_benchmark_query(shape, size)
    relations = instantiate_relations(cq, domain_size=domain, tuple_probability=p, seed=seed)
    rank = AdditiveRankModel(lambda _n, a: float(sum(a.values())))

    t0 = time.perf_counter()
    enum_for_topk = RankedEnumerator(cq, relations, td, rank)
    prep_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    topk_out = enum_for_topk.top_k(k)
    topk_s = time.perf_counter() - t1

    enum_for_delay = RankedEnumerator(cq, relations, td, rank)
    delay_metrics = measure_iterator_delays(iter(enum_for_delay), limit=limit)

    enum_for_full = RankedEnumerator(cq, relations, td, rank)
    t2 = time.perf_counter()
    enum_out = list(enum_for_full)
    enumeration_s = time.perf_counter() - t2

    t3 = time.perf_counter()
    baseline_out = baseline_ranked(cq, relations, td, rank)
    baseline_materialize_sort_s = time.perf_counter() - t3

    return RunResult(
        shape=shape,
        size=size,
        seed=seed,
        rows=len(enum_out),
        prep_s=prep_s,
        topk_s=topk_s,
        first_result_latency_s=delay_metrics.first_result_latency_s,
        mean_delay_s=delay_metrics.mean_delay_s,
        p95_delay_s=delay_metrics.p95_delay_s,
        throughput_rows_per_s=delay_metrics.throughput_rows_per_s,
        enumeration_s=enumeration_s,
        baseline_materialize_sort_s=baseline_materialize_sort_s,
        outputs_match=(enum_out == baseline_out and len(topk_out) <= len(enum_out)),
    )


def print_run(result: RunResult, k: int, limit: int) -> None:
    print(
        "run "
        f"shape={result.shape} size={result.size} seed={result.seed} "
        f"rows={result.rows} prep_s={result.prep_s:.6f} topk_s={result.topk_s:.6f} "
        f"first_s={result.first_result_latency_s:.6f} mean_delay_s={result.mean_delay_s:.6f} "
        f"p95_delay_s={result.p95_delay_s:.6f} throughput_rows_per_s={result.throughput_rows_per_s:.2f} "
        f"enum_s={result.enumeration_s:.6f} baseline_s={result.baseline_materialize_sort_s:.6f} "
        f"outputs_match={result.outputs_match} k={k} limit={limit}"
    )


def print_summary(shape: str, size: int, results: list[RunResult], k: int, limit: int) -> None:
    rows = [float(r.rows) for r in results]
    prep = [r.prep_s for r in results]
    topk = [r.topk_s for r in results]
    first = [r.first_result_latency_s for r in results]
    mean_delay = [r.mean_delay_s for r in results]
    p95 = [r.p95_delay_s for r in results]
    throughput = [r.throughput_rows_per_s for r in results]
    enum = [r.enumeration_s for r in results]
    baseline = [r.baseline_materialize_sort_s for r in results]
    outputs_match_all = all(r.outputs_match for r in results)

    prep_mean = _safe_mean(prep)
    topk_mean = _safe_mean(topk)
    enum_mean = _safe_mean(enum)
    baseline_mean = _safe_mean(baseline)

    print(
        "summary "
        f"shape={shape} size={size} runs={len(results)} "
        f"rows_mean={_safe_mean(rows):.2f} rows_std={_safe_std(rows):.2f} "
        f"prep_mean_s={prep_mean:.6f} prep_std_s={_safe_std(prep):.6f} "
        f"topk_mean_s={topk_mean:.6f} topk_std_s={_safe_std(topk):.6f} "
        f"first_mean_s={_safe_mean(first):.6f} first_std_s={_safe_std(first):.6f} "
        f"mean_delay_s={_safe_mean(mean_delay):.6f} "
        f"p95_delay_s={_safe_mean(p95):.6f} "
        f"throughput_mean_rows_per_s={_safe_mean(throughput):.2f} "
        f"enum_mean_s={enum_mean:.6f} enum_std_s={_safe_std(enum):.6f} "
        f"baseline_mean_s={baseline_mean:.6f} baseline_std_s={_safe_std(baseline):.6f} "
        f"baseline_over_ranked_full={_safe_ratio(baseline_mean, enum_mean):.2f} "
        f"baseline_over_ranked_topk={_safe_ratio(baseline_mean, topk_mean):.2f} "
        f"outputs_match_all={outputs_match_all} k={k} limit={limit}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=["path", "star", "binary_tree", "caterpillar"],
        default=["path", "star", "binary_tree", "caterpillar"],
    )
    parser.add_argument("--sizes", nargs="+", type=int, default=[4])
    parser.add_argument("--domain", type=int, default=20)
    parser.add_argument("--p", type=float, default=0.08)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    for shape in args.shapes:
        for size in args.sizes:
            results: list[RunResult] = []
            for seed in args.seeds:
                result = run_case(shape, size, args.domain, args.p, args.k, args.limit, seed)
                print_run(result, args.k, args.limit)
                results.append(result)
            print_summary(shape, size, results, args.k, args.limit)


if __name__ == "__main__":
    main()
