from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ranked_enumeration.generators import instantiate_relations, make_benchmark_query
from ranked_enumeration.iterator import RankedEnumerator
from ranked_enumeration.metrics import measure_iterator_delays
from ranked_enumeration.ranking import AdditiveRankModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        choices=["path", "star", "binary_tree", "caterpillar"],
        default="path",
    )
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--domain", type=int, default=20)
    parser.add_argument("--p", type=float, default=0.08)
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cq, td = make_benchmark_query(args.shape, args.size)
    relations = instantiate_relations(cq, domain_size=args.domain, tuple_probability=args.p, seed=args.seed)
    rank = AdditiveRankModel(lambda _n, a: float(sum(a.values())))

    t0 = time.perf_counter()
    enum = RankedEnumerator(cq, relations, td, rank)
    prep_s = time.perf_counter() - t0

    metrics = measure_iterator_delays(iter(enum), limit=args.limit)

    print(f"preprocessing_s={prep_s:.6f}")
    print(f"count={metrics.count}")
    print(f"first_result_latency_s={metrics.first_result_latency_s:.6f}")
    print(f"mean_delay_s={metrics.mean_delay_s:.6f}")
    print(f"p95_delay_s={metrics.p95_delay_s:.6f}")
    print(f"throughput_rows_per_s={metrics.throughput_rows_per_s:.2f}")


if __name__ == "__main__":
    main()
