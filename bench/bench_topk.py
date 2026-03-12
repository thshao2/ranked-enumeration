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
from ranked_enumeration.ranking import AdditiveRankModel


def run_case(shape: str, size: int, domain: int, p: float, k: int, seed: int) -> None:
    cq, td = make_benchmark_query(shape, size)
    relations = instantiate_relations(cq, domain_size=domain, tuple_probability=p, seed=seed)
    rank = AdditiveRankModel(lambda _n, a: float(sum(a.values())))

    t0 = time.perf_counter()
    enum = RankedEnumerator(cq, relations, td, rank)
    prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    topk = enum.top_k(k)
    enum_time = time.perf_counter() - t1

    print(
        f"shape={shape} size={size} domain={domain} p={p:.2f} k={k} "
        f"prep_s={prep:.6f} topk_s={enum_time:.6f} returned={len(topk)}"
    )


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
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run_case(args.shape, args.size, args.domain, args.p, args.k, args.seed)


if __name__ == "__main__":
    main()
