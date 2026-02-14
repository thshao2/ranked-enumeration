from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class DelayMetrics:
    count: int
    first_result_latency_s: float
    mean_delay_s: float
    p95_delay_s: float
    throughput_rows_per_s: float


def measure_iterator_delays(iterator: Any, limit: int | None = None) -> DelayMetrics:
    start = time.perf_counter()
    last = start
    delays: list[float] = []
    count = 0

    for item in iterator:
        _ = item
        now = time.perf_counter()
        delays.append(now - last)
        last = now
        count += 1
        if limit is not None and count >= limit:
            break

    if count == 0:
        return DelayMetrics(
            count=0,
            first_result_latency_s=0.0,
            mean_delay_s=0.0,
            p95_delay_s=0.0,
            throughput_rows_per_s=0.0,
        )

    first = delays[0]
    mean = statistics.mean(delays)
    p95 = statistics.quantiles(delays, n=100)[94] if len(delays) >= 20 else max(delays)
    total = last - start
    throughput = float(count / total) if total > 0 else float("inf")

    return DelayMetrics(
        count=count,
        first_result_latency_s=first,
        mean_delay_s=mean,
        p95_delay_s=p95,
        throughput_rows_per_s=throughput,
    )
