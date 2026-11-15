"""Time the native library the installed volengine uses, so two libraries can be compared on one machine.

The library comes from VOLENGINE_LIB or from the wheel's _lib directory. Run it once per library, alternating:

    VOLENGINE_LIB=build/gcc-bench/libvolengine.dll python python/benchmarks/bench_library.py --out a.json
    python python/benchmarks/bench_library.py --out b.json      # the wheel, in a venv without VOLENGINE_LIB

Usage: python python/benchmarks/bench_library.py [--repetitions N] [--out PATH]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import volengine as ve
from numpy.typing import NDArray

SERIES_COUNT = 21
LENGTH = 5000
SEED = 7
MODELS = ("sgarch", "gjrgarch", "egarch")
DISTS = ("norm", "std", "ged", "sstd")
DCC_DISTS = ("mvnorm", "mvt")


def simulate() -> NDArray[np.float64]:
    """GARCH(1,1) with Student-t(8) innovations, in percent units, one column per series."""
    shocks = np.random.default_rng(SEED).standard_t(8, size=(LENGTH, SERIES_COUNT)) * np.sqrt(6 / 8)
    returns = np.empty_like(shocks)
    variance = np.full(SERIES_COUNT, 0.05 / (1 - 0.08 - 0.90))
    for t in range(LENGTH):
        returns[t] = np.sqrt(variance) * shocks[t]
        variance = 0.05 + 0.08 * returns[t] ** 2 + 0.90 * variance
    return returns


def workload(ctx: ve.Context, data: NDArray[np.float64]) -> None:
    """The 12 univariate fits of every series, then a DCC(1,1) over the sGARCH-std fits for each DCC distribution."""
    specs = [ve.Spec(model, p=1, q=1, dist=dist) for model in MODELS for dist in DISTS]
    fits = ctx.fit(specs, data, options=ve.FitOptions(standard_errors=False))
    univariate = list(fits[specs.index(ve.Spec("sgarch", p=1, q=1, dist="std"))])
    for dist in DCC_DISTS:
        ctx.fit_dcc(ve.DccSpec("dcc", dist=dist), univariate, standard_errors=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    data = simulate()
    ctx = ve.Context(threads=0)
    workload(ctx, data)  # warm-up: thread pool, page faults and the SIMD dispatch table
    seconds = []
    for _ in range(args.repetitions):
        start = time.perf_counter()
        workload(ctx, data)
        seconds.append(time.perf_counter() - start)
    result = {
        "median_s": statistics.median(seconds),
        "stdev_s": statistics.pstdev(seconds),
        "library": ve._native.lib._name,
    }
    print(f"median {result['median_s']:.3f} s, stdev {result['stdev_s']:.3f} s over {args.repetitions} runs")
    print(f"library {result['library']}")
    if args.out:
        args.out.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
