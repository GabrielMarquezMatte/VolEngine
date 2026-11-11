# Benchmarks

Results of the C++ Google Benchmark suite (`tests/benchmarks/fit_benchmarks.cpp`). Every number below comes from
the one machine listed here. Compare only runs from the same machine: a number from another machine says nothing about
this library. No other library is measured, so no comparison with one is made.

## Machine and run

| Item | Value |
| --- | --- |
| CPU | AMD Ryzen 7 5700X, 8 cores, 16 hardware threads, 3.4 GHz base (`Get-CimInstance Win32_Processor`) |
| RAM | 32 GiB (31.9 GiB visible to Windows) |
| OS | Windows 10 Pro 10.0.19045 (64-bit) |
| Compiler | g++ 16.2.0 (MSYS2 UCRT64, Rev4), `-O3 -DNDEBUG -ffp-contract=off -std=c++23` (preset `gcc-bench`, Release) |
| CMake | 4.4.2, Ninja |
| Google Benchmark | v1.9.1 |
| Date | 2026-11-11 |
| Code commit | `af6deb8` (the commit benchmarked; the results were recorded in the commit after it) |
| Repetitions | 5 for every benchmark, including `BM_TccGrid`; values are the median over repetitions, with the standard deviation |

Reproduce:

```
cmake --preset gcc-bench
cmake --build --preset gcc-bench
build/gcc-bench/volengine_benchmarks --benchmark_out=build/gcc-bench/bench.json --benchmark_out_format=json --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

The run was not isolated: a normal desktop session, no pinned cores and no fixed frequency. The standard
deviations in the tables are the measured spread over the 5 repetitions: under 5% of the median for nearly every
benchmark, 12% for the `arma11_s11_ged` `Dual` evaluation. Differences smaller than that spread are not results. Runs on
different days on this machine moved some medians by more than the spread (for example `s21_std` `double`), so
re-measure before and after a change instead of comparing against this table.

## Cost of the log-likelihood and of its gradient by AD

`BM_Loglik/<spec>/double` is one evaluation of `loglik<double>`, `BM_Loglik/<spec>/dual` one of `loglik<Dual<N>>`
(N = 8 up to 8 parameters, 16 up to 16), both for T = 5000 observations. The `dual` evaluation returns the value and the
full gradient. The ratio is between two measurements from this run.

| Spec | Parameters | `double` (us) | sd | `Dual<N>` (us) | sd | Dual / double |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s11_norm` | 4 | 51.0 | 1.6 | 177.2 | 2.0 | 3.5 |
| `s21_std` | 6 | 119.3 | 1.9 | 264.1 | 7.1 | 2.2 |
| `arma11_s11_ged` | 7 | 175.7 | 4.7 | 459.8 | 55.2 | 2.6 |
| `arma22_s33_std` | 13 | 172.5 | 0.6 | 1013.0 | 8.2 | 5.9 |

## Single fits

`BM_FitOne/<spec>`: `ve::fit_one` on one series of T = 5000, default `FitConfig`, without the C layer. Every series is
simulated from a model with the structure of the spec (see below). All fits ended `CONVERGED` (the `converged` counter of `bench.json` is 1 and `max_iter` is 0 for the four specs).

| Spec | Median (ms) | sd (ms) | Fits per second |
| --- | ---: | ---: | ---: |
| `s11_norm` | 5.88 | 0.09 | 170.7 |
| `s21_std` | 15.19 | 0.04 | 65.3 |
| `arma11_s11_ged` | 22.03 | 0.04 | 45.5 |
| `arma22_s33_std` | 353.73 | 13.80 | 2.8 |

## Batch scaling

`BM_Batch/threads:N`: one `ve_fit_batch` call with 20 series of T = 5000 and one spec (`s11_norm`), through the C API,
on a context with N threads (`hw` is `ve_context_set_threads(ctx, 0)`, that is 16 here). Time is wall-clock time. All
20 fits ended `CONVERGED` in every repetition. The speedup column is the single-thread time over the row's time, both
from this run.

| Threads | Median (ms) | sd (ms) | Fits per second | Speedup over 1 thread |
| --- | ---: | ---: | ---: | ---: |
| 1 | 109.8 | 0.8 | 182.2 | 1.0 |
| 2 | 58.8 | 1.0 | 339.9 | 1.9 |
| 4 | 32.4 | 0.9 | 616.8 | 3.4 |
| 8 | 21.9 | 1.0 | 911.8 | 5.0 |
| hw (16) | 20.1 | 0.8 | 993.8 | 5.5 |

Only 20 jobs are available, so the tail of the batch limits the gain beyond 8 threads.

## TCC workload

`BM_TccGrid/threads:N`: 20 series of T = 5000 and the grid {p, q in 1..3} x {ARMA(0,0), ARMA(1,1)} x {norm, std, ged}
= 54 specs, 1080 fits in a single `ve_fit_batch` call. This is the reference for the spec goal of under 10 s on an
8-core CPU, which belongs to plan 5; here it is only measured.

| Threads | Median (s) | sd (s) | Fits per second | `CONVERGED` | `MAX_ITER` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 80.6 | 0.61 | 13.4 | 1072 | 8 |
| hw (16) | 12.5 | 0.08 | 86.3 | 1072 | 8 |

On this machine the 1080-fit grid takes 12.5 s with all 16 hardware threads, which is above the 10 s goal.

## What the benchmarks fit

- The series are simulated by `tests/benchmarks/simulated_panel.hpp`, which reuses the portable generators of the unit
  tests (Mersenne Twister with Box-Muller, no `std::normal_distribution`), so every standard library produces the
  same data.
- `BM_Loglik` and `BM_FitOne` use, for each spec, a model with that structure: persistence 0.98 split evenly over the
  ARCH and GARCH lags, ARMA terms of moderate size, and the spec's distribution (std with nu = 8, ged with nu = 1.5).
- `BM_Batch` uses sGARCH(1,1) with normal innovations. The TCC grid uses sGARCH(1,1) with Student-t (nu = 8)
  innovations, like returns, for every spec of the grid. When these numbers were measured, about half of the `std`
  fits of the grid ended `FAILED` with normal innovations (nu runs to infinity), which would have stopped the benchmark
  at its check that every fit ends `CONVERGED` or `MAX_ITER`. That was a precision defect in the Student-t density at
  large nu, since fixed (see [Validation](../guide/validation.md#volengine-student-t-near-the-normal-limit)); the grid
  keeps Student-t innovations so that the numbers stay comparable.
- `s21` means sGARCH(2, 1): ARCH order 2, GARCH order 1.
- The two evaluations of `BM_Loglik` are separate benchmarks (`/double` and `/dual`) so each has its own time.
- Google Benchmark appends `/real_time` and `/iterations:1` to the names of the benchmarks that use wall-clock time;
  `BM_TccGrid` runs one iteration per repetition because one iteration is already seconds long.
- Each benchmark skips with an error if a fit ends neither `CONVERGED` nor `MAX_ITER`, or if the grid yields no jobs.
- "Fits per second" is the median over the repetitions of the benchmark's rate counter (fits divided by the elapsed
  time of the run), not 1 over the median time.
- The `converged`, `max_iter` and `other` counters are the status counts of the last iteration; every iteration
  repeats the same fits. A non-zero `other` skips the benchmark with an error.
- With the Student-t fix in place a normal-innovation run of the grid no longer ends `FAILED`; the grid still uses
  Student-t innovations so that the numbers stay comparable with the earlier ones.
