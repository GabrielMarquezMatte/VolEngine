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
| Code commit | `8b12ef4` (the commit that adds the suite; the results were recorded in the commit after it) |
| Repetitions | 5 for every benchmark, including `BM_TccGrid`; values are the median over repetitions, with the standard deviation |

Reproduce:

```
cmake --preset gcc-bench
cmake --build --preset gcc-bench
build/gcc-bench/volengine_benchmarks --benchmark_out=build/gcc-bench/bench.json --benchmark_out_format=json --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

The run was not isolated: a normal desktop session, no pinned cores and no fixed frequency. Sub-percent differences
are noise.

## Cost of the log-likelihood and of its gradient by AD

`BM_Loglik/<spec>/double` is one evaluation of `loglik<double>`, `BM_Loglik/<spec>/dual` one of `loglik<Dual<N>>`
(N = 8 up to 8 parameters, 16 up to 16), both for T = 5000 observations. The `dual` evaluation returns the value and the
full gradient. The ratio is between two measurements from this run.

| Spec | Parameters | `double` (us) | sd | `Dual<N>` (us) | sd | Dual / double |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s11_norm` | 4 | 51.4 | 0.2 | 177.8 | 5.7 | 3.5 |
| `s21_std` | 6 | 98.8 | 0.1 | 232.6 | 3.7 | 2.4 |
| `arma11_s11_ged` | 7 | 171.6 | 0.3 | 448.0 | 2.2 | 2.6 |
| `arma22_s33_std` | 13 | 150.6 | 0.3 | 962.4 | 49.4 | 6.4 |

## Single fits

`BM_FitOne/<spec>`: `ve::fit_one` on one series of T = 5000, default `FitConfig`, without the C layer. Every series is
simulated from a model with the structure of the spec (see below). All fits ended `CONVERGED`.

| Spec | Median (ms) | sd (ms) | Fits per second |
| --- | ---: | ---: | ---: |
| `s11_norm` | 5.49 | 0.08 | 183.8 |
| `s21_std` | 13.10 | 0.10 | 76.3 |
| `arma11_s11_ged` | 21.21 | 0.16 | 46.6 |
| `arma22_s33_std` | 286.3 | 5.3 | 3.5 |

## Batch scaling

`BM_Batch/threads:N`: one `ve_fit_batch` call with 20 series of T = 5000 and one spec (`s11_norm`), through the C API,
on a context with N threads (`hw` is `ve_context_set_threads(ctx, 0)`, that is 16 here). Time is wall-clock time. All
20 fits ended `CONVERGED` in every repetition. The speedup column is the single-thread time over the row's time, both
from this run.

| Threads | Median (ms) | sd (ms) | Fits per second | Speedup over 1 thread |
| --- | ---: | ---: | ---: | ---: |
| 1 | 138.1 | 11.4 | 144.8 | 1.0 |
| 2 | 59.2 | 1.2 | 337.9 | 2.3 |
| 4 | 31.3 | 0.3 | 638.7 | 4.4 |
| 8 | 21.9 | 0.5 | 912.3 | 6.3 |
| hw (16) | 20.2 | 0.5 | 988.3 | 6.8 |

Only 20 jobs are available, so the tail of the batch limits the gain beyond 8 threads.

## TCC workload

`BM_TccGrid/threads:N`: 20 series of T = 5000 and the grid {p, q in 1..3} x {ARMA(0,0), ARMA(1,1)} x {norm, std, ged}
= 54 specs, 1080 fits in a single `ve_fit_batch` call. This is the reference for the spec goal of under 10 s on an
8-core CPU, which belongs to plan 5; here it is only measured.

| Threads | Median (s) | sd (s) | Fits per second | `CONVERGED` | `MAX_ITER` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 78.8 | 0.33 | 13.7 | 1073 | 7 |
| hw (16) | 12.7 | 0.05 | 85.3 | 1073 | 7 |

On this machine the 1080-fit grid takes 12.7 s with all 16 hardware threads, which is above the 10 s goal.

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
