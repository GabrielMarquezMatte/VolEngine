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
| Date | 2026-11-13 |
| Code commit | `d4b3fae` (the commit benchmarked; the results were recorded in a later commit) |
| Repetitions | 5 for every benchmark in the tables; values are the median over repetitions, with the standard deviation |

Reproduce:

```
cmake --preset gcc-bench
cmake --build --preset gcc-bench
build/gcc-bench/volengine_benchmarks --benchmark_out=build/gcc-bench/bench.json --benchmark_out_format=json --benchmark_repetitions=5 --benchmark_report_aggregates_only=true
```

The run was not isolated: a normal desktop session, no pinned cores and no fixed frequency. The standard
deviations in the tables are the measured spread over the 5 repetitions:
- under 5% of the median for nearly every benchmark;
- 14% for the `Dual` evaluations of `arma22_s33_std` and `arx_sgarch11_norm`;
- 7% for the `double` evaluation of `s21_std`.

Differences smaller than that spread are not results. Runs on different days on this machine moved some medians by
more than the spread. Re-measure before and after a change instead of comparing against these tables, or against the
plan 2a numbers (`af6deb8`, 2026-11-11) that this page showed before.

## Not measured

The run was stopped before its last cases. Not measured:
- `BM_TccGridAllModels/threads:hw`: the TCC grid over the 14 models, 15120 fits.
- `BM_TccGridFull/threads:hw`: 14 models × orders × ARMA × the 8 distributions, 40320 fits.
- `BM_LogBesselK/{fixed_order,free_order}`.

With 5 repetitions the two grids take hours on this machine, so the author decided to leave them out. This page
publishes no number for them.

Also not done:
- The profiles that plans 2b and 2c asked for: `BM_TccGridAllModels`, `BM_TccGridFull`, and `BM_FitOne` for
  `fgarch_allgarch11_norm`, nig and ghyp.
- The assembly of the step-form recursions.
- Settling the before/after effect of plan 2e's step form on the existing cases. This run measures the code after
  the step form, and no A/B run was made.

Plan 2d added no benchmark. Fixed parameters, start values and variance targeting add branches only when the `Layout`
has them, and a spec without them takes the path of plan 2c.

## Cost of the log-likelihood and of its gradient by AD

`BM_Loglik/<series>/double` is one evaluation of `loglik<double>`, `BM_Loglik/<series>/dual` one of
`loglik<Dual<N>>`, with N = 8 up to 8 parameters and 16 up to 16. Both use T = 5000 observations, at the parameters of
the process that simulated the series. The `dual` evaluation returns the value and the full gradient. The ratio is
between two measurements from this run.

| Series | Parameters | `double` (us) | sd | `Dual<N>` (us) | sd | Dual / double |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `s11_norm` | 4 | 49.8 | 0.4 | 151.5 | 1.2 | 3.0 |
| `s21_std` | 6 | 105.3 | 7.8 | 288.9 | 7.7 | 2.7 |
| `arma11_s11_ged` | 7 | 174.4 | 3.7 | 436.5 | 15.4 | 2.5 |
| `arma22_s33_std` | 13 | 156.3 | 2.6 | 755.6 | 107.5 | 4.8 |
| `egarch11_std` | 6 | 182.2 | 4.4 | 391.5 | 29.1 | 2.1 |
| `gjrgarch11_norm` | 5 | 49.4 | 0.1 | 169.6 | 0.5 | 3.4 |
| `aparch11_ged` | 7 | 416.0 | 8.7 | 1025.9 | 50.5 | 2.5 |
| `igarch11_norm` | 4 | 49.7 | 0.1 | 151.6 | 1.0 | 3.0 |
| `csgarch11_std` | 7 | 114.9 | 1.5 | 384.7 | 1.9 | 3.3 |
| `fgarch_tgarch11_norm` | 5 | 88.2 | 0.8 | 356.1 | 1.5 | 4.0 |
| `fgarch_allgarch11_norm` | 7 | 628.2 | 2.2 | 1207.2 | 5.2 | 1.9 |
| `gjrgarch11_sstd` | 7 | 160.3 | 0.7 | 359.1 | 0.7 | 2.2 |
| `aparch11_sged` | 8 | 452.4 | 2.2 | 1104.7 | 10.3 | 2.4 |
| `egarch11_jsu` | 7 | 314.4 | 5.4 | 566.6 | 3.1 | 1.8 |
| `sgarch11_nig` | 6 | 585.5 | 1.5 | 2248.9 | 10.0 | 3.8 |
| `sgarch11_ghyp` | 7 | 591.2 | 4.5 | 22585.9 | 199.3 | 38.2 |
| `arx_sgarch11_norm` | 7 | 66.4 | 1.5 | 210.1 | 46.2 | 3.2 |
| `sgarch11_vx_std` | 6 | 104.6 | 2.0 | 280.8 | 1.1 | 2.7 |
| `archm_gjrgarch11_norm` | 6 | 124.7 | 0.8 | 607.8 | 6.8 | 4.9 |
| `archm2_egarch11_std` | 7 | 190.9 | 1.1 | 778.3 | 20.5 | 4.1 |
| `arfima_sgarch11_norm` | 5 | 2632.5 | 6.3 | 8006.8 | 25.2 | 3.0 |

- **ghyp:** the gradient costs 38 `double` evaluations. Its Bessel K takes the order lambda as a `Dual`, which the
  other distributions do not need (nig fixes the order at −1). Plan 2c named this case for a Bessel K of our own or an
  analytic derivative in lambda (plan 5).
- **`archm`:** these cases run the joint loop of the variance and the mean, which keeps the steps from being
  vectorised over t. Their `Dual` costs about 5 `double` evaluations, against 2 or 3 for the sequential cases.
- **ARFIMA:** almost all of the 2.6 ms of a `double` evaluation at T = 5000 is the O(T²) fractional difference in
  `src/mean/fractional.hpp`, marked `ponytail:` with the FFT as the way out. Its gradient costs 3 evaluations because
  only d and the residual lanes that move get a convolution of their own.

## Single fits

`BM_FitOne/<series>`: `ve::fit_one` on one series of T = 5000, default `FitConfig`, without the C layer. Every fit
ended `CONVERGED` (the `converged` counter of `bench.json` is 1 and `max_iter` is 0), except `arma22_s33_std`. That fit
ended `MAX_ITER` in this run, while at `af6deb8` it ended `CONVERGED`.

| Series | Median (ms) | sd (ms) | Fits per second |
| --- | ---: | ---: | ---: |
| `s11_norm` | 4.69 | 0.29 | 211.9 |
| `s21_std` | 16.4 | 0.50 | 61.0 |
| `arma11_s11_ged` | 21.8 | 1.29 | 46.3 |
| `arma22_s33_std` (`MAX_ITER`) | 377 | 6.5 | 2.7 |
| `egarch11_std` | 17.6 | 0.16 | 57.0 |
| `gjrgarch11_norm` | 5.76 | 0.05 | 174.8 |
| `aparch11_ged` | 63.0 | 0.28 | 16.0 |
| `igarch11_norm` | 3.55 | 0.15 | 283.6 |
| `csgarch11_std` | 30.8 | 0.07 | 32.7 |
| `fgarch_tgarch11_norm` | 13.0 | 0.06 | 78.0 |
| `fgarch_allgarch11_norm` | 83.9 | 0.09 | 12.1 |
| `gjrgarch11_sstd` | 22.2 | 0.14 | 44.5 |
| `aparch11_sged` | 87.4 | 0.58 | 11.5 |
| `egarch11_jsu` | 40.5 | 0.45 | 24.7 |
| `sgarch11_nig` | 103 | 2.95 | 9.7 |
| `sgarch11_ghyp` | 2248 | 21.3 | 0.44 |
| `arx_sgarch11_norm` | 9.53 | 0.22 | 104.3 |
| `sgarch11_vx_std` | 29.2 | 0.74 | 34.1 |
| `archm_gjrgarch11_norm` | 12.5 | 0.08 | 79.6 |
| `archm2_egarch11_std` | 36.5 | 0.95 | 27.6 |
| `arfima_sgarch11_norm` | 224 | 0.93 | 4.5 |

The ghyp case fits the nig series, since the C++ helper does not simulate ghyp.

## Forecast (plan 3a)

`BM_Forecast/<series>/h:60`: one `forecast::forecast` of 60 steps from a fit of the series of the same name, which is
made once, outside the timing. Measured on the same machine on 2026-11-13. The core was at `542657a`, and the benchmark
was registered in the commit after it. Each case ran 5 repetitions; the values are the medians, with the standard
deviation.

| Series | Median (us) | sd (us) |
| --- | ---: | ---: |
| `s11_norm` | 0.96 | 0.02 |
| `egarch11_std` | 1.13 | 0.01 |
| `csgarch11_std` | 3.09 | 0.02 |
| `fgarch_allgarch11_norm` | 14.3 | 0.14 |
| `arfima_sgarch11_norm` | 223.2 | 0.7 |

- **csGARCH:** the closed form of `rugarch` from h = 2 on costs two `pow` per step.
- **fGARCH:** ALLGARCH computes its shock moment κ, a quadrature, once per forecast.
- **ARFIMA:** the forecast sums over the whole history at every step, which is O(n·H) with n = 5000. It is marked
  `ponytail:` in `src/forecast/forecast.cpp`.

A forecast costs well under a fit in every case. For example, the s11_norm forecast takes about 1 µs against 4.7 ms
for its fit.

## Simulation (plan 3b)

`BM_Simulate/<series>/paths:10000/h:60/threads:<N>`: one `sim::simulate` of 10 000 paths × 60 steps from a fit of the
series of the same name, made once outside the timing, with 1 thread and with the 16 threads of the machine (`hw`);
wall time. `BM_Sampler/<dist>/draws:100000`: 10⁵ draws of one distribution, at the parameters of
`Samplers.StandardizedMoments`. Measured on the same machine on 2026-11-13 at `953567b` (the benchmarks were registered
in the commit after it), 5 repetitions; medians, with the standard deviation.

| Series | 1 thread (ms) | sd | 16 threads (ms) | sd |
| --- | ---: | ---: | ---: | ---: |
| `s11_norm` | 125 | 2.8 | 21.3 | 0.9 |
| `egarch11_std` | 286 | 6.4 | 47.7 | 1.5 |
| `sgarch11_ghyp` | 182 | 7.1 | 27.9 | 0.2 |

| Distribution | 10⁵ draws (ms) | sd |
| --- | ---: | ---: |
| norm | 2.92 | 0.03 |
| std | 7.84 | 0.04 |
| ged | 11.3 | 1.4 |
| sstd | 8.90 | 0.11 |
| sged | 12.0 | 0.15 |
| nig | 10.3 | 0.12 |
| ghyp | 10.8 | 0.03 |
| jsu | 5.08 | 0.02 |

- A step costs about 0.2 µs in sGARCH: the forecast of one step from the state of the path, plus one draw. eGARCH
  pays a log of its state per step. These medians predate `274735f`, which keeps that state at a bounded size: before
  it, eGARCH, apARCH and fGARCH cost O(H) per step (apARCH 33 µs a step at H = 4000); after it, the cost per step
  does not grow with the horizon (`Simulate.LongHorizonCostsLinearTime`: 10⁵ apARCH steps in well under a second).
- The 16 threads give about 6×; the paths are independent, so the remainder is the fixed cost of the fit's state
  and of the pool.
- The samplers cost 30–120 ns a draw; ghyp and nig spend it in the GIG. Box-Muller and the state of a step dominate
  over the RNG. `ponytail:` a ziggurat and a step without allocations are the upgrades if a simulation ever needs
  them (plan 5).

## Diagnostics (plan 3c)

Measured on the machine above on 2026-11-13, starting from `f671fae` plus the diagnostics implementation.
`BM_Diagnostics/<series>/<test>` fits 5000 observations once outside the timed loop. Ljung-Box uses lag 5,
ARCH-LM uses lag 5 and fitdf 2, sign bias computes all four tests, and GOF uses 50 groups. Each case ran
5 repetitions; all statistics and p-values were finite. These are CPU-time medians and standard deviations:

| Series | Diagnostic | Median (us) | sd (us) |
| --- | --- | ---: | ---: |
| `s11_norm` | Ljung-Box | 27.87 | 1.06 |
| `s11_norm` | ARCH-LM | 22.95 | 0.27 |
| `s11_norm` | Sign bias | 38.50 | 0.92 |
| `s11_norm` | Pearson GOF | 256.70 | 10.44 |
| `sgarch11_ghyp` | Pearson GOF | 29375.00 | 1767.77 |

The ghyp GOF integrates its density between sorted residuals; it is substantially more expensive than
the closed normal CDF. It does not repeat an integral from negative infinity for every observation. The
ghyp row was measured again after the review fix that integrates a gap much narrower than the density scale
(every gap between the residuals of a fit) with a five-point rule instead of 64 points: 351 562 µs before,
29 375 µs after, with the same Pearson counts on all 108 `rugarch` cases.

Reproduce with `/c/msys64/ucrt64/bin` first on PATH:

```
cmake --preset gcc-bench
cmake --build --preset gcc-bench
build/gcc-bench/volengine_benchmarks --benchmark_filter="BM_Diagnostics|^BM_FitOne/s11_norm$" --benchmark_repetitions=5 --benchmark_out=build/diagnostics_bench.json --benchmark_out_format=json
```

The original before-Nyblom `BM_FitOne/s11_norm` measurement had a CPU median of 5.30 ms (sd 0.15 ms).
After runs concurrent with the release tests and R oracle varied enough to obscure the cost, so a control
executable was built from the original `src/fit/fit.cpp` at `f671fae`, using the same current headers,
benchmark object and other core objects. The control and the new executable then ran consecutively,
5 repetitions each:

| `BM_FitOne/s11_norm` | CPU median (ms) | sd (ms) | Wall median (ms) | sd (ms) |
| --- | ---: | ---: | ---: | ---: |
| Original fit driver (control) | 5.72 | 0.20 | 5.87 | 0.22 |
| With Nyblom | 5.62 | 0.27 | 5.64 | 0.27 |

There is no slowdown distinguishable from this run's noise; the lower median is not evidence of a speedup.
An isolated-kernel check on 5000 x 4 synthetic scores took 61 us per `nyblom_statistics` call over 10000
iterations, below the spread of the full-fit measurements. It reuses the score matrix produced for the
standard errors and adds O(T k²) accumulation and a small Cholesky solve. The singular-score guard rejects
Cholesky pivots at roundoff scale relative to their original diagonal: exact duplicate score columns can
otherwise leave a tiny positive pivot and a finite, invalid joint statistic.

## Batch scaling

`BM_Batch/threads:N` is one `ve_fit_batch` call with 20 series of T = 5000 and one spec (`s11_norm`), through the C API.
It runs on a context with N threads; `hw` is `ve_context_set_threads(ctx, 0)`, which is 16 here. Time is wall-clock
time. All 20 fits ended `CONVERGED` in every repetition. The speedup column is the single-thread time over the row's
time, both from this run.

| Threads | Median (ms) | sd (ms) | Fits per second | Speedup over 1 thread |
| --- | ---: | ---: | ---: | ---: |
| 1 | 89.9 | 0.4 | 222.5 | 1.0 |
| 2 | 47.4 | 1.4 | 421.8 | 1.9 |
| 4 | 27.5 | 0.9 | 728.6 | 3.3 |
| 8 | 20.5 | 0.3 | 974.4 | 4.4 |
| hw (16) | 17.9 | 0.5 | 1117.4 | 5.0 |

Only 20 jobs are available, so the tail of the batch limits the gain beyond 8 threads.

## TCC workload

`BM_TccGrid/threads:N` runs 20 series of T = 5000 against the grid {p, q in 1..3} × {ARMA(0,0), ARMA(1,1)} ×
{norm, std, ged}. That is 54 specs and 1080 fits in a single `ve_fit_batch` call. It is the reference for the spec goal
of under 10 s on an 8-core CPU. The goal belongs to plan 5; here it is only measured.

| Threads | Median (s) | sd (s) | Fits per second | `CONVERGED` | `MAX_ITER` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 68.5 | 0.88 | 15.8 | 1073 | 7 |
| hw (16) | 10.7 | 0.22 | 101.2 | 1073 | 7 |

On this machine the 1080-fit grid takes 10.7 s with all 16 hardware threads, which is above the 10 s goal.

## What the benchmarks fit

- The series are simulated by `tests/benchmarks/simulated_panel.hpp` and `tests/unit/simulated_garch.hpp`. They reuse
  the portable generators of the unit tests (Mersenne Twister with Box-Muller, no `std::normal_distribution`), so every
  standard library produces the same data.
- The plan 2a cases (`s11_norm`, `s21_std`, `arma11_s11_ged`, `arma22_s33_std`) each use a model with the spec's
  structure: persistence 0.98 split evenly over the ARCH and GARCH lags, ARMA terms of moderate size, and the spec's
  distribution (std with nu = 8, ged with nu = 1.5). The later cases use the processes of the oracle suite at their
  own parameters.
- `BM_Batch` uses sGARCH(1,1) with normal innovations. The TCC grid uses sGARCH(1,1) with Student-t (nu = 8)
  innovations, like returns, for every spec of the grid.
  - In plan 2a, about half of the `std` fits of the grid ended `FAILED` with normal innovations, because nu ran to
    infinity. That would have stopped the benchmark at its check that every fit ends `CONVERGED` or `MAX_ITER`.
  - The cause was a precision defect in the Student-t density at large nu, since fixed (see
    [Validation](../guide/validation.md#volengine-student-t-near-the-normal-limit)).
  - The grid keeps Student-t innovations so that the numbers stay comparable.
- `s21` means sGARCH(2, 1): ARCH order 2, GARCH order 1.
- The two evaluations of `BM_Loglik` are separate benchmarks (`/double` and `/dual`) so each has its own time.
- Google Benchmark appends `/real_time` and `/iterations:1` to the names of the benchmarks that use wall-clock time.
  `BM_TccGrid` runs one iteration per repetition because one iteration is already seconds long.
- Each benchmark skips with an error if a fit ends neither `CONVERGED` nor `MAX_ITER`, or if the grid yields no jobs.
- "Fits per second" is the median over the repetitions of the benchmark's rate counter (fits divided by the elapsed
  time of the run), not 1 over the median time.
- The `converged`, `max_iter` and `other` counters are the status counts of the last iteration; every iteration
  repeats the same fits. A non-zero `other` skips the benchmark with an error.
