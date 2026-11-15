# Plan 5a profile

Where the time of the TCC grid goes, measured before any plan 5 optimization. Tools and commands:
[Measuring](measuring.md).

## Machine and run

The machine of [Benchmarks](benchmarks.md): AMD Ryzen 7 5700X (8 cores, 16 hardware threads), Windows 10, g++ 16.2.0
(MSYS2 UCRT64), preset `gcc-bench`. Code at `e7b6021` (the plan 5a measuring tools on top of `5e7650b`). Date
2026-11-14. Desktop session, not isolated.

## The grid

`volengine_grid_profile`: the 1080 fits of `BM_TccGrid` (20 series of T = 5000 from sGARCH(1,1)-std, ν = 8;
{p, q ∈ 1..3} × {ARMA(0,0), ARMA(1,1)} × {norm, std, ged}). One run each.

| Threads | Wall (s) | Sum of fit times (s) | `pool_use` | Converged | `MAX_ITER` |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 67.3 | 67.3 | 1.00 | 1073 | 7 |
| 16 | 10.5 | 142.4 | 0.85 | 1073 | 7 |

- With 16 threads a fit takes about 2.1 times its one-thread time (two threads per core share it), so the sum of fit
  times doubles. `pool_use` 0.85 means 15% of the 16 threads' wall time runs no fit: the tail of the batch.
- The 7 `MAX_ITER` fits (500 iterations each; p2q2 and p3q3 with ARMA(1,1), std and ged) cost 5.1 s of the 142.4 s
  fit time, 3.6%.

### By ARMA, order and distribution (16 threads; shares of the fit time)

| Group | Share | Mean iterations | Mean evaluations | Optimizer share of the fit |
| --- | ---: | ---: | ---: | ---: |
| ARMA(1,1) | 73.9% | 144 | 179 | 91% |
| ARMA(0,0) | 26.1% | 65 | 82 | 85% |
| p3q3 | 23.7% | 202 | 255 | 93% |
| p1q1 | 2.9% | 35 | 42 | 75% |
| std | 41.0% | 119 | 149 | 90% |
| ged | 39.0% | 108 | 132 | 89% |
| norm | 20.0% | 87 | 111 | 88% |

The costliest spec, sGARCH(3,3)-ARMA(1,1)-std, takes 323 iterations per fit on average; sGARCH(1,1) with ARMA(0,0)
takes about 20.

### By parameter count

| Parameters k | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Share of the fit time | 0.2% | 1.4% | 4.8% | 10.4% | 15.0% | 32.2% | 23.7% | 12.3% |

Fits with k = 9, 10 or 11 take 68% of the time. `fit::with_dual_width` runs them on `Dual<16>`, of which 5 to 7 of the
16 derivative lanes carry zeros.

### The work of an iteration

- 1.25 evaluations per iteration over the grid (140 933 evaluations, 113 157 iterations): the line search almost
  always accepts its first step.
- The optimizer is 89% of the fit time. The other 11% is the inference at the estimate: the numerical Hessian (2k
  gradients), the scores, the robust standard errors and Nyblom.
- Every evaluation computes the gradient by forward AD. `BM_Loglik` measures a `Dual` evaluation at 3.0 times a
  `double` one for k = 4 (`s11_norm`, `Dual<8>`) and 4.2 times for k = 13 (`arma22_s33_std`, `Dual<16>`).

## Hot functions

AMD uProf, time-based sampling of `volengine_grid_profile --series 2 --threads 1` (108 fits), built by the
`msvc-profile` preset. uProf does not symbolize the GCC build (see [Measuring](measuring.md)). The MSVC build takes
20.2 s on this slice against 7.7 s for GCC, so the shares below are indicative. The report lists the 10 hottest
functions, 75% of the 20.2 s:

| Function | CPU time (s) | Share |
| --- | ---: | ---: |
| `ad::chain<16>(Dual, Dual, double, double, double)` | 5.37 | 26.6% |
| `garch::sgarch_step<Dual<16>>` | 2.83 | 14.0% |
| `garch::sgarch_step<Dual<8>>` | 1.54 | 7.6% |
| `ad::operator/<16>` | 1.43 | 7.1% |
| `mean::arma_step_of<Dual<16>>` | 1.09 | 5.4% |
| `mean::arma_step_of<Dual<8>>` | 0.78 | 3.9% |
| `ad::chain<16>(Dual, double, double)` | 0.63 | 3.1% |
| `model::sum_of_terms<Dual<16>>` (two instances) | 1.07 | 5.3% |
| `model::mean_square<Dual<16>>` | 0.46 | 2.3% |

Every one of them is `Dual` arithmetic inside the log-likelihood: the forward-mode gradient is where the time goes,
and `Dual<16>` alone is over half of the sampled time.

## Candidates for Task 8

Ordered by the estimated gain on the grid. The first two change the gradient path, which the design leaves to the
author's decision with these numbers.

1. **(e) A `Dual<12>` width** (gradient path). k = 9..11 are 68% of the time on `Dual<16>`; 12 lanes cut their
   derivative work by a quarter. Estimate: 10–20% of the grid. Cost: one more instantiation of every likelihood
   (compile time and binary size).
2. **The gradient itself** (gradient path; the options of the design §5). A `Dual` evaluation costs 3–4 `double`
   ones and the optimizer pays it on every trial. A reverse-mode gradient (generic AD or a hand-written adjoint of
   the recursions) costs about 2–3 `double` evaluations whatever k, so it gains most at k ≥ 9, where the time is.
   This is the largest lever, and the largest change.
3. **(g) Fewer iterations** (optimizer, not gradient). ARMA(1,1) more than doubles the iterations (144 against 65),
   and sGARCH(3,3)-ARMA(1,1)-std takes 323. To look at first: the start values of the ARMA and GARCH lags and the
   scaling of the first step. Task 7 (ridge) may move these counts too.
4. **(c)/(d) Job order and block size.** `pool_use` 0.85: up to ~15% of the wall time with 16 threads. Contiguous
   blocks of 8 per thread put the costly p3q3-ARMA(1,1) jobs of one spec next to each other.
5. **The standard-errors option (Task 5).** Up to 11% of the fit time when the grid skips the inference.
6. **(a) Interpolation in the zoom.** At most a part of the 0.25 extra evaluations per iteration; small.

Dropped:
- **(b) f in `double` before the gradient.** 80% of the trials are accepted and would then pay a `double` evaluation
  on top of the `Dual` one (+0.33 of an evaluation each), more than the 20% rejected trials would save (−0.67 each):
  a net loss of about 13% of the optimizer time.
- **(f) Allocations.** The optimizer's objective already reuses its buffers; no allocation shows in the profile.

## Changes measured (A/B)

Each row ran right after its baseline, in the same session (see [Measuring](measuring.md#ab)).

### Standard-errors option (`ve_fit_options_set_standard_errors`)

The grid with the option on (the default, as before) and off, `volengine_grid_profile`, one run each:

| Threads | On (s) | Off (s) | Off / on |
| --- | ---: | ---: | ---: |
| 1 | 67.7 | 60.4 | 0.89 |
| 16 | 10.3 | 9.4 | 0.91 |

The estimates, iterations and log-likelihoods are the same (`compare-profile`: evaluation ratio 1.0, worst
log-likelihood change 0.0, same status counts); off, the fit evaluates the log-likelihood at the estimate once in
`double` and skips the 2k gradients of the Hessian, the scores and Nyblom.

### L-BFGS flat-step threshold (`ftol` 1e-12 to 1e-15)

A step that decreased f by less than 1e-12·max(1, |f|) used to end a fit `CONVERGED` with a gradient up to √tol
(Task 7, [Validation](../guide/validation.md#volengine-gradient-tolerance-tol--1e-7)). At 1e-15 a fit goes on along an
ill-conditioned ridge until the gradient is below `tol`.

| Measure | `ftol` 1e-12 | `ftol` 1e-15 |
| --- | ---: | ---: |
| Grid, 1 thread (s) | 67.7 | 67.8 |
| Grid evaluations | 1.000 | 1.010 |
| Grid `CONVERGED` / `MAX_ITER` | 1073 / 7 | 1071 / 9 |
| `BM_DccGrid/m:21/T:5000/threads:1` (s, median of 3) | 10.70 | 10.77 (noise) |
| `BM_TccGridSample/threads:1` (ms, median of 3) | 1129 | 1114 |

No log-likelihood of the grid is lower. Five fits on series 11 and 18 now reach a higher one, by 1e-5 to 0.29: the
old stop had labelled them converged on a ridge. Two of them, sGARCH(3,3)-ARMA(1,1)-norm (+0.29) and
sGARCH(1,3)-ARMA(1,1)-std, now end `MAX_ITER` instead of a false `CONVERGED`. In the quick suite `BM_Loglik`, which
`ftol` does not touch, moved by up to 18% between the two runs: that is the noise of this session, and no
`BM_FitOne` moved by more.

### `Dual<12>` (k = 9..12)

`fit::with_dual_width` runs fits with 9 to 12 parameters on `Dual<12>` instead of `Dual<16>`; each lane does the
arithmetic of the same lane of the wider type, so values and gradients are bit-identical
(`LikelihoodGrid.Dual12AgreesWithDual32`).

| Measure | `Dual<16>` | `Dual<12>` |
| --- | ---: | ---: |
| Grid, 1 thread (s) | 70.8 | 61.3 (0.87) |
| Grid, 16 threads (s) | — | 9.5 |
| Grid evaluations, log-likelihoods, statuses | — | identical |
| `BM_TccGridSample/threads:1` (ms) | 1122 | 994 (0.89) |

The quick suite's other cases have k ≤ 8 or 13 and do not move (within noise, `arma22_s33_std/dual` 3%).

### L-BFGS memory (10 to 40)

With k ≤ 12 on the grid, a memory of 40 pairs makes L-BFGS behave like full BFGS
(`Lbfgs.DefaultMemoryCoversTwelveParameters`: 37 iterations against 135 on a 12-parameter ill-conditioned quadratic).
A slice of 2 series first (memory 5: 2.17× the time; 20: 0.57; 30: 0.52; 40: 0.50; 60: 0.50), then:

| Measure | Memory 10 | Memory 40 |
| --- | ---: | ---: |
| Grid, 1 thread (s) | 61.3 | 30.5 (0.50) |
| Grid, 16 threads (s) | 9.5 | 4.5 |
| Grid evaluations | 1.00 | 0.47 |
| Grid `CONVERGED` / `MAX_ITER` | 1066 / 14 | 1080 / 0 |
| `BM_FitOne/arma22_s33_std` (ms) | 383, `MAX_ITER` | 75, `CONVERGED` |
| `BM_FitOne/s21_std` (ms) | 16.7 | 11.7 |
| `BM_DccGrid/m:21/T:5000/threads:1` (s) | 10.7 | 10.4 |
| `BM_TccGridSample/threads:1` (ms) | 995 | 533 |

Every oracle comparison passes. The overparametrized fits of the grid (sGARCH(p, q ≥ 2) on series simulated from
sGARCH(1,1)) have several local optima, and the new path ends at another one on some series: 36 fits reach a
log-likelihood higher by more than 1e-3 (up to +0.93), 19 a lower one (down to −2.83, among them fits that ended
`MAX_ITER` before); the sum over the 1080 fits is −6.6. The author accepted the change with these numbers.

### Job order and block size

`ve::run_batch` ran contiguous blocks of jobs, 8 per thread, in job order; the grid's jobs are spec-major, so the
costly sGARCH(3,3)-ARMA(1,1) fits of one spec ran next to each other. It now runs one fit per block, the jobs by
decreasing parameter count (ties in job order). Results do not depend on the order (each job writes its own slot;
`CapiFit.ResultIndependentOfThreadCount` with 1, 3, 4 and 16 threads). Grid with 16 threads, three runs each:

| Scheduling | Wall (s) | `pool_use` |
| --- | --- | ---: |
| 8 blocks per thread, job order (before) | 4.62, 4.50, 4.48 | 0.92 |
| One fit per block, job order | 4.47, 4.27, 4.26 | 0.98 |
| One fit per block, costliest first (now) | 4.34, 4.23, 4.21 | 1.00 |

The ordering gains little on this grid, whose costly specs are spread over it, but keeps a batch whose costliest
specs come last from ending on one long fit.

### Measured and dropped: interpolation in the zoom

A safeguarded quadratic interpolation in `LineSearch::zoom` (instead of bisection) cut the grid's evaluations by
2.2% and left its one-thread time within noise (30.5 s to 30.9 s), while moving some fits to other points of their
flat optima. With a memory of 40 the line search takes 1.13 evaluations per iteration, so there is little left to
save there; the bisection stays.

### Profile after the changes

The same uProf run as at the top (`msvc-profile`, 2 series, 1 thread) takes 8.1 s instead of 20.2 s. The hottest
functions are the same kind: `Dual<12>` and `Dual<8>` arithmetic in `ad::chain`, `garch::sgarch_step`,
`mean::arma_step_of` and the division. No new hot spot appeared; the forward-mode gradient is still where the time
goes, which is the case for the reverse-mode gradient that follows plan 5a.

## Adjoint gradient (after plan 5a)

Design: `docs/superpowers/specs/2026-11-14-volengine-05-adjoint-design.md`. For sGARCH, iGARCH, gjrGARCH and eGARCH
with norm, std or ged innovations and an ARMA mean (`model::has_adjoint`), the optimizer and the numerical Hessian
take the gradient of the log-likelihood from a hand-written adjoint instead of the forward `Dual`. The forward pass is
`model::loglik<double>`, so the value does not change; each component (density, variance recursion, presample,
ARMA) then runs its reverse sweep. Everything else, and the per-observation scores, stays on the `Dual`.

### What the spike measured first

A throwaway spike compared, on sGARCH-ARMA-std with T = 5000, a generic reverse-mode tape (a `Var` scalar recording
every operation), the `Dual` with AVX2, and a hand-written adjoint. Per evaluation with gradient, against a `double`
evaluation without one: the tape took 5–6 times (about 165 000 nodes, 4 MB, per evaluation), twice the `Dual`; AVX2
cut the `Dual` by 0 to 20%; the adjoint took 1.3–1.5 times. Only the adjoint was built.

### One evaluation (`BM_Loglik`, µs, median of 3)

| Series | k | `double` (no gradient) | `Dual` | Adjoint | Adjoint / `Dual` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `s11_norm` | 4 | 50 | 151 | 91 | 0.60 |
| `igarch11_norm` | 4 | 51 | 158 | 94 | 0.59 |
| `gjrgarch11_norm` | 5 | 51 | 172 | 112 | 0.65 |
| `s21_std` | 6 | 100 | 291 | 190 | 0.65 |
| `egarch11_std` | 6 | 186 | 406 | 327 | 0.81 |
| `arma11_s11_ged` | 7 | 169 | 433 | 373 | 0.86 |
| `arma22_s33_std` | 13 | 155 | 644 | 290 | 0.45 |

The adjoint costs about twice a `double` evaluation, and its gain grows with k. It costs more than the spike's
1.3–1.5 times because the forward is the general `loglik<double>`: the reverse recomputes the transcendentals the
forward already took for the term's partials (`log1p` for std, about a quarter of the adjoint's time; `pow` and `log`
for ged), and eGARCH recomputes log σ² and 1/σ per step. A forward fused with the density partials would remove
that, at the price of a second implementation of the terms.

### Fits (A/B in one session, base `172818c`, adjoint `855a0a6`)

| Measure | `Dual` | Adjoint | Ratio |
| --- | ---: | ---: | ---: |
| Grid, 1 thread, `se:on` (s) | 31.1 | 22.0 | 0.71 |
| Grid, 1 thread, `se:off` (s) | 24.2 | 17.2 | 0.71 |
| Grid, 16 threads, `se:on` (s) | 4.16 | 2.70 | 0.65 |
| Grid, 16 threads, `se:off` (s) | 3.28 | 2.25 | 0.69 |
| Grid evaluations | 1.000 | 1.001 | — |
| Grid `CONVERGED` | 1080 | 1080 | — |
| `BM_TccGridSample/threads:1` (ms) | 539 | 423 | 0.79 |
| `BM_FitOne/s11_norm` (ms) | 4.96 | 3.12 | 0.63 |
| `BM_FitOne/s21_std` (ms) | 12.1 | 8.26 | 0.68 |
| `BM_FitOne/arma11_s11_ged` (ms) | 16.5 | 14.3 | 0.87 |
| `BM_FitOne/arma22_s33_std` (ms) | 75.5 | 36.7 | 0.49 |
| `BM_FitOne/egarch11_std` (ms) | 18.1 | 15.2 | 0.84 |
| `BM_FitOne/gjrgarch11_norm` (ms) | 5.78 | 4.02 | 0.70 |
| `BM_FitOne/igarch11_norm` (ms) | 3.29 | 2.16 | 0.66 |
| `BM_FitOne/aparch11_ged`, not covered (ms) | 45.2 | 45.9 | noise |
| `BM_DccGrid/m:21/T:5000/threads:1` (s) | 10.7 | 10.4 | noise |
| `BM_DccStandardErrors/dcc_mvnorm/m:21` (s) | 2.95 | 2.93 | noise |

By distribution, on the grid with 1 thread and `se:off`: norm 0.63, std 0.70, ged 0.81 of the time; ARMA(1,1) 0.71,
ARMA(0,0) 0.79. The log-likelihoods of the grid move by at most 3.4e-8 (`compare-profile`), every status stays
`CONVERGED`, and the evaluations are the same within 0.2%: the adjoint's gradient differs from the `Dual`'s only in
the last digits (the property test holds it within 1e-10 of the largest partial).
