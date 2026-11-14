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
