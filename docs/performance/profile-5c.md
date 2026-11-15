# Profile and measurements of plans 5c-1 and 5c-2

Plan 5c-1 speeds up the copula fits. This file records the base, the cause of the static Student copula failure and,
below, the A/B of each step. Measures: preset `gcc-bench`, one thread, m = 21, T = 5000, same session, medians of
three repetitions.

## Base (code at `6f7053d`, 2026-11-15)

| Benchmark | Median |
| --- | ---: |
| `BM_CopulaFit/static_mvnorm_kendall_param` | 3.79 s |
| `BM_CopulaFit/static_mvt_kendall_param` | error: "the DCC fit has no finite point" |
| `BM_CopulaFit/static_mvt_ml_param` | error: "the DCC fit has no finite point" |
| `BM_CopulaFit/dcc_mvnorm_param` | 0.945 s |
| `BM_CopulaFit/dcc_mvt_param` | 36.4 s |
| `BM_CopulaFit/adcc_mvnorm_param` | 1.05 s |
| `BM_CopulaFit/adcc_mvt_param` | 36.6 s |
| `BM_DccFit/dcc_mvnorm` | 0.146 s |
| `BM_DccFit/dcc_mvt` | 0.141 s |
| `BM_DccFit/adcc_mvt` | 0.234 s |
| `BM_DccGrid/m:21/T:5000/threads:1` | 2.14 s |

The first base run was stopped before `BM_CopulaGrid` and `BM_CopulaStandardErrors` finished; they were measured in
a later base run (three repetitions): the grid ends with the static Student error, `BM_CopulaStandardErrors` takes
4.44 s.

## The static Student copula failure

Cause: the series are independent, so the profile of the shape rises to the ceiling. Brent evaluates nu = 50 itself,
and that point is the best one, so `profile_shape` returns exactly 50. `fit_copula` then tested `nu < kShapeCeiling`
(the check for a fixed shape outside (4, 50)) on the profiled shape too, rejected it and returned the default
(failed, non-finite) point.
Fix: the open-interval check applies only to a fixed shape; a profiled shape lies in [4, 50] by construction.

## Task 2: copulas on the tangent gradient

`has_fast_gradient` accepts any `copula`; the inner fit of the copula takes the 5b tangent gradient. Medians of three
repetitions, `gcc-bench`, the base measured earlier in the same session (the base run is in the table above).

| Benchmark | Base | Tangent gradient |
| --- | ---: | ---: |
| `BM_CopulaFit/static_mvt_kendall_param` | error | 4.60 s |
| `BM_CopulaFit/static_mvt_ml_param` | error | 2.19 s |
| `BM_CopulaFit/dcc_mvnorm_param` | 0.945 s | 0.208 s |
| `BM_CopulaFit/dcc_mvt_param` | 36.4 s | 8.27 s |
| `BM_CopulaFit/adcc_mvnorm_param` | 1.05 s | 0.238 s |
| `BM_CopulaFit/adcc_mvt_param` | 36.6 s | 9.30 s |
| `BM_CopulaFit/static_mvnorm_kendall_param` (unchanged) | 3.79 s | 3.82 s |
| `BM_DccGrid/m:21/T:5000/threads:1` (unchanged) | 2.14 s | 2.24 s |

`BM_DccFit/dcc_mvt` moved from 0.141 s to 0.205 s although plain DCC specs take the same path as before; the final
table of Task 5 repeats the plain `BM_DccFit` rows five times on both sides.

## Task 3: the shape profile with a warm start

Each profile point starts from the estimate of the evaluated point with the nearest shape, and the maximum is the
point Brent evaluated (no final re-fit). Medians of three repetitions, `gcc-bench`.

| Benchmark | Base | Task 2 | Task 3 |
| --- | ---: | ---: | ---: |
| `BM_CopulaFit/static_mvt_ml_param` | error | 2.19 s | 1.73 s |
| `BM_CopulaFit/dcc_mvt_param` | 36.4 s | 8.27 s | 3.62 s |
| `BM_CopulaFit/adcc_mvt_param` | 36.6 s | 9.30 s | 3.59 s |

Profile points (`fit_dcc` calls) of one fit, counted by a temporary print: 38 on the base (37 Brent points and the
re-fit of the winner), 37 now. A fit is about 0.96 s on the base and 0.098 s now, so the 37 points still dominate
`dcc_mvt_param`: the next step is Newton on the profile through the envelope theorem (the `ponytail:` in
`profile_shape`). `DccFitResult` carries no iteration count, so the mean inner iterations are not recorded.

## Task 4: Kendall's tau-b by Knight's algorithm

O(T log T) per pair of columns (sort by (x, y), merge sort on y counting inversions, ties by runs), with the integer
counts of the old loop: `CopulaStatic.KendallMatchesPairCount` compares the matrix to the pair-count loop to the bit
on panels with and without ties. Medians of three repetitions, `gcc-bench`.

| Benchmark | Base | Task 3 | Task 4 |
| --- | ---: | ---: | ---: |
| `BM_CopulaFit/static_mvnorm_kendall_param` | 3.79 s | 3.92 s | 0.165 s |
| `BM_CopulaFit/static_mvt_kendall_param` | error | 4.57 s | 0.886 s |

The 0.165 s left of `static_mvnorm_kendall_param` is the rest of the fit (the 105 thousand normal quantiles, the
Cholesky of R and the likelihood), not the 210 Kendall entries. The Student version adds about 0.7 s for its 37
profile points, each of which recomputes the 105 thousand Student quantiles (about 20 ms) and the static likelihood.

## Final table (2026-11-15)

`gcc-bench`, one thread unless the name says otherwise, m = 21, T = 5000. New code at `HEAD` of plan 5c-1: five
repetitions, medians. Base (`6f7053d`): the three-repetition medians measured earlier in the same session for
`BM_CopulaFit` and `BM_CopulaStandardErrors`; five repetitions for the secondary rows below. The base run of
`BM_CopulaGrid` ends with "the DCC fit has no finite point" (its static Student specs), so it has no time.

| Benchmark | Base | New | New / base |
| --- | ---: | ---: | ---: |
| `BM_CopulaFit/static_mvnorm_kendall_param` | 3.79 s | 0.160 s | 0.04 |
| `BM_CopulaFit/static_mvt_kendall_param` | error | 0.894 s | |
| `BM_CopulaFit/static_mvt_ml_param` | error | 1.74 s | |
| `BM_CopulaFit/dcc_mvnorm_param` | 0.945 s | 0.197 s | 0.21 |
| `BM_CopulaFit/dcc_mvt_param` | 36.4 s | 3.47 s | 0.10 |
| `BM_CopulaFit/adcc_mvnorm_param` | 1.05 s | 0.218 s | 0.21 |
| `BM_CopulaFit/adcc_mvt_param` | 36.6 s | 3.56 s | 0.10 |
| `BM_CopulaGrid/threads:1` | error | 19.8 s | |
| `BM_CopulaGrid/threads:hw` | error | 4.53 s | |
| `BM_CopulaStandardErrors/dcc_mvnorm_param` | 4.44 s | 4.36 s | 0.98 |

Secondary rows, five repetitions on both sides, no change above the noise:

| Benchmark | Base | New |
| --- | ---: | ---: |
| `BM_DccFit/dcc_mvnorm` | 0.138 s | 0.135 s |
| `BM_DccFit/dcc_mvt` | 0.139 s | 0.144 s |
| `BM_DccFit/adcc_mvt` | 0.224 s | 0.228 s |
| `BM_DccFit/fdcc_mvnorm` | 0.343 s | 0.338 s |
| `BM_DccGrid/m:21/T:5000/threads:1` | 2.13 s | 2.09 s |
| `BM_DccStandardErrors/dcc_mvnorm` | 1.37 s | 1.37 s |

The quick set (`quick.py run` and `compare`, base against new) shows every row within the noise. The one flagged
value, `BM_Loglik/arma22_s33_std/dual` (1.59), is in code this plan does not touch; five repetitions of it alone give
666 and 683 us on the new build against 671 and 682 us on the base.

## Where the rest of the time goes (measured, not optimised)

Two cases stay above one second. Temporary timers (removed) around the stages, one run each, `gcc-bench`:

`static_mvt_ml_param`, 1.74 s, 19 profile points:

| Stage | Time | Calls |
| --- | ---: | ---: |
| the Student score panel (`copula_panel`: 105 thousand Student quantiles) | 0.657 s | 19 |
| `static_ml` (L-BFGS on the 210 free entries) | 0.295 s | 19 |
| `copula_loglik` of the point (builds the panel again, then the static likelihood) | 0.757 s | 20 |

About 80% is quantiles, and each point computes them twice: once for the fit and once more inside `copula_loglik`,
because `copula_terms` rebuilds the panel unless `inputs.scores` holds one at that shape. Candidate for the author:
hand the panel of the point to its loglik.

`BM_CopulaStandardErrors`, 4.7 s (this run, with the timers): `inputs_at` takes 3.25 s in 106 calls (31 ms each: it
recomputes the uniforms of the 21 margins at the perturbed point and the normal scores), the terms of the copula
1.10 s in 325 calls (3.4 ms each). Candidate for the author: the uniforms and scores change only for the series
whose parameter moved, so a perturbation could recompute one column.

# Plan 5c-2: distributions

Measures: preset `gcc-bench`, one thread, same session, medians of three repetitions. The base is the build of
`881e7a7` (plan 5c-2 task 1, code identical to `baf5eb5` but for the new `BM_DistQuantile`), in `build/base-5c2/`.

## 5c-2 base (2026-11-15)

| Benchmark | Base |
| --- | ---: |
| `BM_LogBesselK/fixed_order` (5 thousand arguments) | 1.78 ms |
| `BM_LogBesselK/free_order` | 28.6 ms |
| `BM_Loglik/sgarch11_nig/double` | 0.591 ms |
| `BM_Loglik/sgarch11_nig/dual` | 2.31 ms |
| `BM_FitOne/sgarch11_nig` | 79.6 ms |
| `BM_Loglik/sgarch11_ghyp/double` | 0.589 ms |
| `BM_Loglik/sgarch11_ghyp/dual` | 23.8 ms |
| `BM_FitOne/sgarch11_ghyp` | 1765 ms |
| `BM_Diagnostics/sgarch11_ghyp/gof` | 29.0 ms |
| `BM_DccLoglik/dcc_mvlaplace/double` (m = 21, T = 5000) | 4.12 ms |
| `BM_DccLoglik/dcc_mvlaplace/dual` | 61.0 ms |
| `BM_DistQuantile/nig` (200 quantiles) | 57.1 ms (286 us per call) |
| `BM_DistQuantile/ghyp` (200 quantiles) | 97.8 ms (489 us per call) |
