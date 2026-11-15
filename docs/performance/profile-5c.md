# Profile and measurements of plan 5c-1

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

The base run was stopped before `BM_CopulaGrid` and `BM_CopulaStandardErrors` finished; Task 5 measures those on both
sides with five repetitions.

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
