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
