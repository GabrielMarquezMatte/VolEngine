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

## Task 2: Boost in double

`log_bessel_k` and the tail p-values of `diagnostics/tails.cpp` (`gamma_q`, `ibeta`) took Boost's default policy, which
computes in long double (x87). The other Boost calls in `src/` (`ad/special.cpp`, `dist/cdf.cpp`,
`dcc/copula/scores.cpp`) already carried `promote_double<false>`. The new tests run the same computation with the x87
precision-control field at 64 and at 53 bits and compare bit for bit (`Bessel.SameBitsAtAnyX87Precision`,
`Portmanteau.TailsSameBitsAtAnyX87Precision`, `DistQuantile.HyperbolicSameBitsAtAnyX87Precision`); all three failed
before the change.

| Benchmark | Base | Task 2 |
| --- | ---: | ---: |
| `BM_LogBesselK/fixed_order` | 1.78 ms | 0.398 ms |
| `BM_LogBesselK/free_order` | 28.6 ms | 5.39 ms |
| `BM_Loglik/sgarch11_nig/double` | 0.591 ms | 0.253 ms |
| `BM_Loglik/sgarch11_nig/dual` | 2.31 ms | 0.862 ms |
| `BM_FitOne/sgarch11_nig` | 79.6 ms | 30.3 ms |
| `BM_Loglik/sgarch11_ghyp/double` | 0.589 ms | 0.254 ms |
| `BM_Loglik/sgarch11_ghyp/dual` | 23.8 ms | 5.29 ms |
| `BM_FitOne/sgarch11_ghyp` | 1765 ms | 377 ms |
| `BM_Diagnostics/sgarch11_ghyp/gof` | 29.0 ms | 6.46 ms |
| `BM_DccLoglik/dcc_mvlaplace/double` | 4.12 ms | 3.58 ms |
| `BM_DccLoglik/dcc_mvlaplace/dual` | 61.0 ms | 59.2 ms |
| `BM_DistQuantile/nig` (200 quantiles) | 57.1 ms | 21.7 ms (108 us per call) |
| `BM_DistQuantile/ghyp` (200 quantiles) | 97.8 ms | 28.9 ms (144 us per call) |

The long double evaluation was the cost, not only the bit-identity: the functions dominated by K (the nig and ghyp loglik, fits and GOF, `BM_LogBesselK`, the quantiles) are 2.3x to 5x faster; the mvlaplace DCC, where K is a small part, gains 11% on the `double` path.

## Task 3: slopes of K by the trapezoid rule

The `Dual` overloads of `log_bessel_k` and `log_bessel_k_scaled` with a free order take both slopes from one trapezoid
sum (`detail::log_bessel_k_slopes`); the value is still Boost's, in double. The Richardson difference of the order
slope (step 1e-3) differed from a long double reference by up to 1e-11 relative at x = 300 and 1e-10 at x = 1e-8; the
trapezoid agrees to 1e-12 on the whole grid of the test (orders −3.3 to 200, x 0.01 to 699) and 1e-11 at x = 1e-8.
Outside |order| <= 200, x >= 1e-8 the old Richardson stays, marked `ponytail:`.

| Benchmark | Base | Task 2 | Task 3 |
| --- | ---: | ---: | ---: |
| `BM_LogBesselK/free_order` | 28.6 ms | 5.39 ms | 2.21 ms (443 ns per call) |
| `BM_Loglik/sgarch11_ghyp/dual` | 23.8 ms | 5.29 ms | 2.13 ms |
| `BM_FitOne/sgarch11_ghyp` | 1765 ms | 377 ms | 139 ms |
| `BM_FitOne/sgarch11_nig` | 79.6 ms | 30.3 ms | 30.5 ms |

The ghyp fit is now 4.6 times the nig one (was 22), under the 5x at which the spec asks for a `ponytail:` on SIMD in the
trapezoid.

## Task 4: nig and ghyp quantiles from a table

`dist::QuantileTable` (`src/dist/cdf.*`) is built once per margin in `dcc/future/simulate.cpp`. It keeps the quadrature
cdf of the density at 513 nodes z = 0.25 sinh(w), |z| <= 40, takes the first guess from the cubic Hermite interpolant of
that cdf and finishes with Newton steps on the same quadrature cdf, so its roots are those of `quantile`: the tests
compare them to `quantile` within 1e-13 relative plus the 1e-15 rounding of the cdf divided by the density (in the
upper tail, u = 1 - 1e-12 has four digits left and two roots that both satisfy `|cdf(z) - u| <= 1e-13` can be 1e-5
apart), and require `|cdf(table(u)) - u| <= 1e-13` everywhere.

Ruling on the polish: the plan proposed a short Gauss-Legendre segment from the neighbouring node. The quadrature cdf
beyond |z| = 1 is a scaled tail integral that is not exactly the sum of segment integrals (the two quadratures differ by
up to ~1e-8 on a peaked density, see `quadrature_cdf`), so a segment polish would not land on the root of `quantile`.
The Newton step uses the quadrature cdf itself, with its z-independent parts (the mass below 0 and the two join scales,
`detail::CdfConstants`) computed once: the same arithmetic, so the same bits, at about a quarter of the cost of a cold
call (a cold call was 9.5 us, one step with constants is ~2.5 us).

| Benchmark (200 probabilities) | `quantile` | `QuantileTable` | per call |
| --- | ---: | ---: | ---: |
| nig | 21.8 ms (109 us per call) | 0.539 ms | 2.7 us (40x) |
| ghyp | 28.7 ms (143 us per call) | 0.709 ms | 3.5 us (40x) |
| table construction, nig | | 1.42 ms | |
| table construction, ghyp | | 1.86 ms | |

The base (Task 1) was 286 and 489 us per call; the part of the gain in the `quantile` column is Task 2. A copula
simulation of 1000 paths x 60 steps x 21 nig margins goes from ~6 min (base) to ~3 s plus 21 constructions (~30 ms).

## Task 5: ged and sged at large shapes

Three defects, found by tests against closed forms (a ged at eta2 = 0, the closed cdf of the sged, a brute-force
integral of the density):
- **Edges.** A ged kernel exp(-|y / scale|^nu / 2) is flat to |y| ~ scale and falls within scale / nu: no piece of the
  quadrature resolved it. Relative error of the ged shifted moment against its closed form, before and after splitting
  at nine points around each edge (`kEdgeOffsets`, from nu = 6 on; below it the points and the values are those of
  before):

  | nu | before | after |
  | ---: | ---: | ---: |
  | 5 | 1e-12 | 1e-12 (unchanged) |
  | 20 | 3.5e-6 | < 1e-11 |
  | 50 | 1.3e-3 | < 1e-11 |
  | 200 | 1.2e-2 | < 1e-11 |
  | 1000 | 1.6e-2 | < 1e-11 |

  A power of 1.5 stays at 7e-11 (nu = 2) to 2e-10 (nu = 1000): the cusp of |z|^1.5 at the kink, where 64-node
  Gauss-Legendre converges algebraically. The shifted moment of the sged at nu = 1000 (skew 1.2, eta1 0.3, eta2 0.2,
  power 1.5) was 1.0726 against the uniform limit 1.0845, and is 1.0845.
- **The cdf.** `kernel_cdf` of the ged took 0.5 (|z| / scale)^nu as 0 where it underflows, which at nu = 500 and 1000 is
  well inside the support, and returned the cdf of the centre there (P(z < 0) of the sged 0.61 instead of 0.5). It takes
  P(a, x) = x^a / Gamma(a + 1) from the log of the argument below 1e-290. This is a defect of `dist::cdf`, which the
  GOF and the parametric copula margins use, not of the moments.
- **The gradient.** The derivative in the shape of every ged and sged moment was NaN from nu ~ 100: the quadrature visits
  z ~ 3e3, where 0.5 (|z| / scale)^nu overflows and exp(-inf) times the infinite slope is NaN. `density_at` returns an
  exact 0 where the log density is -inf. It was also wrong at nu = 50 (a gradient of 0.045 in the skew against
  -0.0002), from the same missing edges.

`kEdgeShapeFrom` is 6 because the values below it stay those of before (bit for bit, so the oracles at small shapes do
not move): the old quadrature errs against the closed form by 3.2e-13 at nu = 4, 1.9e-12 at 5 and 2e-11 at 6, so the
moment steps by up to 2e-11 relative at nu = 6, harmless for an optimizer. Lowering it to 4 would move the bits for
nu in (4, 6) by ~1e-12.

No fit benchmark moved: `BM_FitOne/aparch11_ged` 44.5 -> 44.3 ms, `aparch11_sged` 53.0 -> 55.0 ms (noise),
`gjrgarch11_sstd` 15.6 -> 15.5 ms.

## Final table of plan 5c-2 (2026-11-15)

`gcc-bench`, one thread, medians of five repetitions, the base (`881e7a7`, `build/base-5c2/`) and the new code run one
right after the other in the same session.

| Benchmark | Base | 5c-2 | Ratio |
| --- | ---: | ---: | ---: |
| `BM_LogBesselK/fixed_order` (5 thousand arguments) | 1.81 ms | 0.402 ms | 0.22 |
| `BM_LogBesselK/free_order` | 28.9 ms | 2.20 ms | 0.076 |
| `BM_Loglik/sgarch11_nig/double` | 0.602 ms | 0.242 ms | 0.40 |
| `BM_Loglik/sgarch11_nig/dual` | 2.27 ms | 0.870 ms | 0.38 |
| `BM_FitOne/sgarch11_nig` | 79.6 ms | 30.5 ms | 0.38 |
| `BM_Loglik/sgarch11_ghyp/double` | 0.592 ms | 0.245 ms | 0.41 |
| `BM_Loglik/sgarch11_ghyp/dual` | 24.1 ms | 2.13 ms | 0.088 |
| `BM_FitOne/sgarch11_ghyp` | 1785 ms | 141 ms | 0.079 |
| `BM_Diagnostics/sgarch11_ghyp/gof` | 28.9 ms | 6.49 ms | 0.22 |
| `BM_DccLoglik/dcc_mvlaplace/double` (m = 21, T = 5000) | 4.08 ms | 3.65 ms | 0.89 |
| `BM_DccLoglik/dcc_mvlaplace/dual` | 61.4 ms | 61.0 ms | 0.99 |
| `BM_DistQuantile/nig` (200 quantiles, `quantile`) | 58.2 ms | 21.9 ms | 0.38 |
| `BM_DistQuantile/ghyp` | 99.2 ms | 29.1 ms | 0.29 |
| `BM_DistQuantile/nig/table` | | 0.548 ms (2.7 us per call) | 0.009 against the base `quantile` |
| `BM_DistQuantile/ghyp/table` | | 0.724 ms (3.6 us per call) | 0.007 |
| table construction, nig / ghyp | | 1.43 / 1.89 ms | |
| `BM_FitOne/aparch11_ged` | 45.1 ms | 45.6 ms | 1.01 |
| `BM_FitOne/aparch11_sged` | 54.4 ms | 53.4 ms | 0.98 |

The quick suite (`quick.py compare`, base against new): every ratio between 0.94 and 1.01, none above noise on the wrong
side. `BM_Loglik/aparch11_ged/dual` reads 1.00 ms on the base and 1.00 to 1.13 ms on the new code between runs of the
same binary (cv 1%), a bimodality of the machine, not of the change.

What is left of the ghyp: the Dual loglik is 8.7 times its double one (nig: 3.6): 443 ns per call of the trapezoid
against 80 ns of Boost's value. A SIMD sum over the grid nodes is the next step (`ponytail:` not placed: the fit is
4.6 times the nig one, under the 5x at which the spec asked for it).
