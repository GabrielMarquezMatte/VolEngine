# Profile and measurements of plan 5b

Plan 5b puts SIMD inside one evaluation. This file records the profile that orders the work (Task 2) and, below it,
the A/B of each step.

## Profile of the DCC fit (2026-11-14)

AMD uProf, time-based sampling (`--config tbp`), `msvc-profile` preset, one thread, m = 21, T = 5000, the whole fit
of `BM_DccFit/fdcc_mvnorm` (4.2 s on MSVC) and `BM_DccFit/dcc_mvnorm` (2.8 s). MSVC code is slower than GCC's, so
read the shares as indicative (see `measuring.md`).

| Slice | `fdcc_mvnorm` | `dcc_mvnorm` |
|-------|---------------|--------------|
| `linalg::detail::factor_entry<Dual<8>>`, the inner products of the Cholesky factor | 2.72 s (65%) | 1.98 s (72%) |
| `cholesky_factor<Dual<8>>` (the rest of the factor) | 0.27 s (6%) | 0.13 s (5%) |
| `cholesky_solve<Dual<8>>` | 0.32 s (8%) | 0.14 s (5%) |
| **Cholesky and solve, together** | **3.31 s (79%)** | **2.25 s (82%)** |
| `CorrelationRecursion<Dual<8>>::add_news` (the Q_t update) | 0.32 s | 0.20 s |
| `CorrelationRecursion<Dual<8>>::correlation` (R = D Q D) | 0.23 s | 0.13 s |
| `CorrelationRecursion<Dual<8>>::next` | 0.11 s | 0.06 s |
| **Recursion and normalization, together** | **0.66 s (16%)** | **0.39 s (14%)** |
| `observation_term<Dual<8>>` (the density; its logs and the quadratic form) | 0.06 s (1%) | not in the top ten |
| runtime (`vcruntime140`: copies) | 0.11 s | 0.05 s |

Everything is the `Dual<8>` objective, since the optimiser is about 99% of a fit; `reported_loglik` in `double`
(11 ms of a fit, measured in the plan 5b spike) is below the sampling resolution and does not appear.

Ruling: the division is not the expected 30–60% for each kernel. The Cholesky and the solve are about 80% of the
`Dual` objective and the recursion about 15%. The order of the work is therefore:
1. Task 3 writes the Cholesky and solve kernels first and measures them, then the recursion kernels.
2. Task 4 matters more than Task 3 for the fit time: a fit spends its time in the `Dual` Cholesky, which the
   tangent gradient of Task 4 removes (the Cholesky stays in `double`, once per t, with O(m²) work per parameter),
   while Task 3 only speeds up `double` evaluations (`reported_loglik`, standard errors).

## Task 3: the kernels of the value (2026-11-14)

`gcc-bench`, one thread, A/B in the same session (`base` = the build of the commit before Task 3, with the new
`BM_DccLoglik` benchmark copied in; medians of 3 repetitions, or one run of the fit), m = 21, T = 5000.

| Benchmark | base | new | new / base |
|-----------|------|-----|-----------|
| `BM_DccLoglik/dcc_mvnorm/double` | 7.8 ms | 3.3 ms | 0.42 |
| `BM_DccLoglik/fdcc_mvnorm/double` | 7.9 ms | 3.3 ms | 0.42 |
| `BM_DccLoglik/dcc_mvlaplace/double` | 8.7 ms | 4.6 ms | 0.53 (the Bessel K per t is scalar) |
| `BM_DccLoglik/*/dual` | 59 to 61 ms | 59 to 61 ms | 1.0 (the `Dual` path is unchanged) |
| `BM_DccStandardErrors` (finite differences of the `double` likelihood) | 2.9 to 3.1 s | 1.43 to 1.46 s | 0.48 |
| `BM_DccFit/dcc_mvnorm` | 0.87 s | 0.88 s | 1.0 (the gradient is still the `Dual`) |
| `BM_DccFit/fdcc_mvnorm` | 1.54 to 1.58 s | 1.52 to 1.53 s | 1.0 |

Ruling: the Cholesky kernel takes whole matrices of different t in the lanes (`simd::log_det_and_quadratic`), not
the elements of one matrix (the plan's inner products of the rows). The first version, a vector inner product per
entry of the factor and a vector forward and back solve, ran at 16 ms against 8 ms of the scalar code
(`BM_DccLoglik/dcc_mvnorm/double`, the same A/B): a row of the factor has about 7 entries on average, so the
masked tail and the horizontal sum of each inner product cost more than the products, and the scalar loop is
latency bound (each entry waits on the previous one) but not slow. Matrices of different t are independent given
Q_t, so the kernel transposes up to a vector of Q_t, normalizes them, factors them right-looking and forward-solves
in the lanes. A masked store is also slow on AMD cores, so the tails of the elementwise kernels are scalar.
Cost if wrong: none for the results, which agree with the scalar code to 1e-13 relative; the layout (kernel takes
packed Q_t, kBlock = 8 t per call) is the one Task 4 reuses for its per-t pullback.

The elementwise kernels (`multiply_add`, `add_outer`, `correlation_from_q`) never fuse, so Q_t and R_t of the
`double` path are the same bits as the scalar and `Dual` ones; only the Cholesky, the solve and the log differ, by
rounding (`DccLikelihood.DoubleMatchesDualValue`, 1e-12 per term; the measured gap is below 1e-14). The standard
errors and the bit-exact tests (the stepper against the fit's r) needed no tolerance: the stepper runs the same
kernels as the recursion, so it continues the fit to the bit.

Profile of the new `double` evaluation (MSVC, `BM_DccLoglik/dcc_mvnorm/double`, 9.4 ms): the factor 2.1 s, the
transpose and normalization 1.5 s, `add_outer` 1.2 s, the forward solve and the logs 1.1 s, `multiply_add` 0.1 s.
