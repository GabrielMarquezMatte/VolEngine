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
