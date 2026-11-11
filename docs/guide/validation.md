# Validation against reference implementations

VolEngine's sGARCH(1,1) with normal innovations is validated against two independent oracles:
[`rugarch`](https://cran.r-project.org/package=rugarch) (R) for the default conventions and
[`arch`](https://pypi.org/project/arch/) (Python) for the `arch` options. The test is
`python/tests/test_oracles.py`; it reads committed fixtures, so it needs neither R nor `arch`. How the data and
fixtures are produced is described in [`tests/reference/README.md`](../../tests/reference/README.md).

## Data

Three series simulated by `tests/reference/generate_data.py` from one `numpy.random.default_rng(20261110)`
stream, with μ = 0.05, ω = 0.01, α = 0.08, β = 0.9, the variance recursion started at the unconditional
variance ω / (1 − α − β): `t5000`, `t3000` and `t1500` (T = 5000, 3000, 1500). Every oracle estimate is interior
(α̂ between 0.074 and 0.118, β̂ between 0.869 and 0.904, α̂ + β̂ at most 0.988); the test checks this.

## Oracle versions

| Oracle | Version | Environment |
|--------|---------|-------------|
| `rugarch` | 1.5-6 (`packageVersion` 1.5.6) | R 4.6.2 (ucrt, Windows), `jsonlite` 2.0.0, `numDeriv` 2016.8-1.1, `Rsolnp` 2.0.1 |
| `arch` | 8.0.0 | Python 3.14.7 (Windows), NumPy 2.5.4, SciPy 1.18.1 |

## What is compared

| VolEngine call | `rugarch` | `arch` |
|----------------|-----------|--------|
| `ctx.fit(ve.Spec(), y, options=ve.FitOptions(var_init=...))` | `ugarchfit(ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)), mean.model = list(armaOrder = c(0, 0), include.mean = TRUE), distribution.model = "norm"), y, solver = "hybrid")` | `arch_model(y, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(disp="off")` |
| `var_init` | `"sample"` | `"backcast"` |
| `ve.loglik(fit)` | `likelihood(fit)` | `.loglikelihood` |
| `fit.coef().estimate` | `coef(fit)` | `.params` |
| `fit.coef("hessian").std_error` | `fit@fit$matcoef[, " Std. Error"]` | `.std_err` with `cov_type="classic"` |
| `fit.coef("robust").std_error` | QML sandwich from `fit@fit$hessian` and `fit@fit$scores` (see below) | `.std_err` with `cov_type="robust"` |
| `fit.sigma` | `sigma(fit)` | `.conditional_volatility` |
| `ve.aic/bic/hqic/sic(fit, per_obs=...)` | `infocriteria(fit)`, `per_obs=True` | `.aic`, `.bic`, `per_obs=False` |

## Conventions

The oracle sources were read to establish the conventions below; no oracle code is copied or translated.

### `rugarch` (`VE_VARINIT_SAMPLE`, the default)

Read: `R/rugarch-sgarch.R` (`.sgarchLLH`), `R/rugarch-helperfn.R` (`.checkrec`, `.makefitmodel`),
`R/rugarch-tests.R` (`.information.test`), `R/rugarch-numderiv.R` (`robustvcv`, `neweywestcv`),
`src/garchmodels.c` (`sgarchfilterC`) and `src/filters.c` (`sgarchfilter`).

- **Initial variance.** With the default `rec.init = "all"`, every likelihood evaluation sets the
  pre-sample value to m̄ = mean(ε²) over the whole sample, with ε computed at the current μ. The first
  conditional variance is σ²₀ = m̄, and σ²₁ = ω + α ε²₀ + β σ²₀ uses the first in-sample residual.
- **First term.** The t = 0 term enters the log-likelihood: the sum runs over all T observations.
- **Criteria.** Per observation, with n = T and k = 4 estimated parameters: AIC = (−2ℓ + 2k)/n,
  BIC = (−2ℓ + k log n)/n, SIC = −2ℓ/n + log((n + 2k)/n), HQIC = (−2ℓ + 2k log log n)/n.
- **Hessian standard errors.** Square roots of the diagonal of the inverse of a numerical (Richardson)
  Hessian of −ℓ.
- **Robust standard errors.** `rugarch`'s `robust.se.coef` is a Newey-West (Bartlett kernel) sandwich with
  ⌊1.2 T^(1/3)⌋ lags over centered numerical scores. VolEngine's robust standard error is the QML sandwich
  H⁻¹ (SᵀS) H⁻¹ without lags, which is also what `arch` reports, so the two estimators differ by
  construction (up to 12% on these series). The `rugarch` fixture therefore also stores `qml_robust_se`,
  the QML sandwich built in R from `rugarch`'s own numerical Hessian (`fit@fit$hessian`) and per-observation
  scores (`fit@fit$scores`), and the test compares VolEngine against it. The Newey-West values stay in the
  fixture under `robust_se`.

VolEngine's `VE_VARINIT_SAMPLE` already followed these conventions; nothing in the core changed for
`rugarch`.

### `arch` (`VE_VARINIT_BACKCAST`)

Read: `arch/univariate/base.py` (`ARCHModel.fit`, `ARCHModel.compute_param_cov`, `ARCHModel._loglikelihood`),
`arch/univariate/volatility.py` (`VolatilityProcess.backcast`, `VolatilityProcess.variance_bounds`) and
`arch/univariate/recursions_python.py` (`garch_recursion_python`).

- **Backcast.** b = Σ wᵢ ε²ᵢ over the first τ = min(75, T) residuals, wᵢ = 0.94ⁱ / Σⱼ 0.94ʲ. `arch`
  computes b **once**, from the residuals at its starting values (the sample mean for a constant mean, zero
  for a zero mean), and keeps it fixed during the optimization and in the covariance computation.
- **Initial variance.** σ²₀ = ω + α b + β b; the t = 0 term enters the log-likelihood.
- **Criteria.** Totals: AIC = −2ℓ + 2k, BIC = −2ℓ + k log n. `arch` reports no HQIC or SIC.
- **Hessian standard errors.** Inverse of a numerical Hessian of −ℓ.
- **Robust standard errors.** H⁻¹ C H⁻¹ scaled to the sample, where C is the sample covariance of the
  numerical per-observation scores (centered, divisor n − 1). At the optimum the scores average to zero, so
  this equals VolEngine's H⁻¹ (SᵀS) H⁻¹ up to a factor √(n/(n − 1)) on the standard errors.
- **Variance bounds.** `arch` clamps σ²ₜ to loose bounds derived from an EWMA of ε²; they do not bind at
  these estimates (σₜ agrees with VolEngine to 1e-5).

**Core change.** VolEngine originally recomputed b from the residuals at the current μ and differentiated
it through μ. That is a different likelihood from `arch`'s, and on `t1500` it moved μ̂ by 1.4e-3
(relative), σₜ by 4.9e-3 and ℓ by 2.4e-3, failing the tolerances below. `VE_VARINIT_BACKCAST` now centers
the backcast on the sample mean of the series (or zero without a mean) and keeps it fixed, so b does not
depend on any parameter. The analytic gradient remains exact; the unit tests compare it with central
differences and check that σ²₀ does not depend on μ.

## Tolerances

Fixed before the comparison; none was loosened. "Observed" is the worst value over the three series.

| Quantity | Tolerance | Why | Observed, `rugarch` | Observed, `arch` |
|----------|-----------|-----|---------------------|------------------|
| Log-likelihood, one-sided | ℓ_VE ≥ ℓ_oracle − 1e-6·\|ℓ_oracle\| | VolEngine must reach an optimum at least as good as the oracle's | ℓ_VE − ℓ_oracle ≥ −5.0e-8 | ≥ −2.1e-9 |
| Log-likelihood, two-sided | relative 1e-4 | Same likelihood, different optimizers and stopping rules | 1.7e-11 | 5.8e-12 |
| Parameters | relative 1e-3; absolute 1e-5 when \|θ\| < 1e-2 | Flat likelihood near the optimum: parameters agree less than ℓ | 5.5e-5 rel; ω 7.7e-7 abs | 2.8e-5 rel; ω 3.0e-7 abs |
| Hessian SE | relative 5e-2 | Numerical second derivatives on both sides with different step rules | 2.4e-3 | 7.4e-5 |
| Robust SE | relative 5e-2 | Same, plus the oracle's numerical scores | 4.7e-3 (QML sandwich) | 3.3e-4 |
| σₜ | maximum relative 1e-3 | Follows the parameter tolerance through the recursion | 9.1e-6 | 9.0e-6 |
| Criteria | absolute 1e-6 on the oracle's scale | Deterministic functions of ℓ, n and k | 3.3e-11 (per observation) | 5.9e-8 (totals) |

On the totals scale the criteria tolerance is the tightest check: it holds only if ℓ agrees to 5e-7, which it
does (5.9e-8 in AIC, that is 3e-8 in ℓ, on `t5000`).

## Observed differences

VolEngine built from this repository (`gcc-debug`), default `FitOptions` apart from `var_init`.

### `rugarch`

| Series | ℓ_VE − ℓ_oracle | Max param rel | Max SE rel | Max QML robust SE rel | Max NW robust SE rel (not compared) | Max σ rel | Max criteria abs |
|--------|-----------------|---------------|------------|-----------------------|-------------------------------------|-----------|------------------|
| `t5000` | +1.5e-9 | 1.0e-5 | 2.4e-3 | 4.7e-3 | 9.3e-2 | 1.8e-6 | 6.0e-13 |
| `t3000` | −5.0e-8 | 5.5e-5 | 5.9e-4 | 1.2e-3 | 1.1e-1 | 9.1e-6 | 3.3e-11 |
| `t1500` | −3.0e-9 | 4.1e-6 | 2.9e-5 | 5.8e-5 | 1.2e-1 | 2.8e-6 | 4.0e-12 |

### `arch`

| Series | ℓ_VE − ℓ_oracle | Max param rel | Max SE rel | Max robust SE rel | Max σ rel | Max criteria abs |
|--------|-----------------|---------------|------------|-------------------|-----------|------------------|
| `t5000` | +2.9e-8 | 2.8e-5 | 7.4e-5 | 1.0e-4 | 9.0e-6 | 5.9e-8 |
| `t3000` | −2.1e-9 | 2.1e-5 | 6.0e-5 | 1.7e-4 | 3.2e-6 | 4.1e-9 |
| `t1500` | +1.1e-9 | 1.6e-5 | 2.4e-5 | 3.3e-4 | 3.7e-6 | 2.2e-9 |

## Regenerating

See [`tests/reference/README.md`](../../tests/reference/README.md). When an oracle version changes, regenerate
both fixtures, run `python/tests/test_oracles.py`, and update the version and observed-difference tables here.
