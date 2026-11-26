# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
From 1.0 the C ABI only grows: a signature or an enum value is never changed or removed, and `VE_ABI_VERSION` is the
semver major (the Python package and the C++ wrapper refuse a library whose ABI version differs). Before 1.0 it could
break between 0.x releases.

## [Unreleased]

### Fixed

- Python: the regressors of a horizon given to `Fit.forecast`, `DccFit.forecast` or `Context.simulate` as a list, a
  1D array or any input that had to be copied were freed before the library read them, so the forecast or the
  simulation could silently use other values (for example `vxreg`'s in place of `mxreg`'s) or read freed memory. The
  copies now live until the native call is done.
- DCC: with the gcc (MinGW) build, a `ve_dcc_fit_covariance` or `ve_dcc_forecast_cumulative_covariance` whose
  computation failed (for example `VE_OUT_OF_MEMORY` for a large m) made every later call on that handle hang. A
  failed computation now leaves the cache empty and the next call tries again.
- fGARCH TGARCH and AVGARCH (power fixed at 1) with variance regressors: a point where omega + zeta v_t drove
  sigma_t negative was accepted, because its square is a positive sigma2, so the fit could settle outside the model.
  A state sigma^power that is not positive is now outside the domain, in the likelihood and in the forecast.
- jsu innovations in apARCH and fGARCH: at an asymmetry of exactly -1 or 1 (fixed, or reached by the optimizer), the
  shock moment had a NaN gradient and the fit failed. The gradient is now finite.
- C#: `Context.Fit` with more than `int.MaxValue` jobs (specs x series) wrapped the job count around and fitted only
  part of the grid. It now throws `OverflowException` before calling the library.

### Performance

- DCC forecast: `ve_dcc_forecast_create` at m = 21, h = 60 takes 0.18 of its time (one `pow` per step outside FDCC
  instead of one per entry of R), and `ve_dcc_forecast_cumulative_covariance` without an ARMA mean is a running sum,
  linear in the horizon. The results have the same bits.
- Student Copula-DCC: the fits of `dcc`, `adcc` and static ML Student copulas take about 0.64 of their time (21
  series, 5000 rows), with the same bits: each point of the shape profile reuses the scores its inner fit ran on.
- C#: large results are copied once, straight from the native views into the jagged arrays, instead of into a
  flat array that was then split. A DCC simulation of 10 series with R and H (1000 paths, 60 steps) allocates 145 MB
  instead of 246 MB and takes 0.70 of its time, a univariate one (10000 paths) half the memory and 0.78 of the
  time, and `DccFit.Correlation()` 0.35 of the time.
- nig: the Dual log-likelihood and the fit take about 0.78 of their time, with the same bits: the Bessel K of each
  observation and its slope cost two Boost calls instead of four kernels (K_2 from the K_0 and K_1 of Boost's own
  recurrence, and the value computed once).
- Simulation: a simulated step no longer allocates (outside an ARFIMA mean), so `ve_simulate_fit` and
  `ve_simulate_spec` take 0.33 to 0.45 of their time on one thread and 0.43 to 0.58 on all of them (10000 paths of 60
  steps), and `ve_simulate_dcc` about half. The paths have the same bits for the same seed.

## [1.3.0] - 2026-11-26

### Added

- Cooperative cancellation. `ve_cancel_create`, `ve_cancel_request` and `ve_cancel_free` manage a signal that can be
  requested from any thread. `ve_fit_batch_with_cancel`, `ve_dcc_fit_batch_with_cancel`,
  `ve_dcc_standard_errors_with_cancel`, `ve_simulate_fit_with_cancel`, `ve_simulate_spec_with_cancel` and
  `ve_simulate_dcc_with_cancel` take it and return the new status `VE_CANCELLED` soon after it is requested, with every
  output set to NULL. A NULL signal does what the call without `_with_cancel` does.
- Cumulative simulations. `ve_sim_options_set_checkpoints` gives a simulation a list of checkpoints of its horizon;
  `ve_simulate_fit`, `ve_simulate_spec` and `ve_simulate_dcc` then keep only, per path (and series), the sum of the
  simulated series up to each checkpoint, read with `ve_sim_cumulative` / `ve_dcc_sim_cumulative` (and the list with
  `ve_sim_checkpoints` / `ve_dcc_sim_checkpoints`). Memory is paths x checkpoints x series instead of
  paths x horizon x series, and each sum equals the left-to-right sum of the full output with the same seed.
- Cumulative forecasts. `ve_forecast_cumulative_mean` and `ve_forecast_cumulative_sigma` give, for h = 1..horizon,
  the mean and standard deviation of the forecast sum of the next h returns; `ve_dcc_forecast_cumulative_mean` and
  `ve_dcc_forecast_cumulative_covariance` give the mean and covariance for a DCC. The variance weights each step by
  the MA(inf) weights of an ARMA/ARFIMA mean, so it is the plain sum of sigma^2 (or of H) only without one.
- Covariance estimators, the first module outside GARCH (`ve_cov_*`). `ve_cov_sample` gives the sample covariance of
  a panel (divisor t - 1, as `numpy.cov`) and `ve_cov_ledoit_wolf` its Ledoit-Wolf shrinkage towards mu * I with the
  intensity, as `sklearn.covariance.ledoit_wolf`. Both write an exactly symmetric N x N matrix into the caller's
  buffer and refuse panels with ranges, fewer than 2 rows or a value that is not finite.
- Risk measures (`ve_risk_*`). `ve_risk_empirical` gives the VaR and ES of a weighted portfolio over a panel of log
  (`VE_RETURNS_LOG`) or simple (`VE_RETURNS_SIMPLE`) returns, dates or simulated paths in rows, with
  `numpy.quantile`'s linear interpolation; `ve_risk_parametric` gives them for mean + sigma * z with z any of the eight
  standardized densities and its skew, shape and ghlambda (closed-form ES for norm, std and jsu, quadrature for the others).
  Both are positive as losses.
- C++: `ve::risk::empirical` and `ve::risk::parametric` (`volengine/risk/measures.hpp`).
- C++: `ve::cov::sample` and `ve::cov::ledoit_wolf` (`volengine/risk/covariance.hpp`), over a `Panel`.
- C++: `cumulative_mean()` and `cumulative_sigma()` on `Forecast`, `cumulative_mean()` and `cumulative_covariance()` on
  `DccForecast`.
- C#: `Risk.Empirical` and `Risk.Parametric`, returning `RiskMeasure`, with `ShockDistribution` and `ReturnKind`.
- C#: `Covariances.Sample` and `Covariances.LedoitWolf` (with `LedoitWolfResult`), over a T x N `double[,]`.
- C#: `CumulativeMean` and `CumulativeSigma` on `Forecast`, `CumulativeMean` and `CumulativeCovariance` on
  `DccForecast`.
- Python: the `ve.risk` module, with `empirical(returns, weights, kind=, confidence=)` and
  `parametric(mean, sigma, confidence=, dist=, skew=, shape=, ghlambda=)`, both returning `(var, es)`.
- Python: the `ve.cov` module, with `sample(returns)` and `ledoit_wolf(returns)` (the matrix and the intensity).
- Python: `cumulative_mean` and `cumulative_sigma` on `Forecast`, `cumulative_mean` and `cumulative_covariance` on
  `DccForecast`.
- C++: `SimOptions::checkpoints`, and `cumulative()` and `checkpoints()` on `Simulation` and `DccSimulation`.
- C#: `Checkpoints` on the simulation options, and `Cumulative` and `Checkpoints` on the results; `Returns` and `Sigma`
  are empty in that mode.
- Python: the `checkpoints=` keyword of `Context.simulate`, and `cumulative` and `checkpoints` on `Simulation` and
  `DccSimulation`; `returns` and `sigma` raise `VolEngineError` in that mode.
- C++: `fit`, `fit_dcc`, `standard_errors` and `simulate` take an optional `std::stop_token` as their last argument;
  a requested stop fails the call with `ve::Status::cancelled`.
- C#: overloads of `Fit`, `FitDcc`, `StandardErrors` and `Simulate` with a `CancellationToken` as their last parameter;
  a cancelled token throws `OperationCanceledException`. `Status.Cancelled` is the new status.
- Python: `ve.Cancel`, whose `cancel()` may be called from any thread, and the `cancel=` keyword of `Context.fit`,
  `fit_dcc`, `standard_errors`, `simulate` and `DccFit.standard_errors`; a cancelled call raises `ve.Cancelled`, a
  `VolEngineError` with status 6.

### Deprecated

- `ve_fit_batch`, `ve_dcc_fit_batch`, `ve_dcc_standard_errors`, `ve_simulate_fit`, `ve_simulate_spec` and
  `ve_simulate_dcc`: call the `_with_cancel` variant with a NULL `ve_cancel`. They warn at compile time (define
  `VE_NO_DEPRECATED_WARNINGS` to silence it) and are removed in ABI 2.

### Fixed

- C++ wrapper: `ve::Panel::create_strided` with a stride so large that `(rows - 1) * stride_rows + (cols - 1) *
  stride_cols` wrapped around in `size_t` passed the extent check, and the panel then read past the data. Such a stride
  is now refused with `invalid_argument` ("the data is smaller than the shape of the panel").
- DCC: a fit with a fixed weight beside default starts failed or did not start when the default start (`dcca1`,
  `dccb1`, `dccg1`, `fdcca1`, ...) left no room under the persistence ceiling, for example `dcca1 = 0.15` fixed in
  DCC or aDCC, or `fdcca1 = 0.5` fixed in FDCC. The weights left to their default now shrink to fit beside the fixed
  and started ones, so the fit converges. A start the caller gave is still taken exactly.
- DCC: an FDCC fit (30 or more groups) with one weight fixed estimated as many parameters as the widest AD lane but
  the objective differentiated one more, so the fit failed. It now takes the dual-number objective and runs.
- DCC: simulation with the empirical copula from a correlation matrix without a Cholesky factor read a quantile from
  no rank of the sample. Every shock is now NaN, and so is the return.
- A csGARCH fit with `rho` at `P`, or `phi` at the sum of `beta`, had every standard error NaN, where only the
  parameters at the edge can have none: a step up in any weight takes `P` above `rho`, a step down in any `beta` takes
  the sum of `beta` below `phi`. Only `rho`, `phi` and the weights beside them lose their standard errors now.
- A fit whose `omega` underflowed (very short series, `exp(u)` below 1e-308) had `omega = 0`, which failed the validity
  check, every standard error NaN and an infinite unconstrained value that broke it as a warm start or as fixed values.
  `omega` now stops at the smallest normal double, where it has no standard error but the others keep theirs.
- A shape that makes a shock-moment `kappa` subnormal (`alpha` is read back as `t / kappa`, which overflowed) is
  refused as invalid instead of giving an infinite `alpha`.
- Distributions: the standardization moments (the `kappa` of the weights, the probability below zero) of std and sstd
  with the shape near the power (ν near 2), of ged and sged with a small shape (ν = 0.1), and of nig and ghyp
  with a small zeta were NaN or off, up to 7e-2 in `E(z - c)^2`, because the quadrature of the tails overflowed or
  ended before the mass did. They are finite, standardized and differentiable now.
- Johnson SU: the standardization overflowed for `tau` below about 0.053 or `|nu| / tau` above about 354, giving NaN
  or infinite density, constant and scale. It is taken in logs there and stays finite; where the scale itself is
  below the smallest normal double (`tau` below about 0.038) the density, moments and quantiles are NaN, never a scale
  of 0 that made every quantile the center.
- The std (Student t) cdf lost its digits near the center (z near 0, or a large shape), because it was computed through
  a quantity that rounds to 1. It is accurate there, and matches the normal expansion for a large shape.

## [1.2.0] - 2026-11-19

### Added

- Python: `Context.standard_errors(fit, univariate)` adds the two-stage standard errors to a DCC fit on that context's
  threads; `DccFit.standard_errors` takes an optional `context` (without one it still uses every core).

### Changed

- A univariate start or fixed value is checked against the closure of what the parameter map reaches, so every
  estimate passes: the persistence P < 1 (the map keeps it at most 1 - 1e-8 but for rounding), the ARFIMA d in
  rugarch's closed [1e-8, 0.5], |eta1| and the apARCH |gamma| at most 1, and csGARCH P <= rho < 1. A d of 0.5 or a
  gamma of 1, rejected before, is accepted, as in rugarch.
- The ged and sged shape has rugarch's lower bound of 0.1: a start or fixed shape below it ends `VE_FIT_FAILED`.

### Fixed

- A start value on the edge of its domain (a weight of 0, a persistence at its ceiling) no longer freezes it: it mapped
  to a logit of -708 where the optimizer cannot move, and the fit reported convergence there. `alpha1 = 0` in sGARCH
  ended 130 below the optimum in loglik, `dcca1 = 0` or `dccb1 = 0` in DCC and `dccg1 = 0` in aDCC as well. Such a
  start now begins just inside; any other start is taken exactly.
- A fit whose persistence rounds to its ceiling (high-persistence data) had every standard error NaN, and its estimate
  failed as a warm start or as fixed values, because the validity check was stricter than the map. The csGARCH rho
  could also round below P.
- eGARCH (and ged with shape <= 1) standard errors where the mean estimate sits on a kink of |z|, which a concave kink
  attracts: the Hessian no longer differences the gradient across it. On simulated series, eGARCH-ARMA(1,1) fits
  with every standard error NaN fell from 10 in 40 to 1 (an unidentified ARMA), and the plain eGARCH mu standard
  error, 0.2 to 0.33 of the right one on 5 in 40 fits, is right.
- A ged or sged fit on a series with exact zeros (stale prices, no mean) had no maximum: the shape ran to 0 and the fit
  ended `VE_FIT_FAILED` with a loglik of +31775.
- Python: a forecast horizon of 2**64 or more is rejected instead of wrapping; `VOLENGINE_LIB` with a bare file name
  loads that file; broadcast (stride 0) arrays are copied instead of rejected; numpy integers are accepted in `arma`
  and `order`; copying or pickling a fit raises `TypeError` instead of failing in the finalizer; the macOS wheel tag
  keeps the minor version of the deployment target; the `DccSpec` docstring describes fixed parameters as listed.
- C++ wrapper: `Batch::set_regressors({}, {})` clears the regressors (it passed 0 jobs); no handle leaks when an
  allocation fails after a batch fit; the git tag is read only from the VolEngine repository itself, so a vendored
  `cpp/` no longer downloads the release of the enclosing repository's tag; the download verifies TLS and never
  reuses a partial file; musl is detected from the toolchain, not from a loader file on the host; a prerelease tag
  gives its package version (1.1.0 for v1.1.0-rc.1) instead of 0.0.0.
- Release: a manual run publishes to GitHub Releases and PyPI only on a tag, as nuget.org already did; from a branch
  it created a release named after the branch.
- C API: a DCC fit with no parameter (a static copula with Kendall and mvnorm) takes NULL in `ve_dcc_fit_estimates`,
  `_std_errors`, `_t_values` and `_p_values`, which the .NET binding passes for an empty array; `Estimate()` and
  `Coef()` threw there. An absurd count (`1 << 62` jobs, specs, regressors or forecast steps) is
  `VE_INVALID_ARGUMENT`, not `VE_INTERNAL_ERROR`. Forecast, simulation and DCC future regressors with NaN or Inf are
  rejected, as the fit's already were. `ve_sim_options_set_correlations` takes 0 or 1 only, like every other switch.
  `ve_dcc_options_create` and `ve_dcc_batch_create` set `*out` to NULL on failure; recording an error message never
  throws across the C interface; `ve_spec_set_fixed` leaves the spec unchanged when it fails; a NULL DCC parameter
  name says so.
- The Pearson goodness-of-fit test sorts the probabilities before binning, so a CDF one ulp out of order across a bin
  edge can no longer wrap the count of empty bins. `digamma` returns NaN below -1e6 instead of looping forever.
- Docs: the `ve_dcc_fit_criterion` count for a static ML copula, the mvlaplace singularity at z = 0, the
  gjrGARCH (skewed) and csGARCH (h >= 2) forecasts that follow rugarch rather than the mean of simulated paths, the
  C++ `DccOptions` and simulation reproducibility comments, and `Panel.FromSeries` regressors in the C# guide.

## [1.1.0] - 2026-11-19

### Added

- A .NET binding, the `VolEngine` package on nuget.org (.NET 10): every feature of the C API (univariate fits and
  grids over a panel, regressors, standard errors, criteria, diagnostics, forecasts, simulation; DCC, aDCC, FDCC and
  Copula-DCC fits, standard errors, forecasts and simulation) as records, immutable results and exceptions. The
  package carries the native library for linux-x64, linux-arm64, linux-musl-x64, linux-musl-arm64, osx-arm64,
  osx-x64 and win-x64, and is trimming and Native AOT compatible. Its public API is stable from this release. Guide:
  `docs/guide/csharp.md`.

- EWMA (RiskMetrics): `VE_MODEL_EWMA` (C), `ve::Model::ewma` (C++), `Spec("ewma")` (Python). sigma2_t =
  lambda sigma2_{t-1} + (1 - lambda) e_{t-1}^2, the iGARCH(1,1) with omega = 0. Parameters `omega` and `alpha1`
  (derived: 0 and 1 - lambda, NaN standard errors, not counted by the criteria) and `lambda` in [0, 1], estimated by
  maximum likelihood unless fixed (`fixed={"lambda": 0.94}` is RiskMetrics). Every mean, regressor, distribution,
  `var_init`, forecast (flat), simulation from a fit, diagnostic and DCC/copula margin of the other models applies.
  Order (1, 1) only; variance targeting and `ve_simulate_spec` are rejected, as in iGARCH. Validated against `arch`'s
  `EWMAVariance` (three cases).

### Changed

- A free iGARCH weight whose derived `beta_q` sits within 1e-8 of 0 counts as on the boundary, so its standard error
  is NaN instead of a Hessian step that leaves the domain (the EWMA `lambda` next to 1 follows the same rule).

## [1.0.0] - 2026-11-15

The C interface is frozen. The audit of its 133 functions is in `docs/superpowers/notes/2026-11-15-abi-audit-6c.md`.

### Changed (breaking, the last ones)

- Criteria: `ve_fit_loglik` and `ve_fit_criterion(f, which, scale, out)` replace `ve_loglik`, `ve_aic`, `ve_bic`,
  `ve_hqic` and `ve_sic`; `VE_DCC_CRITERION_*` is now `VE_CRITERION_*` and serves both `ve_fit_criterion` and
  `ve_dcc_fit_criterion`. C++: `ve::Criterion` replaces `ve::DccCriterion`, and `Fit::criterion` is new.
- `ve_dcc_fit_standard_errors` is now `ve_dcc_standard_errors`.
- A DCC fit lists every parameter of its model, as a univariate fit does: with a fixed parameter (`DccSpec(fixed=...)`),
  `ve_dcc_fit_param_count`, `_name`, `_estimates`, `_std_errors`, `_t_values` and `_p_values` include it, with its value
  and NaN statistics, so positions follow `parameter_names` whatever is fixed. It used to be left out of the table.
  The criteria still count the estimated parameters only.
- A DCC fit made with `ve_dcc_options_set_standard_errors(o, 0)` answers `ve_dcc_fit_std_errors`, `_t_values` and
  `_p_values` with `VE_INVALID_ARGUMENT`, as the univariate fit does, not with `VE_OK` and NaN. Python: `DccFit.coef()`
  raises `VolEngineError` on such a fit, and `DccFit.estimate` reads its estimates. C++: the same, as an `ve::Error`.

### Fixed

- The log-likelihood, residuals and sigma at the estimate no longer depend on the standard errors option: both settings
  evaluate them in `double` (the Dual pass only supplies the scores), so they agree bit for bit. The inference costs
  one `double` pass more (0.4 to 2.5% of a fit).
- The Windows import libraries: the release `.lib` (MSVC) and the `.dll.a` (MinGW) both record `libvolengine.dll`, and
  the C++ wrapper saves the downloaded DLL under that name for both compilers.

### Added

- `include/volengine/volengine.abi`, one line per declaration of `volengine.h`, generated by
  `scripts/abi_contract.py` and checked by ctest and CI against the header and, after the first stable release,
  against the last release (only additions pass). The Windows export list is made from it.
- An `abidiff` job compares the Linux x64 library with the libabigail dump attached to the last release
  (`volengine-abi-linux-x64.xml`); it fails on an incompatible change.
- `docs/guide/python.md` and `docs/guide/cpp.md`; the Python examples of the README and the guide run as tests.
- Documentation of the DCC orders, groups, parameter names and criterion count in `volengine.h`.
- A header-only C++23 wrapper over the C API in `cpp/` (`#include <volengine/volengine.hpp>`, namespace `ve`): RAII
  handles, `std::expected` errors and `std::span` views, for fits, diagnostics, forecasts, simulations, DCC and
  Copula-DCC. A CMake project consumes it through `FetchContent` (`SOURCE_SUBDIR cpp`) or `find_package(volengine)`;
  the native library comes from the release asset of the tag, or from `VOLENGINE_NATIVE_LIB` for a local build. The
  wrapper calls every `VE_API` function (checked by `cpp/scripts/check_abi_coverage.py`).

## [0.9.0] - 2026-11-15

### Added

- `pip install volengine` without a compiler on Windows x64, macOS arm64 and x64, and Linux glibc (2.25
  or newer) and musl (musllinux_1_2) on x64 and arm64. One `py3-none-<platform>` wheel per platform serves every Python from
  3.11; the library inside has a static C++ runtime and runs on the baseline CPU of its architecture, with the SIMD
  paths chosen at run time.
- A tag `v*` publishes the wheels to PyPI and the native libraries (`volengine-native-<os>-<arch>.<ext>`, plus
  `.dll.a` and `.lib` for Windows) with an SBOM to a GitHub Release; a tag with a `-` is a prerelease.
- `python/scripts/build_native.py`, `check_native_deps.py` and `build_wheel.sh`: build the library for a wheel, fail
  if it needs a compiler runtime a clean machine lacks, and fail if the wheel tag claims an older system than the
  library needs.
- `ve_fit_options_set_standard_errors(o, on)` (Python: `FitOptions(standard_errors=...)`, default on). Off, a fit skips
  the inference at the estimate (Hessian and robust standard errors, scores, Nyblom), about a tenth of its time; the
  standard-error, t, p and Nyblom accessors then return `VE_INVALID_ARGUMENT`. Estimates, log-likelihood, criteria,
  series, forecasts, simulations and DCC fits from it are unchanged.
- Python: `Fit.estimate`, the estimates without the standard errors.

### Changed

- The `double` log-likelihood and the `Dual` one differ by rounding. The `double` instantiation reads the norm, std and
  ged terms of the univariate likelihood, and the Cholesky, the solve and the log determinant of the DCC, from SIMD
  kernels (Google Highway: the vector `log`, `log1p` and `exp`, sums by lanes, fused multiply-adds on AVX2); the
  `Dual` one keeps the scalar libm path. The two agree to 1e-14 relative on the log-likelihood in the tests, and the
  variance and the residuals stay bit for bit. Results on the same machine do not depend on the number of threads;
  across machines with different instruction sets (SSE4, AVX2, AVX-512) the results agree only within a tolerance,
  not bit for bit.
- The L-BFGS optimizer keeps 40 curvature pairs instead of 10, so fits with up to a few dozen parameters behave like
  full BFGS: about half the evaluations, and every fit of the TCC grid now ends `CONVERGED`. The overparametrized fits
  of that grid have several local optima; on some series the new path ends at another one (higher or lower).
- A step is flat only when it decreases f by less than 1e-15·max(1, |f|) (was 1e-12). On an ill-conditioned ridge,
  where every step decreases f by a little, a fit no longer ends `CONVERGED` early on such a step with a gradient up to
  √tol; it goes on until the gradient is below `tol`. A step that cannot decrease f at all (a failed line search, or
  a flat step) still ends the fit `CONVERGED` when the gradient is below √tol·max(1, |f|). The `rugarch` apARCH
  variance-regressor comparison now passes at the default tolerance.
- log Γ comes from Boost.Math in `double` instead of `std::lgamma`, which writes the global `signgam` (a data race
  between the pool's threads). Values move by about an ulp.
- Batches run one fit per task, the specs with the most parameters first: the batch no longer ends on a block of
  costly fits.
- Copula fits with a free Student shape: each point of the shape profile starts from the estimate of the nearest
  point already evaluated, and the maximum is the point Brent evaluated (no final re-fit). The shape and the other
  parameters move within the bracketing precision (shape to 1e-4 relative, log-likelihood to 1e-8 relative); the
  `rmgarch` comparisons hold at their tolerances.

- The derivative of the ghyp log-likelihood in lambda (and in the shape of the Bessel orders) comes from a trapezoid sum
  instead of a Richardson difference: it moves by up to ~1e-10 relative (the old one was off by that much), so an
  optimizer path can end at another digit.

### Fixed

- Results no longer depend on the thread's x87 precision. `log_bessel_k` (nig, ghyp, mvlaplace, GOF, quantiles) and the
  p-values of the diagnostics computed in long double, which on MinGW differs between a pool thread and the Python main
  thread; they now compute in `double`, and the values of fits with those distributions move by about an ulp.
- The moments of the ged and the sged at large shapes. The shifted moment, `abs_moment` of the sged and P(z < 0) lost
  the edge of the density (1.5% off at shape 1000, 3e-6 at 20) and are now exact to 1e-11 for integer powers and 2e-10
  for the others, which moves the apARCH and
  fGARCH persistence of fits with those distributions at shapes above 6. Their derivative in the shape was NaN from
  shape ~100 and is finite. The cdf of the ged and the sged was that of the centre inside the support from shapes of
  ~100 (wrong by 2e-4 at 100, 7e-3 at 200 and 0.1 at 500, where P(z < 0) read 0.61 for 0.5) and is exact.
- The static Student copula (Kendall and ML) with independent series (the profile maximum on the ceiling of the shape)
  failed with a non-finite log-likelihood. A profiled shape of the static copula is now accepted on the whole interval [4, 50];
  only a fixed shape outside (4, 50) fails its spec. (The DCC and aDCC copulas fix the shape in an inner fit that wants
  it inside (4, 50), so their profile maximum stays inside.)

### Performance

- SIMD inside one evaluation (plan 5b; Google Highway 1.2.0, run-time dispatch in a generic x86-64 build). The TCC
  pipeline with all threads (the 1080-fit univariate grid and the 9 DCC fits over 21 series) takes 1.88 s instead of
  5.05 s: the univariate grid 0.48 of its time and the DCC grid 0.25. Three changes: the norm, std and ged terms of the
  likelihood come from one vector loop in every model, fused with the adjoint's partials; the DCC recursion, Cholesky
  and quadratic form run in vector lanes over observations; and the DCC gradient no longer uses forward-mode AD on the
  matrices: tangents of the correlation recursion plus a per-observation pullback (every DCC, aDCC and FDCC fit with
  the mvnorm, mvt or mvlaplace takes 0.15 to 0.25 of its time). The two-stage standard errors of the DCC take 0.47.
  See [docs/performance/profile-5b.md](docs/performance/profile-5b.md).
- sGARCH, iGARCH, gjrGARCH and eGARCH fits with norm, std or ged innovations and an ARMA mean (no regressors,
  ARCH-in-mean, ARFIMA or variance targeting) take the gradient of the log-likelihood from a hand-written adjoint
  instead of forward-mode AD, in the optimizer and in the Hessian of the standard errors: the TCC grid takes 0.71 of
  its time with one thread (31.3 s to 22.0 s; 4.24 s to 2.72 s with all threads). Estimates move only in the last
  digits; other specs are unchanged.
- Copulas (plan 5c-1): the DCC, aDCC and FDCC copula fits take the tangent gradient of plan 5b, the Student shape
  profile warm-starts from the nearest evaluated point, and Kendall's tau-b is Knight's O(T log T) algorithm with the
  bits of the pair count. At m = 21 and T = 5000, one thread: the DCC-t copula goes from 36.4 s to 3.47 s, the aDCC-t
  from 36.6 s to 3.56 s, the DCC-mvnorm copula from 0.945 s to 0.197 s and the static Kendall mvnorm from 3.79 s to
  0.160 s. See [docs/performance/profile-5c.md](docs/performance/profile-5c.md).
- Distributions (plan 5c-2), one thread: the nig and ghyp fits, their GOF and their quantiles are 2 to 5 times faster
  because Boost computes in `double` (no x87 long double; the mvlaplace DCC log-likelihood gains 11%); the ghyp gradient in lambda comes from one trapezoid
  sum for both slopes of log K, so a ghyp fit takes 141 ms instead of 1785 ms (the nig fit 30.5 ms instead of 79.6 ms)
  and its GOF 6.5 ms instead of 28.9 ms. A Copula-DCC simulation with nig or ghyp margins takes the quantile from a
  table built once per margin: 2.7 to 3.5 us per call instead of 109 to 143 us. See
  [docs/performance/profile-5c.md](docs/performance/profile-5c.md).
- Standard errors of the DCC (plan 5c-3): the columns of the cross terms of every fit of a batch (and of a deferred
  `ve_dcc_fit_standard_errors` call) are spread over the pool, so the grid of 9 specs at m = 21, T = 5000 with
  standard errors takes 3.39 s instead of 5.09 s with all threads, and a refiltered series updates only its row and
  column of the moments of the panel. Results are those of before to the bit. See
  [docs/performance/profile-5c.md](docs/performance/profile-5c.md).
- The TCC grid (1080 fits) on the Ryzen 7 5700X: see [docs/performance/benchmarks.md](docs/performance/benchmarks.md)
  and [docs/performance/profile-5a.md](docs/performance/profile-5a.md) for the measurements of each change.

### Changed

- The cdf of the univariate distributions computes in `double` rather than Boost's promoted `long double`, so that
  it gives the same bits on every thread (on MinGW the Python main thread runs the x87 unit at 53 bits and the pool
  threads at 64). Values can move by an ulp.

- Python: `Context.fit` takes its optional arguments as `**keywords` (a `TypedDict`), to carry `mxreg` and `vxreg`
  without a longer signature. The call syntax is the same (`options=`, `starts=`, `lengths=` by name); an unknown
  keyword is a `TypeError`.
- With `vxreg`, the persistence stays limited to P < 1 (as everywhere here), while `rugarch` drops that restriction when
  the model has a `vxreg`. Estimates inside the stationary region agree; the difference is recorded in
  [docs/guide/validation.md](docs/guide/validation.md).
- **Breaking (Python):** `loglik` and the information criteria are members of `Fit` (`fit.loglik`, `fit.aic()`,
  `fit.bic()`, `fit.hqic()`, `fit.sic()`, each criterion with `per_obs=`). The module functions `ve.loglik`,
  `ve.aic`, `ve.bic`, `ve.hqic` and `ve.sic` are removed, with no alias.

### Added

- DCC and aDCC of orders 1 to 6, and FDCC(1,1) with user-defined groups, with multivariate normal, Student-t
  and Laplace innovations. Batched fits share the standardized univariate panel and are bit-identical across
  thread counts. C API DCC handles and Python `DccSpec`, `DccFit`, `Context.fit_dcc` expose
  fixed and start parameters, the full log-likelihood and information criteria, read-only `(T, m, m)`
  correlation and lazy covariance paths, and optional or deferred two-stage Hessian and robust standard errors.
  Public conventions follow the papers; an internal `rmgarch` convention supports reference validation.
  Standard errors with univariate external regressors are unsupported. Benchmarks: `BM_DccFit`, `BM_DccGrid`,
  `BM_DccStandardErrors`.
- Copula-DCC as an option of the DCC spec: Gaussian and Student copulas with static (Kendall or ML), DCC or aDCC
  correlation, on parametric (each fit's cdf) or empirical (rank) margins, in the same batches and results as the
  DCC. The Student shape is the maximum of a profile likelihood. Two-stage standard errors cover the copula rows.
  C API: `VE_DCC_MODEL_STATIC`, `ve_dcc_spec_set_copula`, `ve_dcc_spec_set_static_method`. Python:
  `DccSpec(model="static", copula=..., static_method=...)`. Validated against `rmgarch`'s `cgarchfit`, with the
  public convention following the papers (see `docs/guide/validation.md`). Benchmarks: `BM_CopulaFit`,
  `BM_CopulaGrid`, `BM_CopulaStandardErrors`.
- DCC forecast and simulation, for every DCC, aDCC, FDCC and Copula-DCC fit made by a batch (the fit keeps the final
  state of each series and of its correlation recursion, so neither needs the univariate fits afterwards). Forecast:
  R, H, sigma and mean for h = 1..H as `rmgarch`'s `dccforecast`: the exact recursion for h = 1 and, for h >= 2,
  R_h = (1 - k^(h-1)) Rbar + k^(h-1) R_1 with the total persistence k (the aDCC and the FDCC extend it where
  `rmgarch` refuses or gives no value; see [docs/guide/validation.md](docs/guide/validation.md)). Simulation: P
  multivariate paths of H steps with the shock of the mvnorm, the mvt or the mvlaplace, through the copula and the
  margins of each series in a Copula-DCC, with R and H per path on request. **The seed rule is that of the univariate
  simulation:** path p reads only `Stream(seed, p)`, the same paths for any number of threads and the first paths and
  steps of a longer run, and **a seed reproduces paths only within the same version of VolEngine**. C API:
  `ve_dcc_future_create`, `ve_dcc_future_set`, `ve_dcc_future_free`, `ve_dcc_forecast_create`,
  `ve_dcc_forecast_horizon`, `ve_dcc_forecast_series`, `ve_dcc_forecast_correlation`, `ve_dcc_forecast_covariance`,
  `ve_dcc_forecast_sigma`, `ve_dcc_forecast_mean`, `ve_dcc_forecast_free`, `ve_sim_options_set_correlations`,
  `ve_simulate_dcc`, `ve_dcc_sim_paths`, `ve_dcc_sim_horizon`, `ve_dcc_sim_series`, `ve_dcc_sim_returns`,
  `ve_dcc_sim_sigma`, `ve_dcc_sim_correlation`, `ve_dcc_sim_covariance`, `ve_dcc_sim_free`. Python:
  `DccFit.forecast(horizon, mxreg=, vxreg=)` returning a `DccForecast`, and `Context.simulate(dcc_fit, horizon=,
  paths=, seed=, correlations=, mxreg=, vxreg=)` returning a `DccSimulation`; the per-series regressors are
  sequences with one entry (or `None`) per series. Benchmarks: `BM_DccForecast`, `BM_DccSimulate`.
- Diagnostics reported by `rugarch`'s `show()`: weighted Ljung-Box on standardized residuals or their squares,
  weighted ARCH-LM, Nyblom stability with individual and joint critical values, Engle-Ng sign bias and adjusted
  Pearson goodness of fit for all eight distributions. C API: `ve_fit_ljung_box`, `ve_fit_arch_lm`,
  `ve_fit_nyblom`, `ve_nyblom_critical`, `ve_fit_sign_bias`, `ve_fit_gof`. Python: `Fit.ljung_box`, `Fit.arch_lm`,
  `Fit.nyblom`, `Fit.sign_bias`, `Fit.gof` and the `DiagnosticResult`, `SignBias`, `Nyblom` result types.
  Fits retain only the Nyblom statistics, using the scores already computed for standard errors;
  `ve_fit_nyblom` also reports which parameters were estimated (1) or fixed and derived (0), the count that
  `ve_nyblom_critical` takes.
  Benchmarks: `BM_Diagnostics/<series>/<test>`.
- Forecast: sigma and the conditional mean h = 1..H steps after the last observation of a fit, for every model,
  distribution and option, as `rugarch`'s `ugarchforecast` (`n.roll = 0`), regressors over the horizon included.
  C API: `ve_forecast_create`, `ve_forecast_horizon`, `ve_forecast_sigma`, `ve_forecast_mean`, `ve_forecast_free`.
  Python: `Fit.forecast(horizon, mxreg=, vxreg=)` returning a `Forecast`. In eGARCH, apARCH and fGARCH, sigma for
  h > 1 is `rugarch`'s quantity, not sqrt(E[sigma2]). Validated against `rugarch` (108 cases) and `arch` (15 cases)
  in [docs/guide/validation.md](docs/guide/validation.md); benchmarks `BM_Forecast/<series>/h:60`.
- Simulation: P paths of H steps (returns and sigma, P x H, row = path) from a fit, continuing after its last
  observation (`rugarch`'s `ugarchsim`), or from a spec with every parameter fixed, from its unconditional state
  (`ugarchpath`); every model, distribution and option, regressors over the horizon the same for every path. Philox
  RNG, one stream per path: the same paths for any number of threads, and the first paths and steps of a longer run.
  **A seed reproduces paths only within the same version of VolEngine**; this file says when they change. C API:
  `ve_sim_options_create`, `ve_sim_options_set_regressors`, `ve_sim_options_free`, `ve_simulate_fit`,
  `ve_simulate_spec`, `ve_sim_paths`, `ve_sim_horizon`, `ve_sim_returns`, `ve_sim_sigma`, `ve_sim_free`. Python:
  `Context.simulate(source, horizon=, paths=, seed=, mxreg=, vxreg=)` returning a `Simulation`. Validated
  statistically in [docs/guide/validation.md](docs/guide/validation.md#simulation); benchmarks `BM_Simulate`,
  `BM_Sampler`.
- A fit keeps a small final state (the last max(m, n, p, q) residuals, variances, csGARCH trends and mean
  deviations; every deviation with ARFIMA) for the forecast; no number of a fit changes.
- ARFIMA mean for every model and distribution, following `rugarch`'s `arfima = TRUE`: the parameter `arfima`
  (d in (1e-8, 0.5), after `ma`), a truncated fractional difference of the ARMA residuals at O(T^2) per evaluation.
  It cannot be combined with `archm`. A spec without it gives the numbers of before. C API: `ve_spec_set_arfima`.
  Python: `Spec(arfima=True)`. Validated against `rugarch` ([docs/guide/validation.md](docs/guide/validation.md)),
  with benchmarks `BM_Loglik/arfima_sgarch11_norm` and `BM_FitOne/arfima_sgarch11_norm`, not yet measured on the
  benchmark machine.
- Mean regressors (`mxreg`), variance regressors (`vxreg`) and ARCH-in-mean (`archm`, on the standard deviation or the
  variance) for every model, following `rugarch`: the parameters `archm`, `mxreg1..` and `vxreg1..` (up to 8 of each
  kind) in the `rugarch` order, with start values from the least squares of the series on the mean regressors and
  standard errors as for the other parameters. A spec without regressors and without `archm` gives the numbers of
  before. C API: `ve_spec_set_mxreg`, `ve_spec_set_vxreg`, `ve_spec_set_archm`, `ve_batch_set_regressors` and
  `VE_MAX_REGRESSORS`. Python: `Spec(mxreg=, vxreg=, archm=)` and `ctx.fit(spec, y, mxreg=X, vxreg=V)`. Validated
  against `rugarch` and `arch` ([docs/guide/validation.md](docs/guide/validation.md)).
- `ad::Dual<59>`: the widest layout now holds 59 parameters (`kMaxParameterCount`); the `Dual` buckets are 8, 16, 32
  and 59. This replaces the `Dual<42>` below.
- Benchmarks for the regressor and ARCH-in-mean cases (`BM_Loglik` and `BM_FitOne`), not yet measured on the benchmark
  machine ([docs/performance/benchmarks.md](docs/performance/benchmarks.md)).
- Fixed parameters, start values and variance targeting, as `fixed.pars`, `start.pars` and `variance.targeting` of
  `rugarch`. Each parameter is estimated, fixed or derived (the iGARCH `beta_q`, and `omega` under variance
  targeting); fixed and derived ones have NaN standard errors and do not count in the criteria. A spec with every
  parameter fixed is evaluated at that point without the optimizer. Incompatible specs are
  `VE_INVALID_ARGUMENT`: variance targeting in iGARCH and csGARCH (`rugarch` ignores it in csGARCH and gives
  `omega = 0` in iGARCH; here both are errors on purpose), fixing a derived parameter, and a csGARCH with `eta21`
  fixed and a `beta` free. C API: `ve_spec_set_fixed`, `ve_spec_set_start`, `ve_spec_set_variance_targeting` and
  `ve_spec_validate`. Fixed wins over a start value for the same name, in either order of the setters, so
  `Spec(fixed=..., start=dict(zip(fit.names, fit.coef().estimate)))` is a valid warm start. Python:
  `Spec(fixed=, start=, variance_targeting=)`, which pickles, deep-copies and works with `dataclasses.asdict`. Validated against `rugarch` and `arch`
  ([docs/guide/validation.md](docs/guide/validation.md)).
- Skewed innovations: skewed Student-t (`sstd`), skewed GED (`sged`), normal inverse Gaussian (`nig`),
  generalized hyperbolic (`ghyp`) and Johnson SU (`jsu`), for every model and ARMA mean. C API: `VE_DIST_SSTD`,
  `VE_DIST_SGED`, `VE_DIST_NIG`, `VE_DIST_GHYP` and `VE_DIST_JSU`, values 3 to 7. Python: `Spec(dist=)` with the
  new names; the coefficient table gains `skew` and, for `ghyp`, `ghlambda`.
- Dependency on the headers of Boost.Math (BSL-1.0, fetched by CMake), used only in `src/dist/bessel.cpp`.
- `ad::Dual<42>`: the widest layout now holds 42 parameters (fGARCH-ALLGARCH, ARMA(6,6), orders (6,6), `ghyp`);
  the `Dual` buckets are 8, 16, 32 and 42.
- Validation of the skewed distributions against `rugarch` and `arch`. Three ALLGARCH cases with an asymmetric
  shock stay visible as strict xfails in the oracle suite (`KNOWN_DIVERGENCES`): the optimizer stops at a point
  with a lower log-likelihood than `rugarch`'s ([docs/guide/validation.md](docs/guide/validation.md)).
- Benchmarks for the skewed distributions (`BM_Loglik`, `BM_FitOne`, `BM_LogBesselK`, `BM_TccGridFull`) and an
  `sGARCH(1,1)` `sstd` row against `arch`; they are not measured yet
  ([docs/performance/benchmarks.md](docs/performance/benchmarks.md)).
- GARCH variants next to sGARCH: eGARCH, gjrGARCH, apARCH, iGARCH, csGARCH and fGARCH with its eight submodels
  (garch, tgarch, avgarch, ngarch, nagarch, aparch, gjrgarch, allgarch), each with the three distributions and
  an ARMA mean. C API: `VE_MODEL_*` constants 0 to 13. Python: `Spec(model=, submodel=)`.
- Per-model Hessian steps, boundary rules for standard errors and start values; the fGARCH ALLGARCH shock
  moments use Gauss-Legendre quadrature.
- Validation of the new models against `rugarch` and `arch` ([docs/guide/validation.md](docs/guide/validation.md)).
- Benchmarks for the new models (`BM_Loglik`, `BM_FitOne` per variant and `BM_TccGridAllModels`); they are not
  measured yet ([docs/performance/benchmarks.md](docs/performance/benchmarks.md)).
- Generic model engine: ARMA(m,n)-sGARCH(p,q) with orders up to `VE_MAX_ORDER` (6), a template log-likelihood over
  the scalar type, and forward-mode automatic differentiation (`ad::Dual<N>`) for exact gradients. The fit driver
  picks the narrowest `Dual` that holds the parameters.
- Student-t (`std`) and GED (`ged`) innovations next to the normal; the `rugarch` ARMA presample convention.
- Standard errors near the boundary: alpha or beta below 1e-8, and a `std` shape above 1e8, get NaN standard
  errors and the other parameters are computed on the remaining block.
- C API: `VE_DIST_STD`, `VE_DIST_GED`, `VE_MAX_ORDER` and `ve_spec_set_arma`. Python: `Spec(arma=, dist=)` and
  typed `Context.fit` overloads, checked by mypy.
- Oracle grid against `rugarch` (orders, ARMA and distributions) and `arch` (orders and distributions, no ARMA) in
  [docs/guide/validation.md](docs/guide/validation.md).
- C++ Google Benchmark suite ([docs/performance/benchmarks.md](docs/performance/benchmarks.md)) and a Python
  benchmark against `arch` on matched work ([docs/performance/python-vs-arch.md](docs/performance/python-vs-arch.md)).
- sGARCH with normal innovations: maximum-likelihood fit with L-BFGS on a constrained-to-unconstrained
  reparametrization, with or without a mean. Estimates, log-likelihood and standard errors are equivariant to
  the units of the data.
- Hessian and robust (QML sandwich) standard errors, t-values and p-values; log-likelihood, AIC, BIC, HQIC and
  SIC, per observation (default, `rugarch`) or as totals (`arch`).
- Two variance initializations: `sample` (default, `rugarch`) and `backcast` (`arch`).
- Batch fitting over specs x series on a thread pool owned by the context. Panels may be rectangular or ragged
  (a start and a length per series) and in any stride order. Results are bit-identical for any thread count,
  and a series that does not converge or is too short yields its own fit status without failing the batch.
- C API (`include/volengine/volengine.h`): opaque handles, setters instead of public structs, `ve_status` on
  every fallible call, thread-local `ve_last_error()`, no exception crossing the boundary. A context accepts
  concurrent calls and at most `VE_MAX_THREADS` (1024) threads.
- Python package `volengine` over ctypes: `Context.fit`, `Spec`, `FitOptions`, `Fit`, `CoefTable`, and the
  the `Fit.loglik` property and the `Fit.aic()`, `bic()`, `hqic()` and `sic()` methods; arrays are passed to the
  library without copying.
- Validation against `rugarch` and `arch` with committed fixtures ([docs/guide/validation.md](docs/guide/validation.md)).
- Style checker (`tests/style/check_style.py`), clang-tidy configuration and the CI workflow.

### Fixed
- `std` fits on near-normal data no longer fail at a large shape (the density lost precision for large nu).
