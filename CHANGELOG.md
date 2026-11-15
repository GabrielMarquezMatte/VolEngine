# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until 1.0 the C ABI may break between 0.x releases (`VE_ABI_VERSION` is 0 and the Python package refuses a
library whose ABI version differs).

## [Unreleased]

## [0.9.0] - 2026-11-15

### Added

- `pip install volengine` without a compiler on Windows x64, macOS arm64 and x64, and Linux glibc (manylinux_2_28)
  and musl (musllinux_1_2) on x64 and arm64. One `py3-none-<platform>` wheel per platform serves every Python from
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
