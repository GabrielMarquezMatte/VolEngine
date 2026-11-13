# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until 1.0 the C ABI may break between 0.x releases (`VE_ABI_VERSION` is 0 and the Python package refuses a
library whose ABI version differs).

## [Unreleased]

### Changed

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

- Forecast: sigma and the conditional mean h = 1..H steps after the last observation of a fit, for every model,
  distribution and option, as `rugarch`'s `ugarchforecast` (`n.roll = 0`), regressors over the horizon included.
  C API: `ve_forecast_create`, `ve_forecast_horizon`, `ve_forecast_sigma`, `ve_forecast_mean`, `ve_forecast_free`.
  Python: `Fit.forecast(horizon, mxreg=, vxreg=)` returning a `Forecast`. In eGARCH, apARCH and fGARCH, sigma for
  h > 1 is `rugarch`'s quantity, not sqrt(E[sigma2]). Validated against `rugarch` (108 cases) and `arch` (15 cases)
  in [docs/guide/validation.md](docs/guide/validation.md); benchmarks `BM_Forecast/<series>/h:60`.
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
