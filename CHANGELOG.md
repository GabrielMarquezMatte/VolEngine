# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until 1.0 the C ABI may break between 0.x releases (`VE_ABI_VERSION` is 0 and the Python package refuses a
library whose ABI version differs).

## [Unreleased]

### Changed

- **Breaking (Python):** `loglik` and the information criteria are members of `Fit` (`fit.loglik`, `fit.aic()`,
  `fit.bic()`, `fit.hqic()`, `fit.sic()`, each criterion with `per_obs=`). The module functions `ve.loglik`,
  `ve.aic`, `ve.bic`, `ve.hqic` and `ve.sic` are removed, with no alias.

### Added

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
