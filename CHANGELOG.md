# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Until 1.0 the C ABI may break between 0.x releases (`VE_ABI_VERSION` is 0 and the Python package refuses a
library whose ABI version differs).

## [Unreleased]

### Added
- sGARCH(1,1) with normal innovations: maximum-likelihood fit with L-BFGS on a constrained-to-unconstrained
  reparametrization, with or without a mean.
- Hessian and robust (QML sandwich) standard errors, t-values and p-values; log-likelihood, AIC, BIC, HQIC and
  SIC, per observation (default, `rugarch`) or as totals (`arch`).
- Two variance initializations: `sample` (default, `rugarch`) and `backcast` (`arch`).
- Batch fitting over specs x series on a thread pool owned by the context. Panels may be rectangular or ragged
  (a start and a length per series) and in any stride order. Results are bit-identical for any thread count,
  and a series that does not converge or is too short yields its own fit status without failing the batch.
- C API (`include/volengine/volengine.h`): opaque handles, setters instead of public structs, `ve_status` on
  every fallible call, thread-local `ve_last_error()`, no exception crossing the boundary.
- Python package `volengine` over ctypes: `Context.fit`, `Spec`, `FitOptions`, `Fit`, `CoefTable`, and the
  `loglik`, `aic`, `bic`, `hqic` and `sic` functions; arrays are passed to the library without copying.
- Validation against `rugarch` and `arch` with committed fixtures ([docs/guide/validation.md](docs/guide/validation.md)).
- Style checker (`tests/style/check_style.py`), clang-tidy configuration and the CI workflow.
