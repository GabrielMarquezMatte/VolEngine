# Architecture

A map of how the codebase fits together, not a manual. Start here, then follow the file names into the source:
the code carries its own meaning through naming and flow. What it deliberately does not carry is the evidence
behind a decision (measurements, rejected approaches); that lives in `docs/`, and
[STYLEGUIDE.md](STYLEGUIDE.md#comments) says why.

This document describes what exists today: an ARMA(m,n)-sGARCH(p,q) log-likelihood with normal, Student-t and
GED innovations, a template over the scalar type so that automatic differentiation gives its gradient; and, on
top of it, sGARCH(1,1) with normal innovations fitted in batches behind a C API and a Python package.

## Layers

```
  Python (python/src/volengine, ctypes + numpy)
                 |
        C API  include/volengine/volengine.h        <-- the only stable contract
                 |
        src/capi   handles, error boundary
                 |
        src/fit    driver, batch, standard errors, criteria
          |         \
     src/model     src/optim       layout, likelihood | L-BFGS
          |
   src/mean  src/garch  src/dist   ARMA residuals, variance recursion, densities
          \     |     /
            src/ad                 dual numbers, digamma
              |
        src/core   context, spec, panel
```

Dependencies point one way: `core` <- `ad` <- `mean`, `garch`, `dist` <- `model` <- `fit` <- `capi`, and
`optim` <- `fit`. Nothing in `src/` includes from `src/capi/` except `capi` itself. Everything below the C API is
internal and may change in any release; only the header is a contract, and the Python package is a consumer of it
like any other wrapper would be.

## Where each responsibility lives

| Path | Responsibility |
|---|---|
| `include/volengine/volengine.h` | The C API: `ve_` functions, `VE_` constants, opaque handles, no public struct |
| `src/capi/handles.hpp` | The handle types (`ve_context`, `ve_spec`, `ve_fit_options`, `ve_panel`, `ve_batch`, `ve_fit`), each wrapping one core value |
| `src/capi/error.hpp`, `error.cpp` | Thread-local `ve_last_error()`, `set_error`, and `guard`, the one place exceptions are caught |
| `src/capi/context_api.cpp`, `spec_api.cpp`, `panel_api.cpp`, `batch_api.cpp`, `fit_api.cpp`, `version_api.cpp` | The `extern "C"` functions, one file per handle family |
| `src/core/context.hpp`, `context.cpp` | `ve::Context`: the thread pool and thread count |
| `src/core/spec.hpp` | `ve::Spec` (model, order, distribution, mean) and `ve::FitConfig` (iterations, tolerance, variance initialization): copyable values |
| `src/core/panel.hpp` | `ve::PanelView`: a borrowed pointer, T, N, element strides and optional per-series start and length |
| `src/core/var_init.hpp` | The variance initialization enum (`Sample`, `Backcast`) |
| `src/ad/dual.hpp`, `special.hpp`, `special.cpp` | `ve::ad::Dual<N>`: forward-mode dual numbers and their elementary functions; `digamma` |
| `src/mean/arma.hpp` | ARMA(m,n) residuals, a template over the scalar type |
| `src/garch/sgarch.hpp` | sGARCH(p,q) variance recursion and its presample, a template over the scalar type |
| `src/dist/densities.hpp` | Log-densities of the standardized normal, Student-t and GED, templates over the scalar type |
| `src/model/layout.hpp`, `layout.cpp` | `Layout`: parameter blocks, offsets, count and rugarch names of a `Spec` |
| `src/model/likelihood.hpp`, `likelihood.cpp` | `loglik<T>`: the log-likelihood with its per-observation terms, residuals and variances, and the backcast; instantiated for `double` and `Dual<8>`, `Dual<16>`, `Dual<32>` |
| `src/model/parameter_map.*`, `start_values.*` | Layout-driven map between the unconstrained and natural parameter spaces, a template over the scalar type; validity check and start values |
| `src/optim/lbfgs.hpp` (+ `curvature_memory.hpp`, `line_search.hpp`, `vector_math.hpp`) | L-BFGS minimizer, a template over the objective |
| `src/fit/fit.hpp`, `fit.cpp` | `fit_one`: one series, one spec |
| `src/fit/standard_errors.*`, `cholesky.*` | Hessian and robust standard errors; Cholesky solves, no explicit inverse |
| `src/fit/criteria.*` | Log-likelihood, AIC, BIC, HQIC, SIC |
| `src/fit/batch.hpp`, `batch.cpp` | `check_batch` and `run_batch`: jobs on the thread pool |
| `src/fit/fit_result.hpp` | `ve::FitResult`: status, estimates, standard errors, series, all by value |
| `python/src/volengine/_native.py` | Library discovery (`VOLENGINE_LIB`), ABI check, prototypes, handle ownership |
| `python/src/volengine/_spec.py`, `_fit.py`, `api.py` | `Spec` and `FitOptions`, `Fit` and the criteria, `Context.fit` |
| `tests/unit/{ad,capi,dist,fit,garch,mean,model,optim}` | GoogleTest suites mirroring `src/`; `c_header_check.c` compiles the header as C11 |
| `tests/style/check_style.py` | Style rules that can be checked without a compiler; runs as the `style` ctest |
| `python/tests` | Python tests, including `test_oracles.py` |
| `tests/reference` | Data generator, oracle scripts and the committed fixtures; see [its README](tests/reference/README.md) |
| `docs/guide/validation.md` | Oracle conventions, tolerances and observed differences |

## Stability rules of the C API

Each rule answers a concrete break that happened in the author's earlier libraries (ExcelReader, FixedWidthParser).

| What went wrong before | Rule here |
|---|---|
| A hidden "current" cursor in a handle prevented parallel use | A `ve_fit` is immutable and may be read from several threads. `ve_context` accepts concurrent calls; `ve_context_set_threads` waits for the batches running on it. A spec is copied when a fit runs, so changing it later changes no existing result. |
| Loose positional parameters and `_ex` twins of functions | No public struct. Configuration goes through setters on opaque handles. New configuration is a new function, never a new parameter or an `_ex`. |
| A concept renamed when it was generalized | Names follow the concept (`fit`, not `fit_sgarch`). A model is an enum value, so a new model is a new value, not a new API. |
| A convention assumed silently | Every convention where oracles differ is an explicit option with a documented default (`var_init`, information criteria scale). |
| An enum value removed | Enums are `typedef int32_t` plus `#define`; values are only added, never removed or renumbered. |
| Two parallel code paths duplicating everything | One path: a single series is a batch of one. |
| Mutable global state | None. The thread pool and execution settings live in `ve_context`, passed to each call. |
| A cursor valid after its parent was closed | Handles are independent: freeing one never invalidates another. |

Also: every fallible function returns `ve_status` and puts the message in `ve_last_error()`; no exception crosses
the boundary (`std::bad_alloc` becomes `VE_OUT_OF_MEMORY`, anything else `VE_INTERNAL_ERROR`); sizes use
`int32_t`, `size_t` and `double`. Semantic versioning, with `VE_ABI_VERSION` in the header and `ve_abi_version()`
at run time; the contract freezes at 1.0 and 0.x releases may break it.

## Anatomy of a `ve_fit_batch`

The Python `Context.fit` builds the handles below and then makes one call; any C client does the same.

1. **Handles.** A `ve_panel` borrows the caller's array (pointer, T, N, strides in elements, optional per-series
   start and length). A `ve_batch` holds the specs, the panel, optional `ve_fit_options` and the job list: pairs
   (spec index, series index). Wrappers build the Cartesian product of specs and series.
2. **Validation.** `ve_fit_batch` calls `check_batch` (`src/fit/batch.cpp`). Job indices out of range, ranges
   outside the panel, and a NaN or infinite value inside a series a job uses are argument errors: the whole call
   fails with `VE_INVALID_ARGUMENT`, a message in `ve_last_error()`, and no `ve_fit` is produced.
3. **Scheduling.** `run_batch` splits the jobs into contiguous blocks on the context's thread pool. Each job
   writes only its own slot of the result vector, and a job's computation does not depend on the thread count, so
   results are bit-identical for any number of threads on the same machine.
4. **One fit.** `fit_one` (`src/fit/fit.cpp`) checks the series is long enough (parameters + 2) and not constant,
   and minimizes the negative mean log-likelihood of the series standardized by its sample mean and standard
   deviation with L-BFGS in the unconstrained space (`src/model/parameter_map.cpp`), using the gradient of
   `model::loglik` through `ad::Dual<8>`; the fit therefore does not depend on the units of the data. At the
   estimate it recomputes the variance path and scores and derives Hessian and robust standard errors, with
   finite-difference steps relative to each parameter's scale.
5. **Status per fit.** A fit that does not converge, is too short or is degenerate carries its own status
   (`max_iter`, `failed`, `insufficient_data`); it does not fail the batch or the other fits.
6. **Publishing.** Results are moved into freshly allocated `ve_fit` handles, all created before any is written to
   the output array, so an exception leaves it untouched. The Python layer then wraps each pointer in a `Fit` that
   frees it.

Every `extern "C"` function runs inside `guard` (`src/capi/error.cpp`), the single place that turns exceptions
into statuses.

## Validation

sGARCH(1,1)-normal is compared with `rugarch` (default conventions) and `arch` (the `backcast` option) on
committed fixtures. The conventions, tolerances and observed differences are in
[docs/guide/validation.md](docs/guide/validation.md).
