# Architecture

A map of how the codebase fits together, not a manual. Start here, then follow the file names into the source:
the code carries its own meaning through naming and flow. What it deliberately does not carry is the evidence
behind a decision (measurements, rejected approaches); that lives in `docs/`, and
[STYLEGUIDE.md](STYLEGUIDE.md#comments) says why.

This document describes what exists today: ARMA(m,n)-GARCH(p,q) log-likelihoods for sGARCH, eGARCH, gjrGARCH,
apARCH, iGARCH, csGARCH and fGARCH (eight submodels) with normal, Student-t, GED and skewed innovations, fixed parameters, start values and variance targeting, a
template over the scalar type so that automatic differentiation gives their gradients; and, on top of them, those
models fitted in batches behind a C API and a Python package.

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
| `src/core/spec.hpp` | `ve::Spec` (model, order, distribution, mean, fixed and start values, variance targeting) and `ve::FitConfig` (iterations, tolerance, variance initialization): copyable values |
| `src/core/panel.hpp` | `ve::PanelView`: a borrowed pointer, T, N, element strides and optional per-series start and length |
| `src/core/var_init.hpp` | The variance initialization enum (`Sample`, `Backcast`) |
| `src/ad/dual.hpp`, `special.hpp`, `special.cpp` | `ve::ad::Dual<N>`: forward-mode dual numbers and their elementary functions; `digamma` |
| `src/mean/arma.hpp` | ARMA(m,n) residuals, a template over the scalar type |
| `src/garch/presample.hpp` | The pre-sample rule shared by the recursions: a state held at a value up to a time, and the expected value of a term before the first observation |
| `src/garch/sgarch.hpp`, `egarch.hpp`, `gjrgarch.hpp`, `csgarch.hpp` | The variance recursions of sGARCH (also iGARCH), eGARCH, gjrGARCH and csGARCH, templates over the scalar type |
| `src/garch/power_garch.hpp` | The recursion in sigma^power shared by apARCH and the eight fGARCH submodels |
| `src/dist/densities.hpp` | `Density<T>`, a `std::variant` of the log-density families, and `make_density`, `log_density`, `density_kink`: the one entry point the likelihood uses |
| `src/dist/symmetric.hpp`, `skewed.hpp`, `hyperbolic.hpp`, `johnson.hpp` | The families of the variant: norm, std and ged; sstd and sged (Fernandez-Steel); nig and ghyp; jsu. Templates over the scalar type |
| `src/dist/bessel.hpp`, `bessel.cpp` | `log_bessel_k` and its derivatives. Boost.Math is included only in `bessel.cpp`, so no other file sees it |
| `src/dist/moments.hpp`, `moments.cpp` | Expected absolute shock, P(z < 0) and the fGARCH news moments (Gauss-Legendre quadrature for the shifted and the asymmetric ones) |
| `src/model/layout.hpp`, `layout.cpp` | `Layout`: parameter blocks (gamma, eta1, eta2, power), offsets, count, `ParameterKind` of each parameter, rugarch names of a `Spec`, and the role of each parameter (estimated, fixed or derived; `estimated_count`) |
| `src/model/likelihood.hpp`, `likelihood.cpp` | `loglik<T>`: the log-likelihood (under variance targeting it computes omega from the current residuals, so the AD carries the dependence) with its per-observation terms, residuals and variances, and the backcast; dispatches by model to the recursion; instantiated for `double` and `Dual<8>`, `Dual<16>`, `Dual<32>`, `Dual<42>` |
| `src/model/weight_terms.hpp` | The weight block seen as terms that sum to the persistence: fixed terms are constants and the free ones share the remaining budget (logistic and softmax) |
| `src/model/parameter_map.*`, `start_values.*` | Layout-driven map between the unconstrained and natural parameter spaces, a template over the scalar type; validity check, start values, and `on_boundary` (parameters left out of the standard-error blocks) |
| `src/optim/lbfgs.hpp` (+ `curvature_memory.hpp`, `line_search.hpp`, `vector_math.hpp`) | L-BFGS minimizer, a template over the objective |
| `src/fit/fit.hpp`, `fit.cpp` | `fit_one`: one series, one spec; with no estimated parameter it evaluates the model at the fixed point and skips the optimizer |
| `src/fit/standard_errors.*`, `cholesky.*` | Hessian and robust standard errors; Cholesky solves, no explicit inverse |
| `src/fit/criteria.*` | Log-likelihood, AIC, BIC, HQIC, SIC |
| `src/fit/batch.hpp`, `batch.cpp` | `check_batch` and `run_batch`: jobs on the thread pool |
| `src/fit/fit_result.hpp` | `ve::FitResult`: status, estimates, standard errors, series, all by value |
| `python/src/volengine/_native.py` | Library discovery (`VOLENGINE_LIB`), ABI check, prototypes, handle ownership |
| `python/src/volengine/_spec.py`, `_fit.py`, `api.py` | `Spec` and `FitOptions`, `Fit` and the criteria, `Context.fit` |
| `tests/unit/{ad,capi,dist,fit,garch,mean,model,optim}` | GoogleTest suites mirroring `src/`; `c_header_check.c` compiles the header as C11 |
| `tests/style/check_style.py` | Style rules that can be checked without a compiler; runs as the `style` ctest |
| `python/benchmarks/bench_vs_arch.py` | Timing against `arch` on matched work; writes `docs/performance/python-vs-arch.md` |
| `tests/benchmarks` | Google Benchmark suite (preset `gcc-bench`) |
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
4. **One fit.** `fit_one` (`src/fit/fit.cpp`) checks the series is long enough (more than parameters + 1 and
   than the largest order) and not constant, and minimizes the negative mean log-likelihood of the series
   standardized by its sample mean and standard deviation with L-BFGS in the unconstrained space
   (`src/model/parameter_map.cpp`); the fit therefore does not depend on the units of the data. The gradient comes
   from `model::loglik` evaluated on `ad::Dual<N>` seeded in the unconstrained parameters, with N = 8, 16 or 32,
   the narrowest that holds the k parameters. At the estimate one pass seeded in the natural parameters gives the
   variance path, residuals and per-observation scores; `standard_errors` differentiates the natural gradient by
   central differences, with steps relative to each parameter's scale, and leaves out α and β entries on the
   boundary (below 1e-8) and a std shape with 1/ν below 1e-8, whose standard errors are NaN.
5. **Status per fit.** A fit that does not converge, is too short or is degenerate carries its own status
   (`max_iter`, `failed`, `insufficient_data`); it does not fail the batch or the other fits.
6. **Publishing.** Results are moved into freshly allocated `ve_fit` handles, all created before any is written to
   the output array, so an exception leaves it untouched. The Python layer then wraps each pointer in a `Fit` that
   frees it.

Every `extern "C"` function runs inside `guard` (`src/capi/error.cpp`), the single place that turns exceptions
into statuses.

## The AD flow

The log-likelihood is written once, as a template over the scalar type (`src/model/likelihood.cpp`, with
`src/mean`, `src/garch` and `src/dist` as its pieces). `fit_one` instantiates it with `double` for values and
with `ad::Dual<N>` (N = 8, 16, 32 or 42, the narrowest that holds k parameters; 42 holds the widest layout, orders 6 with every block
and `ghyp`) for gradients: the unconstrained
parameters are seeded as dual variables, `model::parameter_map` maps them to the natural parameters inside the same
computation, and the derivative of the objective comes out of one evaluation. `model::loglik` picks the variance recursion by `Layout::model`. A new model or
distribution adds a template function, a `Layout` entry and its `ParameterKind`s; the driver and the optimizer do not
change, and the standard errors only need the model's boundary rule in `on_boundary`.

## Performance method

The library exists to be fast, so the aim is the most optimized code the problem allows, with readability kept
alongside it ([STYLEGUIDE.md](STYLEGUIDE.md#priorities)). An optimization is accepted on evidence, never on
intuition. A change to a hot path (the likelihood and its recursions, the AD evaluation, the optimizer loop, the
batch scheduling) goes through three steps:

1. **Benchmark.** Run the Google Benchmark suite (preset `gcc-bench`) before and after, on the same machine, and
   compare medians against the spread over repetitions. A new model, distribution or kernel comes with its own
   `BM_Loglik` case. Results go in [docs/performance/benchmarks.md](docs/performance/benchmarks.md).
2. **Profile.** Find where the time goes before changing anything: AMD uProf or Intel VTune on Windows (matching
   the CPU), `perf record`/`perf report` on Linux. A change that targets a function the profile does not show is
   not justified.
3. **Read the assembly** of the hot function, from a `gcc-bench` object: `objdump -d -C -M intel --no-show-raw-insn`
   (or `-S` on a build with `-g`), and `-fopt-info-vec-missed` to find loops that did not vectorize and why. Look
   for calls that should have been inlined, allocations inside loops, copies of `Dual` values, and divisions or
   transcendental calls that can be hoisted.

What a step found (the numbers, the machine, the assembly excerpt that mattered) goes in `docs/performance/`, never
in a comment ([STYLEGUIDE.md](STYLEGUIDE.md#findings-do-not-belong-in-code)). Copy elision on return is checked by
the compiler (`-Wnrvo` on GCC 14+), not by review.

## Validation

The sGARCH(p,q) grid with the three distributions and the GARCH variants are compared with `rugarch` (default
conventions) and `arch` (the `backcast` option, where it has the model) on committed fixtures; the ARMA means are
compared with `rugarch` only. The conventions, tolerances and observed differences are in
[docs/guide/validation.md](docs/guide/validation.md).
