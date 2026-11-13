# Architecture

A map of how the codebase fits together, not a manual. Start here, then follow the file names into the source:
the code carries its own meaning through naming and flow. What it deliberately does not carry is the evidence
behind a decision (measurements, rejected approaches); that lives in `docs/`, and
[STYLEGUIDE.md](STYLEGUIDE.md#comments) says why.

This document describes what exists today: ARMA(m,n)-GARCH(p,q) log-likelihoods for sGARCH, eGARCH, gjrGARCH,
apARCH, iGARCH, csGARCH and fGARCH (eight submodels) with normal, Student-t, GED and skewed innovations, fixed parameters, start values and variance
targeting, a template over the scalar type so that automatic differentiation gives their gradients; and, on top of
them, those models fitted in batches behind a C API and a Python package. A second-stage DCC module estimates
DCC, aDCC and grouped FDCC correlations from selected univariate fits, with multivariate normal, Student-t or
Laplace likelihoods and optional two-stage standard errors.

## Layers

```
  Python (python/src/volengine, ctypes + numpy)
                 |
        C API  include/volengine/volengine.h        <-- the only stable contract
                 |
        src/capi   handles, error boundary
                 |
        src/diagnostics   weighted portmanteau, sign bias, Pearson GOF
                 |
        src/sim    simulated paths: Philox, samplers, start states
                 |
        src/forecast   forecast of a fit, h steps ahead
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

  src/linalg   Cholesky, least squares (depends on nothing; used by model and fit)
```

Dependencies point one way: `core` <- `ad` <- `mean`, `garch`, `dist` <- `model` <- `fit` <- `forecast` <- `sim` <- `capi`,
and `optim` <- `fit`. `fit` includes only `forecast/state.hpp` from `forecast`: the plain `ForecastState` that a
`FitResult` carries. `linalg` sits beside the stack: it depends on nothing, and both `model` (the start values and the
backcast with mean regressors) and `fit` (the standard errors) use it. Nothing in `src/` includes from `src/capi/` except `capi` itself. Everything below the C API is
internal and may change in any release; only the header is a contract, and the Python package is a consumer of it
like any other wrapper would be.

Diagnostics form a separate branch, `fit` <- `diagnostics` <- `capi`: residual tests consume arrays and use
`dist` for CDFs and `linalg` for regression. The Nyblom computation lives in `fit`, where the final score matrix
already exists; only its individual and joint statistics survive that evaluation.

DCC forms another branch, `fit` <- `dcc` <- `capi`. `src/dcc/` standardizes the selected univariate fits
once per batch and shares their panel (z, sigma, sample covariances) among DCC, aDCC and FDCC specifications.
Its templated recursion and multivariate likelihood run with `double` or dual numbers through the existing
L-BFGS optimizer. Each result owns its correlation path; the immutable C handle computes and caches the
covariance path on demand, safely across readers. Two-stage standard errors refilter the univariates under
parameter perturbations and may be requested during fitting or explicitly afterward. Public calls always use
the standard paper conventions; the internal `Rmgarch` convention is reserved for oracle tests.

`src/dcc/copula/` turns the univariate margins into uniforms (each fit's cdf from `dist`, or ranks) and then into
normal or Student scores, and runs the single-spec DCC fit of `src/dcc/` on them, or a static correlation (Kendall,
or ML with an analytic gradient). The batch in `src/dcc/fit.cpp` dispatches copula specs there, so the folder
depends on `dcc` and `dcc` on it for that dispatch only: `dist` <- `dcc/copula` <-> `dcc`.

## Where each responsibility lives

| Path | Responsibility |
|---|---|
| `include/volengine/volengine.h` | The C API: `ve_` functions, `VE_` constants, opaque handles, no public struct |
| `src/capi/handles.hpp` | The handle types (`ve_context`, `ve_spec`, `ve_fit_options`, `ve_panel`, `ve_batch`, `ve_fit`, `ve_forecast`), each wrapping one core value |
| `src/capi/error.hpp`, `error.cpp` | Thread-local `ve_last_error()`, `set_error`, and `guard`, the one place exceptions are caught |
| `src/capi/context_api.cpp`, `spec_api.cpp`, `panel_api.cpp`, `batch_api.cpp`, `version_api.cpp` | The `extern "C"` functions, one file per handle family |
| `src/capi/results/fit_api.cpp`, `forecast_api.cpp`, `sim_api.cpp` | The `extern "C"` functions over results: the accessors of `ve_fit`, `ve_forecast`, and `ve_sim_options` / `ve_simulate_*` / `ve_sim` |
| `src/capi/results/regressor_rows.*` | The checks and copies of the regressors of a horizon, shared by the forecast and the simulation |
| `src/capi/results/diagnostics_api.cpp` | Diagnostic argument checks and caller-owned result arrays over immutable fits |
| `src/capi/dcc/` | DCC spec/options, batch fitting, immutable result accessors, lazy covariance cache and deferred standard-error handles; `dcc_forecast_api.cpp` (`ve_dcc_future`, `ve_dcc_forecast`) and `dcc_sim_api.cpp` (`ve_simulate_dcc`, `ve_dcc_sim`) |
| `src/core/context.hpp`, `context.cpp` | `ve::Context`: the thread pool and thread count |
| `src/core/spec.hpp` | `ve::Spec` (model, order, distribution, mean, fixed and start values, variance targeting) and `ve::FitConfig` (iterations, tolerance, variance initialization): copyable values |
| `src/core/panel.hpp` | `ve::PanelView`: a borrowed pointer, T, N, element strides and optional per-series start and length |
| `src/core/var_init.hpp` | The variance initialization enum (`Sample`, `Backcast`) |
| `src/ad/dual.hpp`, `special.hpp`, `special.cpp` | `ve::ad::Dual<N>`: forward-mode dual numbers and their elementary functions; `digamma` |
| `src/linalg/cholesky.*`, `least_squares.*` | Cholesky factor and solve; minimum-norm least squares by the normal equations, with a Jacobi pseudo-inverse when they are singular |
| `src/linalg/symmetric_eigen.*` | Largest relative eigenvalue for the aDCC stationarity constraint, by power iteration after a Cholesky transform |
| `src/linalg/lu.*` | Pivoted LU solves for the nonsymmetric two-stage standard-error system |
| `src/dcc/spec.hpp`, `panel.*` | DCC models, distributions and internal conventions; standardized panel, covariance moments and aDCC stationarity bound |
| `src/dcc/recursion.hpp`, `likelihood.*` | DCC/aDCC/FDCC correlation paths and multivariate normal, Student-t and Laplace log-densities, templated for AD |
| `src/dcc/parameter_map.*`, `fit.*` | Parameter layout and constraints, L-BFGS estimation, deterministic batches, full likelihood criteria and covariance paths |
| `src/dcc/copula/` | Copula-DCC: margins to uniforms (`margins.*`), scores (`scores.*`), static Kendall/ML correlation (`static_fit.*`), profile fit and copula likelihood (`fit.*`) |
| `src/dcc/future/` | The future of a DCC fit (plan 4c): the final state of the correlation recursion and the `CorrelationStepper` that advances Q and R one step (`state.*`), `forecast_dcc` (`forecast.*`) and `simulate_dcc` (`simulate.*`) |
| `src/dcc/standard_errors.*` | Two-stage Hessian and sandwich standard errors, with univariate refiltering and source-estimate validation |
| `src/mean/arma.hpp` | ARMA(m,n) residuals over the returns or over the deviations from the mean (regressors included), a template over the scalar type |
| `src/mean/fractional.hpp` | The ARFIMA fractional difference (1 - L)^d of the mean residuals, truncated at the first observation, in place, O(T^2) |
| `src/garch/presample.hpp` | The pre-sample rule shared by the recursions: a state held at a value up to a time, and the expected value of a term before the first observation |
| `src/garch/sgarch.hpp`, `egarch.hpp`, `gjrgarch.hpp`, `csgarch.hpp` | The variance recursions of sGARCH (also iGARCH), eGARCH, gjrGARCH and csGARCH, templates over the scalar type |
| `src/garch/power_garch.hpp` | The recursion in sigma^power shared by apARCH and the eight fGARCH submodels |
| `src/dist/densities.hpp` | `Density<T>`, a `std::variant` of the log-density families, and `make_density`, `log_density`, `density_kink`: the one entry point the likelihood uses |
| `src/dist/symmetric.hpp`, `skewed.hpp`, `hyperbolic.hpp`, `johnson.hpp` | The families of the variant: norm, std and ged; sstd and sged (Fernandez-Steel); nig and ghyp; jsu. Templates over the scalar type |
| `src/dist/bessel.hpp`, `bessel.cpp` | `log_bessel_k` and its derivatives. Boost.Math is included only in `.cpp` files (`bessel.cpp`, `dist/cdf.cpp`, `diagnostics/tails.cpp`), so no header exposes it |
| `src/dist/moments.hpp`, `moments.cpp` | Expected absolute shock, P(z < 0) and the fGARCH news moments (Gauss-Legendre quadrature for the shifted and the asymmetric ones) |
| `src/dist/cdf.*` | Standardized CDFs of all eight distributions; sorted accumulation of density quadrature for nig and ghyp; `quantile`, their inverse (closed forms, a root solve for nig and ghyp), used by the Copula-DCC simulation |
| `src/diagnostics/` | Upper probability tails, weighted Ljung-Box and ARCH-LM, sign-bias regression and adjusted Pearson GOF |
| `src/model/data.hpp` | `Data`: the series and its row-major regressor matrices (mean and variance) |
| `src/model/layout.hpp`, `layout.cpp` | `Layout`: parameter blocks (gamma, eta1, eta2, power), offsets, count, `ParameterKind` of each parameter, rugarch names of a `Spec`, and the role of each parameter (estimated, fixed or derived; `estimated_count`) |
| `src/model/likelihood.hpp`, `likelihood.cpp` | `loglik<T>`: the log-likelihood (under variance targeting it computes omega from the current residuals, so the AD carries the dependence) with its per-observation terms, residuals and variances, and the backcast (from the least squares of r on the mean regressors, when there are any); dispatches by model to the recursion; instantiated for `double` and `Dual<8>`, `Dual<16>`, `Dual<32>`, `Dual<59>` |
| `src/model/weight_terms.hpp` | The weight block seen as terms that sum to the persistence: fixed terms are constants and the free ones share the remaining budget (logistic and softmax) |
| `src/model/parameter_map.*`, `start_values.*` | Layout-driven map between the unconstrained and natural parameter spaces, a template over the scalar type; validity check, start values, and `on_boundary` (parameters left out of the standard-error blocks) |
| `src/optim/lbfgs.hpp` (+ `curvature_memory.hpp`, `line_search.hpp`, `vector_math.hpp`) | L-BFGS minimizer, a template over the objective |
| `src/fit/fit.hpp`, `fit.cpp` | `fit_one`: one series, one spec; with no estimated parameter it evaluates the model at the fixed point and skips the optimizer |
| `src/fit/standard_errors.*` | Hessian and robust standard errors; Cholesky solves, no explicit inverse |
| `src/fit/nyblom.*` | Individual and joint score-stability statistics, excluding fixed and derived parameters; Hansen critical values |
| `src/fit/criteria.*` | Log-likelihood, AIC, BIC, HQIC, SIC |
| `src/fit/batch.hpp`, `batch.cpp` | `check_batch` and `run_batch`: jobs on the thread pool |
| `src/fit/fit_result.hpp` | `ve::FitResult`: status, estimates, standard errors, series, forecast state and Nyblom statistics, all by value |
| `src/forecast/state.hpp` | `ForecastState`: the last max(m, n, p, q) residuals, variances, csGARCH trends and mean deviations of a fit (every deviation with ARFIMA), filled by the final evaluation of `fit_one` through the `deviations` and `permanent` outputs of `loglik` |
| `src/forecast/forecast.hpp`, `forecast.cpp` | `forecast`: the mean (ARMA, regressors, `archm`, `rugarch`'s ARFIMA rule) over the variance forecast |
| `src/forecast/variance_forecast.*`, `inputs.hpp` | The variance forecast of each family, in sigma2, log sigma2 or sigma^delta, with future shocks at their expectation; csGARCH by `rugarch`'s closed form from h = 2. The shock moments (pi, E\|z\|, kappa) are computed once per call (`moments_of`) |
| `src/forecast/step.hpp` | `first_step`: the forecast of one step with its mean level and csGARCH q, the step a simulation repeats |
| `src/sim/philox.hpp` | Philox4x32-10 and `Stream`, the stream of one path (key = seed, counter = (block, path)) |
| `src/sim/samplers.*` | Box-Muller, Marsaglia-Tsang gamma, Hormann-Leydold GIG, and `Sampler`: z of the standardized density of a fit, for the eight distributions |
| `src/sim/start.*` | `Start`: the state the paths start from, the final state of a fit or the unconditional state of a spec with every parameter fixed |
| `src/sim/path.*` | `Path`: one path of one series, a step that takes the shock from outside (the univariate `simulate` and the DCC simulation both step through it) |
| `src/sim/simulate.*` | `simulate`: each path repeats `first_step` with a drawn shock; the paths are split among the threads of the context |
| `python/src/volengine/_native.py` | Library discovery (`VOLENGINE_LIB`), ABI check, prototypes, handle ownership |
| `python/src/volengine/_spec.py`, `_fit.py`, `api.py` | `Spec` and `FitOptions`, `Fit` and the criteria, `Context.fit` |
| `python/src/volengine/_diagnostics.py` | `DiagnosticResult`, `SignBias`, `Nyblom` and native diagnostic result readers |
| `python/src/volengine/_dcc.py` | `DccSpec`, immutable `DccFit`, read-only correlation/covariance views and deferred standard errors |
| `tests/unit/{ad,capi,dist,fit,garch,mean,model,optim}` | GoogleTest suites mirroring `src/`; `c_header_check.c` compiles the header as C11 |
| `tests/style/check_style.py` | Style rules that can be checked without a compiler; runs as the `style` ctest |
| `python/benchmarks/bench_vs_arch.py` | Timing against `arch` on matched work; writes `docs/performance/python-vs-arch.md` |
| `tests/benchmarks` | Google Benchmark suite (preset `gcc-bench`) |
| `python/tests` | Python tests; the oracle comparisons in `python/tests/oracles/` |
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
with `ad::Dual<N>` (N = 8, 16, 32 or 59, the narrowest that holds k parameters; 59 is `kMaxParameterCount`, the widest layout: orders 6
with every block, `ghyp`, `archm` and 8 regressors in each of the mean and the variance) for gradients: the unconstrained
parameters are seeded as dual variables, `model::parameter_map` maps them to the natural parameters inside the same
computation, and the derivative of the objective comes out of one evaluation. `model::loglik` picks the variance recursion by `Layout::model`. A new model or
distribution adds a template function, a `Layout` entry and its `ParameterKind`s; the driver and the optimizer do not
change, and the standard errors only need the model's boundary rule in `on_boundary`.

## Regressors and ARCH-in-mean

`model::Data` (`src/model/data.hpp`) is what the likelihood reads: the series plus two row-major regressor matrices
(mean and variance), all borrowed. A series without regressors is `Data{r}`, and the `loglik` overload that takes a
span of returns builds it. The variance recursions in `src/garch` are written in step form: one function computes
sigma^2 at t from the state at t-1 and the intercept shift c_t of the variance regressors, so the same step serves
the plain loop and the joint one. Without `archm` the mean residuals are computed first, for the whole series, and
then the variance (the loop of plan 2d, bit for bit). With `archm` the mean at t needs sigma_t, so one joint loop
advances the variance step and the mean residual together. The pre-sample residuals e^0 (the mean without the
`archm` term) feed the sample initialization and variance targeting. `src/linalg` gives the OLS start of `mu` and the
mean coefficients, and the backcast residuals, by minimum-norm least squares, so a collinear column leaves a finite
start. With ARFIMA (`Layout::arfima`, never together with `archm`) the mean residuals of the ARMA pass through
`mean::fractional_difference` before anything reads them, so the pre-sample, variance targeting, the density and the
reported residuals all see the differenced series. The conventions, with their sources, are in [docs/guide/validation.md](docs/guide/validation.md).

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
