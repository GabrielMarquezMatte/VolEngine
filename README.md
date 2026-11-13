# VolEngine

VolEngine estimates GARCH volatility models and DCC conditional correlations. The numerical core is C++23; its only stable contract is a C API
(`include/volengine/volengine.h`), so other languages can bind to it, and a Python package, `volengine`, is the
first consumer. Many series and many specifications are fitted in one call on a thread pool, with results that do
not depend on the thread count. Univariate default conventions follow R's `rugarch`; those of Python's `arch` are an option.

## Status

Version 0.1.0 (pre-release). Supported models, each with orders 1 to 6: sGARCH, eGARCH, gjrGARCH, apARCH, iGARCH,
csGARCH and fGARCH with its eight submodels (garch, tgarch, avgarch, ngarch, nagarch, aparch, gjrgarch, allgarch).
Every model takes an ARMA(m,n) conditional mean (orders 0 to 6, with or without a constant) and normal (`norm`),
Student-t (`std`) or GED (`ged`) innovations, or the skewed ones: skewed Student-t (`sstd`), skewed GED (`sged`),
normal inverse Gaussian (`nig`), generalized hyperbolic (`ghyp`) and Johnson SU (`jsu`), with Hessian and robust
standard errors and the usual information criteria. Any estimated parameter can be fixed or given a start value
(the derived ones cannot), the variance can be targeted, and the mean and the variance take external regressors
(`mxreg`, `vxreg`), ARCH-in-mean (`archm`) and an ARFIMA mean (`arfima`), as in `rugarch`, forecasts from a fit,
simulated paths and the diagnostics reported by `rugarch`'s `show()`. More features come in
later releases.
Until 1.0 the C ABI may change between releases. See [CHANGELOG.md](CHANGELOG.md).

DCC and asymmetric DCC (aDCC) support orders 1 to 6; flexible DCC (FDCC) uses order (1,1) and user-defined
groups. All three support `mvnorm`, `mvt` and `mvlaplace`, fixed parameters, start values and optional two-stage
Hessian and robust standard errors. Public DCC conventions follow the papers; the internal `rmgarch` convention
exists for numerical validation.

## Python example

```python
import numpy as np
import volengine as ve

returns = np.random.default_rng(1).standard_normal(3000)

ctx = ve.Context()
fit = ctx.fit(ve.Spec(), returns)

print(fit.status)                 # "converged"
print(fit.coef("robust"))         # CoefTable: names, estimate, std_error, t_value, p_value
print(fit.loglik, fit.aic())
print(fit.sigma[:5])              # conditional standard deviation, a read-only view
```

A DCC grid uses one selected univariate fit per series. Here the 21 columns share T = 5000 observations:

```python
panel = np.random.default_rng(3).standard_t(8, size=(5000, 21))
fits = ctx.fit(ve.Spec(dist="std"), panel)
groups = [1] * 7 + [2] * 7 + [3] * 7
specs = [
    ve.DccSpec(model=model, dist=dist, groups=groups if model == "fdcc" else None)
    for model in ("dcc", "adcc", "fdcc")
    for dist in ("mvnorm", "mvt", "mvlaplace")
]
grid = ctx.fit_dcc(specs, fits, standard_errors=False)
best = min((fit for fit in grid if fit.status == "converged"), key=lambda fit: fit.bic())
print(best.standard_errors(fits).coef("robust"))
print(best.R.shape)               # (5000, 21, 21), read-only, zero-copy
print(best.H.shape)               # covariance path, calculated on first access
```

The univariate fits must have the same T (at least 100) and finite estimates; at least two series are required.
Information criteria count the estimated univariate and DCC parameters and the sample correlation entries.
Each DCC result has its own convergence status. Standard errors can be deferred until after model selection;
the explicit call requires the original univariate estimates in the original order. Standard errors with
univariate external regressors are currently unsupported. DCC forecast and simulation are planned separately.

A Copula-DCC is the same spec with `copula="parametric"` (each fit's cdf) or `"empirical"` (ranks), and a static
copula is `model="static"`. Both kinds go in one grid, and their BIC uses the same parameter count:

```python
specs = [ve.DccSpec("dcc", dist="mvt")] + [
    ve.DccSpec(model, dist="mvt", copula=copula)
    for model in ("static", "dcc", "adcc")
    for copula in ("parametric", "empirical")
]
grid = ctx.fit_dcc(specs, fits, standard_errors=False)
best = min(grid, key=lambda fit: fit.bic())
```

The copula's mvt shape comes from a profile search (each point a full DCC fit), so a Copula-DCC-t costs some
30 DCC fits. A static ML copula reports the free entries C1.. of A in R = cov2cor(A A'), as `rmgarch` does; the
correlations are in `R`.

An ARMA mean and another innovation distribution are part of the spec:

```python
spec = ve.Spec(p=2, q=1, arma=(1, 1), dist="std")   # ARMA(1,1)-sGARCH(2,1) with Student-t innovations
fit = ctx.fit(spec, returns)
print(fit.coef("robust"))         # mu, ar1, ma1, omega, alpha1, alpha2, beta1, shape
```

A skewed distribution adds a `skew` coefficient before `shape` (and `ghlambda` after it for `ghyp`):

```python
spec = ve.Spec(model="gjrgarch", dist="sstd")   # gjrGARCH(1,1) with skewed Student-t innovations
fit = ctx.fit(spec, returns)
print(fit.coef("robust"))         # mu, omega, alpha1, beta1, gamma1, skew, shape
```

The model is part of the spec too; fGARCH needs its submodel:

```python
spec = ve.Spec(model="fgarch", submodel="allgarch")   # fGARCH-ALLGARCH(1,1), power lambda estimated
fit = ctx.fit(spec, returns)
print(fit.coef("robust"))         # mu, omega, alpha1, beta1, eta11, eta21, lambda
```

Parameters can be fixed, given start values (a warm start), or `omega` can be derived by variance targeting:

```python
spec = ve.Spec(dist="std", fixed={"shape": 6.0})   # shape is kept at 6, not estimated, standard error NaN
fit = ctx.fit(spec, returns)
warm = ve.Spec(dist="std", fixed={"shape": 6.0}, start=dict(zip(fit.names, fit.coef().estimate)))
print(ctx.fit(warm, returns).status)   # starts from these estimates; the start value of a fixed name is ignored
print(ctx.fit(ve.Spec(variance_targeting=True), returns).coef("robust"))   # omega is derived
```

The criteria count only the estimated parameters. Variance targeting computes `omega` inside the likelihood from the
residuals of the current mean over the whole sample, with m2 = mean(e^2) and P the persistence: `m2 (1 - P)` for
sGARCH and gjrGARCH, `log(m2) (1 - sum(beta))` for eGARCH, `mean(|e|^delta) (1 - P)` for apARCH and fGARCH. P is
`sum(alpha) + sum(beta)` for sGARCH; for gjrGARCH it includes `pi sum(gamma)`, with `pi = P(z < 0)`; for apARCH and
fGARCH it is `sum(alpha kappa) + sum(beta)`. A start value never unfixes a parameter: a name in both `fixed` and `start`
stays fixed. These specs are rejected (`VE_INVALID_ARGUMENT`, `VolEngineError` in Python): variance targeting in iGARCH or csGARCH,
fixing the iGARCH `beta_q` or `omega` under variance targeting (both derived), and a csGARCH with `eta21` fixed and
some `beta` free. The conventions checked against `rugarch` and `arch` are in
[docs/guide/validation.md](docs/guide/validation.md).

Regressors are counted in the spec and passed to `fit`; ARCH-in-mean adds the conditional standard deviation
(`archm=1`) or variance (`archm=2`) to the mean:

```python
rng = np.random.default_rng(2)
X = rng.standard_normal((3000, 2))
spec = ve.Spec(mxreg=2, archm=1)
fit = ctx.fit(spec, returns, mxreg=X)
print(fit.coef("robust"))         # mu, archm, mxreg1, mxreg2, omega, alpha1, beta1
```

The rules, in short: `mxreg` and `vxreg` take a `(T, k)` array (`(T, N, k)` with `(T, N)` data), with k the count in
the spec, and a wrong count or shape is a `ValueError`; the regressor of row t acts on observation t (lag it yourself);
`vxreg` shifts the variance intercept in the scale of the model, and the persistence stays below 1 (`rugarch` drops that
restriction when there is a `vxreg`); under variance targeting `omega` is the target minus the coefficients times the
regressor means. The conventions and the checks against `rugarch` and `arch` are in
[docs/guide/validation.md](docs/guide/validation.md#rugarch-and-arch-regressors-and-arch-in-mean).

ARFIMA adds a fractional difference d in (1e-8, 0.5) to the mean, as `rugarch`'s `arfima = TRUE`: the ARMA residuals
e_t become the truncated (1 - L)^d of the series, sum of pi_k e_(t-k) for k = 0..t. It cannot be combined with
`archm`, and its cost grows as T^2:

```python
fit = ctx.fit(ve.Spec(arma=(1, 0), arfima=True), returns)
print(fit.coef())                 # mu, ar1, arfima, omega, alpha1, beta1
```

The convention and the checks against `rugarch` are in
[docs/guide/validation.md](docs/guide/validation.md#rugarch-arfima).

A fit forecasts σ and the conditional mean h = 1..H steps after its last observation, as `rugarch`'s
`ugarchforecast`; a spec with regressors takes their values over the horizon:

```python
forecast = fit.forecast(10)       # forecast.sigma, forecast.mean: arrays of shape (10,)
```

In eGARCH, apARCH and fGARCH, `sigma` for h > 1 is `rugarch`'s forecast of log σ² or σ^δ taken back to σ, not
√E[σ²]. The rules and the checks against `rugarch` and `arch` are in
[docs/guide/validation.md](docs/guide/validation.md#rugarch-forecast).

The context simulates paths from a fit, continuing after its last observation, or from a spec with every parameter
fixed, starting at its unconditional state. The threads of the context share the paths, and the same seed gives the
same paths for any number of threads (within one version of the library):

```python
sims = ctx.simulate(fit, horizon=60, paths=10_000, seed=42)   # sims.returns, sims.sigma: (10_000, 60)
cumulative = sims.returns.sum(axis=1)
np.percentile(cumulative, [5, 50, 95])
```

How the samplers and the paths are checked is in [docs/guide/validation.md](docs/guide/validation.md#simulation).

Diagnostics are functions on a fit, returning statistics and p-values; Nyblom returns individual and joint
stability statistics with Hansen's critical values at 10%, 5% and 1%:

```python
f = ctx.fit(ve.Spec(arma=(1, 1)), returns)
print(f.ljung_box(1))                  # weighted Ljung-Box of standardized residuals
print(f.ljung_box(5, fitdf=2))          # ARMA(1,1): two fitted mean coefficients
print(f.ljung_box(5, squared=True))    # squared standardized residuals; fitdf is forced to zero
print(f.arch_lm(3, fitdf=2))           # weighted ARCH-LM; p+q = 2
print(f.nyblom())
print(f.sign_bias())                  # sign, negative, positive and joint
print(f.gof(20))                      # adjusted Pearson goodness of fit
```

For mean order sum d = m+n, `show()` uses residual Ljung-Box lags 1 (fitdf 0), max(2, 3d-1) and max(5, 5d-1)
(fitdf d). For squares, d = p+q gives the same lags with fitdf 0. ARCH-LM uses fitdf = p+q and lags fitdf+1,
fitdf+3 and fitdf+5. Pearson uses 20, 30, 40 and 50 groups. The formulas and conventions are in
[docs/guide/validation.md](docs/guide/validation.md#rugarch-diagnostics).

`ctx.fit` also takes a `(T, N)` array (one fit per column), a list of specs, `starts` and `lengths` for series of
different lengths, and `options=ve.FitOptions(var_init="backcast")` for the `arch` convention.

## Build

Needs CMake 3.28+ and a C++23 compiler (GCC 15+, a recent MSVC or Clang); the presets use Ninja, except
`msvc-release` (Visual Studio) and `native-release` (the default generator). GoogleTest, BS::thread_pool and the
headers of Boost.Math (modified Bessel functions for `nig` and `ghyp`) are fetched by CMake.

```
cmake --preset gcc-release
cmake --build --preset gcc-release
ctest --preset gcc-release
```

Other presets: `gcc-debug`, `msvc-release` and `native-release` (the default compiler, taken from `CC` and `CXX`).
Install the Python package with `pip install -e "python[test]"` and point it to the built library with the
`VOLENGINE_LIB` environment variable. See [CONTRIBUTING.md](CONTRIBUTING.md) for details and
[ARCHITECTURE.md](ARCHITECTURE.md) for the code map. Measurements are in
[docs/performance/benchmarks.md](docs/performance/benchmarks.md) (C++) and
[docs/performance/python-vs-arch.md](docs/performance/python-vs-arch.md) (against `arch`).

## License

[MIT](LICENSE).
