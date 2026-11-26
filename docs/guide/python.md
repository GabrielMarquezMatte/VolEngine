# Python quick guide

`pip install volengine` (Python 3.11 or newer, no compiler). Every block below runs as written, one after another.

## Fit a model

One `Context` owns the thread pool; a `Spec` says which model. Every convention is an explicit option (see
[validation.md](validation.md)).

```python
import numpy as np
import volengine as ve

ctx = ve.Context()

# Data to fit: a GARCH(1,1) with Student-t shocks, simulated by the library itself.
truth = ve.Spec(dist="std", fixed={"mu": 0.0003, "omega": 2e-6, "alpha1": 0.08, "beta1": 0.9, "shape": 8.0})
returns = ctx.simulate(truth, horizon=3000, paths=1, seed=1).returns[0]

fit = ctx.fit(ve.Spec(model="gjrgarch", dist="std"), returns)
print(fit.status)                       # "converged"; a fit that does not converge is a status, not an exception
print(fit.coef("robust"))               # names, estimate, std_error, t_value, p_value
print(fit.loglik, fit.aic(), fit.bic(per_obs=False))
print(fit.sigma[:3], fit.std_residuals[:3])   # read-only views of the native results
```

A `(T, N)` array fits one series per column, and a list of specs fits the whole grid in one call on the thread pool;
the numbers do not depend on the thread count:

```python
panel = ctx.simulate(truth, horizon=3000, paths=3, seed=2).returns.T   # (T, N): one column per series
specs = [ve.Spec(model=model, dist=dist) for model in ("sgarch", "egarch") for dist in ("norm", "std")]
grid = ctx.fit(specs, panel)            # one row per spec, one column per series
print(grid.shape)
best = min(grid[:, 0], key=lambda candidate: candidate.bic())   # the best model for the first series
print(best.names)
```

## Diagnostics

```python
print(fit.ljung_box(10, squared=True))  # weighted Ljung-Box of the squared standardized residuals
print(fit.arch_lm(5, fitdf=2))
print(fit.nyblom())                      # individual and joint statistics with critical values
print(fit.sign_bias(), fit.gof(20))
```

## Forecast and simulation

```python
forecast = fit.forecast(10)             # sigma and mean for the next 10 steps
print(forecast.sigma)
sims = ctx.simulate(fit, horizon=60, paths=2000, seed=7)   # the same seed gives the same paths on any thread count
print(np.percentile(sims.returns.sum(axis=1), [5, 50, 95]))
```

The forecast follows `rugarch`, so it is not the mean of the simulated paths for a gjrGARCH with a skewed distribution
(P(z < 0) stands in for E[1{z < 0} z²]) and for a csGARCH from h = 2 (the closed form leaves out the last shock); see
[validation](validation.md).

## DCC and Copula-DCC

A DCC takes univariate fits, one per series, and fits the correlation model on their standardized residuals:

```python
fits = ctx.fit(ve.Spec(dist="std"), panel)
dcc = ctx.fit_dcc(ve.DccSpec("dcc", dist="mvt"), fits)
print(dcc.coef("robust"))               # the DCC parameters: dcca1, dccb1, mshape
print(dcc.bic(), dcc.R.shape)           # R: (T, 3, 3), a read-only view
print(dcc.forecast(5).H[-1])            # the covariance matrix 5 steps ahead
paths = ctx.simulate(dcc, horizon=20, paths=100, seed=3, correlations=True)
print(paths.returns.shape)              # (100, 20, 3)
```

For a model grid chosen by a criterion, skip the standard errors and add them to the winner:

```python
specs = [ve.DccSpec(model, dist="mvnorm") for model in ("dcc", "adcc")]
candidates = ctx.fit_dcc(specs, fits, standard_errors=False)
winner = min(candidates, key=lambda candidate: candidate.bic())
print(winner.estimate)                  # without standard errors `coef()` raises: call `standard_errors`
print(ctx.standard_errors(winner, fits).coef("robust"))  # on ctx's threads
copula = ctx.fit_dcc(ve.DccSpec("dcc", dist="mvt", copula="parametric"), fits)
print(copula.status)
```

The `mvlaplace` density is singular where every standardized residual of a date is exactly 0 (a common holiday with `mu`
fixed at 0): the log-likelihood is then not finite and the fit ends `failed`, with no message besides the status.
Drop such dates before the univariate fits.

## Cancellation

`fit`, `fit_dcc`, `standard_errors` and `simulate` take a keyword-only `cancel=`, a `Cancel`. Its `cancel()` can be
called from any thread, also while the call runs, and the call raises `Cancelled` (a `VolEngineError` with status 6)
within one optimizer iteration, one block of the standard errors or one simulated path; no result is returned. On a
`Context` that other calls keep busy, a cancelled call waits for its turn in the context's queue before it stops, so
give each caller that needs a deadline its own `Context`. The call releases the GIL, so a `threading.Timer` gives a
deadline:

```python
import threading

models = [ve.Spec(model=model, dist="std") for model in ("sgarch", "egarch", "gjrgarch", "aparch")]
c = ve.Cancel()
timer = threading.Timer(0.05, c.cancel)   # the deadline, in seconds
timer.start()
try:
    ctx.fit(models * 10, np.tile(panel, (1, 20)), cancel=c)
except ve.Cancelled:
    print("stopped at the deadline")
finally:
    timer.cancel()
```

A `Cancel` stays requested: make a new one for the next call. One `Cancel` can go to several calls and a request stops
all of them. A request that arrives after the call's last check does not stop it, so the call returns its result.

The README has the long examples: ARMA means, every distribution, fixed parameters, regressors, ARFIMA and variance
targeting.
