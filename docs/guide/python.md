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
print(winner.standard_errors(fits).coef("robust"))
copula = ctx.fit_dcc(ve.DccSpec("dcc", dist="mvt", copula="parametric"), fits)
print(copula.status)
```

The README has the long examples: ARMA means, every distribution, fixed parameters, regressors, ARFIMA and variance
targeting.
