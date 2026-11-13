# Reference data and oracle fixtures

Diagnostics are regenerated with `Rscript tests/reference/diagnostics_rugarch.R` (R 4.6.2,
`rugarch` 1.5.6, `jsonlite`). The script sources the case/specification helpers from
`oracle_rugarch.R`, fits every rugarch case, and writes `fixtures/rugarch_diagnostics.json`.
It calls `nyblom`, `signbias`, `gof`, and the package's callable `Weighted.Box.test` and
`Weighted.LM.test` (the latter two are not exported, so are obtained with `getFromNamespace`).
No package implementation is copied. All-fixed cases use `ugarchfilter` and have no Nyblom statistic.
The fixture includes the coefficients of each new fit: the residual tests evaluate VolEngine at
those exact points, while the Nyblom tests estimate parameters starting from them. Run
`python/.venv/Scripts/python -m pytest python/tests/oracles/test_diagnostics_oracles.py`
after building the release DLL; the tests themselves require neither R nor rugarch.

This directory holds the simulated series and the results of two independent oracles, R's
[`rugarch`](https://cran.r-project.org/package=rugarch) and Python's [`arch`](https://pypi.org/project/arch/),
that `python/tests/oracles/test_oracles.py` compares VolEngine against. Conventions and tolerances are documented in
[`docs/guide/validation.md`](../../docs/guide/validation.md).

| Path | Content |
|------|---------|
| `generate_data.py` | Simulates every series: the plan-1 sGARCH(1,1)-normal series (seed 20261110), the grid series (seed 20261111), the GARCH-variant series (seed 20261112), the skewed-distribution series (seed 20261113) and the regressor and ARCH-in-mean series with their regressors (seed 20261114) |
| `cases.json` | The validation cases: series, model, orders, distribution and the oracles each one is compared with |
| `oracle_rugarch.R` | Fits every case that lists `rugarch` and writes its fixture |
| `oracle_arch.py` | Fits every case that lists `arch` and writes its fixture; with `--forecast`, forecasts the sGARCH and gjrGARCH cases with a constant mean and symmetric shocks at `arch`'s estimate and writes `fixtures/arch_forecast.json` |
| `forecast_rugarch.R` | Forecasts every `rugarch` case at `rugarch`'s estimate (from `fixtures/rugarch.json`) on the first T − 25 observations, 25 steps ahead, and writes `fixtures/rugarch_forecast.json` |
| `data/sgarch11_norm.csv` | Plan-1 series, one column per length (`t5000`, `t3000`, `t1500`), 17 significant digits; shorter series end in empty cells |
| `data/grid.csv` | Grid series, T = 3000 each (`sgarch21_std`, `arma11_sgarch11_ged`, `sgarch12_norm`, `sgarch22_ged`), 17 significant digits |
| `data/grid_2b.csv` | GARCH-variant series, T = 3000 each (`egarch11_std`, `gjrgarch11_norm`, `aparch11_ged`, `igarch11_norm`, `csgarch11_std`, `fgarch_allgarch11_norm`), 17 significant digits |
| `data/grid_2c.csv` | Skewed-distribution series, T = 3000 each (`gjrgarch11_sstd`, `aparch11_sged`, `egarch11_jsu`, `sgarch11_nig`, `sgarch11_ghyp`), 17 significant digits |
| `data/grid_2e.csv` | Regressor and ARCH-in-mean series, T = 3000 each (`arx_sgarch11_norm`, `sgarch11_vx_std`, `archm_gjrgarch11_norm`, `archm2_egarch11_std`), 17 significant digits |
| `data/regressors_2e.csv` | The regressors the cases name, T = 3000 rows aligned with `grid_2e.csv` and every other series: `x1` = sin(t/40), `x2` ~ N(0, 1), `v1` = \|cos(t/25)\|, t = 1..T |
| `copula_probe.R` | Writes `data/copula3.csv` once (seed 20261113) and prints rmgarch's Copula-DCC numbers beside an independent evaluation (the plan 4b notes) |
| `copula_rmgarch.R` | Fits the 13 Copula-DCC cases with `cgarchfit` on `copula3.csv` and writes `fixtures/rmgarch_copula.json` |
| `data/copula3.csv` | Three sGARCH(1,1) series with standardized-t(7) margins and a Student-t(6) DCC(0.04, 0.93) copula, T = 2000, 15 significant digits (`write.csv`) |
| `fixtures/rmgarch_copula.json` | `rmgarch` Copula-DCC results |
| `fixtures/rugarch.json` | `rugarch` results |
| `fixtures/arch.json` | `arch` results |
| `fixtures/arch_forecast.json` | `arch` forecasts, in the format of `rugarch_forecast.json` |
| `fixtures/rugarch_forecast.json` | `rugarch` forecasts: per case `n`, `horizon`, the `fixed` point and the 25 values of `sigma` and `mean` |

The files under `data/` and `fixtures/` are generated and committed. The tests read only these files: running
them needs neither R nor `arch`. Series names are unique across the data files.

## Case format

`cases.json` is a list of

```
{"id": str, "series": str, "model": str, "submodel": str (fgarch only), "p": int, "q": int, "arma": [m, n],
 "include_mean": bool, "dist": "norm" | "std" | "ged" | "sstd" | "sged" | "nig" | "ghyp" | "jsu",
 "fixed": {name: float} (optional), "variance_targeting": bool (optional),
 "mxreg": [column] (optional), "vxreg": [column] (optional), "archm": 1 | 2 (optional), "arfima": true (optional),
 "oracles": ["rugarch", "arch"]}
```

`model` and `submodel` are the names of VolEngine's Python `Spec` (`"sgarch"`, `"egarch"`, `"gjrgarch"`,
`"aparch"`, `"igarch"`, `"csgarch"`, `"fgarch"`; for `"fgarch"` a submodel such as `"allgarch"`). The scripts
translate them: `rugarch`'s `model = "sGARCH"`, `"eGARCH"`, ... and `submodel = "ALLGARCH"`, ...; `arch`'s
`vol = "GARCH"` (with `o = p` for gjrGARCH) or `"EGARCH"`. `arch` covers sGARCH, gjrGARCH and eGARCH only. `p` is
the ARCH order (α terms) and `q` the GARCH order (β terms), as in VolEngine's `Spec` and `arch`, and as
`rugarch`'s `garchOrder = c(p, q)`. The id is `<series>/<model>`. `fixed` maps parameter names (`rugarch`'s, as
below) to values held during the fit, VolEngine's `Spec(fixed=...)` and `rugarch`'s `fixed.pars`;
`variance_targeting` is VolEngine's `Spec(variance_targeting=True)` and `rugarch`'s `variance.targeting = TRUE`.
`mxreg` and `vxreg` list columns of `data/regressors_2e.csv` (VolEngine's `Spec(mxreg=k, vxreg=j)` with the columns
stacked, in order, as `ctx.fit(..., mxreg=X, vxreg=V)`; `rugarch`'s `mean.model$external.regressors` and
`variance.model$external.regressors`; `arch`'s `x` with `mean="LS"`, which includes the constant, so `include_mean`
stays true). `archm` is the power p of ARCH-in-mean (`Spec(archm=p)`; `rugarch`'s `archm = TRUE, archpow = p`). `arch` has
neither `vxreg` nor `archm`, and covers `mxreg` only for a case without ARMA terms: its `ARX` applies the AR to
y, not to y minus the regressors' mean. `arfima` is the ARFIMA mean (`Spec(arfima=True)`; `rugarch`'s `arfima = TRUE`).
`arch` has no ARFIMA, and the ARFIMA cases name `rugarch` only.
A case with every parameter in `fixed` is an *evaluation*: nothing is estimated, `oracle_rugarch.R` runs
`ugarchfilter` and `oracle_arch.py` runs `fix()` at the fixed point. `arch` has no variance targeting and takes
only evaluation cases among those with `fixed`.

## Fixture format

```
{"oracle": str, "oracle_version": str, "cases": [str],
 "results": [{"case": str, "series": str, "loglik": float, "params": {name: float}, "se": {name: float},
              "robust_se": {name: float}, "persistence": float, "sigma": [float], "aic": float, "bic": float,
              "hqic": float | null, "sic": float | null}]}
```

`cases` lists the ids of the cases that name the oracle, in `cases.json` order, and `results` follows it.
Parameter names follow `rugarch`, which VolEngine uses: `mu`, `ar1..`, `ma1..`, `arfima` or `archm`, `mxreg1..`, `omega`,
`alpha1..`, `beta1..`, `gamma1..`, `eta11..`, `eta21..`, `delta` or `lambda`, `vxreg1..`, `skew`, `shape`, `ghlambda`,
in that order; `oracle_arch.py` renames `Const`, `x0..` (its `LS` mean), `alpha[i]`, `gamma[i]`, `beta[j]`, `nu` and
the skew-t's `eta` and `lambda` and reorders them. The skew-t `lambda` (Hansen's skewness) becomes `skew` = ξ = √((1 + λ)/(1 − λ)), with its standard errors by
the delta method (dξ/dλ = 1/(ξ(1 − λ)²)). For eGARCH it also swaps `arch`'s `alpha` (the term
in |z|) and `gamma` (the term in z) into `rugarch`'s `gamma` and `alpha`, and moves the centering of |z| into
`omega` (see `docs/guide/validation.md`). A standard error is `null` where it is not comparable (the eGARCH
`omega` of `arch` outside norm) or not estimated (the derived iGARCH `beta_q`). `persistence` is the oracle's P
(`rugarch`'s `persistence()`, the transitory one for csGARCH; for `arch` computed from the estimates, with the
skew-t's P(z < 0) from `SkewStudent.cdf` for the gjrGARCH γ), used by the interior check. Criteria stay on the oracle's own
scale: per observation for `rugarch`, totals for `arch` (which reports no HQIC or SIC, hence `null`). The
`rugarch` results carry one extra key, `qml_robust_se`: `rugarch`'s `robust_se` is a Newey-West estimator, and
`qml_robust_se` is the QML sandwich (VolEngine's robust definition) rebuilt from `rugarch`'s own Hessian and
per-observation scores.

A fixed or derived parameter (fixed, the iGARCH `beta_q`, the variance-targeting `omega`) appears in `params` with
its value and a `null` standard error; `oracle_rugarch.R` puts `rugarch`'s variance-targeting `omega`, which
`coef()` appends last, back in its place. An evaluation case has every standard error `null`; its criteria are
`rugarch`'s `infocriteria()` of the filter (k = 0) and `null` for `arch`, whose `fix()` counts every parameter.

## Regenerating

Requirements, needed only to regenerate:

- R with `rugarch` and `jsonlite` (`install.packages(c("rugarch", "jsonlite"))`).
- `arch` in the Python test environment (`python/.venv/Scripts/pip install arch`; it brings SciPy, which
  `generate_data.py` uses for the GH draws and Bessel functions). It is deliberately not part of the package's
  `test` extra.

From the repository root:

```
python/.venv/Scripts/python tests/reference/generate_data.py
Rscript tests/reference/oracle_rugarch.R
Rscript tests/reference/forecast_rugarch.R   # after oracle_rugarch.R: it reads fixtures/rugarch.json
python/.venv/Scripts/python tests/reference/oracle_arch.py
python/.venv/Scripts/python tests/reference/oracle_arch.py --forecast   # after oracle_arch.py: it reads fixtures/arch.json
```

The regressor and ARCH-in-mean cases (`mxreg`, `vxreg` or `archm` in `cases.json`) are fitted with a few extra
settings in `oracle_rugarch.R`, each explained in `docs/guide/validation.md`: `start.pars` (`archm` and every `vxreg`
start at 0.01), `hess.d = 1e-2` and `solver.control = list(tol = 1e-10)`. The ARFIMA cases (`arfima` in `cases.json`)
take `hess.d = 1e-2` only.

The scripts stop with an error if an oracle reports non-convergence or an estimate on the boundary: a weight
≤ 1e-4 (the non-negative entries that sum to the persistence P: α and β; β alone in eGARCH; (1 − π)α, β, π(α + γ)
in gjrGARCH, with π = P(z < 0), ½ for the symmetric distributions; ακ and β in apARCH and fGARCH), P ≥ 0.999 (not
for iGARCH, where P = 1), |η₁| or the apARCH |γ| ≥ 0.999, in csGARCH ρ − P ≤ 1e-4 or φ ≤ 1e-4, and for the
distribution |ρ| ≥ 0.999 (nig, ghyp), ξ ∉ (0.05, 20) (sstd, sged), 1/shape ≤ 1e-4 or |ghlambda| ≥ 10.
`oracle_rugarch.R` also stops on a distribution, `archm`, `mxreg` or `vxreg` parameter at `rugarch`'s own bounds (for
example the ghyp shape at 0.25, or a `vxreg` at its lower bound 0; the bounds are [−10, 10] for `archm`, [−100, 100]
for `mxreg` and [0, 100] for `vxreg`). It shrinks `rugarch`'s numerical Hessian step (`hess.d = 1e-3`, and 1e-4 for sged);
`docs/guide/validation.md` explains why. The interior rule holds only the estimated parameters: a weight in a
fixed or derived position, a fixed |η₁| or γ and a fixed distribution parameter are not checked, and evaluation
cases are not checked at all. After regenerating, record the oracle versions and the observed differences there.

Regenerating on another machine can change the last digits: on the plan-2b machine `generate_data.py` rewrites
`grid.csv` at the last digit of some entries, and the plan-1/2a fixture results come back with the same estimates,
ℓ, σ and criteria but standard errors that differ in the ninth digit. The committed plan-1/2a data and results
were therefore kept as they are; only the new series and cases were added (see `docs/guide/validation.md`). The
same holds for the plan-2b data and results, regenerated on the plan-2c machine, and for every earlier result,
regenerated for plan 2d on the plan-2b machine, and for the plan-2e cases: `generate_data.py` rewrote the last
digits of the earlier CSV files and was run again only to produce `grid_2e.csv` and `regressors_2e.csv`, the `arch` rerun
changed three earlier results in their robust standard errors (to 2e-9) and `oracle_rugarch.R` was run for the new
cases only, its results appended to the committed fixture. sged's `hess.d = 1e-4` was chosen
against the committed fixtures and may need re-tuning on another machine or R version. The plan-2f ARFIMA cases
were added the same way: only `grid_2f.csv` was kept from the generator run, and `oracle_rugarch.R` was run on a copy
of `cases.json` holding the ARFIMA cases only, its results appended to the committed fixture.

### DCC oracle

From the repository root, run `Rscript tests/reference/dcc_rmgarch.R`. It requires R 4.6.2,
`rugarch` 1.5.6, `rmgarch` 1.4.3 and `jsonlite`. The script reads the committed CSV files without
regenerating data, fits sGARCH(1,1)-normal margins with `multifit`/`solnp`, then passes those fits
to `dccfit`/`solnp`. It writes `fixtures/rmgarch_dcc.json` with full-precision coefficients,
likelihood, all four information criteria, two-stage standard errors and correlations at
observations 1, floor(T/2), and T. The FDCC names are normalized to `fdcca1`, `fdccb1`, etc.

The two panels are `dcc3.csv` (three series, 1,500 observations) and `grid_2c.csv` (five series,
3,000 observations). The latter is used because `grid.csv` contains only four columns.
Each panel has eleven cases: DCC(1,1), aDCC(1,1), and DCC(2,1), each with normal, Student-t
and Laplace multivariate densities; two-group FDCC-normal; and DCC-t with shape fixed at 8.
The first m-1 series form FDCC group 1, and the last forms group 2.

`tests/unit/dcc/oracles/oracle_test.cpp` contains a numeric table copied from this fixture;
update that table after regeneration. C++ tests intentionally do not parse JSON. They evaluate
the internal `Rmgarch` convention at the exact oracle point. Python reads the JSON and checks
the public paper convention against a separate NumPy recursion and density, with both default
and explicit oracle initial values. Laplace evaluation is covered by C++, since NumPy has no
Bessel K function. See `docs/guide/validation.md` for bounds, weak identification and optimizer
limitations; regeneration must not silently loosen tolerances or replace those disclosures.

### Copula-DCC oracle

From the repository root, run `Rscript tests/reference/copula_rmgarch.R` (R 4.6.2, `rugarch` 1.5.6, `rmgarch`
1.4.3, `jsonlite`). It reads `data/copula3.csv` without regenerating it, fits sGARCH(1,1)-std margins with
`multifit`/`solnp` (tol 1e-10) and passes them to `cgarchfit` with `eval.se = TRUE`. For each of the 13 cases
(static mvnorm Kendall/ML, static mvt Kendall/ML, DCC and aDCC with mvnorm and mvt, parametric and empirical
margins, and DCC-t with `mshape` fixed at 8) it writes the `[Joint]` estimates and robust errors,
`likelihood(fit)` and R at observations 1, T/2 and T. `copula_probe.R` wrote the data once and checks every
formula of `docs/superpowers/plans/2026-11-13-volengine-04b-rmgarch-copula-notes.md`; rerunning it does not
overwrite the CSV.

`tests/unit/dcc/oracles/copula_oracle_test.cpp` holds a table copied from the fixture; the point-parity tables of
`tests/unit/dcc/copula/` hold the same numbers. Update them after regeneration. Python reads the JSON and checks the
public convention against NumPy.

### DCC forecast oracle

From the repository root, run `Rscript tests/reference/dcc_forecast_rmgarch.R` (R 4.6.2, `rugarch` 1.5.6, `rmgarch`
1.4.3, `jsonlite`). It reads `data/dcc3.csv` without regenerating it, fits sGARCH(1,1)-normal margins with
`multifit`/`solnp` (tol 1e-10), fits the DCC and calls `dccforecast` (`n.roll = 0`) for five cases: DCC(1,1) mvnorm,
DCC(2,1) mvnorm and DCC(1,1) mvt with `n.ahead = 10`, and aDCC and FDCC (groups 1, 1, 2) mvnorm with `n.ahead = 1`,
because `rmgarch` refuses `n.ahead > 1` for the aDCC and gives no Q from h = 2 for the FDCC. It writes
`fixtures/rmgarch_dcc_forecast.json` with the univariate estimates of the multifit, the DCC estimates (17 digits) and
`rcor`, `rcov` and `sigma` for h = 1..n.ahead.

`tests/unit/dcc/future/forecast_oracle_test.cpp` holds a table copied from the fixture (the univariate estimates, the
DCC estimates and R, H and sigma at h = 1, 5 and 10): it fixes the univariate fits and every DCC parameter at the
`rmgarch` values, runs the `Rmgarch` convention and compares to 1e-8. Update the table after regeneration; the
univariate estimates of this fixture differ from `dcc3_fits()` in the sixth digit, so the test does not use it.
