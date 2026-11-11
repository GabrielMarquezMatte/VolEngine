# Reference data and oracle fixtures

This directory holds the simulated series and the results of two independent oracles, R's
[`rugarch`](https://cran.r-project.org/package=rugarch) and Python's [`arch`](https://pypi.org/project/arch/),
that `python/tests/test_oracles.py` compares VolEngine against. Conventions and tolerances are documented in
[`docs/guide/validation.md`](../../docs/guide/validation.md).

| Path | Content |
|------|---------|
| `generate_data.py` | Simulates every series: the plan-1 sGARCH(1,1)-normal series (seed 20261110), the grid series (seed 20261111) and the GARCH-variant series (seed 20261112) |
| `cases.json` | The validation cases: series, model, orders, distribution and the oracles each one is compared with |
| `oracle_rugarch.R` | Fits every case that lists `rugarch` and writes its fixture |
| `oracle_arch.py` | Fits every case that lists `arch` and writes its fixture |
| `data/sgarch11_norm.csv` | Plan-1 series, one column per length (`t5000`, `t3000`, `t1500`), 17 significant digits; shorter series end in empty cells |
| `data/grid.csv` | Grid series, T = 3000 each (`sgarch21_std`, `arma11_sgarch11_ged`, `sgarch12_norm`, `sgarch22_ged`), 17 significant digits |
| `data/grid_2b.csv` | GARCH-variant series, T = 3000 each (`egarch11_std`, `gjrgarch11_norm`, `aparch11_ged`, `igarch11_norm`, `csgarch11_std`, `fgarch_allgarch11_norm`), 17 significant digits |
| `fixtures/rugarch.json` | `rugarch` results |
| `fixtures/arch.json` | `arch` results |

The files under `data/` and `fixtures/` are generated and committed. The tests read only these files: running
them needs neither R nor `arch`. Series names are unique across the data files.

## Case format

`cases.json` is a list of

```
{"id": str, "series": str, "model": str, "submodel": str (fgarch only), "p": int, "q": int, "arma": [m, n],
 "include_mean": bool, "dist": "norm" | "std" | "ged", "oracles": ["rugarch", "arch"]}
```

`model` and `submodel` are the names of VolEngine's Python `Spec` (`"sgarch"`, `"egarch"`, `"gjrgarch"`,
`"aparch"`, `"igarch"`, `"csgarch"`, `"fgarch"`; for `"fgarch"` a submodel such as `"allgarch"`). The scripts
translate them: `rugarch`'s `model = "sGARCH"`, `"eGARCH"`, ... and `submodel = "ALLGARCH"`, ...; `arch`'s
`vol = "GARCH"` (with `o = p` for gjrGARCH) or `"EGARCH"`. `arch` covers sGARCH, gjrGARCH and eGARCH only. `p` is
the ARCH order (α terms) and `q` the GARCH order (β terms), as in VolEngine's `Spec` and `arch`, and as
`rugarch`'s `garchOrder = c(p, q)`. The id is `<series>/<model>`.

## Fixture format

```
{"oracle": str, "oracle_version": str, "cases": [str],
 "results": [{"case": str, "series": str, "loglik": float, "params": {name: float}, "se": {name: float},
              "robust_se": {name: float}, "persistence": float, "sigma": [float], "aic": float, "bic": float,
              "hqic": float | null, "sic": float | null}]}
```

`cases` lists the ids of the cases that name the oracle, in `cases.json` order, and `results` follows it.
Parameter names follow `rugarch`, which VolEngine uses: `mu`, `ar1..`, `ma1..`, `omega`, `alpha1..`,
`beta1..`, `gamma1..`, `eta11..`, `eta21..`, `delta` or `lambda`, `shape`, in that order; `oracle_arch.py` renames
`alpha[i]`, `gamma[i]`, `beta[j]` and `nu` and reorders them. For eGARCH it also swaps `arch`'s `alpha` (the term
in |z|) and `gamma` (the term in z) into `rugarch`'s `gamma` and `alpha`, and moves the centering of |z| into
`omega` (see `docs/guide/validation.md`). A standard error is `null` where it is not comparable (the eGARCH
`omega` of `arch` outside norm) or not estimated (the derived iGARCH `beta_q`). `persistence` is the oracle's P
(`rugarch`'s `persistence()`, the transitory one for csGARCH; for `arch` computed from the estimates), used by
the interior check. Criteria stay on the oracle's own
scale: per observation for `rugarch`, totals for `arch` (which reports no HQIC or SIC, hence `null`). The
`rugarch` results carry one extra key, `qml_robust_se`: `rugarch`'s `robust_se` is a Newey-West estimator, and
`qml_robust_se` is the QML sandwich (VolEngine's robust definition) rebuilt from `rugarch`'s own Hessian and
per-observation scores.

## Regenerating

Requirements, needed only to regenerate:

- R with `rugarch` and `jsonlite` (`install.packages(c("rugarch", "jsonlite"))`).
- `arch` in the Python test environment (`python/.venv/Scripts/pip install arch`). It is deliberately not part
  of the package's `test` extra.

From the repository root:

```
python/.venv/Scripts/python tests/reference/generate_data.py
Rscript tests/reference/oracle_rugarch.R
python/.venv/Scripts/python tests/reference/oracle_arch.py
```

The scripts stop with an error if an oracle reports non-convergence or an estimate on the boundary: a weight
≤ 1e-4 (the non-negative entries that sum to the persistence P: α and β; β alone in eGARCH; ½α, β, ½(α + γ) in
gjrGARCH; ακ and β in apARCH and fGARCH), P ≥ 0.999 (not for iGARCH, where P = 1), |η₁| or the apARCH |γ| ≥ 0.999,
or, in csGARCH, ρ − P ≤ 1e-4 or φ ≤ 1e-4. `oracle_rugarch.R` shrinks `rugarch`'s numerical Hessian step
(`hess.d = 1e-3`); `docs/guide/validation.md` explains why. After regenerating, record the oracle versions and
the observed differences there.

Regenerating on another machine can change the last digits: on the plan-2b machine `generate_data.py` rewrites
`grid.csv` at the last digit of some entries, and the plan-1/2a fixture results come back with the same estimates,
ℓ, σ and criteria but standard errors that differ in the ninth digit. The committed plan-1/2a data and results
were therefore kept as they are; only the new series and cases were added (see `docs/guide/validation.md`).
