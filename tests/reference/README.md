# Reference data and oracle fixtures

This directory holds the simulated series and the results of two independent oracles, R's
[`rugarch`](https://cran.r-project.org/package=rugarch) and Python's [`arch`](https://pypi.org/project/arch/),
that `python/tests/test_oracles.py` compares VolEngine against. Conventions and tolerances are documented in
[`docs/guide/validation.md`](../../docs/guide/validation.md).

| Path | Content |
|------|---------|
| `generate_data.py` | Simulates the sGARCH(1,1)-normal series (seed 20261110) |
| `oracle_rugarch.R` | Fits every series with `rugarch` and writes its fixture |
| `oracle_arch.py` | Fits every series with `arch` and writes its fixture |
| `data/sgarch11_norm.csv` | One column per series (`t5000`, `t3000`, `t1500`), 17 significant digits; shorter series end in empty cells |
| `fixtures/rugarch_sgarch11_norm.json` | `rugarch` results |
| `fixtures/arch_sgarch11_norm.json` | `arch` results |

The files under `data/` and `fixtures/` are generated and committed. The tests read only these files: running
them needs neither R nor `arch`.

## Fixture format

```
{"oracle": str, "oracle_version": str, "series": [str],
 "results": [{"series": str, "loglik": float, "params": {name: float}, "se": {name: float},
              "robust_se": {name: float}, "sigma": [float], "aic": float, "bic": float,
              "hqic": float | null, "sic": float | null}]}
```

Parameter names follow `rugarch` (`mu`, `omega`, `alpha1`, `beta1`). Criteria stay on the oracle's own scale:
per observation for `rugarch`, totals for `arch` (which reports no HQIC or SIC, hence `null`). The `rugarch`
results carry one extra key, `qml_robust_se`: `rugarch`'s `robust_se` is a Newey-West estimator, and
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

The scripts stop with an error if an oracle reports non-convergence. After regenerating, record the oracle
versions and the observed differences in `docs/guide/validation.md`.
