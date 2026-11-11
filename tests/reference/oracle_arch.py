"""Fit every reference series with arch and write the oracle fixture."""

import csv
import json
from pathlib import Path
from typing import Any

import arch
import numpy as np
from arch import arch_model
from numpy.typing import NDArray

REFERENCE = Path(__file__).resolve().parent
DATA = REFERENCE / "data" / "sgarch11_norm.csv"
FIXTURE = REFERENCE / "fixtures" / "arch_sgarch11_norm.json"
# arch names the GARCH terms "alpha[1]" and "beta[1]"; VolEngine uses rugarch's names.
NAMES = {"mu": "mu", "omega": "omega", "alpha[1]": "alpha1", "beta[1]": "beta1"}


def load_columns(path: Path) -> dict[str, NDArray[np.float64]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    columns = {}
    for index, name in enumerate(header):
        cells = [row[index] for row in body if row[index] != ""]
        columns[name] = np.array([float(cell) for cell in cells])
    return columns


def renamed(values: Any) -> dict[str, float]:
    return {NAMES[name]: float(value) for name, value in values.items()}


def fit(values: NDArray[np.float64], cov_type: str) -> Any:
    model = arch_model(values, mean="Constant", vol="GARCH", p=1, q=1, dist="normal", rescale=False)
    result = model.fit(disp="off", cov_type=cov_type)
    if result.convergence_flag != 0:
        raise RuntimeError(f"arch did not converge: {result.optimization_result.message}")
    return result


def fit_series(name: str, values: NDArray[np.float64]) -> dict[str, Any]:
    classic = fit(values, "classic")
    robust = fit(values, "robust")
    if not np.array_equal(classic.params, robust.params):
        raise RuntimeError(f"arch estimates differ between covariance types on series {name}")
    return {
        "series": name,
        "loglik": float(classic.loglikelihood),
        "params": renamed(classic.params),
        "se": renamed(classic.std_err),
        "robust_se": renamed(robust.std_err),
        "sigma": [float(value) for value in classic.conditional_volatility],
        "aic": float(classic.aic),
        "bic": float(classic.bic),
        "hqic": None,
        "sic": None,
    }


def main() -> None:
    columns = load_columns(DATA)
    fixture = {
        "oracle": "arch",
        "oracle_version": arch.__version__,
        "series": list(columns),
        "results": [fit_series(name, values) for name, values in columns.items()],
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
