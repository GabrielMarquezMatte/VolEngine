"""Fit every case of cases.json that lists arch and write the oracle fixture."""

import csv
import json
import re
from pathlib import Path
from typing import Any, Literal

import arch
import numpy as np
from arch import arch_model
from numpy.typing import NDArray

REFERENCE = Path(__file__).resolve().parent
CASES = REFERENCE / "cases.json"
DATA = (REFERENCE / "data" / "sgarch11_norm.csv", REFERENCE / "data" / "grid.csv")
FIXTURE = REFERENCE / "fixtures" / "arch.json"
ArchDist = Literal["normal", "t", "ged"]
DISTRIBUTIONS: dict[str, ArchDist] = {"norm": "normal", "std": "t", "ged": "ged"}
# arch names the GARCH terms "alpha[1]", "beta[1]" and the shape "nu"; VolEngine uses rugarch's names.
LAG = re.compile(r"^(alpha|beta)\[(\d+)\]$")


def load_columns(paths: tuple[Path, ...]) -> dict[str, NDArray[np.float64]]:
    columns = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        header, body = rows[0], rows[1:]
        for index, name in enumerate(header):
            cells = [row[index] for row in body if row[index] != ""]
            columns[name] = np.array([float(cell) for cell in cells])
    return columns


def volengine_name(name: str) -> str:
    if name == "nu":
        return "shape"
    match = LAG.match(name)
    if match is None:
        return name
    return f"{match[1]}{match[2]}"


def renamed(values: Any) -> dict[str, float]:
    return {volengine_name(name): float(value) for name, value in values.items()}


def check_supported(case: dict[str, Any]) -> None:
    if case["arma"] != [0, 0] or not case["include_mean"]:
        raise ValueError(f"case {case['id']}: the arch oracle covers only a constant mean")


def check_interior(case_id: str, params: dict[str, float]) -> None:
    weights = {name: value for name, value in params.items() if name.startswith(("alpha", "beta"))}
    if min(weights.values()) <= 1e-4 or sum(weights.values()) >= 0.999:
        raise RuntimeError(f"arch estimate on case {case_id} is on the boundary: {weights}")


def fit(case: dict[str, Any], values: NDArray[np.float64], cov_type: Literal["classic", "robust"]) -> Any:
    dist = DISTRIBUTIONS[case["dist"]]
    model = arch_model(values, mean="Constant", vol="GARCH", p=case["p"], q=case["q"], dist=dist, rescale=False)
    result = model.fit(disp="off", cov_type=cov_type)
    if result.convergence_flag != 0:
        raise RuntimeError(f"arch did not converge on case {case['id']}: {result.optimization_result.message}")
    return result


def fit_case(case: dict[str, Any], values: NDArray[np.float64]) -> dict[str, Any]:
    check_supported(case)
    classic = fit(case, values, "classic")
    robust = fit(case, values, "robust")
    if not np.array_equal(classic.params, robust.params):
        raise RuntimeError(f"arch estimates differ between covariance types on case {case['id']}")
    params = renamed(classic.params)
    check_interior(case["id"], params)
    return {
        "case": case["id"],
        "series": case["series"],
        "loglik": float(classic.loglikelihood),
        "params": params,
        "se": renamed(classic.std_err),
        "robust_se": renamed(robust.std_err),
        "sigma": [float(value) for value in classic.conditional_volatility],
        "aic": float(classic.aic),
        "bic": float(classic.bic),
        "hqic": None,
        "sic": None,
    }


def main() -> None:
    cases = [case for case in json.loads(CASES.read_text(encoding="utf-8")) if "arch" in case["oracles"]]
    columns = load_columns(DATA)
    fixture = {
        "oracle": "arch",
        "oracle_version": arch.__version__,
        "cases": [case["id"] for case in cases],
        "results": [fit_case(case, columns[case["series"]]) for case in cases],
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
