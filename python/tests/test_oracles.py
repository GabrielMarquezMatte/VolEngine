import csv
import json
from functools import cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pytest
from numpy.typing import NDArray

import volengine as ve

REFERENCE = Path(__file__).resolve().parents[2] / "tests" / "reference"
CRITERIA = {"aic": ve.aic, "bic": ve.bic, "hqic": ve.hqic, "sic": ve.sic}


class Oracle(NamedTuple):
    name: str
    var_init: str
    per_obs: bool
    robust_key: str


# rugarch reports a Newey-West robust SE; its QML sandwich (VolEngine's definition) is stored apart.
ORACLES = [
    Oracle("rugarch", "sample", True, "qml_robust_se"),
    Oracle("arch", "backcast", False, "robust_se"),
]


@cache
def load_fixture(oracle: str) -> dict[str, Any]:
    path = REFERENCE / "fixtures" / f"{oracle}_sgarch11_norm.json"
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def load_csv() -> dict[str, NDArray[np.float64]]:
    path = REFERENCE / "data" / "sgarch11_norm.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    return {name: np.array([float(row[i]) for row in body if row[i] != ""]) for i, name in enumerate(header)}


def oracle_result(oracle: str, series: str) -> dict[str, Any]:
    matches = [result for result in load_fixture(oracle)["results"] if result["series"] == series]
    assert len(matches) == 1, f"{oracle} fixture has {len(matches)} results for series {series}"
    return matches[0]


def assert_params(
    estimate: NDArray[np.float64], expected: dict[str, float], rel: float, abs_small: float
) -> None:
    reference = np.array(list(expected.values()))
    small = np.abs(reference) < 1e-2
    error = np.abs(estimate - reference)
    allowed = np.where(small, abs_small, rel * np.abs(reference))
    assert (error <= allowed).all(), f"estimates {estimate} vs {reference}, error {error}, allowed {allowed}"


def assert_rel(values: NDArray[np.float64], expected: dict[str, float], rel: float) -> None:
    reference = np.array(list(expected.values()))
    error = np.abs(values / reference - 1.0)
    assert (error <= rel).all(), f"values {values} vs {reference}, relative error {error}"


def assert_criteria(fit: ve.Fit, result: dict[str, Any], per_obs: bool) -> None:
    for name, criterion in CRITERIA.items():
        if result[name] is None:
            continue
        assert criterion(fit, per_obs=per_obs) == pytest.approx(result[name], abs=1e-6), name


def fixture_series() -> list[str]:
    series = load_fixture("rugarch")["series"]
    assert series == load_fixture("arch")["series"]
    assert sorted(series) == sorted(load_csv())
    return series


@pytest.mark.parametrize("series", fixture_series())
@pytest.mark.parametrize("oracle", ORACLES, ids=[oracle.name for oracle in ORACLES])
def test_matches_oracle(ctx, oracle, series):
    result = oracle_result(oracle.name, series)
    fit = ctx.fit(ve.Spec(), load_csv()[series], options=ve.FitOptions(var_init=oracle.var_init))
    assert fit.status == "converged"
    assert fit.names == tuple(result["params"]) == tuple(result["se"]) == tuple(result[oracle.robust_key])
    assert ve.loglik(fit) >= result["loglik"] - 1e-6 * abs(result["loglik"])
    assert ve.loglik(fit) == pytest.approx(result["loglik"], rel=1e-4)
    assert_params(fit.coef().estimate, result["params"], rel=1e-3, abs_small=1e-5)
    assert_rel(fit.coef("hessian").std_error, result["se"], 5e-2)
    assert_rel(fit.coef("robust").std_error, result[oracle.robust_key], 5e-2)
    assert fit.sigma.shape == (len(result["sigma"]),)
    assert np.max(np.abs(fit.sigma / np.array(result["sigma"]) - 1.0)) <= 1e-3
    assert_criteria(fit, result, oracle.per_obs)


@pytest.mark.parametrize("series", fixture_series())
@pytest.mark.parametrize("oracle", ["rugarch", "arch"])
def test_fixture_estimates_are_interior(oracle, series):
    params = oracle_result(oracle, series)["params"]
    assert params["alpha1"] > 0.05
    assert params["beta1"] > 0.5
    assert params["alpha1"] + params["beta1"] < 0.995
