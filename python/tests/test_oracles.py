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
DATA_FILES = ("sgarch11_norm.csv", "grid.csv")
CRITERIA = ("aic", "bic", "hqic", "sic")
# Tighter than the default gradient tolerance; docs/guide/validation.md says why.
FIT_TOL = 1e-7


class Oracle(NamedTuple):
    name: str
    var_init: str
    per_obs: bool
    robust_key: str


# rugarch reports a Newey-West robust SE; its QML sandwich (VolEngine's definition) is stored apart.
ORACLES = {
    "rugarch": Oracle("rugarch", "sample", True, "qml_robust_se"),
    "arch": Oracle("arch", "backcast", False, "robust_se"),
}


@cache
def load_cases() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads((REFERENCE / "cases.json").read_text(encoding="utf-8")))


@cache
def load_fixture(oracle: str) -> dict[str, Any]:
    path = REFERENCE / "fixtures" / f"{oracle}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def read_columns(path: Path) -> dict[str, NDArray[np.float64]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    return {name: np.array([float(row[i]) for row in body if row[i] != ""]) for i, name in enumerate(header)}


@cache
def load_series() -> dict[str, NDArray[np.float64]]:
    series: dict[str, NDArray[np.float64]] = {}
    for name in DATA_FILES:
        columns = read_columns(REFERENCE / "data" / name)
        assert not set(columns) & set(series), f"{name} repeats a series name"
        series.update(columns)
    return series


def oracle_cases(oracle: str) -> list[dict[str, Any]]:
    return [case for case in load_cases() if oracle in case["oracles"]]


def oracle_result(oracle: str, case_id: str) -> dict[str, Any]:
    matches = [result for result in load_fixture(oracle)["results"] if result["case"] == case_id]
    assert len(matches) == 1, f"{oracle} fixture has {len(matches)} results for case {case_id}"
    return matches[0]


def fit_options(oracle: Oracle) -> ve.FitOptions:
    return ve.FitOptions(var_init=oracle.var_init, tol=FIT_TOL)


def case_spec(case: dict[str, Any]) -> ve.Spec:
    return ve.Spec(
        p=case["p"], q=case["q"], arma=tuple(case["arma"]), dist=case["dist"], include_mean=case["include_mean"]
    )


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
    for name in CRITERIA:
        if result[name] is None:
            continue
        assert getattr(fit, name)(per_obs=per_obs) == pytest.approx(result[name], abs=1e-6), name


ORACLE_CASES = [(ORACLES[name], case) for name in ORACLES for case in oracle_cases(name)]
ORACLE_CASE_IDS = [f"{oracle.name}-{case['id']}" for oracle, case in ORACLE_CASES]


@pytest.mark.parametrize("oracle", list(ORACLES))
def test_fixture_covers_its_cases(oracle):
    expected = [case["id"] for case in oracle_cases(oracle)]
    assert load_fixture(oracle)["cases"] == expected
    assert [result["case"] for result in load_fixture(oracle)["results"]] == expected


def test_cases_name_known_series():
    assert {case["series"] for case in load_cases()} == set(load_series())


@pytest.mark.parametrize(("oracle", "case"), ORACLE_CASES, ids=ORACLE_CASE_IDS)
def test_matches_oracle(ctx, oracle, case):
    result = oracle_result(oracle.name, case["id"])
    fit = ctx.fit(case_spec(case), load_series()[case["series"]], options=fit_options(oracle))
    assert fit.status == "converged"
    assert fit.names == tuple(result["params"]) == tuple(result["se"]) == tuple(result[oracle.robust_key])
    assert fit.loglik >= result["loglik"] - 1e-6 * abs(result["loglik"])
    assert fit.loglik == pytest.approx(result["loglik"], rel=1e-4)
    assert_params(fit.coef().estimate, result["params"], rel=1e-3, abs_small=1e-5)
    assert_rel(fit.coef("hessian").std_error, result["se"], 5e-2)
    assert_rel(fit.coef("robust").std_error, result[oracle.robust_key], 5e-2)
    assert fit.sigma.shape == (len(result["sigma"]),)
    assert np.max(np.abs(fit.sigma / np.array(result["sigma"]) - 1.0)) <= 1e-3
    assert_criteria(fit, result, oracle.per_obs)


@pytest.mark.parametrize(("oracle", "case"), ORACLE_CASES, ids=ORACLE_CASE_IDS)
def test_fixture_estimates_are_interior(oracle, case):
    params = oracle_result(oracle.name, case["id"])["params"]
    weights = [value for name, value in params.items() if name.startswith(("alpha", "beta"))]
    assert len(weights) == case["p"] + case["q"]
    assert min(weights) > 1e-4
    assert sum(weights) < 0.999
