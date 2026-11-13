"""Forecasts against rugarch's ugarchforecast at rugarch's own estimate (tests/reference/forecast_rugarch.R)."""

import dataclasses
import json
from functools import cache
from typing import Any

import numpy as np
import pytest
from test_oracles import REFERENCE, case_regressors, case_spec, load_cases, load_series

import volengine as ve

SIGMA_REL = 1e-8
MEAN_TOL = 1e-8
# rugarch integrates the apARCH and fGARCH kappa and the sstd and sged E|z| of eGARCH with integrate()'s default
# tolerance: its kappa is off by 1.2e-8 (apARCH, norm) and its E|z| by 1.2e-6 (sged) against an integral to 1e-14,
# and the forecast inherits that (docs/guide/validation.md, "rugarch, forecast"). The first step of apARCH and
# fGARCH reads no kappa and stays at FIRST_STEP_REL.
INTEGRATED_SIGMA_REL = 5e-6
FIRST_STEP_REL = 1e-12


@cache
def load_forecasts(oracle: str) -> dict[str, Any]:
    path = REFERENCE / "fixtures" / f"{oracle}_forecast.json"
    return json.loads(path.read_text(encoding="utf-8"))


def case_by_id(case_id: str) -> dict[str, Any]:
    return next(case for case in load_cases() if case["id"] == case_id)


def forecast_ids(oracle: str) -> list[str]:
    return [result["case"] for result in load_forecasts(oracle)["results"]]


def rows(matrix: Any, first: int, last: int) -> Any:
    if matrix is None:
        return None
    return matrix[first:last]


def volengine_forecast(ctx: ve.Context, case: dict[str, Any], result: dict[str, Any], var_init: str) -> ve.Forecast:
    """The model fixed at the oracle's point on the first n observations, forecast over the fixture's horizon."""
    r = load_series()[case["series"]]
    n, horizon = result["n"], result["horizon"]
    x, v = case_regressors(case, "mxreg"), case_regressors(case, "vxreg")
    spec = dataclasses.replace(case_spec(case), fixed=result["fixed"], variance_targeting=False)
    fit = ctx.fit(spec, r[:n], mxreg=rows(x, 0, n), vxreg=rows(v, 0, n), options=ve.FitOptions(var_init=var_init))
    assert fit.status == "converged"
    return fit.forecast(horizon, mxreg=rows(x, n, n + horizon), vxreg=rows(v, n, n + horizon))


def integrated_moment(case: dict[str, Any]) -> bool:
    """Whether rugarch's forecast reads a moment it integrates numerically."""
    return case["model"] in ("aparch", "fgarch") or (case["model"] == "egarch" and case["dist"] in ("sstd", "sged"))


def assert_forecast(forecast: ve.Forecast, result: dict[str, Any], case: dict[str, Any]) -> None:
    rel = INTEGRATED_SIGMA_REL if integrated_moment(case) else SIGMA_REL
    np.testing.assert_allclose(forecast.sigma, result["sigma"], rtol=rel, atol=0.0)
    if case["model"] in ("aparch", "fgarch"):
        assert forecast.sigma[0] == pytest.approx(result["sigma"][0], rel=FIRST_STEP_REL, abs=0.0)
    expected = np.array(result["mean"])
    error = np.abs(forecast.mean - expected)
    assert (error <= MEAN_TOL * (1.0 + np.abs(expected))).all(), f"mean {forecast.mean} vs {expected}"


def test_rugarch_forecast_fixture_covers_its_cases():
    cases = [case["id"] for case in load_cases() if "rugarch" in case["oracles"]]
    assert forecast_ids("rugarch") == cases


@pytest.mark.parametrize("case_id", forecast_ids("rugarch"))
def test_forecast_matches_rugarch(ctx, case_id):
    result = next(result for result in load_forecasts("rugarch")["results"] if result["case"] == case_id)
    case = case_by_id(case_id)
    assert_forecast(volengine_forecast(ctx, case, result, "sample"), result, case)
