"""Diagnostics against rugarch 1.5.6, at its coefficients and at a nearby estimated point."""

import dataclasses
import json
from functools import cache
from typing import Any

import numpy as np
import pytest
from test_oracles import (
    FIT_TOL,
    KNOWN_DIVERGENCES,
    REFERENCE,
    case_regressors,
    case_spec,
    load_cases,
    load_series,
)

import volengine as ve


@cache
def diagnostics_fixture() -> dict[str, Any]:
    path = REFERENCE / "fixtures" / "rugarch_diagnostics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def diagnostic_ids() -> list[str]:
    return [result["id"] for result in diagnostics_fixture()["results"]]


def diagnostic_case(case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    case = next(case for case in load_cases() if case["id"] == case_id)
    result = next(result for result in diagnostics_fixture()["results"] if result["id"] == case_id)
    return case, result


def diagnostic_fit(ctx: ve.Context, case: dict[str, Any], spec: ve.Spec, tol: float = FIT_TOL) -> ve.Fit:
    return ctx.fit(
        spec,
        load_series()[case["series"]],
        mxreg=case_regressors(case, "mxreg"),
        vxreg=case_regressors(case, "vxreg"),
        options=ve.FitOptions(tol=tol),
    )


def fixed_spec(case: dict[str, Any], result: dict[str, Any]) -> ve.Spec:
    fixed = dict(result["params"])
    if case["model"] == "igarch":
        fixed.pop(f"beta{case['q']}")
    return dataclasses.replace(case_spec(case), fixed=fixed, variance_targeting=False)


def assert_test(actual: ve.TestResult, expected: list[float], rel: float) -> None:
    assert actual.statistic == pytest.approx(expected[0], rel=rel, abs=0.0)
    assert actual.p_value == pytest.approx(expected[1], rel=0.0, abs=1e-7)


def residual_tests(fit: ve.Fit, case: dict[str, Any], family: str) -> list[ve.TestResult]:
    d, f = sum(case["arma"]), case["p"] + case["q"]
    if family == "ljung_box":
        lags = ((1, 0), (max(2, 3 * d - 1), d), (max(5, 5 * d - 1), d))
        return [fit.ljung_box(lag, fitdf=fitdf) for lag, fitdf in lags]
    if family == "ljung_box_squared":
        return [fit.ljung_box(lag, squared=True) for lag in (1, max(2, 3 * f - 1), max(5, 5 * f - 1))]
    if family == "arch_lm":
        return [fit.arch_lm(lag, fitdf=f) for lag in (f + 1, f + 3, f + 5)]
    if family == "sign_bias":
        return list(fit.sign_bias())
    return [fit.gof(groups) for groups in (20, 30, 40, 50)]


FAMILIES = ("ljung_box", "ljung_box_squared", "arch_lm", "sign_bias", "gof")
MOMENT_DIFFERENCES = {
    "egarch11_jsu/egarch11_sstd": {"ljung_box_squared", "arch_lm", "sign_bias"},
    "egarch11_jsu/egarch11_sged": set(FAMILIES) - {"gof"},
}


def residual_params() -> list[Any]:
    params = []
    for case_id in diagnostic_ids():
        for family in FAMILIES:
            marks = []
            if family in MOMENT_DIFFERENCES.get(case_id, set()):
                reason = "rugarch's default E|z| quadrature shifts eGARCH residuals (validation.md, diagnostics)"
                marks.append(pytest.mark.xfail(strict=True, reason=reason))
            params.append(pytest.param(case_id, family, id=f"{case_id}-{family}", marks=marks))
    return params


@pytest.fixture(scope="module")
def diagnostic_fits():
    return {}


def test_diagnostics_fixture_covers_rugarch_cases():
    expected = [case["id"] for case in load_cases() if "rugarch" in case["oracles"]]
    assert diagnostic_ids() == expected


@pytest.mark.parametrize(("case_id", "family"), residual_params())
def test_diagnostics_match_rugarch(ctx, diagnostic_fits, case_id, family):
    case, result = diagnostic_case(case_id)
    if case_id not in diagnostic_fits:
        diagnostic_fits[case_id] = diagnostic_fit(ctx, case, fixed_spec(case, result))
    fit = diagnostic_fits[case_id]
    assert fit.status == "converged"
    rel = 1e-4 if case["model"] == "fgarch" else 1e-7
    if family == "gof":
        rel = 0.0
    for actual, expected in zip(residual_tests(fit, case, family), result[family], strict=True):
        if family == "gof":
            assert actual.statistic == pytest.approx(expected[0], rel=0.0, abs=1e-12)
            assert actual.p_value == pytest.approx(expected[1], rel=0.0, abs=1e-7)
            continue
        assert_test(actual, expected, rel)


NYBLOM_IDS = [
    case_id for case_id in diagnostic_ids()
    if f"rugarch-{case_id}" not in KNOWN_DIVERGENCES
]


@pytest.mark.parametrize("case_id", NYBLOM_IDS)
def test_nyblom_matches_rugarch(ctx, case_id):
    case, result = diagnostic_case(case_id)
    start = {name: value for name, value in result["params"].items() if name not in case.get("fixed", {})}
    if case["model"] == "igarch":
        start.pop(f"beta{case['q']}", None)
    if case.get("variance_targeting", False):
        start.pop("omega", None)
    # Accept the oracle start before optimization moves its scores, preserving the estimated mask.
    fit = diagnostic_fit(ctx, case, dataclasses.replace(case_spec(case), start=start), tol=1.0)
    actual = fit.nyblom()
    estimates = dict(zip(fit.names, fit.coef().estimate, strict=True))
    for name in actual.individual:
        assert estimates[name] == pytest.approx(result["params"][name], rel=1e-12, abs=1e-12)
    expected = result["nyblom"]
    assert set(actual.individual) == set(expected["individual"])
    if expected["joint"] is None:
        assert np.isnan(actual.joint)
        return
    assert actual.joint == pytest.approx(expected["joint"], rel=1e-3)
    for name, statistic in expected["individual"].items():
        assert actual.individual[name] == pytest.approx(statistic, rel=1e-3), name
