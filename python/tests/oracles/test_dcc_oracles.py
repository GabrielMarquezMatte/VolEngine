"""Public DCC convention against an independent NumPy recursion and real rmgarch estimates."""

import json
import math
from functools import cache
from pathlib import Path

import numpy as np
import pytest

import volengine as ve

REFERENCE = Path(__file__).resolve().parents[3] / "tests" / "reference"
FIXTURE = json.loads((REFERENCE / "fixtures" / "rmgarch_dcc.json").read_text(encoding="utf-8"))
CASES = [case for case in FIXTURE["results"] if case["dist"] != "mvlaplace"]
ESTIMATE_LIMITS = {
    "dcc3_DCC21_mvt": "default stopping: a2 differs 5.12%, objective 4.42e-4 above oracle point",
    "grid5_DCC_mvnorm": "weak identification: oracle a=4.18e-9, default a=2.85e-4",
    "grid5_DCC_mvt": "weak identification: oracle a=2.09e-4, default a=9.44e-6",
    "grid5_aDCC_mvnorm": "boundary: oracle a=1.60e-10, default a=9.60e-6",
    "grid5_aDCC_mvt": "boundary: oracle a=8.45e-10, default a=1.15e-5",
    "grid5_DCC21_mvnorm": "boundary: oracle a1=9.19e-11, default a1=3.43e-4",
    "grid5_DCC21_mvt": "boundary: oracle a2=9.60e-8, default a2=1.24e-4",
    "grid5_FDCC_mvnorm": "weak identification: oracle a1=1.07e-5, default a1=0.01",
    "grid5_DCC_mvt_fixedshape": "weak identification: oracle a=2.45e-4, default a=4.40e-4",
}


def test_dcc_oracle_fixture_is_complete():
    assert FIXTURE["oracle_version"] == "1.4.3"
    assert len(FIXTURE["results"]) == 22
    for case in FIXTURE["results"]:
        assert set(case["params"]) == set(case["se"])
        assert all(value is not None and math.isfinite(value) for value in case["se"].values())
        m = len(FIXTURE["panels"][case["panel"]]["univariate"])
        assert np.asarray(case["R"]).shape == (3, m, m)


@cache
def panel_fits(panel_id):
    panel = FIXTURE["panels"][panel_id]
    data = np.genfromtxt(REFERENCE / "data" / panel["file"], delimiter=",", skip_header=1)
    data = data[np.isfinite(data).all(axis=1)]
    ctx = ve.Context(threads=1)
    fits = [ctx.fit(ve.Spec(start=params), data[:, i], options=ve.FitOptions(tol=1))
            for i, params in enumerate(panel["univariate"])]
    for fit, params in zip(fits, panel["univariate"], strict=True):
        np.testing.assert_allclose(fit.coef().estimate, list(params.values()), rtol=0, atol=1e-12)
    return fits


def direct_path(z, case, params):
    qbar = np.cov(z, rowvar=False)
    negative = np.minimum(z, 0)
    nbar = negative.T @ negative / len(z)
    p, q = case["order"]
    if case["model"] == "FDCC":
        return fdcc_path(z, qbar, case["groups"], params)
    a = np.array([params[f"dcca{i + 1}"] for i in range(p)])
    b = np.array([params[f"dccb{i + 1}"] for i in range(q)])
    g = np.zeros(p)
    if case["model"] == "aDCC":
        g = np.array([params[f"dccg{i + 1}"] for i in range(p)])
    intercept = (1 - a.sum() - b.sum()) * qbar - g.sum() * nbar
    history, correlations = [], []
    for t in range(len(z)):
        matrix = intercept.copy()
        for lag in range(1, p + 1):
            outer, downside = qbar, nbar
            if t >= lag:
                outer = np.outer(z[t - lag], z[t - lag])
                downside = np.outer(negative[t - lag], negative[t - lag])
            matrix += a[lag - 1] * outer + g[lag - 1] * downside
        for lag in range(1, q + 1):
            previous = qbar
            if t >= lag:
                previous = history[t - lag]
            matrix += b[lag - 1] * previous
        history.append(matrix)
        correlations.append(normalize(matrix))
    return np.array(correlations)


def normalize(matrix):
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


def fdcc_path(z, qbar, groups, params):
    a = np.array([params[f"fdcca{group}"] for group in groups])
    b = np.array([params[f"fdccb{group}"] for group in groups])
    aa, bb = np.outer(a, a), np.outer(b, b)
    intercept = (1 - aa - bb) * qbar
    previous, outer = qbar, qbar
    correlations = []
    for row in z:
        previous = intercept + aa * outer + bb * previous
        correlations.append(normalize(previous))
        outer = np.outer(row, row)
    return np.array(correlations)


def direct_loglik(z, sigma, correlations, params):
    m = z.shape[1]
    quadratic = np.einsum("ti,ti->t", z, np.linalg.solve(correlations, z[..., None])[..., 0])
    logdet = np.linalg.slogdet(correlations)[1]
    terms = -0.5 * (m * math.log(2 * math.pi) + logdet + quadratic)
    if "mshape" in params:
        shape = params["mshape"]
        constant = math.lgamma((shape + m) / 2) - math.lgamma(shape / 2)
        constant -= m / 2 * math.log(math.pi * (shape - 2))
        terms = constant - logdet / 2 - (shape + m) / 2 * np.log1p(quadratic / (shape - 2))
    return np.sum(terms) - np.log(sigma).sum()


@cache
def fitted_case(case_id, oracle_start=False):
    case = next(case for case in CASES if case["id"] == case_id)
    fits = panel_fits(case["panel"])
    model = {"DCC": "dcc", "DCC21": "dcc", "aDCC": "adcc", "FDCC": "fdcc"}[case["model"]]
    groups = None
    if model == "fdcc":
        groups = case["groups"]
    start = {}
    if oracle_start:
        start = case["params"]
    spec = ve.DccSpec(model=model, dist=case["dist"], order=tuple(case["order"]),
                      groups=groups, fixed=case["fixed"] or {}, start=start)
    return ve.Context(threads=1).fit_dcc(spec, fits, standard_errors=False)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("oracle_start", [False, True], ids=["default_start", "oracle_start"])
def test_dcc_matches_paper_convention(case, oracle_start):
    fits = panel_fits(case["panel"])
    fit = fitted_case(case["id"], oracle_start)
    assert fit.status == "converged"
    params = dict(zip(fit.names, fit.coef().estimate, strict=True))
    params.update(case["fixed"] or {})
    z = np.column_stack([univariate.std_residuals for univariate in fits])
    sigma = np.column_stack([univariate.sigma for univariate in fits])
    correlations = direct_path(z, case, params)
    np.testing.assert_allclose(fit.R, correlations, rtol=1e-10, atol=1e-12)
    assert fit.loglik == pytest.approx(direct_loglik(z, sigma, correlations, params), rel=1e-10)
    assert np.isfinite(fit.loglik)


def estimate_cases():
    cases = []
    for case in CASES:
        marks = []
        if case["id"] in ESTIMATE_LIMITS:
            marks.append(pytest.mark.xfail(strict=True, reason=ESTIMATE_LIMITS[case["id"]]))
        cases.append(pytest.param(case, id=case["id"], marks=marks))
    return cases


@pytest.mark.parametrize("case", estimate_cases())
def test_dcc_estimates_near_rmgarch(case):
    fit = fitted_case(case["id"])
    params = dict(zip(fit.names, fit.coef().estimate, strict=True))
    for name in fit.names:
        if case["model"] == "aDCC" and name.startswith("dccg"):
            assert params[name] >= 0
            continue
        assert params[name] == pytest.approx(case["params"][name], rel=5e-2)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_dcc_robust_errors_are_positive(case):
    fit = fitted_case(case["id"], True)
    errors = fit.standard_errors(panel_fits(case["panel"])).coef(se="robust").std_error
    assert np.isfinite(errors).all()
    assert (errors > 0).all()
