"""Public Copula-DCC convention against an independent NumPy evaluation and real rmgarch estimates."""

import json
import math
from functools import cache
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pytest
import volengine as ve

REFERENCE = Path(__file__).resolve().parents[3] / "tests" / "reference"
FIXTURE = json.loads((REFERENCE / "fixtures" / "rmgarch_copula.json").read_text(encoding="utf-8"))
CASES = FIXTURE["results"]
# NumPy has neither the Student quantile nor the univariate cdfs, so the direct evaluation covers the empirical
# mvnorm cases; the C++ oracle covers the rest at rmgarch's convention.
DIRECT = ("dcc_mvnorm_emp", "static_mvnorm_ml_emp")
# Estimates more than 5% from rmgarch's because of the public convention, each confirmed by refitting from rmgarch's
# point, which reaches the same public optimum (or a worse one); docs/guide/validation.md, "rmgarch, Copula-DCC".
CONVENTION_SHIFTS = {
    "dcc_mvt_emp": "rank/(T+1) and the symmetric cut move the empirical tails: mshape 5.771 vs 6.393",
    "adcc_mvnorm_param": "uncentred Nbar: dccg1 0.0160 vs 0.0049, dcca1 0.0406 vs 0.0453",
    "adcc_mvt_param": "uncentred Nbar: dccg1 0.0153 vs 4.8e-8; the public loglik there beats rmgarch's start by 0.90",
}


def model_of(case):
    if not case["time_varying"]:
        return "static"
    if case["asymmetric"]:
        return "adcc"
    return "dcc"


def spec_of(case):
    method = None
    if not case["time_varying"]:
        method = case["method"].lower()
    return ve.DccSpec(model_of(case), dist=case["copula"], copula=case["transformation"], static_method=method,
                      fixed=case["fixed"] or {})


@cache
def univariate_fits():
    data = np.genfromtxt(REFERENCE / "data" / FIXTURE["file"], delimiter=",", skip_header=1)
    ctx = ve.Context(threads=1)
    fits = [ctx.fit(ve.Spec(dist="std", start=params), data[:, i], options=ve.FitOptions(tol=1))
            for i, params in enumerate(FIXTURE["univariate"])]
    for fit, params in zip(fits, FIXTURE["univariate"], strict=True):
        np.testing.assert_allclose(fit.coef().estimate, list(params.values()), rtol=0, atol=1e-12)
    return fits


@cache
def fitted(case_id):
    case = next(case for case in CASES if case["id"] == case_id)
    return ve.Context(threads=1).fit_dcc(spec_of(case), univariate_fits(), standard_errors=False)


def normal_scores(z):
    """u = rank / (T + 1) of each column (ties by order of t), then Phi^-1."""
    ranks = np.argsort(np.argsort(z, axis=0, kind="stable"), axis=0, kind="stable") + 1
    u = ranks / (len(z) + 1)
    inverse = np.vectorize(NormalDist().inv_cdf)
    return inverse(u)


def from_free(x, m):
    a = np.eye(m)
    rows, cols = np.tril_indices(m, -1)
    order = np.lexsort((rows, cols))  # R's lower.tri order: by column, then row
    a[rows[order], cols[order]] = x
    q = a @ a.T
    scale = np.sqrt(np.diag(q))
    return q / np.outer(scale, scale)


def dynamic_path(e, a, b):
    """The public recursion: Qbar = cov(e), e_s e_s' = Qbar for s <= 0."""
    qbar = np.cov(e, rowvar=False)
    previous, outer = qbar, qbar
    correlations = []
    for row in e:
        previous = (1 - a - b) * qbar + a * outer + b * previous
        scale = np.sqrt(np.diag(previous))
        correlations.append(previous / np.outer(scale, scale))
        outer = np.outer(row, row)
    return np.array(correlations)


def mvnorm_copula_terms(e, correlations):
    m = e.shape[1]
    quadratic = np.einsum("ti,ti->t", e, np.linalg.solve(correlations, e[..., None])[..., 0])
    logdet = np.linalg.slogdet(correlations)[1]
    joint = -0.5 * (m * math.log(2 * math.pi) + logdet + quadratic)
    return joint - (-0.5 * math.log(2 * math.pi) - 0.5 * e**2).sum(axis=1)


@pytest.mark.parametrize("case_id", DIRECT)
def test_copula_matches_paper_convention(case_id):
    fits = univariate_fits()
    fit = fitted(case_id)
    assert fit.status == "converged"
    params = dict(zip(fit.names, fit.coef().estimate, strict=True))
    e = normal_scores(np.column_stack([univariate.std_residuals for univariate in fits]))
    if case_id.startswith("dcc"):
        correlations = dynamic_path(e, params["dcca1"], params["dccb1"])
    if case_id.startswith("static"):
        correlations = np.broadcast_to(from_free([params["C1"], params["C2"], params["C3"]], 3), (len(e), 3, 3))
    np.testing.assert_allclose(fit.R, correlations, rtol=1e-10, atol=1e-12)
    expected = sum(univariate.loglik for univariate in fits) + mvnorm_copula_terms(e, correlations).sum()
    assert fit.loglik == pytest.approx(expected, rel=1e-10)


def test_from_free_matches_rmgarch():
    case = next(case for case in CASES if case["id"] == "static_mvt_ml_param")
    r = from_free([case["params"]["C1"], case["params"]["C2"], case["params"]["C3"]], 3)
    np.testing.assert_allclose(r, np.asarray(case["R"][0]), rtol=0, atol=1e-12)


def estimate_cases():
    cases = []
    for case in CASES:
        marks = []
        if case["id"] in CONVENTION_SHIFTS:
            marks.append(pytest.mark.xfail(strict=True, reason=CONVENTION_SHIFTS[case["id"]]))
        cases.append(pytest.param(case, id=case["id"], marks=marks))
    return cases


@pytest.mark.parametrize("case", estimate_cases())
def test_copula_estimates_near_rmgarch(case):
    fit = fitted(case["id"])
    assert fit.status == "converged"
    params = dict(zip(fit.names, fit.coef().estimate, strict=True))
    assert set(params) == set(case["params"])
    for name, expected in case["params"].items():
        if abs(expected) < 1e-6:
            assert abs(params[name]) < 1e-3, name
            continue
        assert params[name] == pytest.approx(expected, rel=5e-2), name


@pytest.mark.parametrize("case", [case for case in CASES if case["params"]], ids=lambda case: case["id"])
def test_copula_robust_errors_are_positive(case):
    errors = fitted(case["id"]).standard_errors(univariate_fits()).coef(se="robust").std_error
    assert np.isfinite(errors).all()
    assert (errors > 0).all()
