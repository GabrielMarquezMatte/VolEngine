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
DATA_FILES = ("sgarch11_norm.csv", "grid.csv", "grid_2b.csv", "grid_2c.csv", "grid_2e.csv", "grid_2f.csv")
CRITERIA = ("aic", "bic", "hqic", "sic")
SYMMETRIC = ("norm", "std", "ged")
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


@cache
def load_regressors() -> dict[str, NDArray[np.float64]]:
    return read_columns(REFERENCE / "data" / "regressors_2e.csv")


def case_regressors(case: dict[str, Any], key: str) -> NDArray[np.float64] | None:
    """The (T, k) matrix of the columns a case lists under "mxreg" or "vxreg", None when it lists none."""
    if key not in case:
        return None
    return np.column_stack([load_regressors()[name] for name in case[key]])


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
        case["model"],
        p=case["p"],
        q=case["q"],
        arma=tuple(case["arma"]),
        dist=case["dist"],
        include_mean=case["include_mean"],
        submodel=case.get("submodel"),
        fixed=case.get("fixed", {}),
        variance_targeting=case.get("variance_targeting", False),
        mxreg=len(case.get("mxreg", [])),
        vxreg=len(case.get("vxreg", [])),
        archm=case.get("archm", 0),
        arfima=case.get("arfima", False),
    )


def not_estimated(case: dict[str, Any]) -> set[str]:
    """The fixed parameters and the derived ones: the iGARCH beta_q and omega under variance targeting."""
    names = set(case.get("fixed", {}))
    if case["model"] == "igarch":
        names.add(f"beta{case['q']}")
    if case.get("variance_targeting", False):
        names.add("omega")
    return names


def is_evaluation(case: dict[str, Any], result: dict[str, Any]) -> bool:
    """Every parameter fixed: the oracle evaluates the model at that point (ugarchfilter, arch's fix())."""
    return set(case.get("fixed", {})) == set(result["params"])


def derived_mask(case: dict[str, Any], names: tuple[str, ...]) -> NDArray[np.bool_]:
    """Fixed and derived parameters are not estimated: their standard error is NaN in VolEngine and NA in
    rugarch."""
    return np.array([name in not_estimated(case) for name in names])


def assert_params(
    estimate: NDArray[np.float64], expected: dict[str, float], rel: float, abs_small: float
) -> None:
    reference = np.array(list(expected.values()))
    small = np.abs(reference) < 1e-2
    error = np.abs(estimate - reference)
    allowed = np.where(small, abs_small, rel * np.abs(reference))
    assert (error <= allowed).all(), f"estimates {estimate} vs {reference}, error {error}, allowed {allowed}"


def assert_rel(values: NDArray[np.float64], expected: dict[str, float | None], rel: float) -> None:
    """A null expected value (not comparable, or the derived iGARCH beta_q) is skipped."""
    reference = np.array(list(expected.values()), dtype=float)
    compared = ~np.isnan(reference)
    error = np.abs(values[compared] / reference[compared] - 1.0)
    assert (error <= rel).all(), f"values {values} vs {reference}, relative error {error}"


def assert_se(fit: ve.Fit, case: dict[str, Any], result: dict[str, Any], robust_key: str) -> None:
    derived = derived_mask(case, fit.names)
    for kind, key in (("hessian", "se"), ("robust", robust_key)):
        std_error = fit.coef(kind).std_error
        assert (np.isnan(std_error) == derived).all(), f"{kind} SE {std_error}, derived {derived}"
        assert_rel(std_error, result[key], 5e-2)


def assert_criteria(fit: ve.Fit, result: dict[str, Any], per_obs: bool) -> None:
    for name in CRITERIA:
        if result[name] is None:
            continue
        assert getattr(fit, name)(per_obs=per_obs) == pytest.approx(result[name], abs=1e-6), name


ORACLE_CASES = [(ORACLES[name], case) for name in ORACLES for case in oracle_cases(name)]
ORACLE_CASE_IDS = [f"{oracle.name}-{case['id']}" for oracle, case in ORACLE_CASES]
OPTIMIZER = "VolEngine optimizer stops below the oracle's loglik (validation.md, known divergence)"
QUADRATURE = "rugarch's fGARCH kappa quadrature enters its variance-targeting omega (validation.md, known divergence)"
# docs/guide/validation.md, "Case selection", known divergences. strict, so that a fix has to remove the entry.
# OPTIMIZER: VolEngine reports Converged below the oracle's log-likelihood. QUADRATURE: rugarch integrates the
# fGARCH kappa with integrate()'s default tolerance; under variance targeting that error moves its likelihood.
KNOWN_DIVERGENCES = {
    "rugarch-aparch11_sged/fgarch_allgarch11_sstd": OPTIMIZER,
    "rugarch-aparch11_sged/fgarch_allgarch11_jsu": OPTIMIZER,
    "rugarch-egarch11_jsu/fgarch_allgarch11_sstd": OPTIMIZER,
    "rugarch-arfima_sgarch11_norm/arfima_csgarch11_norm": OPTIMIZER,
    "rugarch-fgarch_allgarch11_norm/fgarch_allgarch11_norm_vt": QUADRATURE,
}


def comparison_params() -> list[Any]:
    """ORACLE_CASES with the known divergences expected to fail the comparison."""
    assert set(KNOWN_DIVERGENCES) <= set(ORACLE_CASE_IDS)
    params = []
    for (oracle, case), case_id in zip(ORACLE_CASES, ORACLE_CASE_IDS, strict=True):
        marks = []
        if case_id in KNOWN_DIVERGENCES:
            marks.append(pytest.mark.xfail(strict=True, reason=KNOWN_DIVERGENCES[case_id]))
        params.append(pytest.param(oracle, case, id=case_id, marks=marks))
    return params


@pytest.mark.parametrize("oracle", list(ORACLES))
def test_fixture_covers_its_cases(oracle):
    expected = [case["id"] for case in oracle_cases(oracle)]
    assert load_fixture(oracle)["cases"] == expected
    assert [result["case"] for result in load_fixture(oracle)["results"]] == expected


def test_cases_name_known_series():
    assert {case["series"] for case in load_cases()} == set(load_series())


@pytest.mark.parametrize(("oracle", "case"), comparison_params())
def test_matches_oracle(ctx, oracle, case):
    result = oracle_result(oracle.name, case["id"])
    fit = ctx.fit(
        case_spec(case),
        load_series()[case["series"]],
        mxreg=case_regressors(case, "mxreg"),
        vxreg=case_regressors(case, "vxreg"),
        options=fit_options(oracle),
    )
    assert fit.status == "converged"
    assert fit.names == tuple(result["params"]) == tuple(result["se"]) == tuple(result[oracle.robust_key])
    if is_evaluation(case, result):
        assert_evaluation(fit, result, oracle.per_obs)
        return
    assert fit.loglik >= result["loglik"] - 1e-6 * abs(result["loglik"])
    assert fit.loglik == pytest.approx(result["loglik"], rel=1e-4)
    assert_params(fit.coef().estimate, result["params"], rel=1e-3, abs_small=1e-5)
    assert_se(fit, case, result, oracle.robust_key)
    assert fit.sigma.shape == (len(result["sigma"]),)
    assert np.max(np.abs(fit.sigma / np.array(result["sigma"]) - 1.0)) <= 1e-3
    assert_criteria(fit, result, oracle.per_obs)


def assert_evaluation(fit: ve.Fit, result: dict[str, Any], per_obs: bool) -> None:
    """The same computation on both sides, without an optimizer: loglik and sigma to 1e-9 (relative)."""
    assert np.array_equal(fit.coef().estimate, np.array(list(result["params"].values())))
    assert np.isnan(fit.coef("hessian").std_error).all() and np.isnan(fit.coef("robust").std_error).all()
    assert fit.loglik == pytest.approx(result["loglik"], rel=1e-9)
    assert np.max(np.abs(fit.sigma / np.array(result["sigma"]) - 1.0)) <= 1e-9
    assert_criteria(fit, result, per_obs)


def block(params: dict[str, float], name: str) -> list[float]:
    return [value for key, value in params.items() if key.rstrip("0123456789") == name]


def below_zero(case: dict[str, Any], params: dict[str, float], persistence: float) -> float:
    """P(z < 0): 0.5 for the symmetric distributions; for the skewed ones, read back from the oracle's gjrGARCH
    persistence P = sum(alpha) + sum(beta) + pi sum(gamma), which integrates it."""
    if case["dist"] in SYMMETRIC:
        return 0.5
    alpha, beta, gamma = block(params, "alpha"), block(params, "beta"), block(params, "gamma")
    pi = (persistence - sum(alpha) - sum(beta)) / sum(gamma)
    assert 0.0 < pi < 1.0
    return pi


def named(params: dict[str, float], name: str) -> dict[str, float]:
    return {key: value for key, value in params.items() if key.rstrip("0123456789") == name}


def interior_weights(case: dict[str, Any], params: dict[str, float], persistence: float) -> dict[str, float]:
    """The non-negative entries that sum to the persistence P, by position; a power-family weight is
    alpha kappa = P - beta. In gjrGARCH with gamma_i fixed and alpha_i estimated they are alpha_i - max(0, -gamma_i)
    and max(0, -gamma_i) + pi gamma_i."""
    alpha, beta, gamma = named(params, "alpha"), named(params, "beta"), named(params, "gamma")
    assert (len(alpha), len(beta)) == (case["p"], case["q"])
    if case["model"] == "egarch":
        return beta
    if case["model"] == "gjrgarch":
        pi = below_zero(case, params, persistence)
        weights: dict[str, float] = {}
        for (a_name, a), (g_name, g) in zip(alpha.items(), gamma.items(), strict=True):
            if g_name in not_estimated(case) and a_name not in not_estimated(case):
                weights |= {a_name: a - max(0.0, -g), g_name: max(0.0, -g) + pi * g}
                continue
            weights |= {a_name: (1 - pi) * a, g_name: pi * (a + g)}
        return weights | beta
    if case["model"] in ("aparch", "fgarch"):
        assert case["p"] == 1
        return {"alpha1": persistence - sum(beta.values())} | beta
    return alpha | beta


ESTIMATED_CASES = [
    (oracle, case)
    for oracle, case in ORACLE_CASES
    if not is_evaluation(case, oracle_result(oracle.name, case["id"]))
]


@pytest.mark.parametrize(
    ("oracle", "case"), ESTIMATED_CASES, ids=[f"{oracle.name}-{case['id']}" for oracle, case in ESTIMATED_CASES]
)
def test_fixture_estimates_are_interior(oracle, case):
    """The rule holds the estimated parameters only."""
    result = oracle_result(oracle.name, case["id"])
    params, persistence = result["params"], result["persistence"]
    weights = interior_weights(case, params, persistence)
    assert all(value > 1e-4 for name, value in weights.items() if name not in not_estimated(case))
    if case["model"] not in ("igarch", "aparch", "fgarch"):
        # rugarch integrates P(z < 0) for the gjrGARCH gamma numerically.
        assert sum(weights.values()) == pytest.approx(persistence, abs=1e-9)
    if case["model"] != "igarch":
        assert persistence < 0.999
    if case["model"] in ("aparch", "fgarch"):
        asymmetry = named(params, "eta1") | named(params, "gamma")
        assert all(abs(value) < 0.999 for name, value in asymmetry.items() if name not in not_estimated(case))
    if case["model"] == "csgarch":
        assert params["eta11"] - persistence > 1e-4
        assert params["eta21"] > 1e-4
    estimated = {name: value for name, value in params.items() if name not in not_estimated(case)}
    assert_distribution_interior(case, estimated)
    # rugarch bounds a variance regressor's coefficient to [0, 100]: an estimate there is not an optimum.
    assert all(0.0 < value < 100.0 for value in named(estimated, "vxreg").values())
    # rugarch bounds the ARFIMA d to [1e-8, 0.5].
    if "arfima" in estimated:
        assert 1e-6 < estimated["arfima"] < 0.5 - 1e-6


def assert_distribution_interior(case: dict[str, Any], params: dict[str, float]) -> None:
    """|rho| < 0.999 (nig, ghyp), xi in (0.05, 20) (sstd, sged), 1 / shape > 1e-4 and |ghlambda| < 10."""
    if case["dist"] in ("nig", "ghyp") and "skew" in params:
        assert abs(params["skew"]) < 0.999
    if case["dist"] in ("sstd", "sged") and "skew" in params:
        assert 0.05 < params["skew"] < 20.0
    if "shape" in params:
        assert 1.0 / params["shape"] > 1e-4
    if "ghlambda" in params:
        assert abs(params["ghlambda"]) < 10.0
