"""Fit every case of cases.json that lists arch and write the oracle fixture."""

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

import arch
import numpy as np
from arch import arch_model
from arch.univariate import SkewStudent
from numpy.typing import NDArray

REFERENCE = Path(__file__).resolve().parent
CASES = REFERENCE / "cases.json"
DATA = tuple(REFERENCE / "data" / name for name in ("sgarch11_norm.csv", "grid.csv", "grid_2b.csv", "grid_2c.csv"))
FIXTURE = REFERENCE / "fixtures" / "arch.json"
ArchDist = Literal["normal", "t", "ged", "skewt"]
DISTRIBUTIONS: dict[str, ArchDist] = {"norm": "normal", "std": "t", "ged": "ged", "sstd": "skewt"}
# arch's volatility process and its number of asymmetry terms per ARCH term (arch's order o is that times p).
VOLATILITIES = {"sgarch": ("GARCH", 0), "gjrgarch": ("GARCH", 1), "egarch": ("EGARCH", 1)}
# arch names the terms "alpha[1]", "gamma[1]", "beta[1]", the t shape "nu" and the skew-t (Hansen) shape "eta"
# and skewness "lambda"; VolEngine uses rugarch's names.
LAG = re.compile(r"^(alpha|gamma|beta)\[(\d+)\]$")
SHAPES = {"nu": "shape", "eta": "shape", "lambda": "skew"}
# The EGARCH term in |z| is arch's alpha and rugarch's gamma, the term in z the other way round.
EGARCH_NAMES = {"alpha": "gamma", "gamma": "alpha"}
# rugarch's parameter order: VolEngine's.
ORDER = ("mu", "omega", "alpha", "beta", "gamma", "skew", "shape")
SQRT_TWO_OVER_PI = math.sqrt(2.0 / math.pi)


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


def volengine_name(model: str, name: str) -> str:
    if name in SHAPES:
        return SHAPES[name]
    match = LAG.match(name)
    if match is None:
        return name
    if model == "egarch":
        return f"{EGARCH_NAMES.get(match[1], match[1])}{match[2]}"
    return f"{match[1]}{match[2]}"


def order_key(name: str) -> tuple[int, str]:
    return ORDER.index(name.rstrip("0123456789")), name


def renamed(model: str, values: Any) -> dict[str, float | None]:
    named = {volengine_name(model, name): float(value) for name, value in values.items()}
    return {name: named[name] for name in sorted(named, key=order_key)}


def skew_student(raw: Any) -> list[float]:
    """arch's SkewStudent parameters (eta, lambda) from the raw arch estimates."""
    return [float(raw["eta"]), float(raw["lambda"])]


def abs_moment(dist: str, raw: Any) -> float:
    """E|z| of the unit-variance distribution at the raw arch estimates; for the skew-t, -2 E[z; z < 0] (E z = 0)."""
    if dist == "sstd":
        return -2.0 * SkewStudent().partial_moment(1, 0.0, skew_student(raw))
    shape = float(raw.get("nu", 0.0))
    if dist == "std":
        log_ratio = math.lgamma(0.5 * (shape - 1.0)) - math.lgamma(0.5 * shape)
        return math.sqrt(shape - 2.0) * math.exp(log_ratio) / math.sqrt(math.pi)
    if dist == "ged":
        log_scale = 0.5 * (-2.0 / shape * math.log(2.0) + math.lgamma(1.0 / shape) - math.lgamma(3.0 / shape))
        log_moment = log_scale + math.log(2.0) / shape + math.lgamma(2.0 / shape) - math.lgamma(1.0 / shape)
        return math.exp(log_moment)
    return SQRT_TWO_OVER_PI


def block(params: dict[str, Any], name: str) -> list[float]:
    return [value for key, value in params.items() if key.rstrip("0123456789") == name]


def below_zero(dist: str, raw: Any) -> float:
    """P(z < 0): 0.5 for the symmetric distributions, the skew-t CDF at 0 otherwise."""
    if dist == "sstd":
        return float(SkewStudent().cdf(np.zeros(1), skew_student(raw))[0])
    return 0.5


def center_egarch_omega(case: dict[str, Any], params: dict[str, float | None], raw: Any) -> None:
    """arch centers |z| at sqrt(2/pi) for every distribution, rugarch at E|z|: the gap moves into omega."""
    gap = abs_moment(case["dist"], raw) - SQRT_TWO_OVER_PI
    params["omega"] = params["omega"] + gap * sum(block(params, "gamma"))


def convert_skew(params: dict[str, Any], std_errors: tuple[dict[str, Any], ...]) -> None:
    """Hansen's lambda to Fernandez and Steel's xi = sqrt((1 + lambda) / (1 - lambda)), the same skew-t; the
    standard errors by the delta method, d xi / d lambda = 1 / (xi (1 - lambda)^2)."""
    skewness = params["skew"]
    xi = math.sqrt((1.0 + skewness) / (1.0 - skewness))
    params["skew"] = xi
    for se in std_errors:
        se["skew"] = se["skew"] / (xi * (1.0 - skewness) ** 2)


def persistence(model: str, params: dict[str, Any], pi: float) -> float:
    beta = sum(block(params, "beta"))
    if model == "egarch":
        return beta
    return sum(block(params, "alpha")) + beta + pi * sum(block(params, "gamma"))


def interior_weights(model: str, params: dict[str, Any], pi: float) -> list[float]:
    """For gjrGARCH ((1 - pi) alpha, beta, pi (alpha + gamma)), with pi = P(z < 0)."""
    alpha, beta, gamma = block(params, "alpha"), block(params, "beta"), block(params, "gamma")
    if model == "egarch":
        return beta
    if model == "gjrgarch":
        return [(1.0 - pi) * a for a in alpha] + beta + [pi * (a + g) for a, g in zip(alpha, gamma, strict=True)]
    return alpha + beta


def check_supported(case: dict[str, Any]) -> None:
    if case["arma"] != [0, 0] or not case["include_mean"] or case["model"] not in VOLATILITIES:
        raise ValueError(f"case {case['id']}: the arch oracle covers s, gjr and eGARCH with a constant mean")


def distribution_interior(params: dict[str, Any]) -> bool:
    """xi in (0.05, 20) and 1 / shape > 1e-4."""
    if "skew" in params and not 0.05 < params["skew"] < 20.0:
        return False
    return "shape" not in params or 1.0 / params["shape"] > 1e-4


def check_interior(case: dict[str, Any], params: dict[str, Any], pi: float) -> None:
    weights = interior_weights(case["model"], params, pi)
    interior = min(weights) > 1e-4 and persistence(case["model"], params, pi) < 0.999
    if not interior or not distribution_interior(params):
        raise RuntimeError(f"arch estimate on case {case['id']} is on the boundary: {params}")


def fit(case: dict[str, Any], values: NDArray[np.float64], cov_type: Literal["classic", "robust"]) -> Any:
    dist = DISTRIBUTIONS[case["dist"]]
    vol, asymmetry = VOLATILITIES[case["model"]]
    o = asymmetry * case["p"]
    model = arch_model(values, mean="Constant", vol=vol, p=case["p"], o=o, q=case["q"], dist=dist, rescale=False)
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
    params = renamed(case["model"], classic.params)
    se = renamed(case["model"], classic.std_err)
    robust_se = renamed(case["model"], robust.std_err)
    if case["model"] == "egarch":
        center_egarch_omega(case, params, classic.params)
    # The centered omega also depends on alpha and the shape: its standard error is comparable only in norm.
    if case["model"] == "egarch" and case["dist"] != "norm":
        se["omega"] = robust_se["omega"] = None
    if case["dist"] == "sstd":
        convert_skew(params, (se, robust_se))
    pi = below_zero(case["dist"], classic.params)
    check_interior(case, params, pi)
    return {
        "case": case["id"],
        "series": case["series"],
        "loglik": float(classic.loglikelihood),
        "params": params,
        "se": se,
        "robust_se": robust_se,
        "persistence": persistence(case["model"], params, pi),
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
