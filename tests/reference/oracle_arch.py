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
from numpy.typing import NDArray

REFERENCE = Path(__file__).resolve().parent
CASES = REFERENCE / "cases.json"
DATA = tuple(REFERENCE / "data" / name for name in ("sgarch11_norm.csv", "grid.csv", "grid_2b.csv"))
FIXTURE = REFERENCE / "fixtures" / "arch.json"
ArchDist = Literal["normal", "t", "ged"]
DISTRIBUTIONS: dict[str, ArchDist] = {"norm": "normal", "std": "t", "ged": "ged"}
# arch's volatility process and its number of asymmetry terms per ARCH term (arch's order o is that times p).
VOLATILITIES = {"sgarch": ("GARCH", 0), "gjrgarch": ("GARCH", 1), "egarch": ("EGARCH", 1)}
# arch names the terms "alpha[1]", "gamma[1]", "beta[1]" and the shape "nu"; VolEngine uses rugarch's names.
LAG = re.compile(r"^(alpha|gamma|beta)\[(\d+)\]$")
# The EGARCH term in |z| is arch's alpha and rugarch's gamma, the term in z the other way round.
EGARCH_NAMES = {"alpha": "gamma", "gamma": "alpha"}
# rugarch's parameter order: VolEngine's.
ORDER = ("mu", "omega", "alpha", "beta", "gamma", "shape")
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
    if name == "nu":
        return "shape"
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


def abs_moment(dist: str, shape: float) -> float:
    """E|z| of the unit-variance distribution."""
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


def center_egarch_omega(case: dict[str, Any], params: dict[str, float | None]) -> None:
    """arch centers |z| at sqrt(2/pi) for every distribution, rugarch at E|z|: the gap moves into omega."""
    gap = abs_moment(case["dist"], params.get("shape") or 0.0) - SQRT_TWO_OVER_PI
    params["omega"] = params["omega"] + gap * sum(block(params, "gamma"))


def persistence(model: str, params: dict[str, Any]) -> float:
    beta = sum(block(params, "beta"))
    if model == "egarch":
        return beta
    return sum(block(params, "alpha")) + beta + 0.5 * sum(block(params, "gamma"))


def interior_weights(model: str, params: dict[str, Any]) -> list[float]:
    alpha, beta, gamma = block(params, "alpha"), block(params, "beta"), block(params, "gamma")
    if model == "egarch":
        return beta
    if model == "gjrgarch":
        return [0.5 * a for a in alpha] + beta + [0.5 * (a + g) for a, g in zip(alpha, gamma, strict=True)]
    return alpha + beta


def check_supported(case: dict[str, Any]) -> None:
    if case["arma"] != [0, 0] or not case["include_mean"] or case["model"] not in VOLATILITIES:
        raise ValueError(f"case {case['id']}: the arch oracle covers s, gjr and eGARCH with a constant mean")


def check_interior(case: dict[str, Any], params: dict[str, Any]) -> None:
    weights = interior_weights(case["model"], params)
    if min(weights) <= 1e-4 or persistence(case["model"], params) >= 0.999:
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
        center_egarch_omega(case, params)
    # The centered omega also depends on alpha and the shape: its standard error is comparable only in norm.
    if case["model"] == "egarch" and case["dist"] != "norm":
        se["omega"] = robust_se["omega"] = None
    check_interior(case, params)
    return {
        "case": case["id"],
        "series": case["series"],
        "loglik": float(classic.loglikelihood),
        "params": params,
        "se": se,
        "robust_se": robust_se,
        "persistence": persistence(case["model"], params),
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
