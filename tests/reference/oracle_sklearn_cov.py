"""Write fixtures/sklearn_cov.json: numpy.cov and sklearn.covariance.ledoit_wolf on four panels.

Run with `uv run --no-project --with numpy --with scikit-learn python tests/reference/oracle_sklearn_cov.py`.
The cases are `copula3` (data/copula3.csv, T = 2000, N = 3), `single` (its column A), and `wide` (T = 60, N = 40)
and `wider` (T = 30, N = 50), normal draws with seed 20261126 mixed by random loadings. Each case holds its panel
(T rows of N), so the tests read nothing else.
"""

import csv
import json
from pathlib import Path

import numpy as np
import sklearn
from sklearn.covariance import ledoit_wolf

REFERENCE = Path(__file__).resolve().parent
SEED = 20261126


def copula3() -> np.ndarray:
    with (REFERENCE / "data" / "copula3.csv").open(encoding="utf-8") as file:
        rows = list(csv.reader(file))[1:]
    return np.array(rows, dtype=np.float64)


def mixed(rng: np.random.Generator, rows: int, series: int) -> np.ndarray:
    loadings = rng.standard_normal((series, 3))
    common = rng.standard_normal((rows, 3)) @ loadings.T
    return 0.01 * (common + 2.0 * rng.standard_normal((rows, series)))


def case(name: str, returns: np.ndarray) -> dict[str, object]:
    shrunk, shrinkage = ledoit_wolf(returns, assume_centered=False)
    return {
        "id": name,
        "returns": returns.tolist(),
        "sample": np.atleast_2d(np.cov(returns, rowvar=False)).tolist(),
        "ledoit_wolf": shrunk.tolist(),
        "shrinkage": float(shrinkage),
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    panel = copula3()
    cases = [
        case("copula3", panel),
        case("single", panel[:, :1]),
        case("wide", mixed(rng, 60, 40)),
        case("wider", mixed(rng, 30, 50)),
    ]
    fixture = {"numpy_version": np.__version__, "sklearn_version": sklearn.__version__, "results": cases}
    path = REFERENCE / "fixtures" / "sklearn_cov.json"
    path.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
    for item in cases:
        print(item["id"], item["shrinkage"])


if __name__ == "__main__":
    main()
