"""Simulate the sGARCH(1,1)-normal reference series shared by every oracle script and test."""

import csv
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MU = 0.05
OMEGA = 0.01
ALPHA = 0.08
BETA = 0.9
LENGTHS = (5000, 3000, 1500)
SEED = 20261110
OUTPUT = Path(__file__).resolve().parent / "data" / "sgarch11_norm.csv"


def series_name(length: int) -> str:
    return f"t{length}"


def simulate(shocks: NDArray[np.float64]) -> NDArray[np.float64]:
    returns = np.empty(shocks.size)
    variance = OMEGA / (1.0 - ALPHA - BETA)
    for t, shock in enumerate(shocks):
        residual = math.sqrt(variance) * shock
        returns[t] = MU + residual
        variance = OMEGA + ALPHA * residual * residual + BETA * variance
    return returns


def cell(values: NDArray[np.float64], row: int) -> str:
    if row >= values.size:
        return ""
    return f"{values[row]:.17g}"


def write_csv(path: Path, columns: dict[str, NDArray[np.float64]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = max(values.size for values in columns.values())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in range(rows):
            writer.writerow(cell(values, row) for values in columns.values())


def main() -> None:
    rng = np.random.default_rng(SEED)
    columns = {series_name(length): simulate(rng.standard_normal(length)) for length in LENGTHS}
    write_csv(OUTPUT, columns)


if __name__ == "__main__":
    main()
