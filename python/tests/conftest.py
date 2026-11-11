import math

import numpy as np
import pytest
from numpy.typing import NDArray

import volengine as ve

MU = 0.05
OMEGA = 0.01
ALPHA = 0.08
BETA = 0.9
OBSERVATIONS = 3000
SEEDS = (1, 2, 3)


def simulate_sgarch(seed: int) -> NDArray[np.float64]:
    shocks = np.random.default_rng(seed).standard_normal(OBSERVATIONS)
    returns = np.empty(OBSERVATIONS)
    variance = OMEGA / (1.0 - ALPHA - BETA)
    for t, shock in enumerate(shocks):
        residual = math.sqrt(variance) * shock
        returns[t] = MU + residual
        variance = OMEGA + ALPHA * residual * residual + BETA * variance
    return returns


@pytest.fixture(scope="session")
def series() -> NDArray[np.float64]:
    return np.column_stack([simulate_sgarch(seed) for seed in SEEDS])


@pytest.fixture(scope="session")
def ctx() -> ve.Context:
    return ve.Context(threads=2)
