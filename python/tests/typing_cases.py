"""Statically checked by test_typing.py with mypy; never imported or run."""

from typing import assert_type

import numpy as np
from numpy.typing import NDArray

import volengine as ve


def cases(ctx: ve.Context, spec: ve.Spec, one: list[float], two: NDArray[np.float64]) -> None:
    assert_type(ctx.fit(spec, one), ve.Fit)
    assert_type(ctx.fit([spec, spec], one), NDArray[np.object_])
    assert_type(ctx.fit([spec], two), NDArray[np.object_])
    assert_type(ctx.fit(spec, two), ve.Fit | NDArray[np.object_])
