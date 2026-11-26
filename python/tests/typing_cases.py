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


def dcc_cases(ctx: ve.Context, fit: ve.Fit, dcc_fit: ve.DccFit, spec: ve.Spec) -> None:
    assert_type(ctx.simulate(fit, horizon=5, paths=10, seed=1), ve.Simulation)
    assert_type(ctx.simulate(spec, horizon=5, paths=10, seed=1), ve.Simulation)
    assert_type(ctx.simulate(dcc_fit, horizon=5, paths=10, seed=1), ve.DccSimulation)
    assert_type(ctx.simulate(dcc_fit, horizon=5, paths=10, seed=1, correlations=True), ve.DccSimulation)
    assert_type(dcc_fit.forecast(5), ve.DccForecast)
    assert_type(dcc_fit.forecast(5).H, NDArray[np.float64])


def cancel_cases(ctx: ve.Context, fit: ve.Fit, dcc_fit: ve.DccFit, spec: ve.DccSpec, one: list[float]) -> None:
    c = ve.Cancel()
    assert_type(ctx.fit(ve.Spec(), one, cancel=c), ve.Fit)
    assert_type(ctx.fit_dcc(spec, [fit, fit], cancel=c), ve.DccFit)
    assert_type(ctx.standard_errors(dcc_fit, [fit, fit], cancel=c), ve.DccFit)
    assert_type(ctx.simulate(fit, horizon=5, paths=10, seed=1, cancel=c), ve.Simulation)
    assert_type(ctx.simulate(dcc_fit, horizon=5, paths=10, seed=1, cancel=c), ve.DccSimulation)
