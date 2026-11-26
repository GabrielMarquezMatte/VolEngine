import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import volengine as ve

OBSERVATIONS = 1500
HORIZON = 6
PATHS = 40
CANCELLED = 6


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    rng = np.random.default_rng(11)
    loadings = np.array([[1.0, 0.0, 0.0], [0.5, 0.9, 0.0], [0.3, 0.2, 0.8]])
    return rng.standard_normal((OBSERVATIONS, 3)) @ loadings.T


@pytest.fixture(scope="module")
def univariate(ctx: ve.Context, returns: np.ndarray) -> list[ve.Fit]:
    return list(ctx.fit(ve.Spec(), returns))


@pytest.fixture(scope="module")
def dcc_fit(ctx: ve.Context, univariate: list[ve.Fit]) -> ve.DccFit:
    return ctx.fit_dcc(ve.DccSpec("dcc"), univariate, standard_errors=False)


FIXED = {"mu": 0.0, "omega": 1e-6, "alpha1": 0.05, "beta1": 0.9}


def calls(ctx: ve.Context, returns: np.ndarray, univariate: list[ve.Fit], dcc_fit: ve.DccFit) -> dict[str, Callable[[ve.Cancel], Any]]:
    return {
        "fit": lambda c: ctx.fit(ve.Spec(), returns, cancel=c),
        "fit_dcc": lambda c: ctx.fit_dcc(ve.DccSpec("dcc"), univariate, cancel=c),
        "standard_errors": lambda c: ctx.standard_errors(dcc_fit, univariate, cancel=c),
        "simulate_fit": lambda c: ctx.simulate(univariate[0], horizon=HORIZON, paths=PATHS, seed=1, cancel=c),
        "simulate_spec": lambda c: ctx.simulate(ve.Spec(fixed=FIXED), horizon=HORIZON, paths=PATHS, seed=1, cancel=c),
        "simulate_dcc": lambda c: ctx.simulate(dcc_fit, horizon=HORIZON, paths=PATHS, seed=1, cancel=c),
    }


@pytest.mark.parametrize(
    "name", ["fit", "fit_dcc", "standard_errors", "simulate_fit", "simulate_spec", "simulate_dcc"]
)
def test_cancelled_before_the_call_raises(name, ctx, returns, univariate, dcc_fit):
    c = ve.Cancel()
    c.cancel()
    with pytest.raises(ve.Cancelled):
        calls(ctx, returns, univariate, dcc_fit)[name](c)


def test_dcc_fit_standard_errors_method_takes_cancel(ctx, univariate, dcc_fit):
    c = ve.Cancel()
    c.cancel()
    with pytest.raises(ve.Cancelled):
        dcc_fit.standard_errors(univariate, ctx, cancel=c)
    with pytest.raises(ve.Cancelled):
        dcc_fit.standard_errors(univariate, cancel=c)


def test_cancelled_is_a_volengine_error(ctx, returns):
    assert issubclass(ve.Cancelled, ve.VolEngineError)
    c = ve.Cancel()
    c.cancel()
    with pytest.raises(ve.Cancelled) as raised:
        ctx.fit(ve.Spec(), returns, cancel=c)
    assert raised.value.status == CANCELLED


def test_timer_cancels_a_long_fit(ctx):
    rng = np.random.default_rng(5)
    panel = rng.standard_normal((4000, 64)) * 0.01
    specs = [ve.Spec(model=model, dist="std") for model in ("sgarch", "egarch", "gjrgarch", "aparch")]
    c = ve.Cancel()
    timer = threading.Timer(0.05, c.cancel)
    start = time.perf_counter()
    timer.start()
    try:
        with pytest.raises(ve.Cancelled):
            ctx.fit(specs, panel, cancel=c)
    finally:
        timer.cancel()
    assert time.perf_counter() - start < 0.5


def test_unrequested_cancel_gives_the_same_fit(ctx, returns):
    plain = ctx.fit(ve.Spec(), returns[:, 0])
    guarded = ctx.fit(ve.Spec(), returns[:, 0], cancel=ve.Cancel())
    np.testing.assert_array_equal(plain.estimate, guarded.estimate)
