"""The empirical VaR and ES against numpy, simulated paths read without a copy, and refused inputs."""

import numpy as np
import pytest
from numpy.typing import NDArray

import volengine as ve


def numpy_measures(portfolio: NDArray[np.float64], confidence: float) -> tuple[float, float]:
    quantile = float(np.quantile(portfolio, 1.0 - confidence))
    return -quantile, -float(portfolio[portfolio <= quantile].mean())


@pytest.fixture(scope="module")
def log_returns() -> NDArray[np.float64]:
    return 0.015 * np.random.default_rng(8).standard_normal((750, 4))


WEIGHTS = np.array([0.4, 0.3, 0.2, 0.1])


@pytest.mark.parametrize("confidence", [0.9, 0.95, 0.99, 0.999])
def test_empirical_matches_numpy(log_returns: NDArray[np.float64], confidence: float) -> None:
    expected = numpy_measures(np.expm1(log_returns) @ WEIGHTS, confidence)
    var, es = ve.risk.empirical(log_returns, WEIGHTS, confidence=confidence)
    assert var == pytest.approx(expected[0], rel=1e-13)
    assert es == pytest.approx(expected[1], rel=1e-13)


def test_simple_returns(log_returns: NDArray[np.float64]) -> None:
    simple = np.expm1(log_returns)
    from_log = ve.risk.empirical(log_returns, WEIGHTS, confidence=0.99)
    from_simple = ve.risk.empirical(simple, WEIGHTS, kind="simple", confidence=0.99)
    assert from_simple == pytest.approx(from_log, rel=1e-14)


def test_one_dimensional_input_is_one_series() -> None:
    var, es = ve.risk.empirical([-0.05, 0.01, 0.02, -0.02, 0.03], [1.0], kind="simple", confidence=0.8)
    assert var == pytest.approx(0.026, abs=1e-15)
    assert es == pytest.approx(0.05)


def test_dcc_paths_by_checkpoint_without_a_copy(ctx: ve.Context, log_returns: NDArray[np.float64]) -> None:
    fits = list(ctx.fit(ve.Spec(), log_returns[:, :3]))
    dcc = ctx.fit_dcc(ve.DccSpec("dcc"), fits, standard_errors=False)
    paths = ctx.simulate(dcc, horizon=10, paths=500, seed=4, checkpoints=[1, 5, 10])
    weights = np.array([0.5, 0.3, 0.2])
    slice_view = paths.cumulative[:, 1, :]
    assert not slice_view.flags.c_contiguous
    assert ve.risk.empirical(slice_view, weights, confidence=0.99) == ve.risk.empirical(
        slice_view.copy(), weights, confidence=0.99
    )


def test_wrong_weight_count_is_refused(log_returns: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="weights"):
        ve.risk.empirical(log_returns, [0.5, 0.5], confidence=0.99)


@pytest.mark.parametrize("confidence", [0.0, 1.0, float("nan")])
def test_confidence_outside_the_interval_is_refused(log_returns: NDArray[np.float64], confidence: float) -> None:
    with pytest.raises(ve.VolEngineError, match="confidence"):
        ve.risk.empirical(log_returns, WEIGHTS, confidence=confidence)


def test_nan_is_refused(log_returns: NDArray[np.float64]) -> None:
    broken = log_returns.copy()
    broken[3, 1] = np.nan
    with pytest.raises(ve.VolEngineError, match="not finite"):
        ve.risk.empirical(broken, WEIGHTS, confidence=0.99)


def test_unknown_kind_is_refused(log_returns: NDArray[np.float64]) -> None:
    with pytest.raises(ValueError, match="kind"):
        ve.risk.empirical(log_returns, WEIGHTS, kind="percent", confidence=0.99)  # type: ignore[arg-type]


def test_missing_shape_is_refused() -> None:
    with pytest.raises(ve.VolEngineError, match="domain"):
        ve.risk.parametric(0.0, 0.02, confidence=0.99, dist="std")


def test_unknown_dist_is_refused() -> None:
    with pytest.raises(ValueError, match="dist"):
        ve.risk.parametric(0.0, 0.02, confidence=0.99, dist="cauchy")


def test_normal_value() -> None:
    var, es = ve.risk.parametric(0.0, 1.0, confidence=0.99)
    assert var == pytest.approx(2.3263478740408408, rel=1e-14)
    assert es == pytest.approx(2.665214220345808, rel=1e-13)
