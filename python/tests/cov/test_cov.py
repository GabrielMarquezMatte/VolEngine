"""Inputs of the covariance estimators: layouts, shapes and refused panels."""

import numpy as np
import pytest
from numpy.typing import NDArray

import volengine as ve


@pytest.fixture(scope="module")
def returns() -> NDArray[np.float64]:
    rng = np.random.default_rng(5)
    return rng.standard_normal((250, 6)) @ rng.standard_normal((6, 6))


def test_fortran_order_equals_c_order(returns: NDArray[np.float64]) -> None:
    np.testing.assert_array_equal(ve.cov.sample(np.asfortranarray(returns)), ve.cov.sample(returns))


def test_column_slice_equals_its_copy(returns: NDArray[np.float64]) -> None:
    sliced = returns[:, ::2]
    covariance, shrinkage = ve.cov.ledoit_wolf(sliced)
    expected, expected_shrinkage = ve.cov.ledoit_wolf(sliced.copy())
    np.testing.assert_array_equal(covariance, expected)
    assert shrinkage == expected_shrinkage


def test_results_are_exactly_symmetric(returns: NDArray[np.float64]) -> None:
    sample = ve.cov.sample(returns)
    shrunk, _ = ve.cov.ledoit_wolf(returns)
    np.testing.assert_array_equal(sample, sample.T)
    np.testing.assert_array_equal(shrunk, shrunk.T)


def test_one_dimensional_input_is_one_series() -> None:
    covariance = ve.cov.sample([1.0, 4.0, 2.0, 7.0])
    assert covariance.shape == (1, 1)
    assert covariance[0, 0] == pytest.approx(7.0)


def test_integer_lists_are_accepted() -> None:
    np.testing.assert_array_equal(ve.cov.sample([[1, 2], [3, 5], [5, 11]]), [[4.0, 9.0], [9.0, 21.0]])


def test_nan_is_refused(returns: NDArray[np.float64]) -> None:
    broken = returns.copy()
    broken[-1, 2] = np.nan
    with pytest.raises(ve.VolEngineError, match="not finite"):
        ve.cov.ledoit_wolf(broken)


def test_one_row_is_refused() -> None:
    with pytest.raises(ve.VolEngineError, match="at least 2 rows"):
        ve.cov.sample([[1.0, 2.0]])


def test_three_dimensional_input_is_refused() -> None:
    with pytest.raises(ValueError, match="2D"):
        ve.cov.sample(np.zeros((4, 2, 2)))
