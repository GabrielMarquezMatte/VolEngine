"""Covariance estimators over a (T, N) panel of returns, series in columns, each series' mean removed.

The matrices are (N, N) and exactly symmetric. A 1-D input is one series. The panel needs at least 2 rows and finite
values; otherwise `VolEngineError`.
"""

import ctypes
from contextlib import ExitStack

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .._native import check, lib, open_panel
from ..api import panel_array

__all__ = ["ledoit_wolf", "sample"]


def sample(returns: ArrayLike) -> NDArray[np.float64]:
    """The sample covariance Z'Z / (T - 1), as numpy.cov(returns, rowvar=False)."""
    array, _ = panel_array(returns)
    out = np.empty((array.shape[1], array.shape[1]))
    with ExitStack() as stack:
        check(lib.ve_cov_sample(open_panel(stack, array).pointer, out.ctypes.data))
    return out


def ledoit_wolf(returns: ArrayLike) -> tuple[NDArray[np.float64], float]:
    """The Ledoit-Wolf (2004) shrinkage of Z'Z / T towards mu * I, mu the mean variance, and its intensity in [0, 1],
    as sklearn.covariance.ledoit_wolf(returns, assume_centered=False)."""
    array, _ = panel_array(returns)
    out = np.empty((array.shape[1], array.shape[1]))
    shrinkage = ctypes.c_double()
    with ExitStack() as stack:
        panel = open_panel(stack, array)
        check(lib.ve_cov_ledoit_wolf(panel.pointer, out.ctypes.data, ctypes.byref(shrinkage)))
    return out, shrinkage.value
