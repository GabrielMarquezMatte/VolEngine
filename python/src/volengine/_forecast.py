"""Forecasts over ve_forecast handles."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._native import Handle, lib, native_view


class Forecast:
    """The forecast of a fit for h = 1..horizon after its last observation, as rugarch's ugarchforecast.

    `sigma` and `mean` are read-only arrays of shape (horizon,). `sigma` is rugarch's sigmaFor: the square root of
    the expected variance in sGARCH, iGARCH, gjrGARCH and csGARCH, but exp(E[log sigma2] / 2) in eGARCH and
    E[sigma^delta]^(1/delta) in apARCH and fGARCH, which for h > 1 is not sqrt(E[sigma2]). `mean` is the
    conditional mean (rugarch's seriesFor). The forecast does not depend on its fit.
    """

    def __init__(self, handle: Handle) -> None:
        self._handle = handle

    @property
    def sigma(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_forecast_sigma, self._handle.pointer)

    @property
    def mean(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_forecast_mean, self._handle.pointer)
