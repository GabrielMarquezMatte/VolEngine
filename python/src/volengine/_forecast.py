"""Forecasts over ve_forecast handles."""

from __future__ import annotations

import ctypes

import numpy as np
from numpy.typing import NDArray

from ._native import Handle, lib, native_view, read_scalar


class DccForecast:
    """The forecast of a DCC fit for h = 1..horizon after its last observation, as rmgarch's dccforecast.

    `R` and `H` are read-only arrays of shape (horizon, N, N), `sigma` and `mean` of shape (horizon, N). `sigma`
    and `mean` are those of each series' own `Fit.forecast`. `R[0]` is the exact recursion one step after the last
    observation and, for h >= 2, R[h - 1] = (1 - k^(h-1)) Rbar + k^(h-1) R[0] with k the total persistence sum(a) +
    sum(b) (the aDCC g does not enter; the FDCC takes k per element; a static copula keeps its R) and Rbar the
    correlation of Qbar. `H = D R D` with D = diag(sigma). The forecast does not depend on its fit.

    `cumulative_mean` (horizon, N) and `cumulative_covariance` (horizon, N, N) are the mean and covariance of the
    cumulative return R_h = r_{T+1} + ... + r_{T+h} for h = 1..horizon: the left-to-right sum of `mean`, and
    C_h[i, j] = sum_k a_i(h - k) a_j(h - k) H[k], with a(n) = psi_0 + ... + psi_n the MA(inf) weights of each
    series' ARMA/ARFIMA mean (a = 1 without one, so C_h = sum_k H[k]). The covariance is exactly symmetric.
    """

    def __init__(self, handle: Handle) -> None:
        self._handle = handle
        self._series = read_scalar(ctypes.c_size_t, lib.ve_dcc_forecast_series, handle.pointer)
        self._horizon = read_scalar(ctypes.c_size_t, lib.ve_dcc_forecast_horizon, handle.pointer)

    @property
    def R(self) -> NDArray[np.float64]:
        view = native_view(self, lib.ve_dcc_forecast_correlation, self._handle.pointer)
        return view.reshape(self._horizon, self._series, self._series)

    @property
    def H(self) -> NDArray[np.float64]:
        view = native_view(self, lib.ve_dcc_forecast_covariance, self._handle.pointer)
        return view.reshape(self._horizon, self._series, self._series)

    @property
    def sigma(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_dcc_forecast_sigma, self._handle.pointer).reshape(self._horizon, self._series)

    @property
    def mean(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_dcc_forecast_mean, self._handle.pointer).reshape(self._horizon, self._series)

    @property
    def cumulative_mean(self) -> NDArray[np.float64]:
        view = native_view(self, lib.ve_dcc_forecast_cumulative_mean, self._handle.pointer)
        return view.reshape(self._horizon, self._series)

    @property
    def cumulative_covariance(self) -> NDArray[np.float64]:
        view = native_view(self, lib.ve_dcc_forecast_cumulative_covariance, self._handle.pointer)
        return view.reshape(self._horizon, self._series, self._series)


class Forecast:
    """The forecast of a fit for h = 1..horizon after its last observation, as rugarch's ugarchforecast.

    `sigma` and `mean` are read-only arrays of shape (horizon,). `sigma` is rugarch's sigmaFor: the square root of
    the expected variance in sGARCH, iGARCH, gjrGARCH and csGARCH, but exp(E[log sigma2] / 2) in eGARCH and
    E[sigma^delta]^(1/delta) in apARCH and fGARCH, which for h > 1 is not sqrt(E[sigma2]). `mean` is the
    conditional mean (rugarch's seriesFor). The forecast does not depend on its fit.

    `cumulative_mean` and `cumulative_sigma` (shape (horizon,)) are the mean and standard deviation of the cumulative
    return R_h = r_{T+1} + ... + r_{T+h} for h = 1..horizon: the left-to-right sum of `mean`, and
    sqrt(sum_k a(h - k)^2 sigma[k]^2) with a(n) = psi_0 + ... + psi_n the MA(inf) weights of the ARMA/ARFIMA mean
    (a = 1 without one, so the plain root of the summed variances).
    """

    def __init__(self, handle: Handle) -> None:
        self._handle = handle

    @property
    def sigma(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_forecast_sigma, self._handle.pointer)

    @property
    def mean(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_forecast_mean, self._handle.pointer)

    @property
    def cumulative_mean(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_forecast_cumulative_mean, self._handle.pointer)

    @property
    def cumulative_sigma(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_forecast_cumulative_sigma, self._handle.pointer)
