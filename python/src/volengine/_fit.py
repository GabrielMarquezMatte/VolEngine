"""Fit results over ve_fit handles: coefficient tables, read-only views and information criteria."""

from __future__ import annotations

import ctypes
import operator
from collections.abc import Callable
from contextlib import ExitStack
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._forecast import Forecast
from ._native import (
    ELEMENT_BYTES,
    Handle,
    check,
    create_handle,
    lib,
    native_view,
    open_panel,
    read_scalar,
)
from ._spec import code_of

STATUS_NAMES = {0: "converged", 1: "max_iter", 2: "failed", 3: "insufficient_data"}
SE_KINDS = {"hessian": 0, "robust": 1}
IC_PER_OBS = 0
IC_TOTAL = 1


def status_name(code: int) -> str:
    """The name of a VE_FIT_* code; a code newer than this package is "unknown(<code>)"."""
    return STATUS_NAMES.get(code, f"unknown({code})")


class CoefTable(NamedTuple):
    names: tuple[str, ...]
    estimate: NDArray[np.float64]
    std_error: NDArray[np.float64]
    t_value: NDArray[np.float64]
    p_value: NDArray[np.float64]


class Fit:
    """One fitted model. Owns its native handle; the arrays it returns keep it alive.

    `status` is "converged", "max_iter", "failed" or "insufficient_data" ("unknown(<code>)" for a status
    added by a newer native library). A "failed" fit may carry the values of the best finite point the
    optimizer reached, or NaN (with empty series) when it reached none. An "insufficient_data" fit is
    always NaN with empty series. Check `status` before using the values.
    """

    status: str
    nobs: int
    names: tuple[str, ...]

    def __init__(self, handle: Handle) -> None:
        self._handle = handle
        pointer = handle.pointer
        self.status = status_name(read_scalar(ctypes.c_int32, lib.ve_fit_status, pointer))
        self.nobs = read_scalar(ctypes.c_size_t, lib.ve_fit_nobs, pointer)
        count = read_scalar(ctypes.c_size_t, lib.ve_fit_param_count, pointer)
        self.names = tuple(self._param_name(index) for index in range(count))

    def _param_name(self, index: int) -> str:
        text = read_scalar(ctypes.c_char_p, lib.ve_fit_param_name, self._handle.pointer, index)
        return text.decode("utf-8")

    @property
    def sigma(self) -> NDArray[np.float64]:
        return self._view(lib.ve_fit_sigma)

    @property
    def residuals(self) -> NDArray[np.float64]:
        return self._view(lib.ve_fit_residuals)

    @property
    def std_residuals(self) -> NDArray[np.float64]:
        return self._view(lib.ve_fit_std_residuals)

    @property
    def fitted(self) -> NDArray[np.float64]:
        return self._view(lib.ve_fit_fitted)

    @property
    def loglik(self) -> float:
        return read_scalar(ctypes.c_double, lib.ve_loglik, self._handle.pointer)

    def aic(self, *, per_obs: bool = True) -> float:
        return self._criterion(lib.ve_aic, per_obs)

    def bic(self, *, per_obs: bool = True) -> float:
        return self._criterion(lib.ve_bic, per_obs)

    def hqic(self, *, per_obs: bool = True) -> float:
        return self._criterion(lib.ve_hqic, per_obs)

    def sic(self, *, per_obs: bool = True) -> float:
        return self._criterion(lib.ve_sic, per_obs)

    def _criterion(self, function: Callable[..., int], per_obs: bool) -> float:
        scale = IC_PER_OBS if per_obs else IC_TOTAL
        return read_scalar(ctypes.c_double, function, self._handle.pointer, scale)

    def coef(self, se: str = "hessian") -> CoefTable:
        kind = code_of(SE_KINDS, "se", se)
        return CoefTable(
            self.names,
            self._values(lib.ve_fit_estimates),
            self._values(lib.ve_fit_std_errors, kind),
            self._values(lib.ve_fit_t_values, kind),
            self._values(lib.ve_fit_p_values, kind),
        )

    def _values(self, function: Callable[..., int], *args: Any) -> NDArray[np.float64]:
        out = np.empty(len(self.names), dtype=np.float64)
        check(function(self._handle.pointer, *args, out.ctypes.data))
        return out

    def _view(self, function: Callable[..., int]) -> NDArray[np.float64]:
        return native_view(self, function, self._handle.pointer)

    def forecast(self, horizon: int, *, mxreg: ArrayLike | None = None, vxreg: ArrayLike | None = None) -> Forecast:
        """sigma and mean for h = 1..horizon after the last observation, as rugarch's ugarchforecast (n.roll = 0).

        `mxreg` and `vxreg` are the regressors of the horizon, (horizon, k) for a spec with k of them ((horizon,)
        when k = 1), and must be absent when the spec has none. A fit with no finite point raises VolEngineError.
        See `Forecast` for what `sigma` means in eGARCH, apARCH and fGARCH.
        """
        horizon = operator.index(horizon)
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        with ExitStack() as stack:
            mean = self._future(stack, mxreg, "mxreg", horizon)
            variance = self._future(stack, vxreg, "vxreg", horizon)
            handle = create_handle(
                lib.ve_forecast_create, lib.ve_forecast_free, self._handle.pointer, horizon, mean, variance
            )
        return Forecast(handle)

    def _future(self, stack: ExitStack, values: ArrayLike | None, name: str, horizon: int) -> int | None:
        """The native panel of the regressors of the horizon, or None for a spec without them."""
        count = sum(1 for parameter in self.names if parameter.startswith(name))
        return future_panel(stack, values, name, count, horizon)


def future_panel(stack: ExitStack, values: ArrayLike | None, name: str, count: int, horizon: int) -> int | None:
    """The native panel of `count` regressors over the horizon, or None when the spec has none."""
    if count == 0:
        if values is not None:
            raise ValueError(f"{name} was given but the spec has no {name}")
        return None
    if values is None:
        raise ValueError(f"the spec has {name} = {count}, so {name} is required")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1 and count == 1:
        array = array[:, None]
    if array.shape != (horizon, count):
        raise ValueError(f"{name} must have shape {(horizon, count)}, got {array.shape}")
    if any(stride % ELEMENT_BYTES for stride in array.strides):
        array = np.ascontiguousarray(array)
    return open_panel(stack, array).pointer
