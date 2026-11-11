"""Fit results over ve_fit handles: coefficient tables, read-only views and information criteria."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from ._native import Handle, check, lib, read_scalar
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
        data = ctypes.POINTER(ctypes.c_double)()
        length = ctypes.c_size_t()
        check(function(self._handle.pointer, ctypes.byref(data), ctypes.byref(length)))
        if length.value == 0:
            return read_only(np.empty(0, dtype=np.float64))
        address = ctypes.cast(data, ctypes.c_void_p).value
        buffer = (ctypes.c_double * length.value).from_address(address)
        buffer._owner = self
        return read_only(np.frombuffer(buffer, dtype=np.float64))


def read_only(array: NDArray[np.float64]) -> NDArray[np.float64]:
    array.flags.writeable = False
    return array


def loglik(fit: Fit) -> float:
    return read_scalar(ctypes.c_double, lib.ve_loglik, fit._handle.pointer)


def criterion(function: Callable[..., int], fit: Fit, per_obs: bool) -> float:
    scale = IC_PER_OBS if per_obs else IC_TOTAL
    return read_scalar(ctypes.c_double, function, fit._handle.pointer, scale)


def aic(fit: Fit, *, per_obs: bool = True) -> float:
    return criterion(lib.ve_aic, fit, per_obs)


def bic(fit: Fit, *, per_obs: bool = True) -> float:
    return criterion(lib.ve_bic, fit, per_obs)


def hqic(fit: Fit, *, per_obs: bool = True) -> float:
    return criterion(lib.ve_hqic, fit, per_obs)


def sic(fit: Fit, *, per_obs: bool = True) -> float:
    return criterion(lib.ve_sic, fit, per_obs)
