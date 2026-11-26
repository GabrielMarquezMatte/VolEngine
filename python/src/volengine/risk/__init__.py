"""VaR and ES at a confidence in (0, 1) (0.99: the 1% tail), both positive as losses, in units of the return."""

import ctypes
import math
from collections.abc import Callable
from contextlib import ExitStack
from typing import Literal, TypedDict, Unpack

import numpy as np
from numpy.typing import ArrayLike

from .._native import check, lib, open_panel
from .._spec import DISTRIBUTIONS, code_of
from ..api import panel_array

__all__ = ["DistParameters", "empirical", "parametric"]

KINDS = {"log": 0, "simple": 1}


def _measures(call: Callable[..., int], *args: object) -> tuple[float, float]:
    var = ctypes.c_double()
    es = ctypes.c_double()
    check(call(*args, ctypes.byref(var), ctypes.byref(es)))
    return var.value, es.value


def empirical(
    returns: ArrayLike, weights: ArrayLike, *, kind: Literal["log", "simple"] = "log", confidence: float
) -> tuple[float, float]:
    """VaR and ES of the portfolio sum_j weights[j] * s_j over the rows of a (T, N) array (dates or simulated paths),
    s = exp(r) - 1 for log returns and s = r for simple ones; the quantile is numpy.quantile's linear one at
    1 - confidence and the ES minus the mean at or below it. A 1-D array is one series; any strides are read in
    place, so `simulation.cumulative[:, k, :]` needs no copy. The weights are not normalized."""
    array, _ = panel_array(returns)
    weight_array = np.ascontiguousarray(weights, dtype=np.float64).reshape(-1)
    if weight_array.shape[0] != array.shape[1]:
        raise ValueError(f"weights has {weight_array.shape[0]} values for {array.shape[1]} series")
    code = code_of(KINDS, "kind", kind)
    with ExitStack() as stack:
        panel = open_panel(stack, array)
        return _measures(lib.ve_risk_empirical, panel.pointer, weight_array.ctypes.data, code, confidence)


class DistParameters(TypedDict, total=False):
    """The density's parameters, with the fit's names and domains; one the density lacks is ignored."""

    skew: float | None
    shape: float | None
    ghlambda: float | None


def _value(parameters: DistParameters, name: Literal["skew", "shape", "ghlambda"]) -> float:
    value = parameters.get(name)
    if value is None:
        return math.nan
    return value


def parametric(
    mean: float,
    sigma: float,
    *,
    confidence: float,
    dist: str = "norm",
    **parameters: Unpack[DistParameters],
) -> tuple[float, float]:
    """VaR = -(mean + sigma q) and ES = -(mean + sigma E[z | z <= q]) for the return mean + sigma * z, z the
    standardized `dist` ("norm", "std", "ged", "sstd", "sged", "nig", "ghyp", "jsu") with the fit's `skew`, `shape`
    and `ghlambda` (keywords), q its quantile at 1 - confidence. A parameter the density needs and that is missing or
    None raises VolEngineError."""
    code = code_of(DISTRIBUTIONS, "dist", dist)
    return _measures(
        lib.ve_risk_parametric,
        code,
        _value(parameters, "skew"),
        _value(parameters, "shape"),
        _value(parameters, "ghlambda"),
        mean,
        sigma,
        confidence,
    )
