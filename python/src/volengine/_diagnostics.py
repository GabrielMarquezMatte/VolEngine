"""Diagnostic result types and native array readers."""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np

from ._native import check, lib


class DiagnosticResult(NamedTuple):
    statistic: float
    p_value: float


class SignBias(NamedTuple):
    sign: DiagnosticResult
    negative: DiagnosticResult
    positive: DiagnosticResult
    joint: DiagnosticResult


class Nyblom(NamedTuple):
    individual: dict[str, float]
    joint: float
    individual_critical: tuple[float, float, float]
    joint_critical: tuple[float, float, float] | None


def size_argument(value: int, name: str) -> int:
    value = operator.index(value)
    if not 0 <= value < 2**64:
        raise ValueError(f"{name} must be an integer in [0, 2**64), got {value}")
    return value


def read_test(function: Callable[..., int], *args: Any) -> DiagnosticResult:
    out = np.empty(2, dtype=np.float64)
    check(function(*args, out.ctypes.data))
    return DiagnosticResult(float(out[0]), float(out[1]))


def critical_values(parameters: int) -> tuple[float, float, float] | None:
    if not 1 <= parameters <= 20:
        return None
    out = np.empty(3, dtype=np.float64)
    check(lib.ve_nyblom_critical(parameters, out.ctypes.data))
    return float(out[0]), float(out[1]), float(out[2])
