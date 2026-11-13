"""Simulated paths over ve_sim handles."""

from __future__ import annotations

import ctypes

import numpy as np
from numpy.typing import NDArray

from ._native import Handle, lib, native_view, read_scalar


class Simulation:
    """Paths simulated from a fit (after its last observation) or from a spec with every parameter fixed.

    `returns` and `sigma` are read-only arrays of shape (paths, horizon): row p is path p, `returns` the simulated
    series and `sigma` its conditional standard deviation. The same seed gives the same paths for any number of
    threads, but only within one version of the library. The simulation does not depend on its source.
    """

    def __init__(self, handle: Handle) -> None:
        self._handle = handle
        self._shape = (
            read_scalar(ctypes.c_size_t, lib.ve_sim_paths, handle.pointer),
            read_scalar(ctypes.c_size_t, lib.ve_sim_horizon, handle.pointer),
        )

    @property
    def returns(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_sim_returns, self._handle.pointer).reshape(self._shape)

    @property
    def sigma(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_sim_sigma, self._handle.pointer).reshape(self._shape)
