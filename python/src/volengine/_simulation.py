"""Simulated paths over ve_sim handles."""

from __future__ import annotations

import ctypes

import numpy as np
from numpy.typing import NDArray

from ._native import Handle, lib, native_view, read_checkpoints, read_scalar


class DccSimulation:
    """Paths simulated from a DCC fit, after its last observation.

    `returns` and `sigma` are read-only arrays of shape (paths, horizon, N); `R` and `H` of shape (paths, horizon,
    N, N) exist only for a simulation run with `correlations=True` (VolEngineError otherwise). Each step takes R
    from the exact recursion, draws a shock of covariance R (a Gaussian scale mixture for the mvt and the
    mvlaplace), and turns it into the standardized shock of each series: itself in a DCC, or through the copula and
    the margin of the series in a Copula-DCC. Path p reads only the stream (seed, p): the same seed gives the same
    paths for any number of threads, within one version of the library. The simulation does not depend on its fit.

    A simulation run with `checkpoints=` keeps only `cumulative`, shaped (paths, K, N): the sum of the returns of each
    series up to each checkpoint, added in step order. `returns` and `sigma` then raise VolEngineError, and
    `cumulative` raises it for a simulation without checkpoints. `checkpoints` is the tuple of the K checkpoints
    (empty without them).
    """

    def __init__(self, handle: Handle) -> None:
        self._handle = handle
        self._shape = (
            read_scalar(ctypes.c_size_t, lib.ve_dcc_sim_paths, handle.pointer),
            read_scalar(ctypes.c_size_t, lib.ve_dcc_sim_horizon, handle.pointer),
            read_scalar(ctypes.c_size_t, lib.ve_dcc_sim_series, handle.pointer),
        )

    @property
    def returns(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_dcc_sim_returns, self._handle.pointer).reshape(self._shape)

    @property
    def sigma(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_dcc_sim_sigma, self._handle.pointer).reshape(self._shape)

    @property
    def cumulative(self) -> NDArray[np.float64]:
        view = native_view(self, lib.ve_dcc_sim_cumulative, self._handle.pointer)
        return view.reshape(self._shape[0], -1, self._shape[2])

    @property
    def checkpoints(self) -> tuple[int, ...]:
        return read_checkpoints(lib.ve_dcc_sim_checkpoints, self._handle.pointer)

    @property
    def R(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_dcc_sim_correlation, self._handle.pointer).reshape(*self._shape, self._shape[2])

    @property
    def H(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_dcc_sim_covariance, self._handle.pointer).reshape(*self._shape, self._shape[2])


class Simulation:
    """Paths simulated from a fit (after its last observation) or from a spec with every parameter fixed.

    `returns` and `sigma` are read-only arrays of shape (paths, horizon): row p is path p, `returns` the simulated
    series and `sigma` its conditional standard deviation. The same seed gives the same paths for any number of
    threads, but only within one version of the library. The simulation does not depend on its source.

    A simulation run with `checkpoints=` keeps only `cumulative`, shaped (paths, K): the sum of the returns up to each
    checkpoint, added in step order. `returns` and `sigma` then raise VolEngineError, and `cumulative` raises it for a
    simulation without checkpoints. `checkpoints` is the tuple of the K checkpoints (empty without them).
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

    @property
    def cumulative(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_sim_cumulative, self._handle.pointer).reshape(self._shape[0], -1)

    @property
    def checkpoints(self) -> tuple[int, ...]:
        return read_checkpoints(lib.ve_sim_checkpoints, self._handle.pointer)
