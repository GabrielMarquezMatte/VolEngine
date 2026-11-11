"""Public API: the fitting context and the re-exported spec, options and result types."""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from contextlib import ExitStack
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._fit import CoefTable, Fit, aic, bic, hqic, loglik, sic
from ._native import Handle, VolEngineError, check, create_handle, int32, lib
from ._spec import FitOptions, Spec

__all__ = [
    "CoefTable",
    "Context",
    "Fit",
    "FitOptions",
    "Spec",
    "VolEngineError",
    "aic",
    "bic",
    "hqic",
    "loglik",
    "sic",
]

ELEMENT_BYTES = np.dtype(np.float64).itemsize


def panel_array(data: ArrayLike) -> tuple[NDArray[np.float64], bool]:
    array = np.asarray(data, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(-1, 1), True
    if array.ndim == 2:
        return array, False
    raise ValueError(f"data must be 1D (T,) or 2D (T, N), got {array.ndim} dimensions")


def require_integral(array: NDArray[Any], name: str) -> None:
    if array.dtype.kind in "iu" and not (array < 0).any():
        return
    if array.dtype.kind == "f" and np.isfinite(array).all() and (array >= 0).all() and (array == np.floor(array)).all():
        return
    raise ValueError(f"{name} must hold non-negative integers, got dtype {array.dtype} with values {array.tolist()}")


def range_array(values: ArrayLike, series_count: int, name: str) -> NDArray[np.uintp]:
    array = np.asarray(values)
    if array.shape != (series_count,):
        raise ValueError(f"{name} must have one entry per series ({series_count}), got shape {array.shape}")
    require_integral(array, name)
    return np.ascontiguousarray(array, dtype=np.uintp)


def open_panel(stack: ExitStack, array: NDArray[np.float64]) -> Handle:
    panel = stack.enter_context(
        create_handle(lib.ve_panel_create, lib.ve_panel_free, array.ctypes.data, array.shape[0], array.shape[1])
    )
    stride_t, stride_n = (stride // ELEMENT_BYTES for stride in array.strides)
    check(lib.ve_panel_set_strides(panel.pointer, stride_t, stride_n))
    return panel


def set_ranges(panel: Handle, first: NDArray[np.uintp], size: NDArray[np.uintp]) -> None:
    check(lib.ve_panel_set_ranges(panel.pointer, first.ctypes.data, size.ctypes.data))


def set_jobs(batch: Handle, spec_count: int, series_count: int) -> None:
    job_spec = np.repeat(np.arange(spec_count, dtype=np.uintp), series_count)
    job_series = np.tile(np.arange(series_count, dtype=np.uintp), spec_count)
    check(lib.ve_batch_set_jobs(batch.pointer, job_spec.ctypes.data, job_series.ctypes.data, job_spec.size))


def open_batch(stack: ExitStack, specs: Sequence[Spec], panel: Handle, options: FitOptions | None) -> Handle:
    batch = stack.enter_context(create_handle(lib.ve_batch_create, lib.ve_batch_free))
    spec_handles = [stack.enter_context(spec.native_handle()) for spec in specs]
    pointers = (ctypes.c_void_p * len(spec_handles))(*(handle.pointer for handle in spec_handles))
    check(lib.ve_batch_set_specs(batch.pointer, pointers, len(spec_handles)))
    check(lib.ve_batch_set_panel(batch.pointer, panel.pointer))
    if options is not None:
        options_handle = stack.enter_context(options.native_handle())
        check(lib.ve_batch_set_options(batch.pointer, options_handle.pointer))
    return batch


def shaped(fits: list[Fit], shape: tuple[int, ...]) -> NDArray[np.object_]:
    grid = np.empty(len(fits), dtype=object)
    for position, fit in enumerate(fits):
        grid[position] = fit
    return grid.reshape(shape)


class Context:
    """Execution context owning the native thread pool; fits do not depend on it afterwards.

    `threads` is between 1 and 1024 (VE_MAX_THREADS); 0, the default, means the hardware
    concurrency, capped at 1024.
    """

    def __init__(self, threads: int = 0) -> None:
        threads = int32(threads, "threads")
        self._handle = create_handle(lib.ve_context_create, lib.ve_context_free)
        check(lib.ve_context_set_threads(self._handle.pointer, threads))

    def fit(
        self,
        spec: Spec | Sequence[Spec],
        data: ArrayLike,
        *,
        options: FitOptions | None = None,
        starts: ArrayLike | None = None,
        lengths: ArrayLike | None = None,
    ) -> Fit | NDArray[np.object_]:
        """Fit every spec on every series of `data`: (T,) or (T, N), optionally with row ranges.

        One spec and a 1D series give a Fit; one spec and (T, N) data give an object array of
        shape (N,); a list of specs gives (n_specs,) or (n_specs, N).
        """
        if (starts is None) != (lengths is None):
            raise ValueError("starts and lengths must be given together")
        specs = [spec]
        if not isinstance(spec, Spec):
            specs = list(spec)
        array, one_dimensional = panel_array(data)
        series_count = array.shape[1]
        ranges = None
        if starts is not None:
            ranges = (range_array(starts, series_count, "starts"), range_array(lengths, series_count, "lengths"))
        fits = self._run(specs, array, options, ranges)
        if isinstance(spec, Spec) and one_dimensional:
            return fits[0]
        if isinstance(spec, Spec):
            return shaped(fits, (series_count,))
        if one_dimensional:
            return shaped(fits, (len(specs),))
        return shaped(fits, (len(specs), series_count))

    def _run(
        self,
        specs: list[Spec],
        array: NDArray[np.float64],
        options: FitOptions | None,
        ranges: tuple[NDArray[np.uintp], NDArray[np.uintp]] | None,
    ) -> list[Fit]:
        if any(stride % ELEMENT_BYTES for stride in array.strides):
            array = np.ascontiguousarray(array)
        series_count = array.shape[1]
        with ExitStack() as stack:
            panel = open_panel(stack, array)
            if ranges is not None:
                set_ranges(panel, ranges[0], ranges[1])
            batch = open_batch(stack, specs, panel, options)
            set_jobs(batch, len(specs), series_count)
            return self._fit_batch(batch)

    def _fit_batch(self, batch: Handle) -> list[Fit]:
        count = ctypes.c_size_t()
        check(lib.ve_batch_job_count(batch.pointer, ctypes.byref(count)))
        pointers = (ctypes.c_void_p * count.value)()
        check(lib.ve_fit_batch(self._handle.pointer, batch.pointer, pointers))
        handles = [Handle(pointer, lib.ve_fit_free) for pointer in pointers]
        return [Fit(handle) for handle in handles]
