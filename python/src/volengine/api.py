"""Public API: the fitting context and the re-exported spec, options and result types."""

from __future__ import annotations

import ctypes
import operator
from collections.abc import Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any, TypedDict, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._cancel import Cancel, cancel_pointer
from ._dcc import DccFit, DccSpec, dcc_standard_errors, fit_dcc, open_future
from ._diagnostics import DiagnosticResult, Nyblom, SignBias
from ._fit import CoefTable, Fit, future_panel
from ._forecast import DccForecast, Forecast
from ._native import (
    Cancelled,
    Handle,
    VolEngineError,
    bounded,
    check,
    create_handle,
    int32,
    lib,
    native_layout,
    open_panel,
)
from ._simulation import DccSimulation, Simulation
from ._spec import FitOptions, Spec

if TYPE_CHECKING:
    from typing import Unpack

__all__ = [
    "Cancel",
    "Cancelled",
    "CoefTable",
    "Context",
    "DccFit",
    "DccForecast",
    "DccSimulation",
    "DccSpec",
    "DiagnosticResult",
    "Fit",
    "FitOptions",
    "Forecast",
    "Nyblom",
    "SignBias",
    "Simulation",
    "Spec",
    "VolEngineError",
]

Series1D = Sequence[float]


class FitKeywords(TypedDict, total=False):
    """The optional keywords of `Context.fit`."""

    options: FitOptions | None
    starts: ArrayLike | None
    lengths: ArrayLike | None
    mxreg: ArrayLike | None
    vxreg: ArrayLike | None
    cancel: Cancel | None


class FutureKeywords(TypedDict, total=False):
    """The optional keywords of `Context.simulate`: the regressors of the horizon."""

    mxreg: ArrayLike | None
    vxreg: ArrayLike | None


class DccSimulateKeywords(TypedDict, total=False):
    """The optional keywords of `Context.simulate` from a DccFit: `correlations` and one entry per series of each
    regressor kind, as in `DccFit.forecast`."""

    correlations: bool
    mxreg: Sequence[ArrayLike | None] | None
    vxreg: Sequence[ArrayLike | None] | None


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


def regressor_array(
    values: ArrayLike | None, name: str, counts: list[int], data_shape: tuple[int, ...]
) -> NDArray[np.float64] | None:
    """`values` as (T, N, k), or None when no spec has regressors of this kind; the specs with k = 0 ignore it."""
    widths = {count for count in counts if count > 0}
    if not widths and values is not None:
        raise ValueError(f"{name} was given but no spec has {name} > 0")
    if not widths:
        return None
    if values is None:
        raise ValueError(f"a spec has {name} > 0, so {name} is required")
    array = np.asarray(values, dtype=np.float64)
    for width in widths:
        if array.shape != (*data_shape, width):
            raise ValueError(f"{name} must have shape {(*data_shape, width)} for a spec with {name} = {width}, "
                             f"got {array.shape}")
    if len(data_shape) == 1:
        array = array[:, None, :]
    return native_layout(array)


def open_regressors(stack: ExitStack, array: NDArray[np.float64] | None, counts: list[int]) -> NDArray[np.uintp] | None:
    """The panel of job j, in the order of `set_jobs`, for each spec in `counts`: series s of `array`, no copy."""
    if array is None:
        return None
    pointers = [open_panel(stack, array[:, series, :]).pointer for series in range(array.shape[1])]
    absent = [0] * len(pointers)
    jobs: list[int] = []
    for count in counts:
        jobs += pointers if count > 0 else absent
    return np.array(jobs, dtype=np.uintp)


def set_regressors(batch: Handle, mean: NDArray[np.uintp] | None, variance: NDArray[np.uintp] | None) -> None:
    jobs = mean if mean is not None else variance
    if jobs is None:
        return
    mean_data = None
    if mean is not None:
        mean_data = mean.ctypes.data
    variance_data = None
    if variance is not None:
        variance_data = variance.ctypes.data
    check(lib.ve_batch_set_regressors(batch.pointer, mean_data, variance_data, jobs.size))


def shaped(fits: list[Fit], shape: tuple[int, ...]) -> NDArray[np.object_]:
    grid = np.empty(len(fits), dtype=object)
    for position, fit in enumerate(fits):
        grid[position] = fit
    return grid.reshape(shape)


def regressor_counts(source: Fit | Spec) -> tuple[int, int]:
    if isinstance(source, Spec):
        return source.mxreg, source.vxreg
    mean = sum(1 for name in source.names if name.startswith("mxreg"))
    variance = sum(1 for name in source.names if name.startswith("vxreg"))
    return mean, variance


class Context:
    """Execution context owning the native thread pool; fits do not depend on it afterwards.

    `threads` is between 1 and 1024 (VE_MAX_THREADS); 0, the default, means the hardware
    concurrency, capped at 1024.
    """

    def __init__(self, threads: int = 0) -> None:
        threads = int32(threads, "threads")
        self._handle = create_handle(lib.ve_context_create, lib.ve_context_free)
        check(lib.ve_context_set_threads(self._handle.pointer, threads))

    @overload
    def fit(self, spec: Spec, data: Series1D, **keywords: Unpack[FitKeywords]) -> Fit: ...

    @overload
    def fit(self, spec: Spec, data: ArrayLike, **keywords: Unpack[FitKeywords]) -> Fit | NDArray[np.object_]: ...

    @overload
    def fit(self, spec: Sequence[Spec], data: ArrayLike, **keywords: Unpack[FitKeywords]) -> NDArray[np.object_]: ...

    def fit(
        self, spec: Spec | Sequence[Spec], data: ArrayLike, **keywords: Unpack[FitKeywords]
    ) -> Fit | NDArray[np.object_]:
        """Fit every spec on every series of `data`: (T,) or (T, N), optionally with row ranges.

        One spec and a 1D series give a Fit; one spec and (T, N) data give an object array of
        shape (N,); a list of specs gives (n_specs,) or (n_specs, N).

        Keywords (all optional): `options` (FitOptions), `starts` and `lengths` (row ranges, given together),
        `cancel` (a `Cancel`), `mxreg` and `vxreg`, the mean and variance regressors of the specs with mxreg > 0 or vxreg > 0:
        shape (T, k) for 1D data and (T, N, k) for (T, N) data, k being the count of the spec (a different k, a
        different shape or a missing array is a ValueError). Series j uses X[:, j, :] without a copy, and with
        `starts` and `lengths` its rows starts[j] .. starts[j] + lengths[j] - 1 of the regressors. The specs with a
        count of 0 take no regressors; an array given when every spec has a count of 0 is a ValueError.

        `cancel=Cancel()` stops the call with `Cancelled` when its `cancel()` is called, from any thread; no fit is
        returned.
        """
        unknown = sorted(keywords.keys() - FitKeywords.__annotations__.keys())
        if unknown:
            raise TypeError(f"fit() got an unexpected keyword argument {unknown[0]!r}")
        starts, lengths = keywords.get("starts"), keywords.get("lengths")
        if (starts is None) != (lengths is None):
            raise ValueError("starts and lengths must be given together")
        specs: list[Spec] = []
        if isinstance(spec, Spec):
            specs.append(spec)
        if not isinstance(spec, Spec):
            specs.extend(spec)
        array, one_dimensional = panel_array(data)
        series_count = array.shape[1]
        ranges = None
        if starts is not None and lengths is not None:
            ranges = (range_array(starts, series_count, "starts"), range_array(lengths, series_count, "lengths"))
        data_shape = array.shape
        if one_dimensional:
            data_shape = array.shape[:1]
        regressors = (
            regressor_array(keywords.get("mxreg"), "mxreg", [item.mxreg for item in specs], data_shape),
            regressor_array(keywords.get("vxreg"), "vxreg", [item.vxreg for item in specs], data_shape),
        )
        fits = self._run(specs, array, keywords.get("options"), ranges, regressors, keywords.get("cancel"))
        if isinstance(spec, Spec) and one_dimensional:
            return fits[0]
        if isinstance(spec, Spec):
            return shaped(fits, (series_count,))
        if one_dimensional:
            return shaped(fits, (len(specs),))
        return shaped(fits, (len(specs), series_count))

    @overload
    def fit_dcc(
        self, spec: DccSpec, univariate: Sequence[Fit], *, standard_errors: bool = True, cancel: Cancel | None = None
    ) -> DccFit: ...

    @overload
    def fit_dcc(
        self,
        spec: Sequence[DccSpec],
        univariate: Sequence[Fit],
        *,
        standard_errors: bool = True,
        cancel: Cancel | None = None,
    ) -> list[DccFit]: ...

    def fit_dcc(
        self,
        spec: DccSpec | Sequence[DccSpec],
        univariate: Sequence[Fit],
        *,
        standard_errors: bool = True,
        cancel: Cancel | None = None,
    ) -> DccFit | list[DccFit]:
        """Fit one DCC spec or a grid over the chosen ordered univariate fits.

        Fits need at least two series with the same T >= 100 and finite points.
        Disable standard errors for grid selection, then call the selected result's
        standard_errors(univariate). Regressor fits require this option disabled. `cancel` stops the call with
        `Cancelled` (see `Cancel`).
        """
        results = fit_dcc(self._handle, spec, univariate, standard_errors, cancel)
        if isinstance(spec, DccSpec):
            return results[0]
        return results

    def standard_errors(self, fit: DccFit, univariate: Sequence[Fit], *, cancel: Cancel | None = None) -> DccFit:
        """`fit` with its standard errors, on this context's threads; `univariate` are the fits it was made from.

        `cancel` stops the call with `Cancelled` (see `Cancel`).
        """
        return dcc_standard_errors(self._handle, fit, univariate, cancel)

    @overload
    def simulate(
        self,
        source: Fit | Spec,
        *,
        horizon: int,
        paths: int,
        seed: int,
        cancel: Cancel | None = None,
        **regressors: Unpack[FutureKeywords],
    ) -> Simulation: ...

    @overload
    def simulate(
        self,
        source: DccFit,
        *,
        horizon: int,
        paths: int,
        seed: int,
        cancel: Cancel | None = None,
        **options: Unpack[DccSimulateKeywords],
    ) -> DccSimulation: ...

    def simulate(
        self,
        source: Fit | Spec | DccFit,
        *,
        horizon: int,
        paths: int,
        seed: int,
        cancel: Cancel | None = None,
        **regressors: Any,
    ) -> Simulation | DccSimulation:
        """Simulate `paths` paths of `horizon` steps from a fit or from a spec with every parameter fixed.

        From a Fit the paths continue after its last observation (rugarch's ugarchsim, startMethod = "sample");
        from a Spec they start at its unconditional state (rugarch's ugarchpath), which needs every parameter
        fixed, no variance targeting, a stationary model and not iGARCH (VolEngineError otherwise). `mxreg` and
        `vxreg` are the regressors of the horizon, as in `Fit.forecast`, the same for every path. The threads of the
        context share the paths; the same seed gives the same paths for any thread count, within one version of
        the library. `seed` is an integer in [0, 2**64). `cancel` stops the call with `Cancelled` (see `Cancel`).

        From a DccFit (a result of `fit_dcc`, which keeps the final state of each series and needs no univariate
        `Fit`) the paths are multivariate and give a DccSimulation: `mxreg` and `vxreg` are sequences with one entry
        per series, as in `DccFit.forecast`, and `correlations=True` keeps R and H of every path and step (memory:
        paths x horizon x N^2 x 2 doubles). Path p reads only the stream (seed, p), with the same version rule.
        """
        horizon = bounded(horizon, "horizon", 1)
        paths = bounded(paths, "paths", 1)
        seed = bounded(seed, "seed", 0)
        if isinstance(source, DccFit):
            return self._simulate_dcc(source, horizon, paths, seed, regressors, cancel)
        if "correlations" in regressors:
            raise TypeError("correlations is only for a DccFit")
        unknown = sorted(regressors.keys() - FutureKeywords.__annotations__.keys())
        if unknown:
            raise TypeError(f"simulate() got an unexpected keyword argument {unknown[0]!r}")
        counts = regressor_counts(source)
        with ExitStack() as stack:
            mean = future_panel(stack, regressors.get("mxreg"), "mxreg", counts[0], horizon)
            variance = future_panel(stack, regressors.get("vxreg"), "vxreg", counts[1], horizon)
            options = stack.enter_context(
                create_handle(lib.ve_sim_options_create, lib.ve_sim_options_free, horizon, paths, seed)
            )
            check(lib.ve_sim_options_set_regressors(options.pointer, mean, variance))
            if isinstance(source, Spec):
                spec = stack.enter_context(source.native_handle())
                return Simulation(
                    create_handle(
                        lib.ve_simulate_spec_with_cancel,
                        lib.ve_sim_free,
                        self._handle.pointer,
                        spec.pointer,
                        options.pointer,
                        cancel_pointer(cancel),
                    )
                )
            return Simulation(
                create_handle(
                    lib.ve_simulate_fit_with_cancel,
                    lib.ve_sim_free,
                    self._handle.pointer,
                    source._handle.pointer,
                    options.pointer,
                    cancel_pointer(cancel),
                )
            )

    def _simulate_dcc(
        self, source: DccFit, horizon: int, paths: int, seed: int, options: dict[str, Any], cancel: Cancel | None
    ) -> DccSimulation:
        unknown = sorted(options.keys() - DccSimulateKeywords.__annotations__.keys())
        if unknown:
            raise TypeError(f"simulate() got an unexpected keyword argument {unknown[0]!r}")
        with ExitStack() as stack:
            future = open_future(stack, horizon, options.get("mxreg"), options.get("vxreg"))
            handle = stack.enter_context(
                create_handle(lib.ve_sim_options_create, lib.ve_sim_options_free, horizon, paths, seed)
            )
            check(lib.ve_sim_options_set_correlations(handle.pointer, int(bool(options.get("correlations", False)))))
            return DccSimulation(
                create_handle(
                    lib.ve_simulate_dcc_with_cancel,
                    lib.ve_dcc_sim_free,
                    self._handle.pointer,
                    source._handle.pointer,
                    handle.pointer,
                    future,
                    cancel_pointer(cancel),
                )
            )

    def _run(
        self,
        specs: list[Spec],
        array: NDArray[np.float64],
        options: FitOptions | None,
        ranges: tuple[NDArray[np.uintp], NDArray[np.uintp]] | None,
        regressors: tuple[NDArray[np.float64] | None, NDArray[np.float64] | None],
        cancel: Cancel | None,
    ) -> list[Fit]:
        array = native_layout(array)
        series_count = array.shape[1]
        with ExitStack() as stack:
            panel = open_panel(stack, array)
            if ranges is not None:
                set_ranges(panel, ranges[0], ranges[1])
            batch = open_batch(stack, specs, panel, options)
            set_jobs(batch, len(specs), series_count)
            mean = open_regressors(stack, regressors[0], [spec.mxreg for spec in specs])
            variance = open_regressors(stack, regressors[1], [spec.vxreg for spec in specs])
            set_regressors(batch, mean, variance)
            return self._fit_batch(batch, cancel)

    def _fit_batch(self, batch: Handle, cancel: Cancel | None) -> list[Fit]:
        count = ctypes.c_size_t()
        check(lib.ve_batch_job_count(batch.pointer, ctypes.byref(count)))
        pointers = (ctypes.c_void_p * count.value)()
        check(lib.ve_fit_batch_with_cancel(self._handle.pointer, batch.pointer, cancel_pointer(cancel), pointers))
        handles = [Handle(pointer, lib.ve_fit_free) for pointer in pointers]
        return [Fit(handle) for handle in handles]
