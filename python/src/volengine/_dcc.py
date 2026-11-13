"""DCC specifications, batch fitting and immutable native results."""

from __future__ import annotations

import ctypes
import operator
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._fit import IC_PER_OBS, IC_TOTAL, SE_KINDS, CoefTable, Fit, status_name
from ._forecast import DccForecast
from ._native import ELEMENT_BYTES, Handle, check, create_handle, int32, lib, native_view, open_panel, read_scalar
from ._spec import code_of, encoded_name, frozen_values

MODELS = {"dcc": 0, "adcc": 1, "fdcc": 2, "static": 3}
COPULAS = {"parametric": 1, "empirical": 2}
STATIC_METHODS = {"kendall": 1, "ml": 2}
DISTRIBUTIONS = {"mvnorm": 0, "mvt": 1, "mvlaplace": 2}


@dataclass(frozen=True)
class DccSpec:
    """Second-stage DCC model, using the paper convention.

    FDCC needs one group label per series, consecutive from 1, and order (1, 1).
    Fixed parameters are excluded from coefficient tables and information criteria.

    copula="parametric" (each fit's cdf) or "empirical" (ranks) makes it a Copula-DCC on
    the normal or Student scores of the margins; model="static" is a constant-correlation
    copula, with static_method "kendall" (the default) or "ml", whose C1.. are the free
    entries of A in R = cov2cor(A A'), as in rmgarch.
    """

    model: str = "dcc"
    _: KW_ONLY
    order: tuple[int, int] = (1, 1)
    dist: str = "mvnorm"
    groups: Sequence[int] | None = None
    fixed: Mapping[str, float] = field(default_factory=dict)
    start: Mapping[str, float] = field(default_factory=dict)
    copula: str | None = None
    static_method: str | None = None

    def __post_init__(self) -> None:
        if len(self.order) != 2 or any(type(value) is not int for value in self.order):
            raise ValueError("order must be a pair of integers (p, q)")
        object.__setattr__(self, "order", tuple(self.order))
        if self.groups is not None:
            object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "fixed", frozen_values(self.fixed))
        object.__setattr__(self, "start", frozen_values(self.start))
        self.native_handle().close()

    def __hash__(self) -> int:
        groups = None
        if self.groups is not None:
            groups = tuple(self.groups)
        return hash((self.model, self.order, self.dist, groups, frozenset(self.fixed.items()), frozenset(self.start.items()),
                     self.copula, self.static_method))

    def native_handle(self) -> Handle:
        model = code_of(MODELS, "model", self.model)
        dist = code_of(DISTRIBUTIONS, "dist", self.dist)
        handle = create_handle(lib.ve_dcc_spec_create, lib.ve_dcc_spec_free, model)
        check(lib.ve_dcc_spec_set_order(handle.pointer, int32(self.order[0], "p"), int32(self.order[1], "q")))
        check(lib.ve_dcc_spec_set_dist(handle.pointer, dist))
        if self.groups is not None:
            groups = (ctypes.c_int32 * len(self.groups))(*(int32(value, "group") for value in self.groups))
            check(lib.ve_dcc_spec_set_groups(handle.pointer, groups, len(groups)))
        for name, value in self.fixed.items():
            check(lib.ve_dcc_spec_set_fixed(handle.pointer, encoded_name(name), value))
        for name, value in self.start.items():
            check(lib.ve_dcc_spec_set_start(handle.pointer, encoded_name(name), value))
        if self.copula is not None:
            check(lib.ve_dcc_spec_set_copula(handle.pointer, code_of(COPULAS, "copula", self.copula)))
        if self.static_method is not None:
            method = code_of(STATIC_METHODS, "static_method", self.static_method)
            check(lib.ve_dcc_spec_set_static_method(handle.pointer, method))
        return handle


def fit_pointers(fits: Sequence[Fit]) -> Any:
    if any(not isinstance(fit, Fit) for fit in fits):
        raise TypeError("univariate must contain Fit objects")
    return (ctypes.c_void_p * len(fits))(*(fit._handle.pointer for fit in fits))


def fit_dcc(context: Handle, spec: DccSpec | Sequence[DccSpec], univariate: Sequence[Fit], standard_errors: bool) -> list[DccFit]:
    specs: list[DccSpec] = []
    if isinstance(spec, DccSpec):
        specs.append(spec)
    if not isinstance(spec, DccSpec):
        specs = list(spec)
    if not specs:
        raise ValueError("fit_dcc needs at least one spec")
    if any(not isinstance(item, DccSpec) for item in specs):
        raise TypeError("spec must contain DccSpec objects")
    fits = fit_pointers(univariate)
    with ExitStack() as stack:
        handles = [stack.enter_context(item.native_handle()) for item in specs]
        pointers = (ctypes.c_void_p * len(handles))(*(handle.pointer for handle in handles))
        options = stack.enter_context(create_handle(lib.ve_dcc_options_create, lib.ve_dcc_options_free))
        check(lib.ve_dcc_options_set_standard_errors(options.pointer, int(standard_errors)))
        batch = stack.enter_context(create_handle(lib.ve_dcc_batch_create, lib.ve_dcc_batch_free))
        check(lib.ve_dcc_batch_set_specs(batch.pointer, pointers, len(handles)))
        check(lib.ve_dcc_batch_set_fits(batch.pointer, fits, len(univariate)))
        check(lib.ve_dcc_batch_set_options(batch.pointer, options.pointer))
        out = (ctypes.c_void_p * len(specs))()
        check(lib.ve_dcc_fit_batch(context.pointer, batch.pointer, out))
        owned = [Handle(pointer, lib.ve_dcc_fit_free) for pointer in out]
        return [DccFit(handle) for handle in owned]


def future_panel_of(stack: ExitStack, values: ArrayLike | None) -> int | None:
    """The native panel of one series' regressors of the horizon (T x k, or (T,) for one), None for none."""
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"the regressors of a series must be 1D or 2D, got {array.ndim} dimensions")
    if any(stride % ELEMENT_BYTES for stride in array.strides):
        array = np.ascontiguousarray(array)
    return open_panel(stack, array).pointer


def listed(entries: Sequence[ArrayLike | None] | None) -> list[ArrayLike | None] | None:
    if entries is None:
        return None
    return list(entries)


def entry_of(entries: list[ArrayLike | None] | None, series: int) -> ArrayLike | None:
    if entries is None:
        return None
    return entries[series]


def open_future(
    stack: ExitStack,
    horizon: int,
    mxreg: Sequence[ArrayLike | None] | None,
    vxreg: Sequence[ArrayLike | None] | None,
) -> int | None:
    """The native regressors of the horizon, one entry per series, or None when no series has any."""
    if mxreg is None and vxreg is None:
        return None
    mean = listed(mxreg)
    variance = listed(vxreg)
    if mean is not None and variance is not None and len(mean) != len(variance):
        raise ValueError(f"mxreg and vxreg must have one entry per series, got {len(mean)} and {len(variance)}")
    entries = mean if mean is not None else variance
    assert entries is not None
    future = stack.enter_context(create_handle(lib.ve_dcc_future_create, lib.ve_dcc_future_free, len(entries), horizon))
    for series in range(len(entries)):
        mean_panel = future_panel_of(stack, entry_of(mean, series))
        variance_panel = future_panel_of(stack, entry_of(variance, series))
        check(lib.ve_dcc_future_set(future.pointer, series, mean_panel, variance_panel))
    return future.pointer


class DccFit:
    """One DCC result. R and H are read-only views that keep the result alive.

    H is computed once on first access. Standard errors can be added to a new
    result using the original ordered univariate fits; regressor fits are unsupported.

    A fit made by `Context.fit_dcc` keeps the final state of each series and of its correlation recursion, so
    `forecast` and `Context.simulate` do not need the univariate `Fit` objects, which `standard_errors` does need
    because it refilters the whole sample.
    """

    def __init__(self, handle: Handle) -> None:
        self._handle = handle
        self.status = status_name(read_scalar(ctypes.c_int32, lib.ve_dcc_fit_status, handle.pointer))
        self.nobs = read_scalar(ctypes.c_size_t, lib.ve_dcc_fit_nobs, handle.pointer)
        self.series = read_scalar(ctypes.c_size_t, lib.ve_dcc_fit_series, handle.pointer)
        count = read_scalar(ctypes.c_size_t, lib.ve_dcc_fit_param_count, handle.pointer)
        self.names = tuple(self._param_name(index) for index in range(count))

    def _param_name(self, index: int) -> str:
        name = read_scalar(ctypes.c_char_p, lib.ve_dcc_fit_param_name, self._handle.pointer, index)
        return cast(bytes, name).decode("utf-8")

    @property
    def R(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_dcc_fit_correlation, self._handle.pointer).reshape(-1, self.series, self.series)

    @property
    def H(self) -> NDArray[np.float64]:
        return native_view(self, lib.ve_dcc_fit_covariance, self._handle.pointer).reshape(-1, self.series, self.series)

    @property
    def loglik(self) -> float:
        return cast(float, read_scalar(ctypes.c_double, lib.ve_dcc_fit_loglik, self._handle.pointer))

    def aic(self, *, per_obs: bool = True) -> float:
        return self._criterion(0, per_obs)

    def bic(self, *, per_obs: bool = True) -> float:
        return self._criterion(1, per_obs)

    def hqic(self, *, per_obs: bool = True) -> float:
        return self._criterion(2, per_obs)

    def sic(self, *, per_obs: bool = True) -> float:
        return self._criterion(3, per_obs)

    def _criterion(self, kind: int, per_obs: bool) -> float:
        scale = IC_PER_OBS if per_obs else IC_TOTAL
        return cast(float, read_scalar(ctypes.c_double, lib.ve_dcc_fit_criterion, self._handle.pointer, kind, scale))

    def coef(self, se: str = "hessian") -> CoefTable:
        kind = code_of(SE_KINDS, "se", se)
        return CoefTable(
            self.names,
            self._values(lib.ve_dcc_fit_estimates),
            self._values(lib.ve_dcc_fit_std_errors, kind),
            self._values(lib.ve_dcc_fit_t_values, kind),
            self._values(lib.ve_dcc_fit_p_values, kind),
        )

    def _values(self, function: Callable[..., int], *args: Any) -> NDArray[np.float64]:
        out = np.empty(len(self.names), dtype=np.float64)
        check(function(self._handle.pointer, *args, out.ctypes.data))
        return out

    def forecast(
        self,
        horizon: int,
        *,
        mxreg: Sequence[ArrayLike | None] | None = None,
        vxreg: Sequence[ArrayLike | None] | None = None,
    ) -> DccForecast:
        """R, H, sigma and mean for h = 1..horizon after the last observation, as rmgarch's dccforecast (n.roll = 0).

        `mxreg` and `vxreg` hold, for each series, the regressors of the horizon ((horizon, k), or (horizon,) for
        one) or None for a series without; a series' regressors must match its spec (VolEngineError naming the
        series, from 1, otherwise). A fit with no finite point raises VolEngineError. The fit keeps what it needs:
        the univariate `Fit` objects can be freed. See `DccForecast` for the rules of R.
        """
        horizon = operator.index(horizon)
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1, got {horizon}")
        with ExitStack() as stack:
            future = open_future(stack, horizon, mxreg, vxreg)
            handle = create_handle(
                lib.ve_dcc_forecast_create, lib.ve_dcc_forecast_free, self._handle.pointer, horizon, future
            )
        return DccForecast(handle)

    def standard_errors(self, univariate: Sequence[Fit]) -> DccFit:
        fits = fit_pointers(univariate)
        with create_handle(lib.ve_context_create, lib.ve_context_free) as context:
            return DccFit(create_handle(
                lib.ve_dcc_fit_standard_errors, lib.ve_dcc_fit_free,
                context.pointer, self._handle.pointer, fits, len(univariate),
            ))
