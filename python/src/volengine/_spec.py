"""Model specifications and fit options, validated eagerly through the native library."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, dataclass, field
from typing import NoReturn

from ._native import Handle, check, create_handle, int32, lib

MODELS = {"sgarch": 0, "egarch": 1, "gjrgarch": 2, "aparch": 3, "igarch": 4, "csgarch": 5}
SUBMODELS = {
    "garch": 6,
    "tgarch": 7,
    "avgarch": 8,
    "ngarch": 9,
    "nagarch": 10,
    "aparch": 11,
    "gjrgarch": 12,
    "allgarch": 13,
}
DISTRIBUTIONS = {"norm": 0, "std": 1, "ged": 2, "sstd": 3, "sged": 4, "nig": 5, "ghyp": 6, "jsu": 7}
VAR_INITS = {"sample": 0, "backcast": 1}


def code_of(table: dict[str, int], kind: str, name: str) -> int:
    if name not in table:
        listed = list(table)
        if kind == "model":
            listed.append("fgarch")
        raise ValueError(f"unknown {kind} {name!r}; expected one of {sorted(listed)}")
    return table[name]


class _FrozenDict(dict[str, float]):
    """A dict that refuses every mutation; unlike MappingProxyType it pickles, deep-copies and works with asdict."""

    def _refuse(self, *args: object, **kwargs: object) -> NoReturn:
        raise TypeError(f"{type(self).__name__} is read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _refuse

    def __reduce__(self) -> tuple[type[_FrozenDict], tuple[dict[str, float]]]:
        return (type(self), (dict(self),))


def frozen_values(values: Mapping[str, float]) -> Mapping[str, float]:
    return _FrozenDict({name: float(value) for name, value in values.items()})


def encoded_name(name: str) -> bytes:
    if "\0" in name:
        raise ValueError(f"parameter name {name!r} contains a NUL byte")
    return name.encode()


@dataclass(frozen=True)
class Spec:
    """Model specification. Unknown names raise ValueError, native rejections VolEngineError.

    `p` is the ARCH order (number of alpha terms, lagged squared residuals) and `q` the GARCH order
    (number of beta terms, lagged variances), as in arch's `p` and `q` and in rugarch's
    `garchOrder = c(alpha order, beta order)` (rugarch calls those orders q and p, the other way
    round). `arma` is the ARMA(m, n) order of the conditional mean. Orders are limited to 1..6 for
    `p` and `q` and 0..6 for `arma` (VE_MAX_ORDER).

    `model` is one of "sgarch", "egarch", "gjrgarch", "aparch", "igarch", "csgarch" and "fgarch".
    "fgarch" requires `submodel`, one of "garch", "tgarch", "avgarch", "ngarch", "nagarch", "aparch",
    "gjrgarch" and "allgarch"; every other model rejects it. The iGARCH `beta_q` is derived
    (1 - sum of alphas - the other betas), not estimated: it has a name and a value, its standard error
    is NaN and the criteria count the estimated parameters only.

    `dist` is one of "norm", "std", "ged", "sstd", "sged", "nig", "ghyp" and "jsu", all standardized to
    zero mean and unit variance. Distribution parameters come last in the fit names, in the order `skew`,
    `shape`, `ghlambda`: "std" has `shape` (nu > 2), "ged" has `shape` (nu > 0); "sstd" and "sged" are
    Fernandez-Steel skewed versions of those kernels with `skew` xi > 0 and the same `shape`; "nig" and
    "ghyp" are the generalized hyperbolic in the standardized (rho, zeta) parametrization with `skew` rho
    in (-1, 1) and `shape` zeta in (0, 1e6] ("ghyp" adds `ghlambda`, a real number; "nig" fixes it at
    -1/2); "jsu" is Johnson SU with `skew` nu (a real number) and `shape` tau > 0. The "sstd" `skew` is
    xi, not arch's Hansen skewness lambda in (-1, 1): xi = sqrt((1 + lambda) / (1 - lambda)), and the
    standard error of lambda follows by the delta method, d xi / d lambda = 1 / (xi * (1 - lambda)**2);
    the other way, lambda = (xi**2 - 1) / (xi**2 + 1) and SE(lambda) = SE(xi) * 4 * xi / (xi**2 + 1)**2.

    Past zeta = 1e6 double precision no longer standardizes "nig" and "ghyp" (by then a normal to an
    excess kurtosis of order 1 / zeta, 1.5e-5 for nig), and the log-likelihood is -inf there. On
    near-normal data their fits drive zeta up, where rho and zeta are barely identified: the Hessian is
    then often not positive definite, and every standard error is NaN.

    Every parameter is estimated, fixed or derived. `fixed` maps parameter names (those of `Fit.names`:
    "mu", "ar1", "omega", "alpha1", "beta1", "gamma1", "eta11", "skew", "shape", ...) to values the
    fit keeps: they are not optimized, their standard errors are NaN and the criteria do not count them.
    A derived parameter (the iGARCH `beta_q`, and `omega` under variance targeting) is a function of
    the others: it has a name and a value, NaN standard errors, and is not counted either. `start`
    maps names to starting values that replace the defaults; a start value for a derived parameter is
    ignored, so `dict(zip(fit.names, fit.coef().estimate))` is a valid warm start. A start value for a
    name that is also in `fixed` is ignored, because fixed wins: `Spec(fixed=..., start=dict(zip(fit.names,
    fit.coef().estimate)))` is a valid warm start. A name with a NUL character is a ValueError. If the
    starting point is outside the model's domain, the fit ends "failed" with NaN. A spec with every parameter fixed does not optimize: the fit evaluates the model
    at the fixed point and is "converged" when the log-likelihood is finite.

    `variance_targeting=True` makes `omega` derived: it is computed inside the likelihood from the
    residuals of the current mean parameters over the whole sample (m2 = mean of e_t**2, P the
    persistence): omega = m2 * (1 - P) for sgarch and gjrgarch (P = sum(alpha) + sum(beta), plus
    pi * sum(gamma) with pi = P(z < 0) for gjrgarch); omega = log(m2) * (1 - sum(beta)) for egarch;
    omega = mean(|e_t|**delta) * (1 - P) for aparch and fgarch, with delta the power and
    P = sum(alpha_i * kappa_i) + sum(beta). The standard errors of the other parameters come from the
    likelihood in which omega is a function of them (as in rugarch), not from the two-step covariance
    of Francq, Horvath and Zakoian (2011).

    `mxreg` and `vxreg` are the numbers of mean and variance regressor columns (0..8, VE_MAX_REGRESSORS) and
    `archm` is ARCH-in-mean: 0 none, 1 adds xi * sigma_t to the mean, 2 adds xi * sigma_t**2. Their parameters are
    named, in order, "mu", "ar1".., "ma1".., "archm", "mxreg1".., "omega", ... and "vxreg1".. before "skew". The
    data go to `Context.fit(..., mxreg=, vxreg=)`. Mean: m_t = mu + sum(delta_k * x_t,k) + xi * sigma_t**p and
    d_t = r_t - m_t; the ARMA acts on d_t, and the pre-sample residuals (for var_init "sample" and variance
    targeting) are those with xi = 0. Variance: omega becomes omega + sum(zeta_k * v_t,k) in the scale of the
    model (sigma**2, log sigma**2 or sigma**delta; in csgarch it enters q_t). Under variance targeting
    omega = target - sum(zeta_k * mean(v_k)). With var_init "backcast" and `mxreg`, the backcast uses the
    residuals of the OLS of r on [1 (with a mean), X], as arch does. Persistence < 1 is always enforced, unlike
    rugarch, which drops it when there is `vxreg`; a sigma**2 <= 0 gives a log-likelihood of -inf.

    `arfima=True` is rugarch's ARFIMA mean: the parameter "arfima", named after "ma1".. (where "archm" would be;
    a spec cannot have both), is d in (1e-8, 0.5). The ARMA residuals e_t become
    y_t = sum(pi_k * e_(t-k), k = 0..t), pi_0 = 1, pi_k = pi_(k-1) * (k - 1 - d) / k, the (1 - L)**d truncated at
    the start of the series, at a cost of O(T**2); y_t replaces e_t everywhere after.

    Rejected with VolEngineError (the message names the problem): a name the spec does not have
    (`fixed={"shpe": 5.0}`, or "shape" under dist "norm"), fixing the iGARCH `beta_q` or `omega` under
    variance targeting (both derived), variance targeting with "igarch" or "csgarch", and a csgarch
    with `eta21` fixed and a free `beta` (fixing it together with every `beta` is accepted).
    """

    model: str = "sgarch"
    _: KW_ONLY
    p: int = 1
    q: int = 1
    arma: tuple[int, int] = (0, 0)
    dist: str = "norm"
    include_mean: bool = True
    submodel: str | None = None
    fixed: Mapping[str, float] = field(default_factory=dict)
    start: Mapping[str, float] = field(default_factory=dict)
    variance_targeting: bool = False
    mxreg: int = 0
    vxreg: int = 0
    archm: int = 0
    arfima: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "fixed", frozen_values(self.fixed))
        object.__setattr__(self, "start", frozen_values(self.start))
        self.native_handle().close()
        object.__setattr__(self, "arma", tuple(self.arma))

    def __hash__(self) -> int:
        return hash(
            (
                self.model,
                self.p,
                self.q,
                tuple(self.arma),
                self.dist,
                self.include_mean,
                self.submodel,
                frozenset(self.fixed.items()),
                frozenset(self.start.items()),
                self.variance_targeting,
                self.mxreg,
                self.vxreg,
                self.archm,
                self.arfima,
            )
        )

    def arma_orders(self) -> tuple[int, int]:
        arma = self.arma
        if not isinstance(arma, tuple | list) or len(arma) != 2 or not all(type(i) is int for i in arma):
            raise ValueError(f"arma must be a pair of integers (m, n), got {arma!r}")
        return int32(arma[0], "arma m"), int32(arma[1], "arma n")

    def model_code(self) -> int:
        if self.model != "fgarch":
            if self.submodel is not None:
                raise ValueError(f"submodel is only for model 'fgarch', got model {self.model!r}")
            return code_of(MODELS, "model", self.model)
        if self.submodel is None:
            raise ValueError(f"model 'fgarch' requires a submodel, one of {sorted(SUBMODELS)}")
        return code_of(SUBMODELS, "submodel", self.submodel)

    def native_handle(self) -> Handle:
        model = self.model_code()
        dist = code_of(DISTRIBUTIONS, "dist", self.dist)
        arma = self.arma_orders()
        handle = create_handle(lib.ve_spec_create, lib.ve_spec_free, model)
        check(lib.ve_spec_set_order(handle.pointer, int32(self.p, "p"), int32(self.q, "q")))
        check(lib.ve_spec_set_arma(handle.pointer, *arma))
        check(lib.ve_spec_set_dist(handle.pointer, dist))
        check(lib.ve_spec_set_include_mean(handle.pointer, int(self.include_mean)))
        check(lib.ve_spec_set_mxreg(handle.pointer, int32(self.mxreg, "mxreg")))
        check(lib.ve_spec_set_vxreg(handle.pointer, int32(self.vxreg, "vxreg")))
        check(lib.ve_spec_set_archm(handle.pointer, int32(self.archm, "archm")))
        check(lib.ve_spec_set_arfima(handle.pointer, int(self.arfima)))
        self.apply_values(handle)
        return handle

    def apply_values(self, handle: Handle) -> None:
        for name, value in self.fixed.items():
            check(lib.ve_spec_set_fixed(handle.pointer, encoded_name(name), value))
        for name, value in self.start.items():
            check(lib.ve_spec_set_start(handle.pointer, encoded_name(name), value))
        check(lib.ve_spec_set_variance_targeting(handle.pointer, int(self.variance_targeting)))
        check(lib.ve_spec_validate(handle.pointer))


@dataclass(frozen=True)
class FitOptions:
    """Optimizer settings for a fit."""

    _: KW_ONLY
    max_iter: int = 500
    tol: float = 1e-6
    var_init: str = "sample"

    def __post_init__(self) -> None:
        self.native_handle().close()

    def native_handle(self) -> Handle:
        var_init = code_of(VAR_INITS, "var_init", self.var_init)
        handle = create_handle(lib.ve_fit_options_create, lib.ve_fit_options_free)
        check(lib.ve_fit_options_set_max_iter(handle.pointer, int32(self.max_iter, "max_iter")))
        check(lib.ve_fit_options_set_tolerance(handle.pointer, self.tol))
        check(lib.ve_fit_options_set_var_init(handle.pointer, var_init))
        return handle
