"""Model specifications and fit options, validated eagerly through the native library."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

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
    """

    model: str = "sgarch"
    _: KW_ONLY
    p: int = 1
    q: int = 1
    arma: tuple[int, int] = (0, 0)
    dist: str = "norm"
    include_mean: bool = True
    submodel: str | None = None

    def __post_init__(self) -> None:
        self.native_handle().close()
        object.__setattr__(self, "arma", tuple(self.arma))

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
        return handle


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
