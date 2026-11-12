"""Simulate the reference series shared by every oracle script and test.

`data/sgarch11_norm.csv` holds the plan-1 sGARCH(1,1)-normal series (seed 20261110, three lengths);
`data/grid.csv` holds one T = 3000 series per process below, drawn in order from one stream (seed 20261111);
`data/grid_2b.csv` holds one T = 3000 series per GARCH variant, drawn in order from one stream (seed 20261112);
`data/grid_2c.csv` holds one T = 3000 series per skewed distribution, drawn in order from one stream (seed 20261113);
`data/regressors_2e.csv` holds the regressors x1, x2, v1 and `data/grid_2e.csv` one T = 3000 series per regressor or
ARCH-in-mean process, all from one stream (seed 20261114): the x2 draws first, then the shocks of each series.
"""

import csv
import math
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy import special, stats  # type: ignore[import-untyped]

DATA = Path(__file__).resolve().parent / "data"
PLAN_ONE_SEED = 20261110
PLAN_ONE_LENGTHS = (5000, 3000, 1500)
GRID_SEED = 20261111
GRID_LENGTH = 3000
VARIANT_SEED = 20261112
VARIANT_BURN_IN = 1000
VARIANT_MU = 0.05
SKEWED_SEED = 20261113
REGRESSOR_SEED = 20261114

Shocks = Callable[[np.random.Generator, int], NDArray[np.float64]]
# (sigma2, q): the conditional variance and the csGARCH permanent component (unused by the other models).
State = tuple[float, float]
Step = Callable[[State, float], State]
# (state, z, shift): the step with the variance intercept omega moved by shift, the variance regressors' term.
ShiftedStep = Callable[[State, float, float], State]


class Process(NamedTuple):
    """ARMA(m, n)-sGARCH(p, q) with rugarch's signs: r_t - mu = sum ar_i (r_{t-i} - mu) + e_t + sum ma_j e_{t-j}."""

    name: str
    mu: float
    ar: tuple[float, ...]
    ma: tuple[float, ...]
    omega: float
    alpha: tuple[float, ...]
    beta: tuple[float, ...]
    shocks: Shocks


def normal_shocks(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
    return rng.standard_normal(length)


def student_shocks(nu: float) -> Shocks:
    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        return rng.standard_t(nu, length) * math.sqrt((nu - 2.0) / nu)

    return draw


def ged_log_scale(nu: float) -> float:
    return 0.5 * (-2.0 / nu * math.log(2.0) + math.lgamma(1.0 / nu) - math.lgamma(3.0 / nu))


def ged_magnitudes(nu: float) -> Shocks:
    # |z| = lambda (2 u)^(1/nu) with u ~ Gamma(1/nu, 1) has the unit-variance GED density.
    log_lambda = ged_log_scale(nu)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        return math.exp(log_lambda) * (2.0 * rng.gamma(1.0 / nu, 1.0, length)) ** (1.0 / nu)

    return draw


def ged_shocks(nu: float) -> Shocks:
    magnitudes = ged_magnitudes(nu)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        magnitude = magnitudes(rng, length)
        sign = np.where(rng.random(length) < 0.5, -1.0, 1.0)
        return sign * magnitude

    return draw


def student_magnitudes(nu: float) -> Shocks:
    shocks = student_shocks(nu)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        return np.abs(shocks(rng, length))

    return draw


def ged_abs_moment(nu: float) -> float:
    """E|z| of the unit-variance GED."""
    return math.exp(ged_log_scale(nu) + math.log(2.0) / nu + math.lgamma(2.0 / nu) - math.lgamma(1.0 / nu))


def fernandez_steel_shocks(magnitudes: Shocks, mean_abs: float, xi: float) -> Shocks:
    """y = xi |x| with probability xi^2 / (1 + xi^2), else -|x| / xi, for |x| from the unit-variance kernel whose
    E|x| is mean_abs; z = (y - mu_xi) / sigma_xi has mean 0 and variance 1."""
    mean = mean_abs * (xi - 1.0 / xi)
    sd = math.sqrt((1.0 - mean_abs**2) * (xi * xi + 1.0 / (xi * xi)) + 2.0 * mean_abs**2 - 1.0)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        magnitude = magnitudes(rng, length)
        right = rng.random(length) < xi * xi / (1.0 + xi * xi)
        return (np.where(right, xi * magnitude, -magnitude / xi) - mean) / sd

    return draw


class Johnson(NamedTuple):
    """Johnson SU in Rigby and Stasinopoulos' parametrization, standardized: z = c (sqrt(w) sinh(omega) + x),
    with N = -nu + tau asinh(x) standard normal."""

    w: float
    omega: float
    c: float


def johnson(nu: float, tau: float) -> Johnson:
    w = math.exp(1.0 / (tau * tau))
    omega = -nu / tau
    c = (0.5 * (w - 1.0) * (w * math.cosh(2.0 * omega) + 1.0)) ** -0.5
    return Johnson(w, omega, c)


def johnson_shocks(nu: float, tau: float) -> Shocks:
    k = johnson(nu, tau)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        return k.c * (math.sqrt(k.w) * math.sinh(k.omega) + np.sinh((rng.standard_normal(length) + nu) / tau))

    return draw


def johnson_abs_moment(nu: float, tau: float) -> float:
    """E|z| = 2 E[z; z > 0], since E z = 0. z > 0 when N > n0, and E[exp(a N); N > n0] = exp(a^2 / 2) Phi(a - n0)."""
    k = johnson(nu, tau)
    n0 = tau * math.asinh(-math.sqrt(k.w) * math.sinh(k.omega)) - nu
    upper = math.exp(nu / tau) * stats.norm.cdf(1.0 / tau - n0)
    lower = math.exp(-nu / tau) * stats.norm.cdf(-1.0 / tau - n0)
    partial = math.sqrt(k.w) * (math.sinh(k.omega) * stats.norm.cdf(-n0) + 0.5 * (upper - lower))
    return 2.0 * k.c * partial


class Hyperbolic(NamedTuple):
    """The generalized hyperbolic (alpha, beta, delta, mu) with mean 0 and variance 1 at (rho, zeta, lambda)."""

    alpha: float
    beta: float
    delta: float
    mu: float


def hyperbolic(rho: float, zeta: float, lam: float) -> Hyperbolic:
    first = special.kve(lam + 1.0, zeta) / special.kve(lam, zeta)
    second = special.kve(lam + 2.0, zeta) / special.kve(lam, zeta)
    delta = (first / zeta + rho * rho / (1.0 - rho * rho) * (second - first * first)) ** -0.5
    alpha = zeta / (delta * math.sqrt(1.0 - rho * rho))
    beta = rho * alpha
    return Hyperbolic(alpha, beta, delta, -beta * delta * delta * first / zeta)


def nig_shocks(rho: float, zeta: float) -> Shocks:
    """Normal variance-mean mixture z = mu + beta W + sqrt(W) N, W inverse Gaussian with mean delta^2 / zeta
    (= delta / sqrt(alpha^2 - beta^2)) and shape delta^2."""
    gh = hyperbolic(rho, zeta, -0.5)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        mixing = rng.wald(gh.delta**2 / zeta, gh.delta**2, length)
        return gh.mu + gh.beta * mixing + np.sqrt(mixing) * rng.standard_normal(length)

    return draw


def ghyp_shocks(rho: float, zeta: float, lam: float) -> Shocks:
    gh = hyperbolic(rho, zeta, lam)
    law = stats.genhyperbolic(p=lam, a=gh.alpha * gh.delta, b=gh.beta * gh.delta, loc=gh.mu, scale=gh.delta)

    def draw(rng: np.random.Generator, length: int) -> NDArray[np.float64]:
        return np.asarray(law.rvs(size=length, random_state=rng), dtype=np.float64)

    return draw


PLAN_ONE = Process("sgarch11_norm", 0.05, (), (), 0.01, (0.08,), (0.9,), normal_shocks)
GRID = (
    Process("sgarch21_std", 0.05, (), (), 0.01, (0.08, 0.05), (0.82,), student_shocks(6.0)),
    Process("arma11_sgarch11_ged", 0.05, (0.5,), (-0.2,), 0.01, (0.08,), (0.9,), ged_shocks(1.5)),
    Process("sgarch12_norm", 0.05, (), (), 0.01, (0.08,), (0.5, 0.4), normal_shocks),
    Process("sgarch22_ged", 0.05, (), (), 0.01, (0.1, 0.15), (0.2, 0.5), ged_shocks(1.5)),
)


def student_abs_moment(nu: float) -> float:
    """E|z| of the unit-variance Student-t."""
    log_ratio = math.lgamma(0.5 * (nu - 1.0)) - math.lgamma(0.5 * nu)
    return math.sqrt(nu - 2.0) * math.exp(log_ratio) / math.sqrt(math.pi)


def egarch_step(omega: float, alpha: float, gamma: float, beta: float, mean_abs: float) -> Step:
    """log sigma2' = omega + alpha z + gamma (|z| - E|z|) + beta log sigma2."""

    def step(state: State, z: float) -> State:
        news = alpha * z + gamma * (abs(z) - mean_abs)
        return math.exp(omega + news + beta * math.log(state[0])), 0.0

    return step


def gjrgarch_step(omega: float, alpha: float, gamma: float, beta: float) -> Step:
    """sigma2' = omega + (alpha + gamma 1[e < 0]) e^2 + beta sigma2; gamma = 0 is the sGARCH (and iGARCH) step."""

    def step(state: State, z: float) -> State:
        leverage = gamma if z < 0.0 else 0.0
        return omega + (alpha + leverage) * state[0] * z * z + beta * state[0], 0.0

    return step


def aparch_step(omega: float, alpha: float, gamma: float, beta: float, delta: float) -> Step:
    """sigma^delta' = omega + alpha (|e| - gamma e)^delta + beta sigma^delta, with e = sigma z."""

    def step(state: State, z: float) -> State:
        residual = math.sqrt(state[0]) * z
        news = (abs(residual) - gamma * residual) ** delta
        return (omega + alpha * news + beta * state[0] ** (0.5 * delta)) ** (2.0 / delta), 0.0

    return step


def csgarch_step(omega: float, alpha: float, beta: float, rho: float, phi: float) -> Step:
    """q' = omega + rho q + phi (e^2 - sigma2); sigma2' = q' + alpha (e^2 - q) + beta (sigma2 - q)."""

    def step(state: State, z: float) -> State:
        variance, permanent = state
        square = variance * z * z
        next_permanent = omega + rho * permanent + phi * (square - variance)
        return next_permanent + alpha * (square - permanent) + beta * (variance - permanent), next_permanent

    return step


def allgarch_step(omega: float, alpha: float, beta: float, eta: tuple[float, float], power: float) -> Step:
    """sigma^power' = omega + alpha sigma^power (|z - eta2| - eta1 (z - eta2))^power + beta sigma^power."""
    eta1, eta2 = eta

    def step(state: State, z: float) -> State:
        sigma_power = state[0] ** (0.5 * power)
        shifted = z - eta2
        news = (abs(shifted) - eta1 * shifted) ** power
        return (omega + (alpha * news + beta) * sigma_power) ** (2.0 / power), 0.0

    return step


class Variant(NamedTuple):
    """A GARCH variant of order (1, 1) with mean 0.05."""

    name: str
    step: Step
    shocks: Shocks


VARIANTS = (
    Variant("egarch11_std", egarch_step(-0.01, -0.08, 0.15, 0.97, student_abs_moment(6.0)), student_shocks(6.0)),
    Variant("gjrgarch11_norm", gjrgarch_step(0.01, 0.03, 0.09, 0.9), normal_shocks),
    Variant("aparch11_ged", aparch_step(0.02, 0.08, 0.4, 0.9, 1.3), ged_shocks(1.5)),
    Variant("igarch11_norm", gjrgarch_step(0.01, 0.1, 0.0, 0.9), normal_shocks),
    Variant("csgarch11_std", csgarch_step(0.01, 0.06, 0.85, 0.99, 0.04), student_shocks(7.0)),
    Variant("fgarch_allgarch11_norm", allgarch_step(0.02, 0.08, 0.88, (0.3, 0.2), 1.5), normal_shocks),
)
SKEWED = (
    Variant(
        "gjrgarch11_sstd",
        gjrgarch_step(0.01, 0.03, 0.09, 0.9),
        fernandez_steel_shocks(student_magnitudes(6.0), student_abs_moment(6.0), 0.85),
    ),
    Variant(
        "aparch11_sged",
        aparch_step(0.02, 0.08, 0.4, 0.9, 1.3),
        fernandez_steel_shocks(ged_magnitudes(1.4), ged_abs_moment(1.4), 0.9),
    ),
    Variant(
        "egarch11_jsu",
        egarch_step(-0.01, -0.08, 0.15, 0.97, johnson_abs_moment(-0.5, 2.0)),
        johnson_shocks(-0.5, 2.0),
    ),
    Variant("sgarch11_nig", gjrgarch_step(0.02, 0.08, 0.0, 0.9), nig_shocks(-0.3, 1.5)),
    Variant("sgarch11_ghyp", gjrgarch_step(0.02, 0.08, 0.0, 0.9), ghyp_shocks(-0.3, 1.5, 1.0)),
)


class Regressors(NamedTuple):
    """The regressors at t = 1 - VARIANT_BURN_IN .. GRID_LENGTH; the last GRID_LENGTH rows are the series' rows."""

    x1: NDArray[np.float64]
    x2: NDArray[np.float64]
    v1: NDArray[np.float64]


def make_regressors(rng: np.random.Generator) -> Regressors:
    t = np.arange(1 - VARIANT_BURN_IN, GRID_LENGTH + 1, dtype=np.float64)
    return Regressors(np.sin(t / 40.0), rng.standard_normal(t.size), np.abs(np.cos(t / 25.0)))


def shifted(build: Callable[..., Step], omega: float, *rest: float) -> ShiftedStep:
    def step(state: State, z: float, shift: float) -> State:
        return build(omega + shift, *rest)(state, z)

    return step


class Process2e(NamedTuple):
    """r_t = mu + sum delta_k x_t,k + xi sigma_t^power + d_t, with d_t = ar d_(t-1) + sigma_t z_t, and the variance
    intercept omega + zeta v_t."""

    name: str
    mu: float
    ar: float
    delta: tuple[float, float]
    xi: float
    power: float
    step: ShiftedStep
    zeta: float
    shocks: Shocks


PROCESSES_2E = (
    Process2e("arx_sgarch11_norm", 0.05, 0.3, (0.3, -0.2), 0.0, 1.0, shifted(gjrgarch_step, 0.02, 0.08, 0.0, 0.9), 0.0,
              normal_shocks),
    Process2e("sgarch11_vx_std", 0.05, 0.0, (0.0, 0.0), 0.0, 1.0, shifted(gjrgarch_step, 0.01, 0.08, 0.0, 0.88), 0.05,
              student_shocks(6.0)),
    Process2e("archm_gjrgarch11_norm", 0.02, 0.0, (0.0, 0.0), 0.1, 1.0, shifted(gjrgarch_step, 0.02, 0.03, 0.09, 0.9),
              0.0, normal_shocks),
    Process2e("archm2_egarch11_std", 0.02, 0.0, (0.0, 0.0), 0.05, 2.0,
              shifted(egarch_step, -0.01, -0.08, 0.15, 0.97, student_abs_moment(6.0)), 0.0, student_shocks(6.0)),
)


def simulate_2e(process: Process2e, shocks: NDArray[np.float64], regressors: Regressors) -> NDArray[np.float64]:
    """Start at sigma2 = q = 1 and d = 0, and drop the first VARIANT_BURN_IN observations."""
    state: State = (1.0, 1.0)
    deviation = 0.0
    returns = np.empty(shocks.size)
    # The step at t sets the variance of t + 1, whose intercept shift is zeta v_(t+1); the last one is not used.
    shifts = np.append(process.zeta * regressors.v1[1:], 0.0)
    for t, z in enumerate(shocks):
        sigma = math.sqrt(state[0])
        deviation = process.ar * deviation + sigma * z
        regressed = process.delta[0] * regressors.x1[t] + process.delta[1] * regressors.x2[t]
        returns[t] = process.mu + regressed + process.xi * sigma**process.power + deviation
        state = process.step(state, float(z), float(shifts[t]))
    return returns[VARIANT_BURN_IN:]


def lagged(values: list[float], lag: int, presample: float) -> float:
    if lag > len(values):
        return presample
    return values[-lag]


def weighted_lags(weights: tuple[float, ...], values: list[float], presample: float) -> float:
    return sum(weight * lagged(values, lag, presample) for lag, weight in enumerate(weights, start=1))


def simulate(process: Process, shocks: NDArray[np.float64]) -> NDArray[np.float64]:
    """Start the variance at its unconditional value and the ARMA lags at the mean (zero residuals)."""
    unconditional = process.omega / (1.0 - sum(process.alpha) - sum(process.beta))
    deviations: list[float] = []
    residuals: list[float] = []
    squares: list[float] = []
    variances: list[float] = []
    for shock in shocks:
        variance = process.omega + weighted_lags(process.alpha, squares, unconditional)
        variance += weighted_lags(process.beta, variances, unconditional)
        residual = math.sqrt(variance) * shock
        mean = weighted_lags(process.ar, deviations, 0.0) + weighted_lags(process.ma, residuals, 0.0)
        deviations.append(mean + residual)
        residuals.append(residual)
        squares.append(residual * residual)
        variances.append(variance)
    return process.mu + np.array(deviations)


def simulate_plan_one(process: Process, shocks: NDArray[np.float64]) -> NDArray[np.float64]:
    """Plan 1's sGARCH(1,1) recursion, kept so the committed plan-1 series stay bit-identical: `simulate` agrees
    with it only up to rounding."""
    returns = np.empty(shocks.size)
    variance = process.omega / (1.0 - process.alpha[0] - process.beta[0])
    for t, shock in enumerate(shocks):
        residual = math.sqrt(variance) * shock
        returns[t] = process.mu + residual
        variance = process.omega + process.alpha[0] * residual * residual + process.beta[0] * variance
    return returns


def simulate_variant(variant: Variant, shocks: NDArray[np.float64]) -> NDArray[np.float64]:
    """Start at sigma2 = q = 1 and drop the first VARIANT_BURN_IN observations."""
    state: State = (1.0, 1.0)
    returns = np.empty(shocks.size)
    for t, z in enumerate(shocks):
        returns[t] = VARIANT_MU + math.sqrt(state[0]) * z
        state = variant.step(state, float(z))
    return returns[VARIANT_BURN_IN:]


def cell(values: NDArray[np.float64], row: int) -> str:
    if row >= values.size:
        return ""
    return f"{values[row]:.17g}"


def write_csv(path: Path, columns: dict[str, NDArray[np.float64]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = max(values.size for values in columns.values())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        for row in range(rows):
            writer.writerow(cell(values, row) for values in columns.values())


def main() -> None:
    rng = np.random.default_rng(PLAN_ONE_SEED)
    plan_one = {f"t{n}": simulate_plan_one(PLAN_ONE, rng.standard_normal(n)) for n in PLAN_ONE_LENGTHS}
    write_csv(DATA / "sgarch11_norm.csv", plan_one)
    rng = np.random.default_rng(GRID_SEED)
    grid = {process.name: simulate(process, process.shocks(rng, GRID_LENGTH)) for process in GRID}
    write_csv(DATA / "grid.csv", grid)
    rng = np.random.default_rng(VARIANT_SEED)
    length = VARIANT_BURN_IN + GRID_LENGTH
    variants = {variant.name: simulate_variant(variant, variant.shocks(rng, length)) for variant in VARIANTS}
    write_csv(DATA / "grid_2b.csv", variants)
    rng = np.random.default_rng(SKEWED_SEED)
    skewed = {variant.name: simulate_variant(variant, variant.shocks(rng, length)) for variant in SKEWED}
    write_csv(DATA / "grid_2c.csv", skewed)
    rng = np.random.default_rng(REGRESSOR_SEED)
    regressors = make_regressors(rng)
    series_2e = {p.name: simulate_2e(p, p.shocks(rng, length), regressors) for p in PROCESSES_2E}
    write_csv(DATA / "grid_2e.csv", series_2e)
    kept = {name: getattr(regressors, name)[VARIANT_BURN_IN:] for name in Regressors._fields}
    write_csv(DATA / "regressors_2e.csv", kept)


if __name__ == "__main__":
    main()
