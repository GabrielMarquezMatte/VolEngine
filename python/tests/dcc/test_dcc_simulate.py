import numpy as np
import pytest
import volengine as ve

OBSERVATIONS = 1500
HORIZON = 6
PATHS = 40


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    rng = np.random.default_rng(11)
    loadings = np.array([[1.0, 0.0, 0.0], [0.5, 0.9, 0.0], [0.3, 0.2, 0.8]])
    return rng.standard_normal((OBSERVATIONS, 3)) @ loadings.T


@pytest.fixture(scope="module")
def univariate(ctx: ve.Context, returns: np.ndarray) -> list[ve.Fit]:
    return list(ctx.fit(ve.Spec(), returns))


def test_shapes_and_seed(ctx, univariate):
    fit = ctx.fit_dcc(ve.DccSpec("adcc", dist="mvt"), univariate)
    first = ctx.simulate(fit, horizon=HORIZON, paths=PATHS, seed=7, correlations=True)
    again = ctx.simulate(fit, horizon=HORIZON, paths=PATHS, seed=7, correlations=True)
    other = ctx.simulate(fit, horizon=HORIZON, paths=PATHS, seed=8)
    assert isinstance(first, ve.DccSimulation)
    assert first.returns.shape == (PATHS, HORIZON, 3)
    assert first.sigma.shape == (PATHS, HORIZON, 3)
    assert first.R.shape == (PATHS, HORIZON, 3, 3)
    assert first.H.shape == (PATHS, HORIZON, 3, 3)
    for name in ("returns", "sigma", "R", "H"):
        np.testing.assert_array_equal(getattr(first, name), getattr(again, name))
        assert not getattr(first, name).flags.writeable
    assert not np.array_equal(first.returns, other.returns)
    np.testing.assert_array_equal(np.diagonal(first.R, axis1=2, axis2=3), 1.0)


def test_threads_do_not_change_the_paths(univariate):
    fit = ve.Context(threads=1).fit_dcc(ve.DccSpec("dcc"), univariate)
    one = ve.Context(threads=1).simulate(fit, horizon=HORIZON, paths=PATHS, seed=3)
    many = ve.Context(threads=3).simulate(fit, horizon=HORIZON, paths=PATHS, seed=3)
    np.testing.assert_array_equal(one.returns, many.returns)
    np.testing.assert_array_equal(one.sigma, many.sigma)


def test_correlations_are_opt_in(ctx, univariate):
    fit = ctx.fit_dcc(ve.DccSpec("dcc"), univariate)
    simulation = ctx.simulate(fit, horizon=HORIZON, paths=PATHS, seed=1)
    assert simulation.returns.shape == (PATHS, HORIZON, 3)
    with pytest.raises(ve.VolEngineError, match="correlations"):
        simulation.R
    with pytest.raises(ve.VolEngineError, match="correlations"):
        simulation.H
    with pytest.raises(TypeError, match="correlations"):
        ctx.simulate(univariate[0], horizon=HORIZON, paths=PATHS, seed=1, correlations=True)  # type: ignore[call-overload]


def test_copula_dcc_simulates(ctx, univariate):
    for copula in ("parametric", "empirical"):
        fit = ctx.fit_dcc(ve.DccSpec("dcc", copula=copula), univariate)
        simulation = ctx.simulate(fit, horizon=HORIZON, paths=PATHS, seed=5, correlations=True)
        assert np.isfinite(simulation.returns).all()
        assert np.isfinite(simulation.H).all()


def test_first_step_matches_the_forecast(ctx, univariate):
    fit = ctx.fit_dcc(ve.DccSpec("dcc"), univariate)
    forecast = fit.forecast(HORIZON)
    simulation = ctx.simulate(fit, horizon=HORIZON, paths=PATHS, seed=2, correlations=True)
    for path in range(PATHS):
        np.testing.assert_array_equal(simulation.sigma[path, 0], forecast.sigma[0])
        np.testing.assert_array_equal(simulation.R[path, 0], forecast.R[0])


def left_sum(values, k):
    """Sum of the first k values, added left to right (np.cumsum may group differently)."""
    total = 0.0
    for value in values[:k]:
        total += value
    return total


def left_sums(returns, checkpoints):
    """(paths, K, N): the left-to-right sum of the first k steps of each path and series."""
    return np.array([[[left_sum(path[:, s], k) for s in range(path.shape[1])] for k in checkpoints] for path in returns])


@pytest.mark.parametrize("copula", [False, True])
def test_cumulative_matches_the_left_sum(ctx, univariate, copula):
    spec = ve.DccSpec("adcc", dist="mvt")
    if copula:
        spec = ve.DccSpec("dcc", dist="mvt", copula="parametric")
    fit = ctx.fit_dcc(spec, univariate)
    full = ctx.simulate(fit, horizon=HORIZON, paths=PATHS, seed=3)
    cumulative = ctx.simulate(fit, horizon=HORIZON, paths=PATHS, seed=3, checkpoints=[1, 4, HORIZON])
    assert cumulative.checkpoints == (1, 4, HORIZON)
    assert cumulative.cumulative.shape == (PATHS, 3, 3)
    assert np.array_equal(cumulative.cumulative, left_sums(full.returns, [1, 4, HORIZON]))
    assert full.checkpoints == ()
    with pytest.raises(ve.VolEngineError):
        full.cumulative


@pytest.mark.parametrize("checkpoints", [[0, HORIZON], [2, 2, HORIZON], [2, HORIZON - 1]])
def test_checkpoints_are_validated(ctx, univariate, checkpoints):
    fit = ctx.fit_dcc(ve.DccSpec("dcc"), univariate)
    with pytest.raises(ve.VolEngineError):
        ctx.simulate(fit, horizon=HORIZON, paths=PATHS, seed=0, checkpoints=checkpoints)


def test_cumulative_mode_has_no_full_returns_or_correlations(ctx, univariate):
    fit = ctx.fit_dcc(ve.DccSpec("dcc"), univariate)
    sims = ctx.simulate(fit, horizon=HORIZON, paths=PATHS, seed=0, checkpoints=[HORIZON])
    for name in ("returns", "sigma"):
        with pytest.raises(ve.VolEngineError):
            getattr(sims, name)
    with pytest.raises(ve.VolEngineError):
        ctx.simulate(fit, horizon=HORIZON, paths=PATHS, seed=0, checkpoints=[HORIZON], correlations=True)
