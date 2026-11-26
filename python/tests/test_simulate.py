import gc

import numpy as np
import pytest

import volengine as ve

FIXED_SGARCH = {"mu": 0.01, "omega": 0.05, "alpha1": 0.1, "beta1": 0.85}


def test_simulate_shapes_and_read_only(ctx, series):
    fit = ctx.fit(ve.Spec(arma=(1, 1)), series[:, 0])
    sims = ctx.simulate(fit, horizon=60, paths=500, seed=42)
    assert isinstance(sims, ve.Simulation)
    assert sims.returns.shape == (500, 60)
    assert sims.sigma.shape == (500, 60)
    assert not sims.returns.flags.writeable
    assert not sims.sigma.flags.writeable
    assert np.isfinite(sims.returns).all()
    assert (sims.sigma > 0).all()
    np.testing.assert_array_equal(sims.sigma[:, 0], fit.forecast(1).sigma[0])


def test_simulate_from_spec(ctx):
    spec = ve.Spec(fixed=FIXED_SGARCH)
    sims = ctx.simulate(spec, horizon=5, paths=20_000, seed=1)
    assert sims.returns.shape == (20_000, 5)
    assert np.mean(sims.sigma**2) == pytest.approx(0.05 / (1 - 0.95), rel=0.02)


def test_simulate_same_for_any_threads(series):
    one, four = ve.Context(threads=1), ve.Context(threads=4)
    fit = one.fit(ve.Spec(dist="std"), series[:, 1])
    a = one.simulate(fit, horizon=20, paths=300, seed=9)
    b = four.simulate(fit, horizon=20, paths=300, seed=9)
    np.testing.assert_array_equal(a.returns, b.returns)
    np.testing.assert_array_equal(a.sigma, b.sigma)


def test_simulate_with_regressors(ctx, series):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((series.shape[0], 1))
    fit = ctx.fit(ve.Spec(mxreg=1), series[:, 0], mxreg=x)
    one_dimensional = ctx.simulate(fit, horizon=5, paths=10, seed=2, mxreg=np.ones(5))
    two_dimensional = ctx.simulate(fit, horizon=5, paths=10, seed=2, mxreg=np.ones((5, 1)))
    np.testing.assert_array_equal(one_dimensional.returns, two_dimensional.returns)


@pytest.mark.parametrize(
    ("keywords", "word"),
    [
        ({"horizon": 0, "paths": 10, "seed": 0}, "horizon"),
        ({"horizon": 2**64, "paths": 10, "seed": 0}, "horizon"),
        ({"horizon": 5, "paths": -1, "seed": 0}, "paths"),
        ({"horizon": 5, "paths": 10, "seed": -1}, "seed"),
        ({"horizon": 5, "paths": 10, "seed": 2**64}, "seed"),
        ({"horizon": 5, "paths": 10, "seed": 0, "mxreg": np.ones(5)}, "mxreg"),
        ({"horizon": 5, "paths": 10, "seed": 0, "vxreg": np.ones((4, 1))}, "vxreg"),
    ],
)
def test_simulate_bad_arguments_raise(ctx, keywords, word):
    with pytest.raises(ValueError, match=word):
        ctx.simulate(ve.Spec(fixed=FIXED_SGARCH), **keywords)


def test_simulate_too_large_raises(ctx):
    with pytest.raises(ve.VolEngineError, match="too large"):
        ctx.simulate(ve.Spec(fixed=FIXED_SGARCH), horizon=60, paths=2**62, seed=0)


def test_simulate_spec_with_free_parameter_raises(ctx):
    with pytest.raises(ve.VolEngineError, match="beta1"):
        ctx.simulate(ve.Spec(fixed={"mu": 0.0, "omega": 0.05, "alpha1": 0.1}), horizon=5, paths=10, seed=0)


def test_simulate_outlives_sources(series):
    context = ve.Context(threads=2)
    fit = context.fit(ve.Spec(), series[:, 2])
    sims = context.simulate(fit, horizon=10, paths=50, seed=3)
    before = sims.returns.copy()
    del fit, context
    gc.collect()
    np.testing.assert_array_equal(sims.returns, before)


CHECKPOINTS = [1, 5, 20]


def left_sum(values, k):
    """Sum of the first k values, added left to right (np.cumsum may group differently)."""
    total = 0.0
    for value in values[:k]:
        total += value
    return total


def left_sums(returns, checkpoints):
    return np.array([[left_sum(row, k) for k in checkpoints] for row in returns])


@pytest.mark.parametrize("source", ["fit", "spec"])
def test_cumulative_matches_the_left_sum(ctx, series, source):
    model = ve.Spec(fixed=FIXED_SGARCH)
    if source == "fit":
        model = ctx.fit(ve.Spec(), series[:, 0])
    full = ctx.simulate(model, horizon=20, paths=50, seed=5)
    cumulative = ctx.simulate(model, horizon=20, paths=50, seed=5, checkpoints=CHECKPOINTS)
    assert cumulative.checkpoints == (1, 5, 20)
    assert cumulative.cumulative.shape == (50, 3)
    assert not cumulative.cumulative.flags.writeable
    assert np.array_equal(cumulative.cumulative, left_sums(full.returns, CHECKPOINTS))
    assert full.checkpoints == ()
    with pytest.raises(ve.VolEngineError):
        full.cumulative


@pytest.mark.parametrize("checkpoints", [[0, 20], [5, 5, 20], [5, 10]])
def test_checkpoints_are_validated(ctx, checkpoints):
    with pytest.raises(ve.VolEngineError):
        ctx.simulate(ve.Spec(fixed=FIXED_SGARCH), horizon=20, paths=10, seed=0, checkpoints=checkpoints)


@pytest.mark.parametrize("checkpoints", [[True, 20], [-1, 20], [1.5, 20]])
def test_checkpoints_reject_non_integers(ctx, checkpoints):
    with pytest.raises((TypeError, ValueError)):
        ctx.simulate(ve.Spec(fixed=FIXED_SGARCH), horizon=20, paths=10, seed=0, checkpoints=checkpoints)


def test_cumulative_mode_has_no_full_returns(ctx):
    sims = ctx.simulate(ve.Spec(fixed=FIXED_SGARCH), horizon=20, paths=10, seed=0, checkpoints=[20])
    with pytest.raises(ve.VolEngineError):
        sims.returns
    with pytest.raises(ve.VolEngineError):
        sims.sigma
