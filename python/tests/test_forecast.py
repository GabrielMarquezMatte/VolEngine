import gc

import numpy as np
import pytest

import volengine as ve


def test_forecast_shapes_and_read_only(ctx, series):
    forecast = ctx.fit(ve.Spec(arma=(1, 1)), series[:, 0]).forecast(10)
    assert isinstance(forecast, ve.Forecast)
    assert forecast.sigma.shape == (10,)
    assert forecast.mean.shape == (10,)
    assert not forecast.sigma.flags.writeable
    assert not forecast.mean.flags.writeable
    assert np.isfinite(forecast.sigma).all()
    assert (forecast.sigma > 0).all()


def test_forecast_first_step_is_the_next_sigma_of_sgarch(ctx, series):
    fit = ctx.fit(ve.Spec(), series[:, 0])
    mu, omega, alpha, beta = fit.coef().estimate
    expected = omega + alpha * fit.residuals[-1] ** 2 + beta * fit.sigma[-1] ** 2
    forecast = fit.forecast(1)
    assert forecast.sigma[0] ** 2 == pytest.approx(expected, rel=1e-12)
    assert forecast.mean[0] == pytest.approx(mu, rel=1e-15)


def test_forecast_with_regressors(ctx, series):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((series.shape[0], 2))
    v = 0.5 + rng.random((series.shape[0], 1))
    fit = ctx.fit(ve.Spec(mxreg=2, vxreg=1), series[:, 0], mxreg=x, vxreg=v)
    two_columns = fit.forecast(5, mxreg=np.ones((5, 2)), vxreg=np.full((5, 1), 0.7))
    one_dimensional = fit.forecast(5, mxreg=np.ones((5, 2)), vxreg=np.full(5, 0.7))
    np.testing.assert_array_equal(two_columns.sigma, one_dimensional.sigma)
    np.testing.assert_array_equal(two_columns.mean, one_dimensional.mean)
    mu, delta1, delta2 = fit.coef().estimate[:3]
    assert two_columns.mean[0] == pytest.approx(mu + delta1 + delta2, rel=1e-14)


@pytest.mark.parametrize(
    ("keywords", "word"),
    [
        ({}, "mxreg"),
        ({"mxreg": np.ones((4, 2)), "vxreg": np.ones((5, 1))}, "mxreg"),
        ({"mxreg": np.ones((5, 3)), "vxreg": np.ones((5, 1))}, "mxreg"),
        ({"mxreg": np.ones((5, 2))}, "vxreg"),
    ],
)
def test_forecast_bad_regressors_raise(ctx, series, keywords, word):
    rng = np.random.default_rng(4)
    x = rng.standard_normal((series.shape[0], 2))
    v = 0.5 + rng.random((series.shape[0], 1))
    fit = ctx.fit(ve.Spec(mxreg=2, vxreg=1), series[:, 0], mxreg=x, vxreg=v)
    with pytest.raises(ValueError, match=word):
        fit.forecast(5, **keywords)


def test_forecast_regressors_without_spec_regressors_raise(ctx, series):
    fit = ctx.fit(ve.Spec(), series[:, 0])
    with pytest.raises(ValueError, match="mxreg"):
        fit.forecast(5, mxreg=np.ones((5, 1)))
    with pytest.raises(ValueError, match="vxreg"):
        fit.forecast(5, vxreg=np.ones((5, 1)))


@pytest.mark.parametrize("horizon", [0, -1])
def test_forecast_bad_horizon_raises(ctx, series, horizon):
    fit = ctx.fit(ve.Spec(), series[:, 0])
    with pytest.raises(ValueError, match="horizon"):
        fit.forecast(horizon)


def test_forecast_of_a_fit_without_finite_point_raises(ctx):
    fit = ctx.fit(ve.Spec(), np.full(500, 0.01))
    assert fit.status == "failed"
    with pytest.raises(ve.VolEngineError, match="finite"):
        fit.forecast(5)


def test_forecast_outlives_fit(ctx, series):
    fit = ctx.fit(ve.Spec(), series[:, 0])
    forecast = fit.forecast(20)
    before = forecast.sigma.copy()
    del fit
    gc.collect()
    np.testing.assert_array_equal(forecast.sigma, before)


def test_forecast_horizon_beyond_size_t_raises(ctx, series):
    fit = ctx.fit(ve.Spec(), series[:, 0])
    with pytest.raises(ValueError, match="horizon"):
        fit.forecast(2**64 + 3)


def left_to_right_sum(values):
    total = 0.0
    sums = []
    for value in values:
        total += float(value)
        sums.append(total)
    return np.array(sums)


def ar1_series(phi, length=1500):
    rng = np.random.default_rng(21)
    shocks = rng.standard_normal(length)
    values = np.zeros(length)
    for t in range(1, length):
        values[t] = phi * values[t - 1] + shocks[t]
    return 0.01 * values


@pytest.mark.parametrize("arma", [(0, 0), (1, 1)])
def test_cumulative_mean_is_the_left_to_right_sum_of_the_means(ctx, series, arma):
    forecast = ctx.fit(ve.Spec(arma=arma), series[:, 0]).forecast(12)
    expected = left_to_right_sum(forecast.mean)
    assert forecast.cumulative_mean.shape == (12,)
    assert not forecast.cumulative_mean.flags.writeable
    assert np.array_equal(forecast.cumulative_mean.view(np.uint64), expected.view(np.uint64))
    assert forecast.cumulative_mean[0] == forecast.mean[0]
    assert forecast.cumulative_sigma[0] == pytest.approx(forecast.sigma[0], rel=1e-14)


def test_cumulative_sigma_without_arma_is_the_root_of_the_summed_variances(ctx, series):
    forecast = ctx.fit(ve.Spec(), series[:, 0]).forecast(12)
    assert forecast.cumulative_sigma.shape == (12,)
    assert not forecast.cumulative_sigma.flags.writeable
    np.testing.assert_allclose(forecast.cumulative_sigma**2, np.cumsum(forecast.sigma**2), rtol=1e-14)


def test_cumulative_sigma_with_positive_ar_exceeds_the_plain_root_sum(ctx):
    forecast = ctx.fit(ve.Spec(arma=(1, 0)), ar1_series(0.6)).forecast(10)
    plain = np.sqrt(np.cumsum(forecast.sigma**2))
    assert forecast.cumulative_sigma[0] == pytest.approx(plain[0], rel=1e-14)
    assert (forecast.cumulative_sigma[1:] > plain[1:]).all()


def test_cumulative_views_outlive_the_fit(ctx, series):
    fit = ctx.fit(ve.Spec(), series[:, 0])
    forecast = fit.forecast(6)
    mean, sigma = forecast.cumulative_mean.copy(), forecast.cumulative_sigma.copy()
    del fit
    gc.collect()
    np.testing.assert_array_equal(forecast.cumulative_mean, mean)
    np.testing.assert_array_equal(forecast.cumulative_sigma, sigma)
