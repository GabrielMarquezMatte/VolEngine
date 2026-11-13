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
