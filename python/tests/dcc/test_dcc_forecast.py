import gc

import numpy as np
import pytest
import volengine as ve

OBSERVATIONS = 1500
HORIZON = 8


def make_spec(model: str, copula: str | None = None) -> ve.DccSpec:
    if model == "fdcc":
        return ve.DccSpec(model, groups=(1, 1, 2), copula=copula)
    return ve.DccSpec(model, copula=copula)


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    rng = np.random.default_rng(11)
    loadings = np.array([[1.0, 0.0, 0.0], [0.5, 0.9, 0.0], [0.3, 0.2, 0.8]])
    return rng.standard_normal((OBSERVATIONS, 3)) @ loadings.T


@pytest.fixture(scope="module")
def univariate(ctx: ve.Context, returns: np.ndarray) -> list[ve.Fit]:
    return list(ctx.fit(ve.Spec(), returns))


def normalized_qbar(z: np.ndarray) -> np.ndarray:
    """The public convention: R-bar = cov2cor(cov(z)) of the standardized residuals."""
    qbar = np.cov(z.T)
    scale = 1.0 / np.sqrt(np.diag(qbar))
    return qbar * scale[:, None] * scale[None, :]


def test_shapes_and_read_only_views(ctx, univariate):
    fit = ctx.fit_dcc(make_spec("dcc"), univariate)
    forecast = fit.forecast(HORIZON)
    assert forecast.R.shape == (HORIZON, 3, 3)
    assert forecast.H.shape == (HORIZON, 3, 3)
    assert forecast.sigma.shape == (HORIZON, 3)
    assert forecast.mean.shape == (HORIZON, 3)
    for view in (forecast.R, forecast.H, forecast.sigma, forecast.mean):
        assert not view.flags.writeable
    np.testing.assert_array_equal(np.diagonal(forecast.R, axis1=1, axis2=2), 1.0)
    with pytest.raises(ValueError, match="horizon"):
        fit.forecast(0)


@pytest.mark.parametrize("model", ["dcc", "adcc"])
def test_correlations_follow_the_formula(ctx, univariate, model):
    fit = ctx.fit_dcc(make_spec(model), univariate)
    forecast = fit.forecast(HORIZON)
    z = np.column_stack([item.std_residuals for item in univariate])
    bar = normalized_qbar(z)
    persistence = fit.coef().estimate[:2].sum()
    first = forecast.R[0]
    for h in range(1, HORIZON):
        weight = persistence**h
        np.testing.assert_allclose(forecast.R[h], (1.0 - weight) * bar + weight * first, atol=1e-12)
    sigma = forecast.sigma
    for h in range(HORIZON):
        np.testing.assert_allclose(forecast.H[h], sigma[h][:, None] * forecast.R[h] * sigma[h][None, :], rtol=1e-14)


def test_series_forecast_is_the_univariate_forecast(ctx, univariate):
    fit = ctx.fit_dcc(make_spec("dcc"), univariate)
    forecast = fit.forecast(HORIZON)
    for i, item in enumerate(univariate):
        own = item.forecast(HORIZON)
        np.testing.assert_array_equal(forecast.sigma[:, i], own.sigma)
        np.testing.assert_array_equal(forecast.mean[:, i], own.mean)


def test_copula_dcc_forecasts(ctx, univariate):
    fit = ctx.fit_dcc(make_spec("dcc", copula="parametric"), univariate)
    forecast = fit.forecast(HORIZON)
    assert np.isfinite(forecast.R).all()
    static = ctx.fit_dcc(ve.DccSpec("static", copula="parametric"), univariate)
    constant = static.forecast(HORIZON)
    for h in range(1, HORIZON):
        np.testing.assert_array_equal(constant.R[h], constant.R[0])


def test_forecast_survives_the_univariate_fits_and_standard_errors(returns):
    context = ve.Context(threads=1)
    fits = list(context.fit(ve.Spec(), returns))
    dcc = context.fit_dcc(make_spec("dcc"), fits, standard_errors=False)
    with_errors = dcc.standard_errors(fits)
    expected = dcc.forecast(HORIZON)
    again = with_errors.forecast(HORIZON)
    for name in ("R", "H", "sigma", "mean"):
        np.testing.assert_array_equal(getattr(again, name), getattr(expected, name))
    del fits
    gc.collect()
    after = dcc.forecast(HORIZON)
    np.testing.assert_array_equal(after.R, expected.R)
    np.testing.assert_array_equal(after.sigma, expected.sigma)


def test_mixed_regressors(ctx, returns):
    x = 0.1 * np.sin(0.3 * np.arange(OBSERVATIONS + HORIZON))
    fits = [
        ctx.fit(ve.Spec(), returns[:, 0]),
        ctx.fit(ve.Spec(mxreg=1), returns[:, 1], mxreg=x[:OBSERVATIONS, None]),
        ctx.fit(ve.Spec(), returns[:, 2]),
    ]
    dcc = ctx.fit_dcc(make_spec("dcc"), fits, standard_errors=False)
    future = x[OBSERVATIONS:]
    forecast = dcc.forecast(HORIZON, mxreg=[None, future, None])
    own = fits[1].forecast(HORIZON, mxreg=future)
    np.testing.assert_array_equal(forecast.mean[:, 1], own.mean)
    np.testing.assert_array_equal(forecast.sigma[:, 1], own.sigma)
    with pytest.raises(ve.VolEngineError, match="series"):
        dcc.forecast(HORIZON, mxreg=[None, future])
    with pytest.raises(ve.VolEngineError, match="series 2"):
        dcc.forecast(HORIZON)
    with pytest.raises(ve.VolEngineError, match="series 1"):
        dcc.forecast(HORIZON, mxreg=[future, future, None])
    with pytest.raises(ValueError, match="mxreg and vxreg"):
        dcc.forecast(HORIZON, mxreg=[None, None, None], vxreg=[None])
