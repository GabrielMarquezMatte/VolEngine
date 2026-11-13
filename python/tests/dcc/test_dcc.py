import gc
import math

import numpy as np
import pytest
import volengine as ve

MODELS = ("dcc", "adcc", "fdcc")
DISTS = ("mvnorm", "mvt", "mvlaplace")
GROUPS = (1, 1, 2)
OBSERVATIONS = 1500


def make_spec(model: str, dist: str) -> ve.DccSpec:
    return ve.DccSpec(model, dist=dist, groups=GROUPS if model == "fdcc" else None)


def grid() -> list[ve.DccSpec]:
    return [make_spec(model, dist) for model in MODELS for dist in DISTS]


@pytest.fixture(scope="module")
def returns() -> np.ndarray:
    rng = np.random.default_rng(11)
    loadings = np.array([[1.0, 0.0, 0.0], [0.5, 0.9, 0.0], [0.3, 0.2, 0.8]])
    return rng.standard_normal((OBSERVATIONS, 3)) @ loadings.T


@pytest.fixture(scope="module")
def univariate(ctx: ve.Context, returns: np.ndarray) -> list[ve.Fit]:
    return list(ctx.fit(ve.Spec(), returns))


def test_fit_dcc_grid_and_bic(ctx, univariate):
    fits = ctx.fit_dcc(grid(), univariate)
    assert isinstance(fits, list)
    assert len(fits) == 9
    for fit in fits:
        assert isinstance(fit, ve.DccFit)
        assert fit.status == "converged"
        assert fit.nobs == OBSERVATIONS
        assert fit.series == 3
        assert math.isfinite(fit.bic())
        assert math.isfinite(fit.aic(per_obs=False))
        assert math.isfinite(fit.hqic())
        assert math.isfinite(fit.sic())
        assert math.isfinite(fit.loglik)
        assert fit.R.shape == (OBSERVATIONS, 3, 3)
        assert fit.H.shape == (OBSERVATIONS, 3, 3)
        assert not fit.R.flags.writeable
        assert not fit.H.flags.writeable
    assert fits[0].names == ("dcca1", "dccb1")
    assert fits[4].names == ("dcca1", "dccb1", "dccg1", "mshape")
    assert fits[8].names == ("fdcca1", "fdcca2", "fdccb1", "fdccb2")


def test_dcc_tables_and_views(ctx, univariate):
    fit = ctx.fit_dcc(make_spec("dcc", "mvnorm"), univariate)
    assert isinstance(fit, ve.DccFit)
    for kind in ("hessian", "robust"):
        table = fit.coef(se=kind)
        assert table.names == fit.names
        assert np.all(np.isfinite(table.std_error))
        np.testing.assert_allclose(table.t_value, table.estimate / table.std_error)
        assert np.all((table.p_value >= 0) & (table.p_value <= 1))
    with pytest.raises(ValueError, match="se"):
        fit.coef(se="bogus")
    sigma = np.column_stack([item.sigma for item in univariate])
    np.testing.assert_allclose(fit.H[10, 1, 2], sigma[10, 1] * fit.R[10, 1, 2] * sigma[10, 2], rtol=1e-14)
    np.testing.assert_array_equal(np.diagonal(fit.R, axis1=1, axis2=2), 1.0)
    assert fit.H is not fit.R
    assert fit.R.base is not None


def test_dcc_views_outlive_the_inputs(univariate, returns):
    context = ve.Context(threads=1)
    fits = list(context.fit(ve.Spec(), returns))
    fit = context.fit_dcc(make_spec("adcc", "mvt"), fits)
    expected_r = fit.R.copy()
    expected_h = fit.H.copy()
    coefficients = fit.coef()
    del fits, context
    gc.collect()
    np.testing.assert_array_equal(fit.R, expected_r)
    np.testing.assert_array_equal(fit.H, expected_h)
    np.testing.assert_array_equal(fit.coef().estimate, coefficients.estimate)
    view = fit.R
    del fit
    gc.collect()
    np.testing.assert_array_equal(view, expected_r)


def test_dcc_standard_errors_later(ctx, univariate):
    spec = make_spec("dcc", "mvt")
    on = ctx.fit_dcc(spec, univariate)
    off = ctx.fit_dcc(spec, univariate, standard_errors=False)
    assert np.all(np.isnan(off.coef(se="hessian").std_error))
    assert np.all(np.isnan(off.coef(se="robust").t_value))
    later = off.standard_errors(univariate)
    assert later is not off
    for kind in ("hessian", "robust"):
        np.testing.assert_array_equal(later.coef(se=kind).std_error, on.coef(se=kind).std_error)
        np.testing.assert_array_equal(later.coef(se=kind).p_value, on.coef(se=kind).p_value)
    np.testing.assert_array_equal(later.coef().estimate, on.coef().estimate)
    assert np.all(np.isnan(off.coef().std_error))
    np.testing.assert_array_equal(later.R, on.R)


def test_dcc_fixed_parameters_leave_the_table(ctx, univariate):
    spec = ve.DccSpec("dcc", fixed={"dcca1": 0.03}, start={"dccb1": 0.9})
    fit = ctx.fit_dcc(spec, univariate)
    assert fit.names == ("dccb1",)
    assert fit.coef().estimate.shape == (1,)
    assert fit.coef(se="robust").std_error.shape == (1,)


def test_dcc_non_converged_spec_keeps_its_own_status(ctx, univariate):
    fits = ctx.fit_dcc([ve.DccSpec("dcc"), ve.DccSpec("dcc", fixed={"dccb1": 1.2})], univariate)
    assert [fit.status for fit in fits] == ["converged", "failed"]
    assert fits[1].R.shape == (0, 3, 3)
    assert fits[1].H.shape == (0, 3, 3)
    assert math.isnan(fits[1].loglik)
    assert math.isnan(fits[1].bic())


def test_dcc_bad_inputs_raise(ctx, univariate, returns):
    spec = ve.DccSpec("dcc")
    with pytest.raises(ValueError, match="model"):
        ve.DccSpec("garch")
    with pytest.raises(ValueError, match="dist"):
        ve.DccSpec("dcc", dist="norm")
    with pytest.raises(ve.VolEngineError, match="FDCC"):
        ve.DccSpec("dcc", groups=GROUPS)
    with pytest.raises(ve.VolEngineError):
        ve.DccSpec("dcc", order=(0, 1))
    with pytest.raises(ve.VolEngineError, match="at least 2 series"):
        ctx.fit_dcc(spec, univariate[:1])
    with pytest.raises(ve.VolEngineError, match="group 2 has no series") as excinfo:
        ctx.fit_dcc(ve.DccSpec("fdcc", groups=(1, 3, 3)), univariate)
    assert excinfo.value.status == 1
    with pytest.raises(ve.VolEngineError, match="spec 2"):
        ctx.fit_dcc([spec, ve.DccSpec("fdcc")], univariate)
    with pytest.raises(ve.VolEngineError, match="dccz9"):
        ctx.fit_dcc(ve.DccSpec("dcc", fixed={"dccz9": 0.1}), univariate)
    with pytest.raises(TypeError, match="Fit"):
        ctx.fit_dcc(spec, [univariate[0], "series"])
    with pytest.raises(ValueError, match="at least one"):
        ctx.fit_dcc([], univariate)
    shorter = ctx.fit(ve.Spec(), returns[:-1, 2])
    with pytest.raises(ve.VolEngineError, match="series 3"):
        ctx.fit_dcc(spec, [univariate[0], univariate[1], shorter])


def test_dcc_regressors_block_standard_errors(ctx, returns):
    x = np.sin(0.3 * np.arange(OBSERVATIONS))[:, None, None] * np.ones((1, 3, 1))
    fits = list(ctx.fit(ve.Spec(mxreg=1), returns, mxreg=x))
    spec = ve.DccSpec("dcc")
    with pytest.raises(ve.VolEngineError, match="regressors") as excinfo:
        ctx.fit_dcc(spec, fits)
    assert excinfo.value.status == 3
    plain = ctx.fit_dcc(spec, fits, standard_errors=False)
    assert plain.status == "converged"
    with pytest.raises(ve.VolEngineError, match="regressors") as later:
        plain.standard_errors(fits)
    assert later.value.status == 3


def test_dcc_standard_errors_reject_other_fits(ctx, univariate):
    fit = ctx.fit_dcc(ve.DccSpec("dcc"), univariate, standard_errors=False)
    reordered = [univariate[1], univariate[0], univariate[2]]
    with pytest.raises(ve.VolEngineError, match="differ") as excinfo:
        fit.standard_errors(reordered)
    assert excinfo.value.status == 1
    with pytest.raises(ve.VolEngineError, match="3 series"):
        fit.standard_errors(univariate[:2])


def test_dcc_spec_is_hashable_and_immutable():
    one = ve.DccSpec("fdcc", groups=[1, 1, 2], fixed={"fdcca1": 0.2})
    two = ve.DccSpec("fdcc", groups=(1, 1, 2), fixed={"fdcca1": 0.2})
    assert one == two
    assert hash(one) == hash(two)
    assert one.groups == (1, 1, 2)
    with pytest.raises(TypeError):
        one.fixed["fdccb1"] = 0.9  # type: ignore[index]
    with pytest.raises(AttributeError):
        one.model = "dcc"  # type: ignore[misc]
