import math
from pathlib import Path

import numpy as np
import pytest
import volengine as ve

OBSERVATIONS = 2000
# A Student-t(6) DCC copula with std(7) margins (plan 4b notes, section 1).
DATA = Path(__file__).resolve().parents[3] / "tests" / "reference" / "data" / "copula3.csv"


@pytest.fixture(scope="module")
def univariate(ctx: ve.Context) -> list[ve.Fit]:
    returns = np.genfromtxt(DATA, delimiter=",", skip_header=1)
    return list(ctx.fit(ve.Spec(dist="std"), returns))


def copula_grid() -> list[ve.DccSpec]:
    specs = []
    for copula in ("parametric", "empirical"):
        specs.append(ve.DccSpec("static", copula=copula))
        specs.append(ve.DccSpec("static", dist="mvt", copula=copula))
        specs.append(ve.DccSpec("static", dist="mvt", copula=copula, static_method="ml"))
        for model in ("dcc", "adcc"):
            for dist in ("mvnorm", "mvt"):
                specs.append(ve.DccSpec(model, dist=dist, copula=copula))
    return specs


def test_copula_grid_and_bic(ctx, univariate):
    specs = [ve.DccSpec("dcc"), *copula_grid()]
    fits = ctx.fit_dcc(specs, univariate, standard_errors=False)
    assert len(fits) == 15
    for spec, fit in zip(specs, fits, strict=True):
        assert fit.status == "converged", spec
        assert math.isfinite(fit.bic()), spec
        assert math.isfinite(fit.loglik), spec
        assert fit.R.shape == (OBSERVATIONS, 3, 3)
        assert not fit.R.flags.writeable
    assert fits[1].names == ()
    assert fits[2].names == ("mshape",)
    assert fits[3].names == ("C1", "C2", "C3", "mshape")
    assert fits[5].names == ("dcca1", "dccb1", "mshape")


def test_copula_static_r_is_constant(ctx, univariate):
    fit = ctx.fit_dcc(ve.DccSpec("static", dist="mvt", copula="empirical"), univariate, standard_errors=False)
    assert np.array_equal(fit.R, np.broadcast_to(fit.R[0], fit.R.shape))
    assert np.allclose(np.diagonal(fit.R[0]), 1.0)


@pytest.mark.parametrize(
    ("spec", "words"),
    [
        (ve.DccSpec("static"), "needs a copula"),
        (ve.DccSpec("fdcc", groups=(1, 1, 2), copula="parametric"), "fdcc model has no copula"),
        (ve.DccSpec("dcc", dist="mvlaplace", copula="empirical"), "mvlaplace copula"),
        (ve.DccSpec("dcc", copula="parametric", static_method="ml"), "static_method"),
        (ve.DccSpec("dcc", dist="mvt", copula="parametric", start={"mshape": 8.0}), "mshape takes no start"),
    ],
)
def test_copula_bad_combinations_raise(ctx, univariate, spec, words):
    with pytest.raises(ve.VolEngineError, match=words):
        ctx.fit_dcc(spec, univariate)


def test_copula_bad_values_raise():
    with pytest.raises(ValueError, match="copula"):
        ve.DccSpec("dcc", copula="spd")
    with pytest.raises(ValueError, match="static_method"):
        ve.DccSpec("static", copula="parametric", static_method="tau")


def test_copula_spec_hash_and_equality():
    a = ve.DccSpec("static", copula="parametric", static_method="ml")
    b = ve.DccSpec("static", copula="parametric", static_method="ml")
    c = ve.DccSpec("static", copula="empirical", static_method="ml")
    assert a == b
    assert hash(a) == hash(b)
    assert a != c


def test_copula_standard_errors_later(ctx, univariate):
    spec = ve.DccSpec("dcc", dist="mvt", copula="parametric")
    light = ctx.fit_dcc(spec, univariate, standard_errors=False)
    assert np.isnan(light.coef().std_error).all()
    later = light.standard_errors(univariate)
    eager = ctx.fit_dcc(spec, univariate)
    np.testing.assert_array_equal(later.coef(se="robust").std_error, eager.coef(se="robust").std_error)
    assert (later.coef().std_error > 0).all()
