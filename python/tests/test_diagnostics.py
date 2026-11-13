import numpy as np
import pytest

import volengine as ve


@pytest.fixture(scope="module")
def fit(ctx, series):
    return ctx.fit(ve.Spec(), series[:, 0])


@pytest.mark.parametrize("method,args,kwargs", [
    ("ljung_box", (5,), {}), ("ljung_box", (5,), {"squared": True}),
    ("arch_lm", (5,), {"fitdf": 2}), ("gof", (), {}),
])
def test_diagnostic_result_shape(fit, method, args, kwargs):
    result = getattr(fit, method)(*args, **kwargs)
    assert isinstance(result, ve.TestResult)
    assert np.isfinite(result).all()
    assert 0 <= result.p_value <= 1


def test_diagnostics_shapes(fit):
    sign = fit.sign_bias()
    assert isinstance(sign, ve.SignBias)
    assert np.isfinite(sign).all()
    nyblom = fit.nyblom()
    assert isinstance(nyblom, ve.Nyblom)
    assert set(nyblom.individual) == set(fit.names)
    assert np.isfinite(nyblom.joint)
    assert nyblom.individual_critical == (0.353, 0.470, 0.748)
    assert nyblom.joint_critical == (1.07, 1.24, 1.60)


def test_nyblom_individual_names(ctx, series):
    fit = ctx.fit(ve.Spec(fixed={"mu": 0.05}), series[:, 0])
    assert set(fit.nyblom().individual) == set(fit.names) - {"mu"}
    fixed = dict(zip(fit.names, fit.coef().estimate, strict=True))
    evaluation = ctx.fit(ve.Spec(fixed=fixed), series[:, 0])
    assert evaluation.nyblom().individual == {}
    assert np.isnan(evaluation.nyblom().joint)
    assert evaluation.nyblom().joint_critical is None


@pytest.mark.parametrize("value", [-1, 2**64])
@pytest.mark.parametrize("method,kwargs,argument", [
    ("ljung_box", {}, "lag"), ("ljung_box", {"lag": 5}, "fitdf"),
    ("arch_lm", {"fitdf": 2}, "lag"), ("arch_lm", {"lag": 5}, "fitdf"), ("gof", {}, "groups"),
])
def test_diagnostics_integer_bounds(fit, value, method, kwargs, argument):
    with pytest.raises(ValueError):
        getattr(fit, method)(**kwargs, **{argument: value})


@pytest.mark.parametrize("method,kwargs,message", [
    ("ljung_box", {"lag": 0}, "lag"), ("ljung_box", {"lag": 4, "fitdf": 2}, "fitdf|lag"),
    ("arch_lm", {"lag": 5, "fitdf": 0}, "fitdf"), ("gof", {"groups": 1}, "groups"),
    ("gof", {"groups": 2**63}, "groups"),
])
def test_diagnostics_bad_arguments_raise(fit, method, kwargs, message):
    with pytest.raises(ve.VolEngineError, match=message):
        getattr(fit, method)(**kwargs)


@pytest.mark.parametrize("method,kwargs", [
    ("ljung_box", {"lag": 1}), ("arch_lm", {"lag": 3, "fitdf": 2}),
    ("nyblom", {}), ("sign_bias", {}), ("gof", {}),
])
def test_empty_fit_diagnostics(ctx, method, kwargs):
    fit = ctx.fit(ve.Spec(), np.ones(10))
    with pytest.raises(ve.VolEngineError, match="finite"):
        getattr(fit, method)(**kwargs)


@pytest.mark.parametrize("spec, excluded", [(ve.Spec(variance_targeting=True), "omega"),
                                            (ve.Spec("igarch"), "beta1")])
def test_nyblom_excludes_derived(ctx, series, spec, excluded):
    fit = ctx.fit(spec, series[:, 0])
    assert set(fit.nyblom().individual) == set(fit.names) - {excluded}


def test_nyblom_retains_estimated_nan(fit, monkeypatch):
    import ctypes

    from volengine._native import lib

    def zero_scores(pointer, individual, joint):
        values = (ctypes.c_double * len(fit.names)).from_address(individual)
        for i in range(len(values)):
            values[i] = float("nan")
        ctypes.cast(joint, ctypes.POINTER(ctypes.c_double))[0] = float("nan")
        return 0

    monkeypatch.setattr(lib, "ve_fit_nyblom", zero_scores)
    result = fit.nyblom()
    assert set(result.individual) == set(fit.names)
    assert all(np.isnan(value) for value in result.individual.values())
    assert result.joint_critical == (1.07, 1.24, 1.60)
