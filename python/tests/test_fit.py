import copy
import ctypes
import dataclasses
import gc
import math
import pickle
import weakref

import numpy as np
import pytest

import volengine as ve
from volengine import _native as native

RAGGED_STARTS = [0, 1000, 2995]
RAGGED_LENGTHS = [3000, 2000, 5]


def test_single_spec_1d_returns_fit(ctx, series):
    assert isinstance(ctx.fit(ve.Spec(), series[:, 0]), ve.Fit)


def test_single_spec_2d_returns_object_array_of_fits(ctx, series):
    fits = ctx.fit(ve.Spec(), series)
    assert fits.shape == (3,)
    assert fits.dtype == object
    assert all(isinstance(fit, ve.Fit) for fit in fits)


def test_spec_list_1d_returns_one_fit_per_spec(ctx, series):
    fits = ctx.fit([ve.Spec(), ve.Spec(include_mean=False)], series[:, 0])
    assert fits.shape == (2,)
    assert [fit.names for fit in fits] == [("mu", "omega", "alpha1", "beta1"), ("omega", "alpha1", "beta1")]


def test_grid_shape(ctx, series):
    fits = ctx.fit([ve.Spec(), ve.Spec(include_mean=False)], series)
    assert fits.shape == (2, 3)


def test_grid_is_cartesian_spec_major(ctx, series):
    fits = ctx.fit([ve.Spec(), ve.Spec(include_mean=False)], series)
    assert [len(fits[0, j].names) for j in range(3)] == [4, 4, 4]
    assert [len(fits[1, j].names) for j in range(3)] == [3, 3, 3]
    single = ctx.fit(ve.Spec(include_mean=False), series[:, 2])
    assert np.array_equal(fits[1, 2].coef().estimate, single.coef().estimate)


def test_fit_converges_with_expected_metadata(ctx, series):
    fit = ctx.fit(ve.Spec(), series[:, 0])
    assert fit.status == "converged"
    assert fit.nobs == 3000
    assert fit.names == ("mu", "omega", "alpha1", "beta1")


def test_views_zero_copy_and_readonly(ctx, series):
    f = ctx.fit(ve.Spec(), series[:, 0])
    assert not f.sigma.flags.writeable
    assert f.sigma.shape == (3000,)


@pytest.mark.parametrize("name", ["sigma", "residuals", "std_residuals", "fitted"])
def test_every_view_is_readonly_float64_of_nobs(ctx, series, name):
    view = getattr(ctx.fit(ve.Spec(), series[:, 0]), name)
    assert view.dtype == np.float64
    assert view.shape == (3000,)
    assert not view.flags.writeable
    with pytest.raises(ValueError):
        view[0] = 0.0


def test_residuals_are_returns_minus_fitted(ctx, series):
    fit = ctx.fit(ve.Spec(), series[:, 0])
    assert np.allclose(fit.residuals + fit.fitted, series[:, 0])
    assert np.allclose(fit.std_residuals, fit.residuals / fit.sigma)


def test_view_survives_fit_reference_drop(ctx, series):
    s = ctx.fit(ve.Spec(), series[:, 0]).sigma
    gc.collect()
    assert np.isfinite(s).all()


def test_view_keeps_fit_alive_until_view_dropped(ctx, series):
    fit = ctx.fit(ve.Spec(), series[:, 0])
    alive = weakref.ref(fit)
    view = fit.sigma
    del fit
    gc.collect()
    assert alive() is not None
    del view
    gc.collect()
    assert alive() is None


def test_fortran_and_reversed_match(ctx, series):
    a = ctx.fit(ve.Spec(), series)
    b = ctx.fit(ve.Spec(), np.asfortranarray(series))
    assert all(np.array_equal(x.coef().estimate, y.coef().estimate) for x, y in zip(a, b))


def test_negative_stride_matches_contiguous_copy(ctx, series):
    reversed_view = series[::-1]
    assert reversed_view.strides[0] < 0
    a = ctx.fit(ve.Spec(), reversed_view)
    b = ctx.fit(ve.Spec(), np.ascontiguousarray(reversed_view))
    assert all(np.array_equal(x.coef().estimate, y.coef().estimate) for x, y in zip(a, b))


def test_strided_column_view_matches_contiguous_copy(ctx, series):
    column = series[:, 1]
    a = ctx.fit(ve.Spec(), column)
    b = ctx.fit(ve.Spec(), column.copy())
    assert np.array_equal(a.coef().estimate, b.coef().estimate)


def test_input_is_not_modified(ctx, series):
    before = series.copy()
    ctx.fit(ve.Spec(), series)
    assert np.array_equal(series, before)


def test_list_input_is_converted_to_float64(ctx, series):
    a = ctx.fit(ve.Spec(), series[:, 0].tolist())
    b = ctx.fit(ve.Spec(), series[:, 0])
    assert np.array_equal(a.coef().estimate, b.coef().estimate)


def test_ragged(ctx, series):
    fits = ctx.fit(ve.Spec(), series, starts=RAGGED_STARTS, lengths=RAGGED_LENGTHS)
    assert [f.nobs for f in fits] == RAGGED_LENGTHS
    assert fits[2].status == "insufficient_data"


def test_ragged_series_match_direct_fit_on_the_slice(ctx, series):
    fits = ctx.fit(ve.Spec(), series, starts=RAGGED_STARTS, lengths=RAGGED_LENGTHS)
    direct = ctx.fit(ve.Spec(), series[1000:3000, 1])
    assert np.array_equal(fits[1].coef().estimate, direct.coef().estimate)


def test_insufficient_data_fit_is_nan_with_empty_views(ctx, series):
    fit = ctx.fit(ve.Spec(), series, starts=RAGGED_STARTS, lengths=RAGGED_LENGTHS)[2]
    assert np.isnan(fit.coef().estimate).all()
    assert np.isnan(fit.loglik)
    assert fit.sigma.shape == (0,)
    assert fit.sigma.dtype == np.float64
    assert not fit.sigma.flags.writeable


@pytest.mark.parametrize("keyword", ["starts", "lengths"])
def test_starts_without_lengths_or_reverse_raises(ctx, series, keyword):
    with pytest.raises(ValueError, match="together"):
        ctx.fit(ve.Spec(), series, **{keyword: [0, 0, 0]})


@pytest.mark.parametrize("starts, lengths", [([0, 0], [10, 10]), ([0, 0, 0], [10, 10]), ([[0, 0, 0]], [[1, 1, 1]])])
def test_ranges_with_wrong_length_raise_value_error(ctx, series, starts, lengths):
    with pytest.raises(ValueError, match="one entry per series"):
        ctx.fit(ve.Spec(), series, starts=starts, lengths=lengths)


def test_range_beyond_panel_is_a_native_error(ctx, series):
    with pytest.raises(ve.VolEngineError) as error:
        ctx.fit(ve.Spec(), series, starts=[0, 0, 2999], lengths=[3000, 3000, 10])
    assert error.value.status == 1


@pytest.mark.parametrize("shape", [(), (2, 3, 4)])
def test_unsupported_ndim_raises_value_error(ctx, shape):
    with pytest.raises(ValueError, match="dimensions"):
        ctx.fit(ve.Spec(), np.zeros(shape))


def test_criteria_scale(ctx, series):
    f = ctx.fit(ve.Spec(), series[:, 0])
    assert f.aic(per_obs=False) == pytest.approx(f.aic() * f.nobs)


@pytest.mark.parametrize("name", ["aic", "bic", "hqic", "sic"])
def test_every_criterion_total_is_per_obs_times_nobs(ctx, series, name):
    f = ctx.fit(ve.Spec(), series[:, 0])
    criterion = getattr(f, name)
    assert criterion(per_obs=False) == pytest.approx(criterion() * f.nobs)


def test_fit_statistics_are_members(ctx, series):
    f = ctx.fit(ve.Spec(), series[:, 0])
    n, k, ll = f.nobs, len(f.names), f.loglik
    assert ll == native.read_scalar(ctypes.c_double, native.lib.ve_loglik, f._handle.pointer)
    penalties = {"aic": 2.0 * k, "bic": k * np.log(n), "hqic": 2.0 * k * np.log(np.log(n))}
    for name, penalty in penalties.items():
        assert getattr(f, name)(per_obs=False) == pytest.approx(-2.0 * ll + penalty, rel=1e-14)
        assert getattr(f, name)() == pytest.approx((-2.0 * ll + penalty) / n, rel=1e-14)
    sic = -2.0 * ll / n + np.log((n + 2.0 * k) / n)
    assert f.sic() == pytest.approx(sic, rel=1e-14)
    assert f.sic(per_obs=False) == pytest.approx(sic * n, rel=1e-14)
    assert not hasattr(ve, "aic")
    assert not hasattr(ve, "loglik")


def test_coef_table_fields_and_robust_difference(ctx, series):
    fit = ctx.fit(ve.Spec(), series[:, 0])
    hessian = fit.coef()
    robust = fit.coef(se="robust")
    assert hessian.names == fit.names
    assert hessian.estimate.shape == (4,)
    assert np.array_equal(hessian.estimate, robust.estimate)
    assert not np.array_equal(hessian.std_error, robust.std_error)
    assert np.allclose(hessian.t_value, hessian.estimate / hessian.std_error)
    assert ((hessian.p_value >= 0.0) & (hessian.p_value <= 1.0)).all()


def test_coef_unknown_se_raises(ctx, series):
    with pytest.raises(ValueError, match="unknown se"):
        ctx.fit(ve.Spec(), series[:, 0]).coef(se="bootstrap")


def test_options_are_forwarded_to_the_fit(ctx, series):
    default = ctx.fit(ve.Spec(), series[:, 0])
    backcast = ctx.fit(ve.Spec(), series[:, 0], options=ve.FitOptions(var_init="backcast"))
    assert not np.array_equal(default.coef().estimate, backcast.coef().estimate)
    capped = ctx.fit(ve.Spec(), series[:, 0], options=ve.FitOptions(max_iter=1))
    assert capped.status == "max_iter"


def test_nan_raises(ctx, series):
    s = series.copy()
    s[10, 0] = np.nan
    with pytest.raises(ve.VolEngineError, match="NaN"):
        ctx.fit(ve.Spec(), s)


def test_error_carries_status_and_next_call_still_works(ctx, series):
    s = series.copy()
    s[10, 0] = np.nan
    with pytest.raises(ve.VolEngineError) as error:
        ctx.fit(ve.Spec(), s)
    assert error.value.status == 1
    assert ctx.fit(ve.Spec(), series[:, 0]).status == "converged"


def test_empty_spec_list_is_a_native_error(ctx, series):
    with pytest.raises(ve.VolEngineError):
        ctx.fit([], series)


def test_results_do_not_depend_on_the_context(series):
    context = ve.Context(threads=1)
    fit = context.fit(ve.Spec(), series[:, 0])
    del context
    gc.collect()
    assert fit.status == "converged"
    assert np.isfinite(fit.sigma).all()


def test_results_are_identical_across_thread_counts(series):
    one = ve.Context(threads=1).fit(ve.Spec(), series)
    four = ve.Context(threads=4).fit(ve.Spec(), series)
    assert all(np.array_equal(a.coef().estimate, b.coef().estimate) for a, b in zip(one, four))


def test_unknown_dist_raises():
    with pytest.raises(ValueError):
        ve.Spec(dist="cauchy")


def test_unknown_distribution():
    with pytest.raises(ValueError) as error:
        ve.Spec(dist="ghst")
    for name in ("norm", "std", "ged", "sstd", "sged", "nig", "ghyp", "jsu"):
        assert repr(name) in str(error.value)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        ve.Spec(model="figarch")


def test_unknown_model_message_lists_fgarch():
    with pytest.raises(ValueError, match="fgarch"):
        ve.Spec(model="fgarh")


def test_invalid_order_is_a_native_error():
    with pytest.raises(ve.VolEngineError) as error:
        ve.Spec(q=0)
    assert error.value.status == 1


def test_unknown_var_init_raises():
    with pytest.raises(ValueError, match="unknown var_init"):
        ve.FitOptions(var_init="zero")


@pytest.mark.parametrize("keywords", [{"max_iter": 0}, {"tol": 0.0}, {"tol": float("nan")}])
def test_invalid_fit_options_are_native_errors(keywords):
    with pytest.raises(ve.VolEngineError) as error:
        ve.FitOptions(**keywords)
    assert error.value.status == 1


def test_negative_threads_is_a_native_error():
    with pytest.raises(ve.VolEngineError):
        ve.Context(threads=-1)


def test_threads_above_the_cap_are_a_native_error():
    with pytest.raises(ve.VolEngineError, match="threads") as error:
        ve.Context(threads=2049)
    assert error.value.status == 1


INT32_OVERFLOWS = [2**31, -(2**31) - 1, 2**32 + 1]


@pytest.mark.parametrize("value", INT32_OVERFLOWS)
@pytest.mark.parametrize(
    "build",
    [
        lambda value: ve.Context(threads=value),
        lambda value: ve.Spec(p=value),
        lambda value: ve.Spec(q=value),
        lambda value: ve.FitOptions(max_iter=value),
    ],
    ids=["threads", "p", "q", "max_iter"],
)
def test_integers_outside_int32_raise_value_error(build, value):
    with pytest.raises(ValueError, match="32-bit"):
        build(value)


def test_unknown_native_status_is_named_by_its_code(monkeypatch, ctx, series):
    def newer_status(pointer, out):
        out._obj.value = 99
        return 0

    monkeypatch.setattr(native.lib, "ve_fit_status", newer_status)
    assert ctx.fit(ve.Spec(), series[:, 0]).status == "unknown(99)"


HANDLE_KINDS = [
    ("ve_spec_create", "ve_spec_free"),
    ("ve_fit_options_create", "ve_fit_options_free"),
    ("ve_panel_create", "ve_panel_free"),
    ("ve_batch_create", "ve_batch_free"),
]


def count_calls(monkeypatch, name):
    calls = []
    original = getattr(native.lib, name)

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(native.lib, name, counted)
    return calls


def balanced_calls(monkeypatch, create, free):
    created = count_calls(monkeypatch, create)
    freed = count_calls(monkeypatch, free)
    return created, freed


@pytest.mark.parametrize("create, free", HANDLE_KINDS)
def test_temporary_handles_are_freed_exactly_once(monkeypatch, ctx, series, create, free):
    options = ve.FitOptions()
    created, freed = balanced_calls(monkeypatch, create, free)
    ctx.fit([ve.Spec(), ve.Spec(include_mean=False)], series, options=options)
    gc.collect()
    assert len(created) > 0
    assert len(freed) == len(created)


@pytest.mark.parametrize("create, free", HANDLE_KINDS)
def test_temporary_handles_are_freed_exactly_once_on_error(monkeypatch, ctx, series, create, free):
    options = ve.FitOptions()
    poisoned = series.copy()
    poisoned[10, 0] = np.nan
    created, freed = balanced_calls(monkeypatch, create, free)
    with pytest.raises(ve.VolEngineError):
        ctx.fit([ve.Spec(), ve.Spec(include_mean=False)], poisoned, options=options)
    gc.collect()
    assert len(created) > 0
    assert len(freed) == len(created)


def test_fit_handles_are_freed_once_the_fits_are_dropped(monkeypatch, ctx, series):
    freed = count_calls(monkeypatch, "ve_fit_free")
    fits = ctx.fit(ve.Spec(), series)
    assert len(freed) == 0
    del fits
    gc.collect()
    assert len(freed) == 3


def test_stride_not_multiple_of_element_size_matches_contiguous_copy(ctx, series):
    packed = np.zeros(3000, dtype=np.dtype([("flag", "u1"), ("x", "<f8")]))
    packed["x"] = series[:, 0]
    column = packed["x"]
    assert column.strides[0] % 8 != 0
    a = ctx.fit(ve.Spec(), column)
    b = ctx.fit(ve.Spec(), series[:, 0].copy())
    assert a.status == "converged"
    assert np.array_equal(a.coef().estimate, b.coef().estimate)
    assert np.array_equal(a.sigma, b.sigma)


@pytest.mark.parametrize("keyword", ["starts", "lengths"])
@pytest.mark.parametrize(
    "bad",
    [[0.5, 0, 0], [-1, 0, 0], [0, np.nan, 0], [0, np.inf, 0], [True, False, True], ["a", "b", "c"]],
)
def test_malformed_ranges_raise_value_error_before_native_calls(monkeypatch, ctx, series, keyword, bad):
    created = count_calls(monkeypatch, "ve_panel_create")
    good = [0, 0, 0]
    keywords = {"starts": good, "lengths": [10, 10, 10]}
    keywords[keyword] = bad
    with pytest.raises(ValueError, match="non-negative integers"):
        ctx.fit(ve.Spec(), series, **keywords)
    assert created == []


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64, np.float64])
def test_integral_ranges_of_any_numeric_dtype_are_accepted(ctx, series, dtype):
    starts = np.array(RAGGED_STARTS, dtype=dtype)
    lengths = np.array(RAGGED_LENGTHS, dtype=dtype)
    fits = ctx.fit(ve.Spec(), series, starts=starts, lengths=lengths)
    assert [f.nobs for f in fits] == RAGGED_LENGTHS


def test_arma_std_spec_fits(ctx, series):
    f = ctx.fit(ve.Spec(p=2, arma=(1, 1), dist="std"), series[:, 0])
    assert f.names == ("mu", "ar1", "ma1", "omega", "alpha1", "alpha2", "beta1", "shape")


def test_ged_spec_fits(ctx, series):
    f = ctx.fit(ve.Spec(dist="ged"), series[:, 0])
    assert f.names == ("mu", "omega", "alpha1", "beta1", "shape")


@pytest.mark.parametrize("arma", [(1,), (1, 2, 3), (1.5, 0), "11"])
def test_bad_arma_raises_value_error(arma):
    with pytest.raises(ValueError, match="arma"):
        ve.Spec(arma=arma)


def test_arma_list_equals_tuple_and_hashes():
    assert ve.Spec(arma=[1, 1]) == ve.Spec(arma=(1, 1))
    assert hash(ve.Spec(arma=[1, 1])) == hash(ve.Spec(arma=(1, 1)))


def test_order_above_max_raises():
    with pytest.raises(ve.VolEngineError, match="VE_MAX_ORDER"):
        ve.Spec(p=7)
    with pytest.raises(ve.VolEngineError, match="VE_MAX_ORDER"):
        ve.Spec(arma=(0, 7))


def simulate_igarch(seed):
    shocks = np.random.default_rng(seed).standard_normal(4000)
    returns = np.empty(4000)
    variance = 1.0
    for t, shock in enumerate(shocks):
        residual = math.sqrt(variance) * shock
        returns[t] = 0.05 + residual
        variance = 0.01 + 0.1 * residual * residual + 0.9 * variance
    return returns[1000:]


@pytest.mark.parametrize(
    ("model", "submodel", "extra"),
    [
        ("egarch", None, ("gamma1",)),
        ("csgarch", None, ("eta11", "eta21")),
        ("fgarch", "allgarch", ("eta11", "eta21", "lambda")),
    ],
)
def test_variant_names(ctx, series, model, submodel, extra):
    fit = ctx.fit(ve.Spec(model, submodel=submodel), series[:, 0], options=ve.FitOptions(max_iter=30))
    assert fit.names == ("mu", "omega", "alpha1", "beta1", *extra)


@pytest.mark.parametrize(
    ("dist", "extra"),
    [
        ("sstd", ("skew", "shape")),
        ("nig", ("skew", "shape")),
        ("ghyp", ("skew", "shape", "ghlambda")),
        ("jsu", ("skew", "shape")),
    ],
)
def test_distribution_names(ctx, series, dist, extra):
    fit = ctx.fit(ve.Spec(dist=dist), series[:, 0], options=ve.FitOptions(max_iter=30))
    assert fit.names == ("mu", "omega", "alpha1", "beta1", *extra)


def test_fgarch_requires_submodel():
    with pytest.raises(ValueError, match="submodel"):
        ve.Spec(model="fgarch")


def test_submodel_only_for_fgarch():
    with pytest.raises(ValueError, match="submodel"):
        ve.Spec(model="egarch", submodel="tgarch")


def test_unknown_submodel_raises():
    with pytest.raises(ValueError, match="submodel"):
        ve.Spec(model="fgarch", submodel="cgarch")


def test_igarch_derived_beta_has_nan_se(ctx):
    fit = ctx.fit(ve.Spec("igarch"), simulate_igarch(1))
    std_error = dict(zip(fit.names, fit.coef().std_error, strict=True))
    assert math.isnan(std_error["beta1"])
    assert all(math.isfinite(value) for name, value in std_error.items() if name != "beta1")


def test_fixed_parameter_is_kept_with_nan_se(ctx, series):
    fit = ctx.fit(ve.Spec(dist="std", fixed={"shape": 6.0}), series[:, 0], options=ve.FitOptions(max_iter=100))
    table = dict(zip(fit.names, zip(fit.coef().estimate, fit.coef().std_error, strict=True), strict=True))
    assert table["shape"][0] == 6.0
    assert math.isnan(table["shape"][1])
    assert all(math.isfinite(se) for name, (_, se) in table.items() if name != "shape")


def test_start_does_not_change_the_optimum(ctx, series):
    plain = ctx.fit(ve.Spec(), series[:, 0])
    start = dict(zip(plain.names, plain.coef().estimate, strict=True))
    warm = ctx.fit(ve.Spec(start=start), series[:, 0])
    assert np.allclose(warm.coef().estimate, plain.coef().estimate, atol=1e-6, rtol=0.0)


def test_variance_targeting_omega_has_nan_se(ctx, series):
    fit = ctx.fit(ve.Spec(variance_targeting=True), series[:, 0])
    std_error = dict(zip(fit.names, fit.coef().std_error, strict=True))
    assert math.isnan(std_error["omega"])
    assert all(math.isfinite(value) for name, value in std_error.items() if name != "omega")
    estimate = dict(zip(fit.names, fit.coef().estimate, strict=True))
    variance = float(np.mean(fit.residuals**2))
    assert estimate["omega"] == pytest.approx(variance * (1.0 - estimate["alpha1"] - estimate["beta1"]))


def test_unknown_fixed_name_raises():
    with pytest.raises(ve.VolEngineError, match="shpe"):
        ve.Spec(fixed={"shpe": 5.0})


def test_fixed_wins_over_start_for_the_same_name(ctx, series):
    fit = ctx.fit(ve.Spec(fixed={"alpha1": 0.08}, start={"alpha1": 0.4}), series[:, 0])
    table = dict(zip(fit.names, zip(fit.coef().estimate, fit.coef().std_error, strict=True), strict=True))
    assert table["alpha1"][0] == 0.08
    assert math.isnan(table["alpha1"][1])


def test_warm_start_keeps_fixed(ctx, series):
    spec = ve.Spec(dist="std", fixed={"shape": 6.0})
    plain = ctx.fit(spec, series[:, 0], options=ve.FitOptions(max_iter=100))
    warm_spec = ve.Spec(dist="std", fixed={"shape": 6.0}, start=dict(zip(plain.names, plain.coef().estimate, strict=True)))
    warm = ctx.fit(warm_spec, series[:, 0], options=ve.FitOptions(max_iter=100))
    assert warm.coef().estimate[plain.names.index("shape")] == 6.0
    assert np.allclose(warm.coef().estimate, plain.coef().estimate, atol=1e-6, rtol=0.0)


def test_nul_in_a_parameter_name_raises():
    for field in ("fixed", "start"):
        with pytest.raises(ValueError, match="mu"):
            ve.Spec(**{field: {"mu\0x": 0.0}})


@pytest.mark.parametrize(
    "spec",
    [
        ve.Spec(),
        ve.Spec(dist="std", fixed={"shape": 6.0}, start={"alpha1": 0.1}, variance_targeting=True),
    ],
)
def test_spec_pickles_deepcopies_and_converts_with_asdict(spec):
    for clone in (pickle.loads(pickle.dumps(spec)), copy.deepcopy(spec)):
        assert clone == spec
        assert hash(clone) == hash(spec)
        assert dict(clone.fixed) == dict(spec.fixed)
        assert dict(clone.start) == dict(spec.start)
    as_dict = dataclasses.asdict(spec)
    assert as_dict["fixed"] == dict(spec.fixed)
    assert as_dict["start"] == dict(spec.start)
    assert ve.Spec(**as_dict) == spec


def test_spec_with_values_hashes():
    source = {"mu": 0.0}
    spec = ve.Spec(fixed=source)
    assert spec == ve.Spec(fixed={"mu": 0.0})
    assert hash(spec) == hash(ve.Spec(fixed={"mu": 0.0}))
    assert spec != ve.Spec(start={"mu": 0.0})
    source["mu"] = 1.0
    assert spec.fixed["mu"] == 0.0
    assert spec == ve.Spec(fixed={"mu": 0.0})


def test_spec_values_are_read_only():
    spec = ve.Spec(fixed={"mu": 0.0}, start={"alpha1": 0.1})
    with pytest.raises(TypeError):
        spec.fixed["mu"] = 1.0  # type: ignore[index]
    with pytest.raises(TypeError):
        spec.start["alpha1"] = 1.0  # type: ignore[index]
    for values in (spec.fixed, spec.start):
        for mutate in (values.clear, values.popitem, lambda: values.pop("mu"), lambda: values.update(mu=1.0)):
            with pytest.raises(TypeError):
                mutate()  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            del values["mu"]  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            values.setdefault("mu", 1.0)  # type: ignore[attr-defined]
        with pytest.raises(TypeError):
            values |= {"mu": 1.0}  # type: ignore[assignment,operator]


def test_variance_targeting_in_igarch_raises():
    with pytest.raises(ve.VolEngineError, match="variance targeting"):
        ve.Spec("igarch", variance_targeting=True)
