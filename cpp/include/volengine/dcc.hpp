#ifndef VOLENGINE_CPP_DCC_HPP
#define VOLENGINE_CPP_DCC_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>
#include <volengine/error.hpp>
#include <volengine/fit.hpp>
#include <volengine/input.hpp>

namespace ve
{

// VE_DCC_MODEL_STATIC is DccModel::static_ (`static` is a keyword).
enum class DccModel : std::int32_t
{
    dcc = VE_DCC_MODEL_DCC,
    adcc = VE_DCC_MODEL_ADCC,
    fdcc = VE_DCC_MODEL_FDCC,
    static_ = VE_DCC_MODEL_STATIC,
};

enum class DccDist : std::int32_t
{
    mvnorm = VE_DCC_DIST_MVNORM,
    mvt = VE_DCC_DIST_MVT,
    mvlaplace = VE_DCC_DIST_MVLAPLACE,
};

// How a Copula-DCC turns the margins into uniforms: each fit's cdf (parametric) or the ranks (empirical).
enum class Copula : std::int32_t
{
    none = VE_DCC_COPULA_NONE,
    parametric = VE_DCC_COPULA_PARAMETRIC,
    empirical = VE_DCC_COPULA_EMPIRICAL,
};

// Only for DccModel::static_ with a copula: Kendall's tau (the default) or maximum likelihood.
enum class StaticMethod : std::int32_t
{
    default_ = VE_DCC_STATIC_DEFAULT,
    kendall = VE_DCC_STATIC_KENDALL,
    ml = VE_DCC_STATIC_ML,
};

enum class DccCriterion : std::int32_t
{
    aic = VE_DCC_CRITERION_AIC,
    bic = VE_DCC_CRITERION_BIC,
    hqic = VE_DCC_CRITERION_HQIC,
    sic = VE_DCC_CRITERION_SIC,
};

namespace detail
{

using DccSpecHandle = Owned<ve_dcc_spec, ve_dcc_spec_free>;
using DccOptionsHandle = Owned<ve_dcc_options, ve_dcc_options_free>;
using DccBatchHandle = Owned<ve_dcc_batch, ve_dcc_batch_free>;

} // namespace detail

// The second stage: a DCC, aDCC, FDCC or static model over the univariate fits, with a plain or a copula
// likelihood. `groups` (FDCC only) is one label per series, consecutive from 1. `fixed` and `start` give parameters by
// the names of DccFit::param_names.
struct DccSpec
{
    DccModel model = DccModel::dcc;
    int p = 1;
    int q = 1;
    DccDist dist = DccDist::mvnorm;
    std::vector<std::int32_t> groups{};
    NamedValues fixed{};
    NamedValues start{};
    Copula copula = Copula::none;
    StaticMethod static_method = StaticMethod::default_;

    Result<detail::DccSpecHandle> make_handle() const
    {
        ve_dcc_spec* raw = nullptr;
        if (auto created = detail::check(ve_dcc_spec_create(static_cast<std::int32_t>(model), &raw)); !created)
        {
            return detail::failure(created);
        }
        detail::DccSpecHandle handle{raw};
        auto applied =
            detail::check(ve_dcc_spec_set_order(raw, p, q))
                .and_then([&] { return detail::check(ve_dcc_spec_set_dist(raw, static_cast<std::int32_t>(dist))); })
                .and_then([&] { return set_groups(raw); })
                .and_then([&] { return detail::apply_named(raw, fixed, ve_dcc_spec_set_fixed); })
                .and_then([&] { return detail::apply_named(raw, start, ve_dcc_spec_set_start); })
                .and_then([&] { return detail::check(ve_dcc_spec_set_copula(raw, static_cast<std::int32_t>(copula))); })
                .and_then([&]
                    { return detail::check(ve_dcc_spec_set_static_method(raw, static_cast<std::int32_t>(static_method))); });
        if (!applied)
        {
            return detail::failure(applied);
        }
        return handle;
    }

private:
    Result<void> set_groups(ve_dcc_spec* spec) const
    {
        if (groups.empty())
        {
            return {};
        }
        return detail::check(ve_dcc_spec_set_groups(spec, groups.data(), groups.size()));
    }
};

struct DccOptions
{
    // Two-stage Hessian and robust standard errors. Off, DccFit::std_errors, t_values and p_values are NaN (they do
    // not fail, unlike a univariate Fit); Context::standard_errors computes them later. Univariate fits with
    // regressors need it off: with it on they are unsupported.
    bool standard_errors = true;

    Result<detail::DccOptionsHandle> make_handle() const
    {
        ve_dcc_options* raw = nullptr;
        if (auto created = detail::check(ve_dcc_options_create(&raw)); !created)
        {
            return detail::failure(created);
        }
        detail::DccOptionsHandle handle{raw};
        if (auto applied = detail::check(ve_dcc_options_set_standard_errors(raw, standard_errors ? 1 : 0)); !applied)
        {
            return detail::failure(applied);
        }
        return handle;
    }
};

// A fitted second stage. Move-only. Its views (names, correlations) are valid until it is destroyed.
class DccFit
{
public:
    // Takes ownership of a handle made by ve_dcc_fit_batch or ve_dcc_fit_standard_errors.
    explicit DccFit(ve_dcc_fit* raw) noexcept : handle_(raw)
    {
    }

    Result<FitStatus> status() const
    {
        std::int32_t code = 0;
        if (auto read = detail::check(ve_dcc_fit_status(handle_.get(), &code)); !read)
        {
            return detail::failure(read);
        }
        return static_cast<FitStatus>(code);
    }

    Result<std::size_t> nobs() const
    {
        return detail::count(handle_.get(), ve_dcc_fit_nobs);
    }

    // The number of series m.
    Result<std::size_t> series() const
    {
        return detail::count(handle_.get(), ve_dcc_fit_series);
    }

    Result<std::size_t> param_count() const
    {
        return detail::count(handle_.get(), ve_dcc_fit_param_count);
    }

    Result<std::vector<std::string_view>> param_names() const
    {
        return detail::names(handle_.get(), param_count(), ve_dcc_fit_param_name);
    }

    Result<std::vector<double>> estimates() const
    {
        return per_parameter([&](double* out) { return ve_dcc_fit_estimates(handle_.get(), out); });
    }

    // The two-stage standard errors; NaN when the fit ran with standard_errors = false.
    Result<std::vector<double>> std_errors(SeKind kind) const
    {
        return per_parameter(
            [&](double* out) { return ve_dcc_fit_std_errors(handle_.get(), static_cast<std::int32_t>(kind), out); });
    }

    Result<std::vector<double>> t_values(SeKind kind) const
    {
        return per_parameter(
            [&](double* out) { return ve_dcc_fit_t_values(handle_.get(), static_cast<std::int32_t>(kind), out); });
    }

    Result<std::vector<double>> p_values(SeKind kind) const
    {
        return per_parameter(
            [&](double* out) { return ve_dcc_fit_p_values(handle_.get(), static_cast<std::int32_t>(kind), out); });
    }

    Result<double> loglik() const
    {
        return detail::scalar([&](double* out) { return ve_dcc_fit_loglik(handle_.get(), out); });
    }

    Result<double> criterion(DccCriterion which, IcScale scale = IcScale::per_obs) const
    {
        return detail::scalar(
            [&](double* out)
            {
                return ve_dcc_fit_criterion(
                    handle_.get(), static_cast<std::int32_t>(which), static_cast<std::int32_t>(scale), out);
            });
    }

    // T x m x m, row-major: the conditional correlations R_t.
    Result<std::span<const double>> correlation() const
    {
        return detail::view(handle_.get(), ve_dcc_fit_correlation);
    }

    // T x m x m, row-major: H_t = D_t R_t D_t. Computed on first access.
    Result<std::span<const double>> covariance() const
    {
        return detail::view(handle_.get(), ve_dcc_fit_covariance);
    }

    const ve_dcc_fit* native() const noexcept
    {
        return handle_.get();
    }

private:
    template <class Fill>
    Result<std::vector<double>> per_parameter(Fill fill) const
    {
        const auto size = param_count();
        if (!size)
        {
            return std::unexpected(size.error());
        }
        return detail::values(*size, fill);
    }

    detail::Owned<ve_dcc_fit, ve_dcc_fit_free> handle_;
};

} // namespace ve

#endif
