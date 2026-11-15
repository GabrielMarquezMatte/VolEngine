#ifndef VOLENGINE_CPP_FIT_HPP
#define VOLENGINE_CPP_FIT_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <utility>
#include <vector>
#include <volengine/error.hpp>
#include <volengine/input.hpp>

namespace ve
{

enum class FitStatus : std::int32_t
{
    converged = VE_FIT_CONVERGED,
    max_iter = VE_FIT_MAX_ITER,
    failed = VE_FIT_FAILED,
    insufficient_data = VE_FIT_INSUFFICIENT_DATA,
};

enum class SeKind : std::int32_t
{
    hessian = VE_SE_HESSIAN,
    robust = VE_SE_ROBUST,
};

// Information criteria per observation (rugarch's convention) or in total.
enum class IcScale : std::int32_t
{
    per_obs = VE_IC_PER_OBS,
    total = VE_IC_TOTAL,
};

// The information criteria of Fit::criterion and DccFit::criterion.
enum class Criterion : std::int32_t
{
    aic = VE_CRITERION_AIC,
    bic = VE_CRITERION_BIC,
    hqic = VE_CRITERION_HQIC,
    sic = VE_CRITERION_SIC,
};

// [statistic, p-value] of a residual test; NaN where the statistic is undefined.
struct Test
{
    double statistic;
    double p_value;
};

struct SignBias
{
    Test sign;
    Test negative;
    Test positive;
    Test joint;
};

// The Nyblom stability statistics. estimated[i] is 1 for a parameter the fit estimated (the count of 1s is what
// nyblom_critical takes for the joint statistic) and 0 for a fixed or derived one, whose individual statistic is NaN.
struct Nyblom
{
    std::vector<double> individual;
    double joint;
    std::vector<std::int32_t> estimated;
};

// Critical values of the joint Nyblom statistic at 10%, 5% and 1%, for 1 .. 20 estimated parameters.
inline Result<std::array<double, 3>> nyblom_critical(std::size_t parameters)
{
    std::array<double, 3> values{};
    if (auto read = detail::check(ve_nyblom_critical(parameters, values.data())); !read)
    {
        return detail::failure(read);
    }
    return values;
}

namespace detail
{

// A read-only view `(data, length)` of a C handle; the pointer lives as long as the handle.
template <class Handle>
Result<std::span<const double>> view(
    const Handle* handle, ve_status (*function)(const Handle*, const double**, std::size_t*))
{
    const double* data = nullptr;
    std::size_t length = 0;
    if (auto read = check(function(handle, &data, &length)); !read)
    {
        return failure(read);
    }
    return std::span<const double>(data, length);
}

} // namespace detail

// The forecast of a fit h = 1 .. horizon steps after its last observation. It copies what it needs: it stays valid
// after the Fit is destroyed. The views are valid until the Forecast is destroyed.
class Forecast
{
public:
    explicit Forecast(ve_forecast* raw) noexcept : handle_(raw)
    {
    }

    Result<std::size_t> horizon() const
    {
        std::size_t value = 0;
        if (auto read = detail::check(ve_forecast_horizon(handle_.get(), &value)); !read)
        {
            return detail::failure(read);
        }
        return value;
    }

    // The conditional standard deviation, rugarch's sigmaFor.
    Result<std::span<const double>> sigma() const
    {
        return detail::view(handle_.get(), ve_forecast_sigma);
    }

    // The conditional mean, rugarch's seriesFor.
    Result<std::span<const double>> mean() const
    {
        return detail::view(handle_.get(), ve_forecast_mean);
    }

private:
    detail::Owned<ve_forecast, ve_forecast_free> handle_;
};

// One model fitted on one series. Move-only. The views it returns (names, series) are valid until the Fit is
// destroyed. Check status() before using the values: a fit that did not converge has the best finite point the
// optimizer reached, or NaN.
class Fit
{
public:
    // Takes ownership of a handle made by ve_fit_batch.
    explicit Fit(ve_fit* raw) noexcept : handle_(raw)
    {
    }

    Result<FitStatus> status() const
    {
        std::int32_t code = 0;
        if (auto read = detail::check(ve_fit_status(handle_.get(), &code)); !read)
        {
            return detail::failure(read);
        }
        return static_cast<FitStatus>(code);
    }

    Result<std::size_t> nobs() const
    {
        return detail::count(handle_.get(), ve_fit_nobs);
    }

    Result<std::size_t> param_count() const
    {
        return detail::count(handle_.get(), ve_fit_param_count);
    }

    Result<std::vector<std::string_view>> param_names() const
    {
        return detail::names(handle_.get(), param_count(), ve_fit_param_name);
    }

    Result<std::vector<double>> estimates() const
    {
        return per_parameter([&](double* out) { return ve_fit_estimates(handle_.get(), out); });
    }

    // NaN, with success, when the fit has no finite point.
    Result<double> loglik() const
    {
        return detail::scalar([&](double* out) { return ve_fit_loglik(handle_.get(), out); });
    }

    // Standard errors, t values and p values of the estimated parameters (NaN for fixed and derived ones). They fail
    // with invalid_argument when the fit ran with standard_errors = false.
    Result<std::vector<double>> std_errors(SeKind kind) const
    {
        return per_parameter([&](double* out) { return ve_fit_std_errors(handle_.get(), as_int(kind), out); });
    }

    Result<std::vector<double>> t_values(SeKind kind) const
    {
        return per_parameter([&](double* out) { return ve_fit_t_values(handle_.get(), as_int(kind), out); });
    }

    Result<std::vector<double>> p_values(SeKind kind) const
    {
        return per_parameter([&](double* out) { return ve_fit_p_values(handle_.get(), as_int(kind), out); });
    }

    Result<double> criterion(Criterion which, IcScale scale = IcScale::per_obs) const
    {
        return detail::scalar([&](double* out)
                              { return ve_fit_criterion(handle_.get(), as_int(which), as_int(scale), out); });
    }

    Result<double> aic(IcScale scale = IcScale::per_obs) const
    {
        return criterion(Criterion::aic, scale);
    }

    Result<double> bic(IcScale scale = IcScale::per_obs) const
    {
        return criterion(Criterion::bic, scale);
    }

    Result<double> hqic(IcScale scale = IcScale::per_obs) const
    {
        return criterion(Criterion::hqic, scale);
    }

    Result<double> sic(IcScale scale = IcScale::per_obs) const
    {
        return criterion(Criterion::sic, scale);
    }

    // Views of nobs values, valid until the Fit is destroyed; sigma is the conditional standard deviation. A fit
    // without a finite point has no series: the view is empty.
    Result<std::span<const double>> sigma() const
    {
        return detail::view(handle_.get(), ve_fit_sigma);
    }

    Result<std::span<const double>> residuals() const
    {
        return detail::view(handle_.get(), ve_fit_residuals);
    }

    Result<std::span<const double>> std_residuals() const
    {
        return detail::view(handle_.get(), ve_fit_std_residuals);
    }

    Result<std::span<const double>> fitted() const
    {
        return detail::view(handle_.get(), ve_fit_fitted);
    }

    // Ljung-Box needs 1 <= lag < nobs and lag >= 3 * fitdf - 1 (fitdf is ignored for squared residuals).
    Result<Test> ljung_box(std::size_t lag, bool squared = false, std::size_t fitdf = 0) const
    {
        return test([&](double* out) { return ve_fit_ljung_box(handle_.get(), squared ? 1 : 0, lag, fitdf, out); });
    }

    // ARCH-LM needs 1 <= fitdf < lag < nobs.
    Result<Test> arch_lm(std::size_t lag, std::size_t fitdf) const
    {
        return test([&](double* out) { return ve_fit_arch_lm(handle_.get(), lag, fitdf, out); });
    }

    // Needs nobs >= 6.
    Result<SignBias> sign_bias() const
    {
        std::array<double, 8> values{};
        if (auto read = detail::check(ve_fit_sign_bias(handle_.get(), values.data())); !read)
        {
            return detail::failure(read);
        }
        return SignBias{{values[0], values[1]}, {values[2], values[3]}, {values[4], values[5]}, {values[6], values[7]}};
    }

    // Needs 2 <= groups <= nobs.
    Result<Test> gof(std::size_t groups = 20) const
    {
        return test([&](double* out) { return ve_fit_gof(handle_.get(), groups, out); });
    }

    Result<Nyblom> nyblom() const
    {
        const auto size = param_count();
        if (!size)
        {
            return std::unexpected(size.error());
        }
        Nyblom result{std::vector<double>(*size), 0.0, std::vector<std::int32_t>(*size)};
        if (auto read = detail::check(
                ve_fit_nyblom(handle_.get(), result.individual.data(), &result.joint, result.estimated.data()));
            !read)
        {
            return detail::failure(read);
        }
        return result;
    }

    // The regressors of the horizon, panels of exactly `horizon` rows, or null when the spec has none.
    Result<Forecast> forecast(std::size_t horizon, const Panel* mxreg = nullptr, const Panel* vxreg = nullptr) const
    {
        ve_forecast* raw = nullptr;
        if (auto created = detail::check(ve_forecast_create(
                handle_.get(), horizon, detail::native_or_null(mxreg), detail::native_or_null(vxreg), &raw));
            !created)
        {
            return detail::failure(created);
        }
        return Forecast{raw};
    }

    const ve_fit* native() const noexcept
    {
        return handle_.get();
    }

private:
    using Handle = detail::Owned<ve_fit, ve_fit_free>;

    // `fill` writes param_count doubles.
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

    template <class Enum>
    static std::int32_t as_int(Enum value) noexcept
    {
        return static_cast<std::int32_t>(value);
    }

    // `fill` writes [statistic, p-value].
    template <class Fill>
    static Result<Test> test(Fill fill)
    {
        std::array<double, 2> values{};
        if (auto read = detail::check(fill(values.data())); !read)
        {
            return detail::failure(read);
        }
        return Test{values[0], values[1]};
    }

    Handle handle_;
};

} // namespace ve

#endif
