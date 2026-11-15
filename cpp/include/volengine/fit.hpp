#ifndef VOLENGINE_CPP_FIT_HPP
#define VOLENGINE_CPP_FIT_HPP

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
        return count(ve_fit_nobs);
    }

    Result<std::size_t> param_count() const
    {
        return count(ve_fit_param_count);
    }

    Result<std::vector<std::string_view>> param_names() const
    {
        const auto size = param_count();
        if (!size)
        {
            return std::unexpected(size.error());
        }
        std::vector<std::string_view> names;
        for (std::size_t i = 0; i < *size; ++i)
        {
            const char* name = nullptr;
            if (auto read = detail::check(ve_fit_param_name(handle_.get(), i, &name)); !read)
            {
                return detail::failure(read);
            }
            names.emplace_back(name);
        }
        return names;
    }

    Result<std::vector<double>> estimates() const
    {
        return per_parameter([&](double* out) { return ve_fit_estimates(handle_.get(), out); });
    }

    // NaN, with success, when the fit has no finite point.
    Result<double> loglik() const
    {
        double value = 0.0;
        if (auto read = detail::check(ve_loglik(handle_.get(), &value)); !read)
        {
            return detail::failure(read);
        }
        return value;
    }

    const ve_fit* native() const noexcept
    {
        return handle_.get();
    }

private:
    using Handle = detail::Owned<ve_fit, ve_fit_free>;

    Result<std::size_t> count(ve_status (*function)(const ve_fit*, std::size_t*)) const
    {
        std::size_t value = 0;
        if (auto read = detail::check(function(handle_.get(), &value)); !read)
        {
            return detail::failure(read);
        }
        return value;
    }

    // `fill` writes param_count doubles.
    template <class Fill>
    Result<std::vector<double>> per_parameter(Fill fill) const
    {
        const auto size = param_count();
        if (!size)
        {
            return std::unexpected(size.error());
        }
        std::vector<double> values(*size);
        if (auto read = detail::check(fill(values.data())); !read)
        {
            return detail::failure(read);
        }
        return values;
    }

    Handle handle_;
};

} // namespace ve

#endif
