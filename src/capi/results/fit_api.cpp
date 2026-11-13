#include "capi/error.hpp"
#include "capi/handles.hpp"
#include "fit/criteria.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

using ve::capi::guard;
using ve::capi::set_error;

namespace
{

template <typename T>
ve_status check_fit(const ve_fit* f, const T* out, const char* out_name)
{
    if (f == nullptr)
    {
        return set_error(VE_INVALID_HANDLE, "f must not be null");
    }
    if (out == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, std::string(out_name) + " must not be null");
    }
    return VE_OK;
}

ve_status bad_value(const char* name, int32_t value)
{
    return set_error(VE_INVALID_ARGUMENT, std::string(name) + " has unknown value " + std::to_string(value));
}

ve_status std_errors_of(const ve::FitResult& result, int32_t se_kind, const std::vector<double>*& errors)
{
    if (se_kind == VE_SE_HESSIAN)
    {
        errors = &result.se_hessian;
        return VE_OK;
    }
    if (se_kind == VE_SE_ROBUST)
    {
        errors = &result.se_robust;
        return VE_OK;
    }
    return bad_value("se_kind", se_kind);
}

ve_status to_scale(int32_t value, ve::IcScale& scale)
{
    if (value == VE_IC_PER_OBS)
    {
        scale = ve::IcScale::PerObs;
        return VE_OK;
    }
    if (value == VE_IC_TOTAL)
    {
        scale = ve::IcScale::Total;
        return VE_OK;
    }
    return bad_value("scale", value);
}

double std_error_value(double, double std_error)
{
    return std_error;
}

double t_value(double estimate, double std_error)
{
    return estimate / std_error;
}

// 2 (1 - Phi(|t|)), written with erfc to keep the accuracy in the tails.
double p_value(double estimate, double std_error)
{
    return std::erfc(std::fabs(t_value(estimate, std_error)) / std::numbers::sqrt2);
}

using Statistic = double (*)(double estimate, double std_error);

ve_status fill_statistic(const ve_fit* f, int32_t se_kind, double* out, Statistic statistic)
{
    return guard(
        [&]
        {
            const ve_status status = check_fit(f, out, "out");
            if (status != VE_OK)
            {
                return status;
            }
            const std::vector<double>* errors = nullptr;
            const ve_status kind_status = std_errors_of(f->result, se_kind, errors);
            if (kind_status != VE_OK)
            {
                return kind_status;
            }
            for (size_t i = 0; i < errors->size(); ++i)
            {
                out[i] = statistic(f->result.estimate[i], (*errors)[i]);
            }
            return VE_OK;
        });
}

using Criterion = double (*)(const ve::FitResult&, ve::IcScale);

ve_status read_criterion(const ve_fit* f, int32_t scale, double* out, Criterion criterion)
{
    return guard(
        [&]
        {
            const ve_status status = check_fit(f, out, "out");
            if (status != VE_OK)
            {
                return status;
            }
            ve::IcScale parsed = ve::IcScale::PerObs;
            const ve_status scale_status = to_scale(scale, parsed);
            if (scale_status != VE_OK)
            {
                return scale_status;
            }
            *out = criterion(f->result, parsed);
            return VE_OK;
        });
}

using SeriesMember = std::vector<double> ve::FitResult::*;

// An empty series is reported as a NULL view of length 0.
ve_status read_series(const ve_fit* f, SeriesMember member, const double** data, size_t* len)
{
    return guard(
        [&]
        {
            const ve_status status = check_fit(f, data, "data");
            if (status != VE_OK)
            {
                return status;
            }
            if (len == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "len must not be null");
            }
            const std::vector<double>& values = f->result.*member;
            *data = nullptr;
            if (!values.empty())
            {
                *data = values.data();
            }
            *len = values.size();
            return VE_OK;
        });
}

}

extern "C" ve_status ve_fit_status(const ve_fit* f, int32_t* out)
{
    return guard(
        [&]
        {
            const ve_status status = check_fit(f, out, "out");
            if (status != VE_OK)
            {
                return status;
            }
            *out = static_cast<int32_t>(f->result.status);
            return VE_OK;
        });
}

extern "C" ve_status ve_fit_nobs(const ve_fit* f, size_t* out)
{
    return guard(
        [&]
        {
            const ve_status status = check_fit(f, out, "out");
            if (status != VE_OK)
            {
                return status;
            }
            *out = f->result.nobs;
            return VE_OK;
        });
}

extern "C" ve_status ve_fit_param_count(const ve_fit* f, size_t* out)
{
    return guard(
        [&]
        {
            const ve_status status = check_fit(f, out, "out");
            if (status != VE_OK)
            {
                return status;
            }
            *out = f->result.names.size();
            return VE_OK;
        });
}

extern "C" ve_status ve_fit_param_name(const ve_fit* f, size_t i, const char** out)
{
    return guard(
        [&]
        {
            const ve_status status = check_fit(f, out, "out");
            if (status != VE_OK)
            {
                return status;
            }
            if (i >= f->result.names.size())
            {
                return set_error(VE_INVALID_ARGUMENT, "i must be < param_count");
            }
            *out = f->result.names[i].data();
            return VE_OK;
        });
}

extern "C" ve_status ve_fit_estimates(const ve_fit* f, double* out)
{
    return guard(
        [&]
        {
            const ve_status status = check_fit(f, out, "out");
            if (status != VE_OK)
            {
                return status;
            }
            std::ranges::copy(f->result.estimate, out);
            return VE_OK;
        });
}

extern "C" ve_status ve_fit_std_errors(const ve_fit* f, int32_t se_kind, double* out)
{
    return fill_statistic(f, se_kind, out, std_error_value);
}

extern "C" ve_status ve_fit_t_values(const ve_fit* f, int32_t se_kind, double* out)
{
    return fill_statistic(f, se_kind, out, t_value);
}

extern "C" ve_status ve_fit_p_values(const ve_fit* f, int32_t se_kind, double* out)
{
    return fill_statistic(f, se_kind, out, p_value);
}

extern "C" ve_status ve_loglik(const ve_fit* f, double* out)
{
    return guard(
        [&]
        {
            const ve_status status = check_fit(f, out, "out");
            if (status != VE_OK)
            {
                return status;
            }
            *out = f->result.loglik;
            return VE_OK;
        });
}

extern "C" ve_status ve_aic(const ve_fit* f, int32_t scale, double* out)
{
    return read_criterion(f, scale, out, ve::aic);
}

extern "C" ve_status ve_bic(const ve_fit* f, int32_t scale, double* out)
{
    return read_criterion(f, scale, out, ve::bic);
}

extern "C" ve_status ve_hqic(const ve_fit* f, int32_t scale, double* out)
{
    return read_criterion(f, scale, out, ve::hqic);
}

extern "C" ve_status ve_sic(const ve_fit* f, int32_t scale, double* out)
{
    return read_criterion(f, scale, out, ve::sic);
}

extern "C" ve_status ve_fit_sigma(const ve_fit* f, const double** data, size_t* len)
{
    return read_series(f, &ve::FitResult::sigma, data, len);
}

extern "C" ve_status ve_fit_residuals(const ve_fit* f, const double** data, size_t* len)
{
    return read_series(f, &ve::FitResult::residuals, data, len);
}

extern "C" ve_status ve_fit_std_residuals(const ve_fit* f, const double** data, size_t* len)
{
    return read_series(f, &ve::FitResult::std_residuals, data, len);
}

extern "C" ve_status ve_fit_fitted(const ve_fit* f, const double** data, size_t* len)
{
    return read_series(f, &ve::FitResult::fitted, data, len);
}

extern "C" void ve_fit_free(ve_fit* f)
{
    std::unique_ptr<ve_fit> owner(f);
}
