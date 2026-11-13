#include "capi/dcc/boundary.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

using ve::capi::guard;
using ve::capi::set_error;
using ve::capi::dcc::check;

#include <numbers>

namespace
{

using Statistic = double (*)(double, double);

double standard_error(double, double se)
{
    return se;
}

double t_value(double estimate, double se)
{
    return estimate / se;
}

double p_value(double estimate, double se)
{
    return std::erfc(std::fabs(estimate / se) / std::numbers::sqrt2);
}

ve_status fill_statistic(const ve_dcc_fit* f, int32_t kind, double* out, Statistic statistic)
{
    return guard(
        [&]
        {
            const ve_status status = check(f, out);
            if (status != VE_OK)
            {
                return status;
            }
            if (kind != VE_SE_HESSIAN && kind != VE_SE_ROBUST)
            {
                return set_error(VE_INVALID_ARGUMENT, "unknown standard error kind");
            }
            const auto& errors = kind == VE_SE_HESSIAN ? f->result.se_hessian : f->result.se_robust;
            for (size_t i = 0; i < errors.size(); ++i)
            {
                out[i] = statistic(f->result.estimate[i], errors[i]);
            }
            return VE_OK;
        });
}

ve_status read_path(const ve_dcc_fit* f, const double** data, size_t* len, bool covariance)
{
    return guard(
        [&]
        {
            const ve_status status = check(f, data);
            if (status != VE_OK)
            {
                return status;
            }
            if (len == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "len must not be null");
            }
            *data = nullptr;
            *len = 0;
            if (!f->result.r)
            {
                return VE_OK;
            }
            if (covariance)
            {
                std::call_once(f->covariance_once, [&] { f->covariance = ve::dcc::covariance_path(f->result); });
            }
            const auto& path = covariance ? f->covariance : *f->result.r;
            if (!path.empty())
            {
                *data = path.data();
            }
            *len = path.size();
            return VE_OK;
        });
}

}

extern "C" ve_status ve_dcc_fit_status(const ve_dcc_fit* f, int32_t* out)
{
    return guard(
        [&]
        {
            const ve_status status = check(f, out);
            if (status != VE_OK)
            {
                return status;
            }
            *out = static_cast<int32_t>(f->result.status);
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_fit_nobs(const ve_dcc_fit* f, size_t* out)
{
    return guard(
        [&]
        {
            const ve_status status = check(f, out);
            if (status != VE_OK)
            {
                return status;
            }
            *out = f->result.nobs;
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_fit_series(const ve_dcc_fit* f, size_t* out)
{
    return guard(
        [&]
        {
            const ve_status status = check(f, out);
            if (status != VE_OK)
            {
                return status;
            }
            *out = f->result.series;
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_fit_param_count(const ve_dcc_fit* f, size_t* out)
{
    return guard(
        [&]
        {
            const ve_status status = check(f, out);
            if (status != VE_OK)
            {
                return status;
            }
            *out = f->result.names.size();
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_fit_loglik(const ve_dcc_fit* f, double* out)
{
    return guard(
        [&]
        {
            const ve_status status = check(f, out);
            if (status != VE_OK)
            {
                return status;
            }
            *out = f->result.loglik;
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_fit_param_name(const ve_dcc_fit* f, size_t i, const char** out)
{
    return guard(
        [&]
        {
            const ve_status status = check(f, out);
            if (status != VE_OK)
            {
                return status;
            }
            if (i >= f->result.names.size())
            {
                return set_error(VE_INVALID_ARGUMENT, "i must be < param_count");
            }
            *out = f->result.names[i].c_str();
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_fit_estimates(const ve_dcc_fit* f, double* out)
{
    return guard(
        [&]
        {
            const ve_status status = check(f, out);
            if (status != VE_OK)
            {
                return status;
            }
            std::ranges::copy(f->result.estimate, out);
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_fit_std_errors(const ve_dcc_fit* f, int32_t se_kind, double* out)
{
    return fill_statistic(f, se_kind, out, standard_error);
}

extern "C" ve_status ve_dcc_fit_t_values(const ve_dcc_fit* f, int32_t se_kind, double* out)
{
    return fill_statistic(f, se_kind, out, t_value);
}

extern "C" ve_status ve_dcc_fit_p_values(const ve_dcc_fit* f, int32_t se_kind, double* out)
{
    return fill_statistic(f, se_kind, out, p_value);
}

extern "C" ve_status ve_dcc_fit_criterion(const ve_dcc_fit* f, int32_t which, int32_t scale, double* out)
{
    return guard(
        [&]
        {
            const ve_status status = check(f, out);
            if (status != VE_OK)
            {
                return status;
            }
            if (which < VE_DCC_CRITERION_AIC || which > VE_DCC_CRITERION_SIC)
            {
                return set_error(VE_INVALID_ARGUMENT, "unknown criterion");
            }
            if (scale != VE_IC_PER_OBS && scale != VE_IC_TOTAL)
            {
                return set_error(VE_INVALID_ARGUMENT, "unknown criterion scale");
            }
            *out = ve::dcc::dcc_criterion(f->result, static_cast<ve::dcc::Criterion>(which), scale == VE_IC_PER_OBS);
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_fit_correlation(const ve_dcc_fit* f, const double** data, size_t* len)
{
    return read_path(f, data, len, false);
}

extern "C" ve_status ve_dcc_fit_covariance(const ve_dcc_fit* f, const double** data, size_t* len)
{
    return read_path(f, data, len, true);
}

extern "C" void ve_dcc_fit_free(ve_dcc_fit* f)
{
    std::unique_ptr<ve_dcc_fit> owner(f);
}
