#include "capi/error.hpp"
#include "capi/handles.hpp"
#include "diagnostics/gof.hpp"
#include "diagnostics/portmanteau.hpp"
#include "diagnostics/sign_bias.hpp"
#include "fit/nyblom.hpp"
#include "model/layout.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

using ve::capi::guard;
using ve::capi::set_error;

namespace
{

ve_status check_fit(const ve_fit* f, const double* out)
{
    if (f == nullptr)
    {
        return set_error(VE_INVALID_HANDLE, "f must not be null");
    }
    if (out == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, "out must not be null");
    }
    if (f->result.std_residuals.empty())
    {
        return set_error(VE_INVALID_ARGUMENT, "the fit has no finite point");
    }
    return VE_OK;
}

ve_status write_test(ve::diagnostics::TestResult result, double* out)
{
    out[0] = result.statistic;
    out[1] = result.p_value;
    return VE_OK;
}

}

extern "C" ve_status ve_fit_ljung_box(const ve_fit* f, int32_t squared, size_t lag, size_t fitdf, double* out)
{
    return guard([&]
    {
        const ve_status status = check_fit(f, out);
        if (status != VE_OK)
        {
            return status;
        }
        if (squared != 0 && squared != 1)
        {
            return set_error(VE_INVALID_ARGUMENT, "squared must be 0 or 1");
        }
        const auto problem = ve::diagnostics::ljung_box_problem(f->result.nobs, lag, fitdf, squared != 0);
        if (problem)
        {
            return set_error(VE_INVALID_ARGUMENT, *problem);
        }
        return write_test(ve::diagnostics::weighted_ljung_box(f->result.std_residuals, lag, fitdf, squared != 0), out);
    });
}

extern "C" ve_status ve_fit_arch_lm(const ve_fit* f, size_t lag, size_t fitdf, double* out)
{
    return guard([&]
    {
        const ve_status status = check_fit(f, out);
        if (status != VE_OK)
        {
            return status;
        }
        const auto problem = ve::diagnostics::arch_lm_problem(f->result.nobs, lag, fitdf);
        if (problem)
        {
            return set_error(VE_INVALID_ARGUMENT, *problem);
        }
        return write_test(ve::diagnostics::weighted_arch_lm(f->result.residuals, f->result.sigma, lag, fitdf), out);
    });
}

extern "C" ve_status ve_fit_nyblom(const ve_fit* f, double* individual, double* joint, int32_t* estimated)
{
    return guard([&]
    {
        const ve_status status = check_fit(f, individual);
        if (status != VE_OK)
        {
            return status;
        }
        if (joint == nullptr || estimated == nullptr)
        {
            return set_error(VE_INVALID_ARGUMENT, "joint and estimated must not be null");
        }
        const ve::model::Layout layout = ve::model::layout_of(f->result.spec);
        for (std::size_t i = 0; i < layout.count(); ++i)
        {
            estimated[i] = layout.is_estimated(i) ? 1 : 0;
        }
        std::ranges::copy(f->result.nyblom.individual, individual);
        *joint = f->result.nyblom.joint;
        return VE_OK;
    });
}

extern "C" ve_status ve_nyblom_critical(size_t parameters, double* out)
{
    return guard([&]
    {
        if (out == nullptr)
        {
            return set_error(VE_INVALID_ARGUMENT, "out must not be null");
        }
        const auto critical = ve::fit::nyblom_critical(parameters);
        if (!critical)
        {
            return set_error(VE_INVALID_ARGUMENT, "parameters must be in 1..20");
        }
        std::ranges::copy(*critical, out);
        return VE_OK;
    });
}

extern "C" ve_status ve_fit_sign_bias(const ve_fit* f, double* out)
{
    return guard([&]
    {
        const ve_status status = check_fit(f, out);
        if (status != VE_OK)
        {
            return status;
        }
        if (f->result.nobs < 6)
        {
            return set_error(VE_INVALID_ARGUMENT, "sign bias needs at least 6 observations");
        }
        const auto result = ve::diagnostics::sign_bias(f->result.std_residuals, f->result.residuals);
        const std::array tests{result.sign, result.negative, result.positive, result.joint};
        for (std::size_t i = 0; i < tests.size(); ++i)
        {
            write_test(tests[i], out + 2 * i);
        }
        return VE_OK;
    });
}

extern "C" ve_status ve_fit_gof(const ve_fit* f, size_t groups, double* out)
{
    return guard([&]
    {
        const ve_status status = check_fit(f, out);
        if (status != VE_OK)
        {
            return status;
        }
        if (groups < 2 || groups > f->result.nobs)
        {
            return set_error(VE_INVALID_ARGUMENT, "groups must be between 2 and nobs");
        }
        const auto layout = ve::model::layout_of(f->result.spec);
        const auto parameters = ve::model::shape_parameters_of<double>(f->result.estimate, layout);
        return write_test(ve::diagnostics::pearson_gof(f->result.std_residuals, layout.dist, parameters, groups), out);
    });
}
