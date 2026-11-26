#include "capi/error.hpp"
#include "capi/handles.hpp"
#include "cov/estimators.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

using ve::capi::guard;
using ve::capi::set_error;

namespace
{

bool all_finite(const ve::PanelView& view)
{
    for (std::size_t row = 0; row < view.t; ++row)
    {
        for (std::size_t series = 0; series < view.n; ++series)
        {
            if (!std::isfinite(view.at(row, series)))
            {
                return false;
            }
        }
    }
    return true;
}

// VE_OK when the estimators may run on the panel; otherwise the error, with its message set.
ve_status check_panel(const ve_panel* panel, const double* out)
{
    if (panel == nullptr)
    {
        return set_error(VE_INVALID_HANDLE, "panel must not be null");
    }
    if (out == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, "out must not be null");
    }
    const ve::PanelView& view = panel->view;
    if (view.has_ranges())
    {
        return set_error(VE_INVALID_ARGUMENT, "the covariance estimators need a rectangular panel (no ranges)");
    }
    if (view.t < 2 || view.n == 0)
    {
        return set_error(VE_INVALID_ARGUMENT, "the covariance estimators need at least 2 rows and 1 series");
    }
    if (!all_finite(view))
    {
        return set_error(VE_INVALID_ARGUMENT, "the panel has a value that is not finite");
    }
    return VE_OK;
}

}

extern "C" ve_status ve_cov_sample(const ve_panel* panel, double* out)
{
    return guard(
        [&]
        {
            if (const ve_status status = check_panel(panel, out); status != VE_OK)
            {
                return status;
            }
            const auto covariance = ve::cov::sample(panel->view);
            std::ranges::copy(covariance, out);
            return VE_OK;
        });
}

extern "C" ve_status ve_cov_ledoit_wolf(const ve_panel* panel, double* out, double* shrinkage)
{
    return guard(
        [&]
        {
            if (const ve_status status = check_panel(panel, out); status != VE_OK)
            {
                return status;
            }
            if (shrinkage == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "shrinkage must not be null");
            }
            const auto result = ve::cov::ledoit_wolf(panel->view);
            std::ranges::copy(result.covariance, out);
            *shrinkage = result.shrinkage;
            return VE_OK;
        });
}
