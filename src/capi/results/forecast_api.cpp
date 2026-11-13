#include "capi/error.hpp"
#include "capi/handles.hpp"
#include "forecast/forecast.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

using ve::capi::guard;
using ve::capi::set_error;

namespace
{

// The panel of the horizon copied row-major into rows, or why it does not fit the `columns` the spec has.
ve_status future_rows(const ve_panel* panel, int columns, std::size_t horizon, const std::string& name,
                      std::vector<double>& rows)
{
    if (columns == 0)
    {
        if (panel != nullptr)
        {
            return set_error(VE_INVALID_ARGUMENT, name + " was given but the spec has no " + name);
        }
        return VE_OK;
    }
    if (panel == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT,
                         name + " must not be null: the spec has " + std::to_string(columns) + " " + name);
    }
    const ve::PanelView& view = panel->view;
    std::size_t count = static_cast<std::size_t>(columns);
    if (view.has_ranges() || view.t != horizon || view.n != count)
    {
        return set_error(VE_INVALID_ARGUMENT, name + " must be a panel of " + std::to_string(horizon) + " rows x " +
                                                  std::to_string(count) + " columns without ranges");
    }
    rows.resize(horizon * count);
    for (std::size_t t = 0; t < horizon; ++t)
    {
        for (std::size_t k = 0; k < count; ++k)
        {
            rows[t * count + k] = view.at(t, k);
        }
    }
    return VE_OK;
}

ve_status check_request(const ve_fit* f, std::size_t horizon)
{
    if (f == nullptr)
    {
        return set_error(VE_INVALID_HANDLE, "f must not be null");
    }
    if (horizon == 0)
    {
        return set_error(VE_INVALID_ARGUMENT, "horizon must be at least 1");
    }
    if (f->result.state.residuals.empty())
    {
        return set_error(VE_INVALID_ARGUMENT, "the fit has no finite point to forecast from (see ve_fit_status)");
    }
    return VE_OK;
}

using ForecastMember = std::vector<double> ve::forecast::Forecast::*;

ve_status read_view(const ve_forecast* fc, ForecastMember member, const double** data, size_t* len)
{
    return guard(
        [&]
        {
            if (fc == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "fc must not be null");
            }
            if (data == nullptr || len == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "data and len must not be null");
            }
            const std::vector<double>& values = fc->forecast.*member;
            *data = values.data();
            *len = values.size();
            return VE_OK;
        });
}

}

extern "C" ve_status ve_forecast_create(const ve_fit* f, size_t horizon, const ve_panel* mxreg,
                                        const ve_panel* vxreg, ve_forecast** out)
{
    return guard(
        [&]
        {
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            *out = nullptr;
            ve_status status = check_request(f, horizon);
            if (status != VE_OK)
            {
                return status;
            }
            const ve::Spec& spec = f->result.spec;
            std::vector<double> mean;
            std::vector<double> variance;
            status = future_rows(mxreg, spec.mxreg, horizon, "mxreg", mean);
            if (status != VE_OK)
            {
                return status;
            }
            status = future_rows(vxreg, spec.vxreg, horizon, "vxreg", variance);
            if (status != VE_OK)
            {
                return status;
            }
            ve::forecast::FutureRegressors future{{mean, static_cast<std::size_t>(spec.mxreg)},
                                                  {variance, static_cast<std::size_t>(spec.vxreg)}};
            auto handle = std::make_unique<ve_forecast>();
            handle->forecast = ve::forecast::forecast(f->result, horizon, future);
            *out = handle.release();
            return VE_OK;
        });
}

extern "C" ve_status ve_forecast_horizon(const ve_forecast* fc, size_t* out)
{
    return guard(
        [&]
        {
            if (fc == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "fc must not be null");
            }
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            *out = fc->forecast.sigma.size();
            return VE_OK;
        });
}

extern "C" ve_status ve_forecast_sigma(const ve_forecast* fc, const double** data, size_t* len)
{
    return read_view(fc, &ve::forecast::Forecast::sigma, data, len);
}

extern "C" ve_status ve_forecast_mean(const ve_forecast* fc, const double** data, size_t* len)
{
    return read_view(fc, &ve::forecast::Forecast::mean, data, len);
}

extern "C" void ve_forecast_free(ve_forecast* fc)
{
    std::unique_ptr<ve_forecast> owner(fc);
}
