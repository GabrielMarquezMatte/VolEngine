#include "capi/dcc/boundary.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

using ve::capi::guard;
using ve::capi::set_error;
using ve::capi::store_rows;

namespace
{

using ForecastMember = std::vector<double> ve::dcc::future::DccForecast::*;
using ForecastSize = std::size_t ve::dcc::future::DccForecast::*;

ve_status read_view(const ve_dcc_forecast* fc, ForecastMember member, const double** data, size_t* len)
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

ve_status read_size(const ve_dcc_forecast* fc, ForecastSize member, size_t* out)
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
            *out = fc->forecast.*member;
            return VE_OK;
        });
}

}

extern "C" ve_status ve_dcc_future_create(size_t series, size_t horizon, ve_dcc_future** out)
{
    return guard(
        [&]
        {
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            *out = nullptr;
            if (series == 0 || !ve::capi::dcc::valid_count<ve::capi::StoredRows>(series))
            {
                return set_error(VE_INVALID_ARGUMENT, "series must be at least 1 and fit in memory");
            }
            if (horizon == 0)
            {
                return set_error(VE_INVALID_ARGUMENT, "horizon must be at least 1");
            }
            auto handle = std::make_unique<ve_dcc_future>();
            handle->series = series;
            handle->horizon = horizon;
            handle->mxreg.resize(series);
            handle->vxreg.resize(series);
            *out = handle.release();
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_future_set(ve_dcc_future* x, size_t series, const ve_panel* mxreg, const ve_panel* vxreg)
{
    return guard(
        [&]
        {
            if (x == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "x must not be null");
            }
            if (series >= x->series)
            {
                return set_error(VE_INVALID_ARGUMENT, "series is out of range: x holds " + std::to_string(x->series));
            }
            ve::capi::StoredRows mean;
            ve::capi::StoredRows variance;
            ve_status status = store_rows(mxreg, x->horizon, "mxreg", mean);
            if (status != VE_OK)
            {
                return status;
            }
            status = store_rows(vxreg, x->horizon, "vxreg", variance);
            if (status != VE_OK)
            {
                return status;
            }
            x->mxreg[series] = std::move(mean);
            x->vxreg[series] = std::move(variance);
            return VE_OK;
        });
}

extern "C" void ve_dcc_future_free(ve_dcc_future* x)
{
    std::unique_ptr<ve_dcc_future> owner(x);
}

extern "C" ve_status ve_dcc_forecast_create(const ve_dcc_fit* f, size_t horizon, const ve_dcc_future* x,
                                            ve_dcc_forecast** out)
{
    return guard(
        [&]
        {
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            *out = nullptr;
            if (f == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "f must not be null");
            }
            ve::dcc::future::SeriesRegressors future;
            ve_status status = ve::capi::dcc::series_regressors(f, horizon, x, future);
            if (status != VE_OK)
            {
                return status;
            }
            status = ve::capi::dcc::check_future(f, horizon, future);
            if (status != VE_OK)
            {
                return status;
            }
            auto handle = std::make_unique<ve_dcc_forecast>();
            handle->forecast = ve::dcc::future::forecast_dcc(f->result, horizon, future);
            *out = handle.release();
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_forecast_horizon(const ve_dcc_forecast* fc, size_t* out)
{
    return read_size(fc, &ve::dcc::future::DccForecast::horizon, out);
}

extern "C" ve_status ve_dcc_forecast_series(const ve_dcc_forecast* fc, size_t* out)
{
    return read_size(fc, &ve::dcc::future::DccForecast::series, out);
}

extern "C" ve_status ve_dcc_forecast_correlation(const ve_dcc_forecast* fc, const double** data, size_t* len)
{
    return read_view(fc, &ve::dcc::future::DccForecast::r, data, len);
}

extern "C" ve_status ve_dcc_forecast_covariance(const ve_dcc_forecast* fc, const double** data, size_t* len)
{
    return read_view(fc, &ve::dcc::future::DccForecast::h, data, len);
}

extern "C" ve_status ve_dcc_forecast_sigma(const ve_dcc_forecast* fc, const double** data, size_t* len)
{
    return read_view(fc, &ve::dcc::future::DccForecast::sigma, data, len);
}

extern "C" ve_status ve_dcc_forecast_mean(const ve_dcc_forecast* fc, const double** data, size_t* len)
{
    return read_view(fc, &ve::dcc::future::DccForecast::mean, data, len);
}

extern "C" void ve_dcc_forecast_free(ve_dcc_forecast* fc)
{
    std::unique_ptr<ve_dcc_forecast> owner(fc);
}
