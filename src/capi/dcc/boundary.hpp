#pragma once

#include "capi/error.hpp"
#include "capi/handles.hpp"

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace ve::capi::dcc
{

template <class Handle, class Output>
ve_status check(const Handle* handle, const Output* out)
{
    if (handle == nullptr)
    {
        return set_error(VE_INVALID_HANDLE, "handle must not be null");
    }
    if (out == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, "output must not be null");
    }
    return VE_OK;
}

template <class Handle>
ve_status create(Handle** out)
{
    if (out == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, "out must not be null");
    }
    *out = std::make_unique<Handle>().release();
    return VE_OK;
}

template <class Value>
bool valid_count(std::size_t count)
{
    return count > 0 && count <= std::vector<Value>().max_size()
        && count <= std::numeric_limits<std::ptrdiff_t>::max() / sizeof(Value);
}

inline ve_status fit_pointers(const ve_fit* const* fits, std::size_t count,
                              std::vector<const ve::FitResult*>& out)
{
    if (fits == nullptr || !valid_count<const ve_fit*>(count))
    {
        return set_error(VE_INVALID_ARGUMENT, "fits must not be null and count must be valid and positive");
    }
    for (std::size_t i = 0; i < count; ++i)
    {
        if (fits[i] == nullptr)
        {
            return set_error(VE_INVALID_HANDLE, "fits must not contain null handles");
        }
    }
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        out.push_back(&fits[i]->result);
    }
    return VE_OK;
}

// The regressors of x as one entry per series of f (none for a NULL x), or why x does not fit f and the horizon.
inline ve_status series_regressors(const ve_dcc_fit* f, std::size_t horizon, const ve_dcc_future* x,
                                   ve::dcc::future::SeriesRegressors& out)
{
    const std::size_t series = f->result.series;
    out.assign(series, {});
    if (x == nullptr)
    {
        return VE_OK;
    }
    if (x->series != series || x->horizon != horizon)
    {
        return set_error(VE_INVALID_ARGUMENT, "x holds " + std::to_string(x->series) + " series of " +
                                                  std::to_string(x->horizon) + " rows, the fit needs " +
                                                  std::to_string(series) + " series of " + std::to_string(horizon));
    }
    for (std::size_t i = 0; i < series; ++i)
    {
        out[i] = {{x->mxreg[i].values, x->mxreg[i].columns}, {x->vxreg[i].values, x->vxreg[i].columns}};
    }
    return VE_OK;
}

// What forecast_problem says about f, the horizon and the regressors of x, as an error.
inline ve_status check_future(const ve_dcc_fit* f, std::size_t horizon, const ve::dcc::future::SeriesRegressors& future)
{
    if (const auto problem = ve::dcc::future::forecast_problem(f->result, horizon, future))
    {
        return set_error(VE_INVALID_ARGUMENT, *problem);
    }
    return VE_OK;
}

}
