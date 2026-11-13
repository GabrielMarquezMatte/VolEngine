#include "capi/dcc/boundary.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using ve::capi::guard;
using ve::capi::set_error;

namespace
{

using SimMember = std::vector<double> ve::dcc::future::DccSimulation::*;
using SimSize = std::size_t ve::dcc::future::DccSimulation::*;

// The doubles of the result: returns and sigma (2 per series and step) and, with correlations, R and H (2 m^2).
constexpr std::size_t kMaxSeries = std::size_t{1} << 20;

bool fits_in_memory(const ve_sim_options* o, std::size_t series)
{
    constexpr std::size_t kLimit = std::numeric_limits<std::size_t>::max() / sizeof(double);
    if (series > kMaxSeries)
    {
        return false;
    }
    std::size_t per_step = 2 * series;
    if (o->correlations)
    {
        per_step += 2 * series * series;
    }
    return o->paths <= kLimit / o->horizon / per_step;
}

ve_status check_request(ve_context* ctx, const ve_dcc_fit* f, const ve_sim_options* o, ve_dcc_sim** out)
{
    if (out == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, "out must not be null");
    }
    *out = nullptr;
    if (ctx == nullptr || f == nullptr || o == nullptr)
    {
        return set_error(VE_INVALID_HANDLE, "ctx, f and o must not be null");
    }
    if (o->mxreg.given || o->vxreg.given)
    {
        return set_error(VE_INVALID_ARGUMENT,
                         "o must have no regressors of its own: a DCC takes them per series in x");
    }
    if (!fits_in_memory(o, f->result.series))
    {
        return set_error(VE_INVALID_ARGUMENT, "the simulation is too large: paths x horizon x series does not fit in memory");
    }
    return VE_OK;
}

ve_status read_view(const ve_dcc_sim* sim, SimMember member, bool needs_correlations, const double** data, size_t* len)
{
    return guard(
        [&]
        {
            if (sim == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "sim must not be null");
            }
            if (data == nullptr || len == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "data and len must not be null");
            }
            if (needs_correlations && !sim->correlations)
            {
                return set_error(VE_INVALID_ARGUMENT,
                                 "the simulation kept no correlations: turn them on in its options");
            }
            const std::vector<double>& values = sim->simulation.*member;
            *data = values.data();
            *len = values.size();
            return VE_OK;
        });
}

ve_status read_size(const ve_dcc_sim* sim, SimSize member, size_t* out)
{
    return guard(
        [&]
        {
            if (sim == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "sim must not be null");
            }
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            *out = sim->simulation.*member;
            return VE_OK;
        });
}

}

extern "C" ve_status ve_simulate_dcc(ve_context* ctx, const ve_dcc_fit* f, const ve_sim_options* o,
                                     const ve_dcc_future* x, ve_dcc_sim** out)
{
    return guard(
        [&]
        {
            ve_status status = check_request(ctx, f, o, out);
            if (status != VE_OK)
            {
                return status;
            }
            ve::dcc::future::SeriesRegressors future;
            status = ve::capi::dcc::series_regressors(f, o->horizon, x, future);
            if (status != VE_OK)
            {
                return status;
            }
            status = ve::capi::dcc::check_future(f, o->horizon, future);
            if (status != VE_OK)
            {
                return status;
            }
            ve::dcc::future::DccRequest request{o->horizon, o->paths, o->seed, o->correlations, std::move(future)};
            auto handle = std::make_unique<ve_dcc_sim>();
            handle->simulation = ve::dcc::future::simulate_dcc(ctx->context, f->result, request);
            handle->correlations = o->correlations;
            *out = handle.release();
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_sim_paths(const ve_dcc_sim* sim, size_t* out)
{
    return read_size(sim, &ve::dcc::future::DccSimulation::paths, out);
}

extern "C" ve_status ve_dcc_sim_horizon(const ve_dcc_sim* sim, size_t* out)
{
    return read_size(sim, &ve::dcc::future::DccSimulation::horizon, out);
}

extern "C" ve_status ve_dcc_sim_series(const ve_dcc_sim* sim, size_t* out)
{
    return read_size(sim, &ve::dcc::future::DccSimulation::series, out);
}

extern "C" ve_status ve_dcc_sim_returns(const ve_dcc_sim* sim, const double** data, size_t* len)
{
    return read_view(sim, &ve::dcc::future::DccSimulation::returns, false, data, len);
}

extern "C" ve_status ve_dcc_sim_sigma(const ve_dcc_sim* sim, const double** data, size_t* len)
{
    return read_view(sim, &ve::dcc::future::DccSimulation::sigma, false, data, len);
}

extern "C" ve_status ve_dcc_sim_correlation(const ve_dcc_sim* sim, const double** data, size_t* len)
{
    return read_view(sim, &ve::dcc::future::DccSimulation::r, true, data, len);
}

extern "C" ve_status ve_dcc_sim_covariance(const ve_dcc_sim* sim, const double** data, size_t* len)
{
    return read_view(sim, &ve::dcc::future::DccSimulation::h, true, data, len);
}

extern "C" void ve_dcc_sim_free(ve_dcc_sim* sim)
{
    std::unique_ptr<ve_dcc_sim> owner(sim);
}
