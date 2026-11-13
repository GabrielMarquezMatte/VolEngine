#include "capi/error.hpp"
#include "capi/handles.hpp"
#include "capi/results/regressor_rows.hpp"
#include "sim/simulate.hpp"
#include "sim/start.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using ve::capi::check_stored;
using ve::capi::guard;
using ve::capi::set_error;
using ve::capi::store_rows;

namespace
{

ve_status check_handles(const ve_context* ctx, const void* source, const char* source_name, const ve_sim_options* o,
                        ve_sim** out)
{
    if (out == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, "out must not be null");
    }
    *out = nullptr;
    if (ctx == nullptr || source == nullptr || o == nullptr)
    {
        return set_error(VE_INVALID_HANDLE, std::string("ctx, ") + source_name + " and o must not be null");
    }
    return VE_OK;
}

// The request of o for a spec, or why its regressors do not match the spec.
ve_status request_of(const ve_sim_options* o, const ve::Spec& spec, ve::sim::Request& request)
{
    ve_status status = check_stored(o->mxreg, spec.mxreg, o->horizon, "mxreg");
    if (status != VE_OK)
    {
        return status;
    }
    status = check_stored(o->vxreg, spec.vxreg, o->horizon, "vxreg");
    if (status != VE_OK)
    {
        return status;
    }
    request = {o->horizon,
               o->paths,
               o->seed,
               {{o->mxreg.values, o->mxreg.columns}, {o->vxreg.values, o->vxreg.columns}}};
    return VE_OK;
}

ve_status publish(ve_context* ctx, const ve::sim::Start& start, const ve::sim::Request& request, ve_sim** out)
{
    auto handle = std::make_unique<ve_sim>();
    handle->simulation = ve::sim::simulate(ctx->context, start, request);
    *out = handle.release();
    return VE_OK;
}

using SimValues = std::vector<double> ve::sim::Simulation::*;

ve_status read_view(const ve_sim* sim, SimValues member, const double** data, size_t* len)
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
            const std::vector<double>& values = sim->simulation.*member;
            *data = values.data();
            *len = values.size();
            return VE_OK;
        });
}

using SimSize = std::size_t ve::sim::Simulation::*;

ve_status read_size(const ve_sim* sim, SimSize member, size_t* out)
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

extern "C" ve_status ve_sim_options_create(size_t horizon, size_t paths, uint64_t seed, ve_sim_options** out)
{
    return guard(
        [&]
        {
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            *out = nullptr;
            if (horizon == 0)
            {
                return set_error(VE_INVALID_ARGUMENT, "horizon must be at least 1");
            }
            if (paths == 0)
            {
                return set_error(VE_INVALID_ARGUMENT, "paths must be at least 1");
            }
            // returns and sigma: 2 paths x horizon doubles.
            constexpr std::size_t kLimit = std::numeric_limits<std::size_t>::max() / (2 * sizeof(double));
            if (paths > kLimit / horizon)
            {
                return set_error(VE_INVALID_ARGUMENT, "paths x horizon is too large");
            }
            *out = new ve_sim_options{horizon, paths, seed};
            return VE_OK;
        });
}

extern "C" ve_status ve_sim_options_set_regressors(ve_sim_options* o, const ve_panel* mxreg, const ve_panel* vxreg)
{
    return guard(
        [&]
        {
            if (o == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "o must not be null");
            }
            ve::capi::StoredRows mean;
            ve::capi::StoredRows variance;
            ve_status status = store_rows(mxreg, o->horizon, "mxreg", mean);
            if (status != VE_OK)
            {
                return status;
            }
            status = store_rows(vxreg, o->horizon, "vxreg", variance);
            if (status != VE_OK)
            {
                return status;
            }
            o->mxreg = std::move(mean);
            o->vxreg = std::move(variance);
            return VE_OK;
        });
}

extern "C" void ve_sim_options_free(ve_sim_options* o)
{
    std::unique_ptr<ve_sim_options> owner(o);
}

extern "C" ve_status ve_simulate_fit(ve_context* ctx, const ve_fit* f, const ve_sim_options* o, ve_sim** out)
{
    return guard(
        [&]
        {
            ve_status status = check_handles(ctx, f, "f", o, out);
            if (status != VE_OK)
            {
                return status;
            }
            if (f->result.state.residuals.empty())
            {
                return set_error(VE_INVALID_ARGUMENT,
                                 "the fit has no finite point to simulate from (see ve_fit_status)");
            }
            ve::sim::Request request;
            status = request_of(o, f->result.spec, request);
            if (status != VE_OK)
            {
                return status;
            }
            return publish(ctx, ve::sim::start_of_fit(f->result), request, out);
        });
}

extern "C" ve_status ve_simulate_spec(ve_context* ctx, const ve_spec* s, const ve_sim_options* o, ve_sim** out)
{
    return guard(
        [&]
        {
            ve_status status = check_handles(ctx, s, "s", o, out);
            if (status != VE_OK)
            {
                return status;
            }
            ve::sim::Request request;
            status = request_of(o, s->spec, request);
            if (status != VE_OK)
            {
                return status;
            }
            auto start = ve::sim::start_of_spec(s->spec, request.future);
            if (!start.has_value())
            {
                return set_error(VE_INVALID_ARGUMENT, start.error());
            }
            return publish(ctx, *start, request, out);
        });
}

extern "C" ve_status ve_sim_paths(const ve_sim* sim, size_t* out)
{
    return read_size(sim, &ve::sim::Simulation::paths, out);
}

extern "C" ve_status ve_sim_horizon(const ve_sim* sim, size_t* out)
{
    return read_size(sim, &ve::sim::Simulation::horizon, out);
}

extern "C" ve_status ve_sim_returns(const ve_sim* sim, const double** data, size_t* len)
{
    return read_view(sim, &ve::sim::Simulation::returns, data, len);
}

extern "C" ve_status ve_sim_sigma(const ve_sim* sim, const double** data, size_t* len)
{
    return read_view(sim, &ve::sim::Simulation::sigma, data, len);
}

extern "C" void ve_sim_free(ve_sim* sim)
{
    std::unique_ptr<ve_sim> owner(sim);
}
