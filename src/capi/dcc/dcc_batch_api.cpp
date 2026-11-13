#include "capi/dcc/boundary.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <utility>

using ve::capi::guard;
using ve::capi::set_error;
using ve::capi::dcc::check;

#include "dcc/standard_errors.hpp"

extern "C" ve_status ve_dcc_batch_create(ve_dcc_batch** out)
{
    return guard(
        [&]
        {
            return ve::capi::dcc::create(out);
        });
}

extern "C" ve_status ve_dcc_batch_set_specs(ve_dcc_batch* b, const ve_dcc_spec* const* specs, size_t n)
{
    return guard(
        [&]
        {
            if (b == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "b must not be null");
            }
            if (specs == nullptr || !ve::capi::dcc::valid_count<ve::dcc::DccSpec>(n))
            {
                return set_error(VE_INVALID_ARGUMENT, "specs must not be null and count must be valid and positive");
            }
            for (size_t i = 0; i < n; ++i)
            {
                if (specs[i] == nullptr)
                {
                    return set_error(VE_INVALID_HANDLE, "specs must not contain null handles");
                }
            }
            std::vector<ve::dcc::DccSpec> copies;
            copies.reserve(n);
            for (size_t i = 0; i < n; ++i)
            {
                copies.push_back(specs[i]->spec);
            }
            b->specs = std::move(copies);
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_batch_set_fits(ve_dcc_batch* b, const ve_fit* const* fits, size_t m)
{
    return guard(
        [&]
        {
            if (b == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "b must not be null");
            }
            if (!ve::capi::dcc::valid_count<ve::FitResult>(m))
            {
                return set_error(VE_INVALID_ARGUMENT, "fits count must be valid and positive");
            }
            std::vector<const ve::FitResult*> pointers;
            const ve_status status = ve::capi::dcc::fit_pointers(fits, m, pointers);
            if (status != VE_OK)
            {
                return status;
            }
            std::vector<ve::FitResult> copies;
            copies.reserve(m);
            for (const auto* fit : pointers)
            {
                copies.push_back(*fit);
            }
            b->fits = std::move(copies);
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_batch_set_options(ve_dcc_batch* b, const ve_dcc_options* o)
{
    return guard(
        [&]
        {
            if (b == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "b must not be null");
            }
            if (o == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "o must not be null");
            }
            b->options = o->options;
            return VE_OK;
        });
}

extern "C" ve_status ve_dcc_batch_spec_count(const ve_dcc_batch* b, size_t* out)
{
    return guard(
        [&]
        {
            const ve_status status = check(b, out);
            if (status != VE_OK)
            {
                return status;
            }
            *out = b->specs.size();
            return VE_OK;
        });
}

namespace
{

void publish(std::vector<ve::dcc::DccFitResult> results, ve_dcc_fit** out)
{
    std::vector<std::unique_ptr<ve_dcc_fit>> handles;
    handles.reserve(results.size());
    for (auto& result : results)
    {
        auto handle = std::make_unique<ve_dcc_fit>();
        handle->result = std::move(result);
        handles.push_back(std::move(handle));
    }
    for (size_t i = 0; i < handles.size(); ++i)
    {
        out[i] = handles[i].release();
    }
}

}

extern "C" ve_status ve_dcc_fit_batch(ve_context* ctx, const ve_dcc_batch* b, ve_dcc_fit** out)
{
    return guard(
        [&]
        {
            if (b != nullptr && out != nullptr)
            {
                std::fill_n(out, b->specs.size(), nullptr);
            }
            if (ctx == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "ctx must not be null");
            }
            const ve_status status = check(b, out);
            if (status != VE_OK)
            {
                return status;
            }
            std::vector<const ve::FitResult*> fits;
            fits.reserve(b->fits.size());
            for (const auto& fit : b->fits)
            {
                fits.push_back(&fit);
            }
            auto validation_options = b->options;
            validation_options.standard_errors = false;
            if (const auto problem = ve::dcc::check_dcc_batch(fits, b->specs, validation_options))
            {
                return set_error(VE_INVALID_ARGUMENT, *problem);
            }
            if (b->options.standard_errors)
            {
                if (const auto problem = ve::dcc::standard_errors_problem(fits))
                {
                    return set_error(VE_UNSUPPORTED, *problem);
                }
            }
            publish(ve::dcc::fit_dcc_batch(ctx->context, fits, b->specs, b->options), out);
            return VE_OK;
        });
}

extern "C" void ve_dcc_batch_free(ve_dcc_batch* b)
{
    std::unique_ptr<ve_dcc_batch> owner(b);
}
