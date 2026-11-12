#include "capi/error.hpp"
#include "capi/handles.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

using ve::capi::guard;
using ve::capi::set_error;

namespace
{

ve_status check_fit_batch_arguments(const ve_context* ctx, const ve_batch* batch, ve_fit** fits_out)
{
    if (ctx == nullptr)
    {
        return set_error(VE_INVALID_HANDLE, "ctx must not be null");
    }
    if (batch == nullptr)
    {
        return set_error(VE_INVALID_HANDLE, "batch must not be null");
    }
    if (fits_out == nullptr)
    {
        return set_error(VE_INVALID_ARGUMENT, "fits_out must not be null");
    }
    return VE_OK;
}

// All handles are created before any is published, so an exception leaves fits_out untouched.
void publish_results(std::vector<ve::FitResult>&& results, ve_fit** fits_out)
{
    std::vector<std::unique_ptr<ve_fit>> handles;
    handles.reserve(results.size());
    for (ve::FitResult& result : results)
    {
        auto handle = std::make_unique<ve_fit>();
        handle->result = std::move(result);
        handles.push_back(std::move(handle));
    }
    for (std::size_t job = 0; job < handles.size(); ++job)
    {
        fits_out[job] = handles[job].release();
    }
}

}

extern "C" ve_status ve_batch_create(ve_batch** out)
{
    if (out != nullptr)
    {
        *out = nullptr;
    }
    return guard(
        [&]
        {
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            *out = std::make_unique<ve_batch>().release();
            return VE_OK;
        });
}

extern "C" ve_status ve_batch_set_specs(ve_batch* b, const ve_spec* const* specs, size_t n_specs)
{
    return guard(
        [&]
        {
            if (b == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "b must not be null");
            }
            if (n_specs == 0 || specs == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "specs must not be null and n_specs must be >= 1");
            }
            std::vector<ve::Spec> copies;
            copies.reserve(n_specs);
            for (size_t i = 0; i < n_specs; ++i)
            {
                if (specs[i] == nullptr)
                {
                    return set_error(VE_INVALID_HANDLE, "specs must not contain null entries");
                }
                copies.push_back(specs[i]->spec);
            }
            b->batch.specs = std::move(copies);
            return VE_OK;
        });
}

extern "C" ve_status ve_batch_set_panel(ve_batch* b, const ve_panel* panel)
{
    return guard(
        [&]
        {
            if (b == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "b must not be null");
            }
            if (panel == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "panel must not be null");
            }
            b->batch.panel = &panel->view;
            return VE_OK;
        });
}

extern "C" ve_status ve_batch_set_jobs(
    ve_batch* b, const size_t* job_spec, const size_t* job_series, size_t n_jobs)
{
    return guard(
        [&]
        {
            if (b == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "b must not be null");
            }
            if (n_jobs == 0 || job_spec == nullptr || job_series == nullptr)
            {
                return set_error(
                    VE_INVALID_ARGUMENT, "job_spec and job_series must not be null and n_jobs must be >= 1");
            }
            std::vector<ve::FitJob> jobs(n_jobs);
            for (size_t i = 0; i < n_jobs; ++i)
            {
                jobs[i] = ve::FitJob{job_spec[i], job_series[i]};
            }
            b->batch.jobs = std::move(jobs);
            return VE_OK;
        });
}

extern "C" ve_status ve_batch_set_options(ve_batch* b, const ve_fit_options* options)
{
    return guard(
        [&]
        {
            if (b == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "b must not be null");
            }
            if (options == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "options must not be null");
            }
            b->batch.config = options->config;
            return VE_OK;
        });
}

namespace
{

std::vector<const ve::PanelView*> panel_views(const ve_panel* const* panels, size_t n_jobs)
{
    std::vector<const ve::PanelView*> views;
    if (panels == nullptr)
    {
        return views;
    }
    views.reserve(n_jobs);
    for (size_t job = 0; job < n_jobs; ++job)
    {
        views.push_back(panels[job] == nullptr ? nullptr : &panels[job]->view);
    }
    return views;
}

}

extern "C" ve_status ve_batch_set_regressors(
    ve_batch* b, const ve_panel* const* mxreg, const ve_panel* const* vxreg, size_t n_jobs)
{
    return guard(
        [&]
        {
            if (b == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "b must not be null");
            }
            if (n_jobs == 0)
            {
                return set_error(VE_INVALID_ARGUMENT, "n_jobs must be >= 1");
            }
            auto mean = panel_views(mxreg, n_jobs);
            auto variance = panel_views(vxreg, n_jobs);
            b->batch.mean_regressors = std::move(mean);
            b->batch.variance_regressors = std::move(variance);
            return VE_OK;
        });
}

extern "C" ve_status ve_batch_job_count(const ve_batch* b, size_t* out)
{
    return guard(
        [&]
        {
            if (b == nullptr)
            {
                return set_error(VE_INVALID_HANDLE, "b must not be null");
            }
            if (out == nullptr)
            {
                return set_error(VE_INVALID_ARGUMENT, "out must not be null");
            }
            *out = b->batch.jobs.size();
            return VE_OK;
        });
}

extern "C" void ve_batch_free(ve_batch* b)
{
    std::unique_ptr<ve_batch> owner(b);
}

extern "C" ve_status ve_fit_batch(ve_context* ctx, const ve_batch* batch, ve_fit** fits_out)
{
    if (batch != nullptr && fits_out != nullptr)
    {
        std::fill_n(fits_out, batch->batch.jobs.size(), nullptr);
    }
    return guard(
        [&]
        {
            const ve_status status = check_fit_batch_arguments(ctx, batch, fits_out);
            if (status != VE_OK)
            {
                return status;
            }
            if (const auto problem = ve::check_batch(batch->batch))
            {
                return set_error(VE_INVALID_ARGUMENT, *problem);
            }
            publish_results(ve::run_batch(ctx->context, batch->batch), fits_out);
            return VE_OK;
        });
}
