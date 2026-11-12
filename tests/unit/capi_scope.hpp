#pragma once

#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>
#include <volengine/volengine.h>

namespace ve::testing
{

struct ContextFree
{
    void operator()(ve_context* handle) const
    {
        ve_context_free(handle);
    }
};

struct SpecFree
{
    void operator()(ve_spec* handle) const
    {
        ve_spec_free(handle);
    }
};

struct PanelFree
{
    void operator()(ve_panel* handle) const
    {
        ve_panel_free(handle);
    }
};

struct BatchFree
{
    void operator()(ve_batch* handle) const
    {
        ve_batch_free(handle);
    }
};

using ContextPtr = std::unique_ptr<ve_context, ContextFree>;
using SpecPtr = std::unique_ptr<ve_spec, SpecFree>;
using PanelPtr = std::unique_ptr<ve_panel, PanelFree>;
using BatchPtr = std::unique_ptr<ve_batch, BatchFree>;

inline ContextPtr make_context(int32_t threads)
{
    ve_context* raw = nullptr;
    EXPECT_EQ(ve_context_create(&raw), VE_OK);
    ContextPtr context(raw);
    EXPECT_EQ(ve_context_set_threads(raw, threads), VE_OK);
    return context;
}

inline SpecPtr make_spec(int32_t include_mean)
{
    ve_spec* raw = nullptr;
    EXPECT_EQ(ve_spec_create(VE_MODEL_SGARCH, &raw), VE_OK);
    SpecPtr spec(raw);
    EXPECT_EQ(ve_spec_set_include_mean(raw, include_mean), VE_OK);
    return spec;
}

inline PanelPtr make_panel(const double* data, std::size_t t, std::size_t n)
{
    ve_panel* raw = nullptr;
    EXPECT_EQ(ve_panel_create(data, t, n, &raw), VE_OK);
    return PanelPtr(raw);
}

// Parts left empty (or null) are not set on the batch.
struct BatchInputs
{
    std::vector<const ve_spec*> specs;
    const ve_panel* panel = nullptr;
    std::vector<std::size_t> job_spec;
    std::vector<std::size_t> job_series;
    const ve_fit_options* options = nullptr;
    // One entry per job; an empty vector is a null array.
    std::vector<const ve_panel*> mxreg;
    std::vector<const ve_panel*> vxreg;
};

inline BatchPtr build_batch(const BatchInputs& inputs)
{
    ve_batch* raw = nullptr;
    EXPECT_EQ(ve_batch_create(&raw), VE_OK);
    BatchPtr batch(raw);
    if (!inputs.specs.empty())
    {
        EXPECT_EQ(ve_batch_set_specs(raw, inputs.specs.data(), inputs.specs.size()), VE_OK);
    }
    if (inputs.panel != nullptr)
    {
        EXPECT_EQ(ve_batch_set_panel(raw, inputs.panel), VE_OK);
    }
    if (!inputs.job_spec.empty())
    {
        const std::size_t count = inputs.job_spec.size();
        EXPECT_EQ(ve_batch_set_jobs(raw, inputs.job_spec.data(), inputs.job_series.data(), count), VE_OK);
    }
    if (inputs.options != nullptr)
    {
        EXPECT_EQ(ve_batch_set_options(raw, inputs.options), VE_OK);
    }
    if (!inputs.mxreg.empty() || !inputs.vxreg.empty())
    {
        const ve_panel* const* mean = inputs.mxreg.empty() ? nullptr : inputs.mxreg.data();
        const ve_panel* const* variance = inputs.vxreg.empty() ? nullptr : inputs.vxreg.data();
        EXPECT_EQ(ve_batch_set_regressors(raw, mean, variance, inputs.job_spec.size()), VE_OK);
    }
    return batch;
}

// Owns the ve_fit handles written by ve_fit_batch.
class FitSet
{
public:
    explicit FitSet(std::size_t count) : fits_(count, nullptr)
    {
    }

    FitSet(const FitSet&) = delete;
    FitSet& operator=(const FitSet&) = delete;
    FitSet(FitSet&& other) noexcept : fits_(std::move(other.fits_))
    {
        other.fits_.clear();
    }

    FitSet& operator=(FitSet&& other) noexcept
    {
        std::swap(fits_, other.fits_);
        return *this;
    }

    ~FitSet()
    {
        for (ve_fit* fit : fits_)
        {
            ve_fit_free(fit);
        }
    }

    ve_fit** out()
    {
        return fits_.data();
    }

    ve_fit* operator[](std::size_t index) const
    {
        return fits_.at(index);
    }

    std::size_t size() const
    {
        return fits_.size();
    }

private:
    std::vector<ve_fit*> fits_;
};

// Runs the batch and expects success.
inline FitSet fit_all(ve_context* context, const ve_batch* batch)
{
    std::size_t count = 0;
    EXPECT_EQ(ve_batch_job_count(batch, &count), VE_OK);
    FitSet fits(count);
    EXPECT_EQ(ve_fit_batch(context, batch, fits.out()), VE_OK);
    return fits;
}

struct Snapshot
{
    int32_t status = -1;
    std::size_t nobs = 0;
    std::vector<double> estimate;
    double loglik = 0.0;
    std::vector<double> sigma;

    bool operator==(const Snapshot&) const = default;
};

inline Snapshot read_snapshot(const ve_fit* fit)
{
    Snapshot snapshot;
    EXPECT_EQ(ve_fit_status(fit, &snapshot.status), VE_OK);
    EXPECT_EQ(ve_fit_nobs(fit, &snapshot.nobs), VE_OK);
    std::size_t count = 0;
    EXPECT_EQ(ve_fit_param_count(fit, &count), VE_OK);
    snapshot.estimate.resize(count);
    EXPECT_EQ(ve_fit_estimates(fit, snapshot.estimate.data()), VE_OK);
    EXPECT_EQ(ve_loglik(fit, &snapshot.loglik), VE_OK);
    const double* data = nullptr;
    std::size_t length = 0;
    EXPECT_EQ(ve_fit_sigma(fit, &data, &length), VE_OK);
    snapshot.sigma.assign(data, data + length);
    return snapshot;
}

}
