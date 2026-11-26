#pragma once

#include "capi/dcc/table.hpp"
#include "capi/results/regressor_rows.hpp"
#include "core/cancel.hpp"
#include "core/context.hpp"
#include "dcc/fit.hpp"
#include "dcc/future/forecast.hpp"
#include "dcc/future/simulate.hpp"
#include "core/panel.hpp"
#include "core/spec.hpp"
#include "fit/batch.hpp"
#include "fit/fit_result.hpp"
#include "forecast/forecast.hpp"
#include "sim/simulate.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <vector>
#include <volengine/volengine.h>

struct ve_context
{
    ve::Context context;
};

struct ve_cancel
{
    ve::CancelFlag flag;
};

namespace ve::capi
{

// Null for a null handle, which never cancels.
inline const CancelFlag* flag_of(const ve_cancel* c)
{
    return c == nullptr ? nullptr : &c->flag;
}

// Values computed on the first read and shared by every reader after it. A computation that throws leaves
// them unset, so the next read tries again. Not std::call_once: libstdc++ over pthread_once (MinGW) keeps
// the flag busy after a throw, so every later read of the handle would hang.
class LazyValues
{
public:
    template <class Compute> const std::vector<double>& get(Compute compute) const
    {
        std::scoped_lock lock(mutex_);
        if (!ready_)
        {
            values_ = compute();
            ready_ = true;
        }
        return values_;
    }

private:
    mutable std::mutex mutex_;
    mutable bool ready_ = false;
    mutable std::vector<double> values_;
};

}

struct ve_spec
{
    ve::Spec spec;
};

struct ve_fit_options
{
    ve::FitConfig config;
};

struct ve_panel
{
    ve::PanelView view;
};

struct ve_batch
{
    ve::Batch batch;
};

struct ve_fit
{
    ve::FitResult result;
};

struct ve_forecast
{
    ve::forecast::Forecast forecast;
};

struct ve_sim_options
{
    std::size_t horizon = 0;
    std::size_t paths = 0;
    std::uint64_t seed = 0;
    ve::capi::StoredRows mxreg{};
    ve::capi::StoredRows vxreg{};
    bool correlations = false;
    // Empty, or strictly increasing from 1 and ending at horizon (see checkpoints_problem).
    std::vector<std::size_t> checkpoints{};
};

namespace ve::capi
{

// Why ks cannot be the checkpoints of a horizon, or null when they can (an empty list always can).
inline const char* checkpoints_problem(std::span<const std::size_t> ks, std::size_t horizon)
{
    if (ks.empty())
    {
        return nullptr;
    }
    std::size_t previous = 0;
    for (std::size_t k : ks)
    {
        if (k <= previous)
        {
            return "checkpoints must be strictly increasing, from 1, and end at the horizon";
        }
        previous = k;
    }
    if (previous != horizon)
    {
        return "checkpoints must be strictly increasing, from 1, and end at the horizon";
    }
    return nullptr;
}

}

struct ve_sim
{
    ve::sim::Simulation simulation;
};

struct ve_dcc_spec
{
    ve::dcc::DccSpec spec;
};

struct ve_dcc_options
{
    ve::dcc::DccOptions options;
};

struct ve_dcc_batch
{
    std::vector<ve::dcc::DccSpec> specs;
    std::vector<ve::FitResult> fits;
    ve::dcc::DccOptions options;
};

struct ve_dcc_fit
{
    ve::dcc::DccFitResult result;
    // Every parameter, fixed ones included; result holds the estimated ones (src/capi/dcc/table.hpp).
    ve::capi::dcc::DccTable table;
    ve::capi::LazyValues covariance;
};

// The regressors of the horizon of each series of a DCC fit.
struct ve_dcc_future
{
    std::size_t series = 0;
    std::size_t horizon = 0;
    std::vector<ve::capi::StoredRows> mxreg;
    std::vector<ve::capi::StoredRows> vxreg;
};

struct ve_dcc_forecast
{
    ve::dcc::future::DccForecast forecast;
    ve::capi::LazyValues cumulative_covariance;
};

struct ve_dcc_sim
{
    ve::dcc::future::DccSimulation simulation;
    bool correlations = false;
};
