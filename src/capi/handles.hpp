#pragma once

#include "capi/results/regressor_rows.hpp"
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
#include <volengine/volengine.h>

struct ve_context
{
    ve::Context context;
};

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
};

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
    mutable std::once_flag covariance_once;
    mutable std::vector<double> covariance;
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
};

struct ve_dcc_sim
{
    ve::dcc::future::DccSimulation simulation;
    bool correlations = false;
};
