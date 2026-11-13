#pragma once

#include "capi/results/regressor_rows.hpp"
#include "core/context.hpp"
#include "core/panel.hpp"
#include "core/spec.hpp"
#include "fit/batch.hpp"
#include "fit/fit_result.hpp"
#include "forecast/forecast.hpp"
#include "sim/simulate.hpp"

#include <cstddef>
#include <cstdint>
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
};

struct ve_sim
{
    ve::sim::Simulation simulation;
};
