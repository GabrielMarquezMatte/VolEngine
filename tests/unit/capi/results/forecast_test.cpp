#include "capi/handles.hpp"
#include "capi_scope.hpp"
#include "forecast/forecast.hpp"
#include "simulated_garch.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <volengine/volengine.h>

namespace
{

using ve::testing::FitSet;
using ve::testing::PanelPtr;
using ve::testing::SpecPtr;

constexpr std::size_t kT = 800;
constexpr std::size_t kHorizon = 10;

struct ForecastFree
{
    void operator()(ve_forecast* handle) const
    {
        ve_forecast_free(handle);
    }
};

using ForecastPtr = std::unique_ptr<ve_forecast, ForecastFree>;

std::vector<double> view(ve_status (*read)(const ve_forecast*, const double**, size_t*), const ve_forecast* fc)
{
    const double* data = nullptr;
    size_t len = 0;
    EXPECT_EQ(read(fc, &data, &len), VE_OK);
    return {data, data + len};
}

bool mentions(const std::string& word)
{
    return std::string(ve_last_error()).find(word) != std::string::npos;
}

// One job on `series`, with the spec built by `configure` and `columns` mean regressors when given.
struct OneFit
{
    std::vector<double> series;
    std::vector<double> regressors;
    SpecPtr spec = ve::testing::make_spec(1);
    PanelPtr data;
    PanelPtr mean;
    FitSet fits{0};

    OneFit(std::vector<double> r, std::size_t columns) : series(std::move(r))
    {
        regressors.resize(series.size() * columns);
        for (std::size_t i = 0; i < regressors.size(); ++i)
        {
            regressors[i] = std::sin(0.3 * static_cast<double>(i));
        }
        EXPECT_EQ(ve_spec_set_arma(spec.get(), 1, 1), VE_OK);
        EXPECT_EQ(ve_spec_set_mxreg(spec.get(), static_cast<int32_t>(columns)), VE_OK);
        data = ve::testing::make_panel(series.data(), series.size(), 1);
        ve::testing::BatchInputs inputs;
        inputs.specs = {spec.get()};
        inputs.panel = data.get();
        inputs.job_spec = {0};
        inputs.job_series = {0};
        if (columns > 0)
        {
            mean = ve::testing::make_panel(regressors.data(), series.size(), columns);
            inputs.mxreg = {mean.get()};
        }
        ve::testing::BatchPtr batch = ve::testing::build_batch(inputs);
        ve::testing::ContextPtr context = ve::testing::make_context(1);
        fits = ve::testing::fit_all(context.get(), batch.get());
    }

    ve_fit* fit() const
    {
        return fits[0];
    }
};

ForecastPtr create(const ve_fit* fit, std::size_t horizon, const ve_panel* mxreg = nullptr,
                   const ve_panel* vxreg = nullptr)
{
    ve_forecast* raw = nullptr;
    EXPECT_EQ(ve_forecast_create(fit, horizon, mxreg, vxreg, &raw), VE_OK) << ve_last_error();
    return ForecastPtr(raw);
}

TEST(CapiForecast, MatchesCore)
{
    OneFit one(ve::testing::simulate_garch_series(5, kT), 0);
    ForecastPtr fc = create(one.fit(), kHorizon);
    ve::forecast::Forecast core = ve::forecast::forecast(one.fit()->result, kHorizon);
    EXPECT_EQ(view(ve_forecast_sigma, fc.get()), core.sigma);
    EXPECT_EQ(view(ve_forecast_mean, fc.get()), core.mean);
    size_t horizon = 0;
    EXPECT_EQ(ve_forecast_horizon(fc.get(), &horizon), VE_OK);
    EXPECT_EQ(horizon, kHorizon);
}

TEST(CapiForecast, WithMeanRegressors)
{
    OneFit one(ve::testing::simulate_garch_series(5, kT), 2);
    std::vector<double> future(kHorizon * 2, 0.4);
    PanelPtr panel = ve::testing::make_panel(future.data(), kHorizon, 2);
    ForecastPtr fc = create(one.fit(), kHorizon, panel.get());
    ve::forecast::Forecast core =
        ve::forecast::forecast(one.fit()->result, kHorizon, {{future, 2}, {}});
    EXPECT_EQ(view(ve_forecast_mean, fc.get()), core.mean);
}

// Review Focus 4: the forecast keeps its own copy.
TEST(CapiForecast, OutlivesFit)
{
    auto one = std::make_unique<OneFit>(ve::testing::simulate_garch_series(5, kT), 0);
    ForecastPtr fc = create(one->fit(), kHorizon);
    std::vector<double> before = view(ve_forecast_sigma, fc.get());
    one.reset();
    EXPECT_EQ(view(ve_forecast_sigma, fc.get()), before);
}

TEST(CapiForecast, SameFromTwoThreads)
{
    OneFit one(ve::testing::simulate_garch_series(5, kT), 0);
    std::vector<double> a;
    std::vector<double> b;
    std::thread first([&] { a = view(ve_forecast_sigma, create(one.fit(), kHorizon).get()); });
    std::thread second([&] { b = view(ve_forecast_sigma, create(one.fit(), kHorizon).get()); });
    first.join();
    second.join();
    EXPECT_EQ(a, b);
    EXPECT_EQ(a.size(), kHorizon);
}

ve_status try_create(const ve_fit* fit, std::size_t horizon, const ve_panel* mxreg, const ve_panel* vxreg)
{
    ve_forecast* raw = reinterpret_cast<ve_forecast*>(0x1);
    ve_status status = ve_forecast_create(fit, horizon, mxreg, vxreg, &raw);
    EXPECT_EQ(raw, nullptr);
    return status;
}

// Review Focus 1 and 2.
TEST(CapiForecast, InvalidArguments)
{
    OneFit plain(ve::testing::simulate_garch_series(5, kT), 0);
    OneFit regressed(ve::testing::simulate_garch_series(5, kT), 2);
    OneFit constant(std::vector<double>(kT, 0.01), 0);
    std::vector<double> values(4 * kHorizon, 0.4);
    PanelPtr good = ve::testing::make_panel(values.data(), kHorizon, 2);
    PanelPtr short_rows = ve::testing::make_panel(values.data(), kHorizon - 1, 2);
    PanelPtr wide = ve::testing::make_panel(values.data(), kHorizon, 3);
    PanelPtr ranged = ve::testing::make_panel(values.data(), kHorizon, 2);
    const size_t starts[2] = {0, 0};
    const size_t lengths[2] = {kHorizon, kHorizon};
    ASSERT_EQ(ve_panel_set_ranges(ranged.get(), starts, lengths), VE_OK);

    EXPECT_EQ(try_create(plain.fit(), 0, nullptr, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("horizon")) << ve_last_error();
    EXPECT_EQ(try_create(constant.fit(), kHorizon, nullptr, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("finite")) << ve_last_error();
    EXPECT_EQ(try_create(regressed.fit(), kHorizon, nullptr, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("mxreg")) << ve_last_error();
    for (const ve_panel* bad : {short_rows.get(), wide.get(), ranged.get()})
    {
        EXPECT_EQ(try_create(regressed.fit(), kHorizon, bad, nullptr), VE_INVALID_ARGUMENT);
        EXPECT_TRUE(mentions("mxreg")) << ve_last_error();
    }
    EXPECT_EQ(try_create(plain.fit(), kHorizon, good.get(), nullptr), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("mxreg")) << ve_last_error();
    EXPECT_EQ(try_create(plain.fit(), kHorizon, nullptr, good.get()), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("vxreg")) << ve_last_error();
    EXPECT_EQ(try_create(nullptr, kHorizon, nullptr, nullptr), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_forecast_create(plain.fit(), kHorizon, nullptr, nullptr, nullptr), VE_INVALID_ARGUMENT);
}

TEST(CapiForecast, FailedWithFinitePointForecasts)
{
    ve_fit fit;
    fit.result.status = ve::FitStatus::Failed;
    fit.result.estimate = {0.05, 0.02, 0.08, 0.9};
    fit.result.state = {{0.4}, {1.3}, {}, {0.4}};
    ForecastPtr fc = create(&fit, kHorizon);
    for (double sigma : view(ve_forecast_sigma, fc.get()))
    {
        EXPECT_TRUE(std::isfinite(sigma));
    }
}

TEST(CapiForecast, NullViewArguments)
{
    OneFit one(ve::testing::simulate_garch_series(5, kT), 0);
    ForecastPtr fc = create(one.fit(), kHorizon);
    const double* data = nullptr;
    size_t len = 0;
    EXPECT_EQ(ve_forecast_sigma(nullptr, &data, &len), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_forecast_mean(fc.get(), nullptr, &len), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_forecast_mean(fc.get(), &data, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_forecast_horizon(fc.get(), nullptr), VE_INVALID_ARGUMENT);
    ve_forecast_free(nullptr);
}

}
