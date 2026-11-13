#include "capi/handles.hpp"
#include "capi_scope.hpp"
#include "core/context.hpp"
#include "sim/simulate.hpp"
#include "sim/start.hpp"
#include "simulated_garch.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <volengine/volengine.h>

namespace
{

using ve::testing::ContextPtr;
using ve::testing::FitSet;
using ve::testing::PanelPtr;
using ve::testing::SpecPtr;

constexpr std::size_t kT = 800;
constexpr std::size_t kHorizon = 10;
constexpr std::size_t kPaths = 300;

struct OptionsFree
{
    void operator()(ve_sim_options* handle) const
    {
        ve_sim_options_free(handle);
    }
};

struct SimFree
{
    void operator()(ve_sim* handle) const
    {
        ve_sim_free(handle);
    }
};

using OptionsPtr = std::unique_ptr<ve_sim_options, OptionsFree>;
using SimPtr = std::unique_ptr<ve_sim, SimFree>;

OptionsPtr make_options(std::size_t horizon, std::size_t paths, std::uint64_t seed)
{
    ve_sim_options* raw = nullptr;
    EXPECT_EQ(ve_sim_options_create(horizon, paths, seed, &raw), VE_OK) << ve_last_error();
    return OptionsPtr(raw);
}

std::vector<double> view(ve_status (*read)(const ve_sim*, const double**, size_t*), const ve_sim* sim)
{
    const double* data = nullptr;
    size_t len = 0;
    EXPECT_EQ(read(sim, &data, &len), VE_OK);
    return {data, data + len};
}

bool mentions(const std::string& word)
{
    return std::string(ve_last_error()).find(word) != std::string::npos;
}

// One job on `series` with `columns` mean regressors.
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
        ContextPtr context = ve::testing::make_context(1);
        fits = ve::testing::fit_all(context.get(), batch.get());
    }

    ve_fit* fit() const
    {
        return fits[0];
    }
};

ve::sim::Simulation core(const ve::sim::Start& start, std::size_t paths, std::uint64_t seed,
                         const ve::forecast::FutureRegressors& future = {})
{
    ve::Context context;
    return ve::sim::simulate(context, start, ve::sim::Request{kHorizon, paths, seed, future});
}

SpecPtr fixed_spec()
{
    SpecPtr spec = ve::testing::make_spec(1);
    EXPECT_EQ(ve_spec_set_fixed(spec.get(), "mu", 0.01), VE_OK);
    EXPECT_EQ(ve_spec_set_fixed(spec.get(), "omega", 0.05), VE_OK);
    EXPECT_EQ(ve_spec_set_fixed(spec.get(), "alpha1", 0.1), VE_OK);
    EXPECT_EQ(ve_spec_set_fixed(spec.get(), "beta1", 0.85), VE_OK);
    return spec;
}

TEST(CapiSim, FitMatchesCore)
{
    OneFit one(ve::testing::simulate_garch_series(5, kT), 0);
    ContextPtr context = ve::testing::make_context(4);
    OptionsPtr options = make_options(kHorizon, kPaths, 7);
    ve_sim* raw = nullptr;
    ASSERT_EQ(ve_simulate_fit(context.get(), one.fit(), options.get(), &raw), VE_OK) << ve_last_error();
    SimPtr sim(raw);
    ve::sim::Simulation expected = core(ve::sim::start_of_fit(one.fit()->result), kPaths, 7);
    EXPECT_EQ(view(ve_sim_returns, sim.get()), expected.returns);
    EXPECT_EQ(view(ve_sim_sigma, sim.get()), expected.sigma);
    size_t paths = 0;
    size_t horizon = 0;
    EXPECT_EQ(ve_sim_paths(sim.get(), &paths), VE_OK);
    EXPECT_EQ(ve_sim_horizon(sim.get(), &horizon), VE_OK);
    EXPECT_EQ(paths, kPaths);
    EXPECT_EQ(horizon, kHorizon);
}

TEST(CapiSim, RegressorsMatchCore)
{
    OneFit one(ve::testing::simulate_garch_series(5, kT), 2);
    std::vector<double> future(kHorizon * 2);
    for (std::size_t i = 0; i < future.size(); ++i)
    {
        future[i] = 0.1 * static_cast<double>(i);
    }
    PanelPtr panel = ve::testing::make_panel(future.data(), kHorizon, 2);
    ContextPtr context = ve::testing::make_context(2);
    OptionsPtr options = make_options(kHorizon, kPaths, 3);
    ASSERT_EQ(ve_sim_options_set_regressors(options.get(), panel.get(), nullptr), VE_OK) << ve_last_error();
    ve_sim* raw = nullptr;
    ASSERT_EQ(ve_simulate_fit(context.get(), one.fit(), options.get(), &raw), VE_OK) << ve_last_error();
    SimPtr sim(raw);
    ve::sim::Simulation expected = core(ve::sim::start_of_fit(one.fit()->result), kPaths, 3, {{future, 2}, {}});
    EXPECT_EQ(view(ve_sim_returns, sim.get()), expected.returns);
}

TEST(CapiSim, SpecMatchesCore)
{
    SpecPtr spec = fixed_spec();
    ContextPtr context = ve::testing::make_context(3);
    OptionsPtr options = make_options(kHorizon, kPaths, 11);
    ve_sim* raw = nullptr;
    ASSERT_EQ(ve_simulate_spec(context.get(), spec.get(), options.get(), &raw), VE_OK) << ve_last_error();
    SimPtr sim(raw);
    auto start = ve::sim::start_of_spec(spec->spec, {});
    ASSERT_TRUE(start.has_value());
    ve::sim::Simulation expected = core(*start, kPaths, 11);
    EXPECT_EQ(view(ve_sim_returns, sim.get()), expected.returns);
    EXPECT_EQ(view(ve_sim_sigma, sim.get()), expected.sigma);
}

// Review Focus 3: the simulation keeps its own copy.
TEST(CapiSim, OutlivesSources)
{
    auto one = std::make_unique<OneFit>(ve::testing::simulate_garch_series(5, kT), 0);
    SpecPtr spec = fixed_spec();
    ContextPtr context = ve::testing::make_context(2);
    OptionsPtr options = make_options(kHorizon, kPaths, 5);
    ve_sim* from_fit = nullptr;
    ve_sim* from_spec = nullptr;
    ASSERT_EQ(ve_simulate_fit(context.get(), one->fit(), options.get(), &from_fit), VE_OK);
    ASSERT_EQ(ve_simulate_spec(context.get(), spec.get(), options.get(), &from_spec), VE_OK);
    SimPtr fit_sim(from_fit);
    SimPtr spec_sim(from_spec);
    std::vector<double> fit_returns = view(ve_sim_returns, fit_sim.get());
    std::vector<double> spec_sigma = view(ve_sim_sigma, spec_sim.get());
    one.reset();
    spec.reset();
    options.reset();
    context.reset();
    EXPECT_EQ(view(ve_sim_returns, fit_sim.get()), fit_returns);
    EXPECT_EQ(view(ve_sim_sigma, spec_sim.get()), spec_sigma);
}

ve_status try_options(std::size_t horizon, std::size_t paths)
{
    ve_sim_options* raw = reinterpret_cast<ve_sim_options*>(0x1);
    ve_status status = ve_sim_options_create(horizon, paths, 0, &raw);
    EXPECT_EQ(raw, nullptr);
    return status;
}

ve_status try_fit(ve_context* context, const ve_fit* fit, const ve_sim_options* options)
{
    ve_sim* raw = reinterpret_cast<ve_sim*>(0x1);
    ve_status status = ve_simulate_fit(context, fit, options, &raw);
    EXPECT_EQ(raw, nullptr);
    return status;
}

ve_status try_spec(ve_context* context, const ve_spec* spec, const ve_sim_options* options)
{
    ve_sim* raw = reinterpret_cast<ve_sim*>(0x1);
    ve_status status = ve_simulate_spec(context, spec, options, &raw);
    EXPECT_EQ(raw, nullptr);
    return status;
}

// Review Focus 1 and the regressor rules.
TEST(CapiSim, InvalidArguments)
{
    EXPECT_EQ(try_options(0, 10), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("horizon")) << ve_last_error();
    EXPECT_EQ(try_options(10, 0), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("paths")) << ve_last_error();
    EXPECT_EQ(try_options(60, std::size_t{1} << 62), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("too large")) << ve_last_error();
    EXPECT_EQ(ve_sim_options_create(10, 10, 0, nullptr), VE_INVALID_ARGUMENT);

    OneFit plain(ve::testing::simulate_garch_series(5, kT), 0);
    OneFit regressed(ve::testing::simulate_garch_series(5, kT), 2);
    OneFit constant(std::vector<double>(kT, 0.01), 0);
    ContextPtr context = ve::testing::make_context(1);
    OptionsPtr options = make_options(kHorizon, 5, 0);
    EXPECT_EQ(try_fit(context.get(), constant.fit(), options.get()), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("finite")) << ve_last_error();
    SpecPtr free_alpha = ve::testing::make_spec(1);
    EXPECT_EQ(ve_spec_set_fixed(free_alpha.get(), "omega", 0.05), VE_OK);
    EXPECT_EQ(try_spec(context.get(), free_alpha.get(), options.get()), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("mu")) << ve_last_error();
    EXPECT_EQ(try_fit(context.get(), regressed.fit(), options.get()), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("mxreg")) << ve_last_error();

    std::vector<double> values(4 * kHorizon, 0.4);
    PanelPtr good = ve::testing::make_panel(values.data(), kHorizon, 2);
    PanelPtr short_rows = ve::testing::make_panel(values.data(), kHorizon - 1, 2);
    PanelPtr wide = ve::testing::make_panel(values.data(), kHorizon, 3);
    PanelPtr ranged = ve::testing::make_panel(values.data(), kHorizon, 2);
    const size_t starts[2] = {0, 0};
    const size_t lengths[2] = {kHorizon, kHorizon};
    ASSERT_EQ(ve_panel_set_ranges(ranged.get(), starts, lengths), VE_OK);
    for (const ve_panel* bad : {short_rows.get(), ranged.get()})
    {
        EXPECT_EQ(ve_sim_options_set_regressors(options.get(), bad, nullptr), VE_INVALID_ARGUMENT);
        EXPECT_TRUE(mentions("mxreg")) << ve_last_error();
    }
    EXPECT_EQ(ve_sim_options_set_regressors(options.get(), wide.get(), nullptr), VE_OK);
    EXPECT_EQ(try_fit(context.get(), regressed.fit(), options.get()), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("mxreg")) << ve_last_error();
    EXPECT_EQ(ve_sim_options_set_regressors(options.get(), good.get(), nullptr), VE_OK);
    EXPECT_EQ(try_fit(context.get(), plain.fit(), options.get()), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("mxreg")) << ve_last_error();
    EXPECT_EQ(ve_sim_options_set_regressors(options.get(), nullptr, good.get()), VE_OK);
    EXPECT_EQ(try_fit(context.get(), plain.fit(), options.get()), VE_INVALID_ARGUMENT);
    EXPECT_TRUE(mentions("vxreg")) << ve_last_error();

    EXPECT_EQ(ve_sim_options_set_regressors(nullptr, nullptr, nullptr), VE_INVALID_HANDLE);
    EXPECT_EQ(try_fit(nullptr, plain.fit(), options.get()), VE_INVALID_HANDLE);
    EXPECT_EQ(try_fit(context.get(), nullptr, options.get()), VE_INVALID_HANDLE);
    EXPECT_EQ(try_fit(context.get(), plain.fit(), nullptr), VE_INVALID_HANDLE);
    EXPECT_EQ(try_spec(context.get(), nullptr, options.get()), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_simulate_fit(context.get(), plain.fit(), options.get(), nullptr), VE_INVALID_ARGUMENT);
    size_t out = 0;
    EXPECT_EQ(ve_sim_paths(nullptr, &out), VE_INVALID_HANDLE);
    const double* data = nullptr;
    EXPECT_EQ(ve_sim_returns(nullptr, &data, &out), VE_INVALID_HANDLE);
}

// Review Focus 4.
TEST(CapiSim, SeedEdges)
{
    SpecPtr spec = fixed_spec();
    ContextPtr context = ve::testing::make_context(1);
    std::vector<std::vector<double>> returns;
    for (std::uint64_t seed : {std::uint64_t{0}, std::numeric_limits<std::uint64_t>::max()})
    {
        OptionsPtr options = make_options(kHorizon, 4, seed);
        ve_sim* raw = nullptr;
        ASSERT_EQ(ve_simulate_spec(context.get(), spec.get(), options.get(), &raw), VE_OK) << ve_last_error();
        SimPtr sim(raw);
        returns.push_back(view(ve_sim_returns, sim.get()));
    }
    EXPECT_NE(returns[0], returns[1]);
}

}
