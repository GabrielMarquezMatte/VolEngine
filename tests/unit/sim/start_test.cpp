#include "core/context.hpp"
#include "model/layout.hpp"
#include "model/parameter_map.hpp"
#include "sim/simulate.hpp"
#include "sim/start.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::Dist;
using ve::Model;

constexpr std::size_t kPaths = 20000;
constexpr std::size_t kHorizon = 5;
constexpr double kStandardErrors = 5.0;

ve::sim::Start start_of(const ve::Spec& spec, const ve::forecast::FutureRegressors& future = {})
{
    auto start = ve::sim::start_of_spec(spec, future);
    EXPECT_TRUE(start.has_value()) << (start.has_value() ? std::string{} : start.error());
    return start.value_or(ve::sim::Start{});
}

ve::sim::Simulation run(const ve::sim::Start& start, std::size_t paths, std::size_t horizon, int threads = 4)
{
    ve::Context context;
    context.set_threads(threads);
    return ve::sim::simulate(context, start, ve::sim::Request{horizon, paths, 42, {}});
}

void expect_path_mean(const ve::sim::Simulation& sim, std::size_t h, const std::function<double(double)>& f,
                      double target, bool on_returns = false)
{
    const std::vector<double>& values = on_returns ? sim.returns : sim.sigma;
    double sum = 0.0;
    double squares = 0.0;
    for (std::size_t p = 0; p < sim.paths; ++p)
    {
        double value = f(values[p * sim.horizon + h]);
        sum += value;
        squares += value * value;
    }
    double n = static_cast<double>(sim.paths);
    double mean = sum / n;
    double error = std::sqrt(std::max(squares / n - mean * mean, 0.0) / n);
    EXPECT_NEAR(mean, target, kStandardErrors * error + 1e-12) << "h = " << h + 1;
}

ve::Spec fixed_spec(Model model, Dist dist, std::vector<ve::ParameterValue> fixed)
{
    ve::Spec spec{.model = model, .dist = dist};
    spec.fixed = std::move(fixed);
    return spec;
}

double persistence_of(const ve::sim::Start& start)
{
    return ve::model::persistence<double>(start.theta, start.layout);
}

TEST(SimulateSpec, StationaryInExpectation)
{
    auto square = [](double s) { return s * s; };
    {
        SCOPED_TRACE("sGARCH");
        ve::sim::Start start =
            start_of(fixed_spec(Model::SGarch, Dist::Norm, {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", 0.1}, {"beta1", 0.85}}));
        ve::sim::Simulation sim = run(start, kPaths, kHorizon);
        for (std::size_t h = 0; h < kHorizon; ++h)
        {
            expect_path_mean(sim, h, square, 0.05 / 0.05);
        }
    }
    {
        SCOPED_TRACE("gjrGARCH");
        ve::sim::Start start = start_of(fixed_spec(
            Model::GjrGarch, Dist::Norm, {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", 0.05}, {"beta1", 0.85}, {"gamma1", 0.1}}));
        ve::sim::Simulation sim = run(start, kPaths, kHorizon);
        for (std::size_t h = 0; h < kHorizon; ++h)
        {
            expect_path_mean(sim, h, square, 0.05 / 0.05);
        }
    }
    {
        SCOPED_TRACE("eGARCH");
        ve::sim::Start start = start_of(fixed_spec(Model::EGarch, Dist::Std,
                                                   {{"mu", 0.0}, {"omega", -0.1}, {"alpha1", -0.05}, {"beta1", 0.95},
                                                    {"gamma1", 0.15}, {"shape", 8.0}}));
        ve::sim::Simulation sim = run(start, kPaths, kHorizon);
        for (std::size_t h = 0; h < kHorizon; ++h)
        {
            expect_path_mean(sim, h, [](double s) { return 2.0 * std::log(s); }, -0.1 / (1.0 - 0.95));
        }
    }
    {
        SCOPED_TRACE("apARCH");
        ve::sim::Start start = start_of(fixed_spec(Model::ApArch, Dist::Ged,
                                                   {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", 0.1}, {"beta1", 0.85},
                                                    {"gamma1", 0.2}, {"delta", 1.5}, {"shape", 1.5}}));
        double level = 0.05 / (1.0 - persistence_of(start));
        ve::sim::Simulation sim = run(start, kPaths, kHorizon);
        for (std::size_t h = 0; h < kHorizon; ++h)
        {
            expect_path_mean(sim, h, [](double s) { return std::pow(s, 1.5); }, level);
        }
    }
    {
        SCOPED_TRACE("csGARCH");
        ve::sim::Start start = start_of(fixed_spec(Model::CsGarch, Dist::Norm,
                                                   {{"mu", 0.0}, {"omega", 0.02}, {"alpha1", 0.05}, {"beta1", 0.8},
                                                    {"eta11", 0.98}, {"eta21", 0.03}}));
        ASSERT_EQ(start.state.permanent.size(), start.state.sigma2.size());
        EXPECT_NEAR(start.state.permanent[0], 1.0, 1e-14);
        ve::sim::Simulation sim = run(start, kPaths, kHorizon);
        for (std::size_t h = 0; h < kHorizon; ++h)
        {
            expect_path_mean(sim, h, square, 1.0);
        }
    }
    {
        SCOPED_TRACE("ARMA(1,1) mean");
        ve::Spec spec{.ar = 1, .ma = 1};
        spec.fixed = {{"mu", 0.3}, {"ar1", 0.5}, {"ma1", 0.2}, {"omega", 0.05}, {"alpha1", 0.1}, {"beta1", 0.85}};
        ve::sim::Simulation sim = run(start_of(spec), kPaths, kHorizon);
        for (std::size_t h = 0; h < kHorizon; ++h)
        {
            expect_path_mean(sim, h, [](double r) { return r; }, 0.3, true);
        }
    }
}

TEST(SimulateSpec, VxregShiftsTheLevel)
{
    ve::Spec spec{.vxreg = 1};
    spec.fixed = {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", 0.1}, {"beta1", 0.85}, {"vxreg1", 0.01}};
    std::vector<double> v{2.0, 3.0};
    ve::forecast::FutureRegressors future{{}, {v, 1}};
    ve::sim::Start start = start_of(spec, future);
    double level = (0.05 + 2.0 * 0.01) / (1.0 - 0.95);
    ASSERT_FALSE(start.state.sigma2.empty());
    EXPECT_NEAR(start.state.sigma2[0], level, 1e-14 * level);
    EXPECT_NEAR(start.presample_sigma, std::sqrt(level), 1e-14);
}

void expect_rejected(const ve::Spec& spec, const std::string& cites)
{
    auto start = ve::sim::start_of_spec(spec, {});
    ASSERT_FALSE(start.has_value()) << cites;
    EXPECT_NE(start.error().find(cites), std::string::npos) << start.error();
}

TEST(SimulateSpec, Rejections)
{
    expect_rejected(fixed_spec(Model::SGarch, Dist::Norm, {{"mu", 0.0}, {"omega", 0.05}, {"beta1", 0.85}}), "alpha1");
    ve::Spec targeting{.variance_targeting = true};
    targeting.fixed = {{"mu", 0.0}, {"alpha1", 0.1}, {"beta1", 0.85}};
    expect_rejected(targeting, "variance targeting");
    expect_rejected(fixed_spec(Model::IGarch, Dist::Norm, {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", 0.1}}), "iGARCH");
    expect_rejected(fixed_spec(Model::SGarch, Dist::Norm, {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", 0.2}, {"beta1", 0.82}}),
                    "stationary");
    expect_rejected(fixed_spec(Model::SGarch, Dist::Std,
                               {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", 0.1}, {"beta1", 0.85}, {"shape", 1.5}}),
                    "domain");
    // Inside the distribution's domain and below persistence 1, but outside the model's.
    expect_rejected(fixed_spec(Model::SGarch, Dist::Norm, {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", -0.3}, {"beta1", 0.85}}),
                    "model's domain");
    expect_rejected(fixed_spec(Model::SGarch, Dist::Norm, {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", 0.1}, {"beta1", -0.5}}),
                    "model's domain");
    expect_rejected(fixed_spec(Model::GjrGarch, Dist::Norm,
                               {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", 0.05}, {"beta1", 0.85}, {"gamma1", -0.2}}),
                    "model's domain");
    expect_rejected(fixed_spec(Model::ApArch, Dist::Norm,
                               {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", 0.01}, {"beta1", 0.85}, {"gamma1", 1.5},
                                {"delta", 2.0}}),
                    "model's domain");
}

TEST(SimulateSpec, PresampleComesFirst)
{
    ve::Spec spec{.p = 2, .q = 1};
    spec.fixed = {{"mu", 0.0}, {"omega", 0.05}, {"alpha1", 0.05}, {"alpha2", 0.05}, {"beta1", 0.85}};
    ve::sim::Start start = start_of(spec);
    EXPECT_EQ(start.presample, 2u);
    ve::sim::Simulation sim = run(start, 50, 10);
    EXPECT_NE(sim.sigma[0], sim.sigma[10]);
    ve::sim::Simulation shorter = run(start, 10, 5, 1);
    for (std::size_t p = 0; p < 10; ++p)
    {
        for (std::size_t h = 0; h < 5; ++h)
        {
            ASSERT_EQ(shorter.returns[p * 5 + h], sim.returns[p * 10 + h]) << p << ", " << h;
            ASSERT_EQ(shorter.sigma[p * 5 + h], sim.sigma[p * 10 + h]) << p << ", " << h;
        }
    }
}

}
