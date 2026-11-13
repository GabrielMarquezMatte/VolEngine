#include "core/context.hpp"
#include "fit/fit.hpp"
#include "forecast/forecast.hpp"
#include "model/layout.hpp"
#include "forecast/step.hpp"
#include "sim/path.hpp"
#include "sim/philox.hpp"
#include "sim/samplers.hpp"
#include "sim/simulate.hpp"
#include "sim/start.hpp"
#include "simulated_garch.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace
{

using ve::Dist;
using ve::Model;

constexpr std::size_t kObservations = 600;
constexpr std::size_t kPaths = 20000;
constexpr std::size_t kHorizon = 10;
constexpr double kStandardErrors = 5.0;

std::vector<double> columns(std::size_t rows, std::size_t count, double base)
{
    std::vector<double> x(rows * count);
    for (std::size_t t = 0; t < rows; ++t)
    {
        for (std::size_t k = 0; k < count; ++k)
        {
            double s = std::sin(0.37 * static_cast<double>((k + 1) * t) + static_cast<double>(k));
            x[t * count + k] = base + 0.5 * s * s;
        }
    }
    return x;
}

// A fit of spec on a simulated series, with the regressors of columns() over the sample and the horizon.
struct Case
{
    ve::FitResult fit;
    std::vector<double> x;
    std::vector<double> v;
    ve::forecast::FutureRegressors future;
};

Case fitted(const ve::Spec& spec, std::size_t horizon = kHorizon)
{
    Case c;
    std::size_t k_mean = static_cast<std::size_t>(spec.mxreg);
    std::size_t k_variance = static_cast<std::size_t>(spec.vxreg);
    c.x = columns(kObservations + horizon, k_mean, -0.25);
    c.v = columns(kObservations + horizon, k_variance, 0.5);
    std::vector<double> r = ve::testing::simulate_garch_series(5, kObservations);
    ve::model::Data sample{r,
                           {std::span<const double>(c.x.data(), kObservations * k_mean), k_mean},
                           {std::span<const double>(c.v.data(), kObservations * k_variance), k_variance}};
    c.fit = ve::fit_one(sample, spec, ve::FitConfig{});
    c.future = {{std::span<const double>(c.x).last(horizon * k_mean), k_mean},
                {std::span<const double>(c.v).last(horizon * k_variance), k_variance}};
    return c;
}

ve::sim::Simulation run(const Case& c, std::size_t paths, std::size_t horizon, std::uint64_t seed = 42,
                        int threads = 4)
{
    ve::Context context;
    context.set_threads(threads);
    ve::sim::Request request{horizon, paths, seed, c.future};
    return ve::sim::simulate(context, ve::sim::start_of_fit(c.fit), request);
}

// The mean over the paths of f(sigma) (f(r) with on_returns) at step h lies within kStandardErrors of target.
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
    EXPECT_NEAR(mean, target, kStandardErrors * error + 1e-12 * (1.0 + std::abs(target))) << "h = " << h + 1;
}

std::vector<ve::Spec> family_specs()
{
    return {ve::Spec{.model = Model::SGarch},
            ve::Spec{.model = Model::IGarch, .dist = Dist::Std},
            ve::Spec{.model = Model::GjrGarch, .dist = Dist::Sstd},
            ve::Spec{.model = Model::EGarch, .dist = Dist::Std},
            ve::Spec{.model = Model::ApArch, .dist = Dist::Ged},
            ve::Spec{.model = Model::FGarchAllGarch},
            ve::Spec{.model = Model::CsGarch},
            ve::Spec{.ar = 1, .ma = 1, .mxreg = 1, .vxreg = 1},
            ve::Spec{.model = Model::GjrGarch, .archm = 1},
            ve::Spec{.ar = 1, .ma = 1, .arfima = true}};
}

TEST(Simulate, FirstStepIsTheForecast)
{
    for (const ve::Spec& spec : family_specs())
    {
        SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(spec.model));
        Case c = fitted(spec);
        ASSERT_FALSE(c.fit.sigma.empty());
        ve::forecast::Forecast forecast = ve::forecast::forecast(c.fit, kHorizon, c.future);
        ve::sim::Simulation sim = run(c, 200, kHorizon);
        ASSERT_EQ(sim.returns.size(), 200 * kHorizon);
        for (std::size_t p = 0; p < sim.paths; ++p)
        {
            ASSERT_EQ(sim.sigma[p * kHorizon], forecast.sigma[0]) << "path " << p;
        }
    }
}

double power_of_fit(const ve::FitResult& fit)
{
    ve::model::Layout layout = ve::model::layout_of(fit.spec);
    return ve::model::power_of<double>(fit.estimate, layout);
}

TEST(Simulate, MeanMatchesForecast)
{
    auto square = [](double s) { return s * s; };
    // sigma^2: the quadratic families; gjrGARCH only with symmetric shocks, as rugarch's forecast takes P(z < 0)
    // for E[z^2 1(z < 0)].
    for (const ve::Spec& spec : {ve::Spec{}, ve::Spec{.model = Model::IGarch, .dist = Dist::Std},
                                 ve::Spec{.model = Model::GjrGarch}, ve::Spec{.model = Model::GjrGarch, .dist = Dist::Std}})
    {
        SCOPED_TRACE(::testing::Message() << "sigma2, model " << static_cast<int>(spec.model));
        Case c = fitted(spec);
        ve::forecast::Forecast forecast = ve::forecast::forecast(c.fit, kHorizon, c.future);
        ve::sim::Simulation sim = run(c, kPaths, kHorizon);
        for (std::size_t h = 0; h < kHorizon; ++h)
        {
            expect_path_mean(sim, h, square, square(forecast.sigma[h]));
        }
    }
    {
        SCOPED_TRACE("eGARCH, log sigma2");
        Case c = fitted(ve::Spec{.model = Model::EGarch, .dist = Dist::Std});
        ve::forecast::Forecast forecast = ve::forecast::forecast(c.fit, kHorizon, c.future);
        ve::sim::Simulation sim = run(c, kPaths, kHorizon);
        for (std::size_t h = 0; h < kHorizon; ++h)
        {
            expect_path_mean(sim, h, [](double s) { return 2.0 * std::log(s); }, 2.0 * std::log(forecast.sigma[h]));
        }
    }
    for (const ve::Spec& spec : {ve::Spec{.model = Model::ApArch, .dist = Dist::Ged}, ve::Spec{.model = Model::FGarchAllGarch}})
    {
        SCOPED_TRACE(::testing::Message() << "sigma^delta, model " << static_cast<int>(spec.model));
        Case c = fitted(spec);
        double delta = power_of_fit(c.fit);
        ve::forecast::Forecast forecast = ve::forecast::forecast(c.fit, kHorizon, c.future);
        ve::sim::Simulation sim = run(c, kPaths, kHorizon);
        for (std::size_t h = 0; h < kHorizon; ++h)
        {
            expect_path_mean(sim, h, [delta](double s) { return std::pow(s, delta); }, std::pow(forecast.sigma[h], delta));
        }
    }
}

// csGARCH(1,1) with the shocks at their expectation, E e^2 = E sigma2: E q_h = omega + rho E q_{h-1} and
// E sigma2_h = E q_h + (alpha + beta) (E sigma2_{h-1} - E q_{h-1}), from the deterministic step 1. Not rugarch's
// closed form, which leaves out the last shock.
TEST(Simulate, CsGarchMeanFollowsExpectedRecursion)
{
    Case c = fitted(ve::Spec{.model = Model::CsGarch});
    ve::model::Layout layout = ve::model::layout_of(c.fit.spec);
    const std::vector<double>& theta = c.fit.estimate;
    double omega = theta[layout.omega()];
    double alpha = theta[layout.alpha_begin()];
    double beta = theta[layout.beta_begin()];
    double rho = theta[layout.eta1_begin()];
    double phi = theta[layout.eta2_begin()];
    const ve::forecast::ForecastState& s = c.fit.state;
    double e = s.residuals.back();
    double q = omega + rho * s.permanent.back() + phi * (e * e - s.sigma2.back());
    double sigma2 = q + alpha * (e * e - s.permanent.back()) + beta * (s.sigma2.back() - s.permanent.back());
    ve::sim::Simulation sim = run(c, kPaths, kHorizon);
    for (std::size_t h = 0; h < kHorizon; ++h)
    {
        expect_path_mean(sim, h, [](double v) { return v * v; }, sigma2);
        double next_q = omega + rho * q;
        sigma2 = next_q + (alpha + beta) * (sigma2 - q);
        q = next_q;
    }
}

TEST(Simulate, MeanReturnsMatchForecast)
{
    for (const ve::Spec& spec : {ve::Spec{.ar = 1, .ma = 1, .mxreg = 1}, ve::Spec{.ar = 1, .ma = 1, .arfima = true}})
    {
        SCOPED_TRACE(::testing::Message() << "arfima " << spec.arfima);
        Case c = fitted(spec);
        ve::forecast::Forecast forecast = ve::forecast::forecast(c.fit, kHorizon, c.future);
        ve::sim::Simulation sim = run(c, kPaths, kHorizon);
        for (std::size_t h = 0; h < kHorizon; ++h)
        {
            expect_path_mean(sim, h, [](double r) { return r; }, forecast.mean[h], true);
        }
    }
}

bool same_bits(const std::vector<double>& a, const std::vector<double>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0;
}

TEST(Simulate, SameForAnyThreadCount)
{
    Case c = fitted(ve::Spec{.model = Model::GjrGarch, .dist = Dist::Nig, .mxreg = 1});
    ve::sim::Simulation one = run(c, 1000, kHorizon, 7, 1);
    for (int threads : {3, 16})
    {
        ve::sim::Simulation many = run(c, 1000, kHorizon, 7, threads);
        EXPECT_TRUE(same_bits(one.returns, many.returns)) << threads << " threads";
        EXPECT_TRUE(same_bits(one.sigma, many.sigma)) << threads << " threads";
    }
}

TEST(Simulate, Prefixes)
{
    Case c = fitted(ve::Spec{.dist = Dist::Ghyp}, 120);
    ve::sim::Simulation small = run(c, 100, 60);
    ve::sim::Simulation more_paths = run(c, 1000, 60);
    ve::sim::Simulation longer = run(c, 100, 120);
    for (std::size_t p = 0; p < 100; ++p)
    {
        for (std::size_t h = 0; h < 60; ++h)
        {
            ASSERT_EQ(small.returns[p * 60 + h], more_paths.returns[p * 60 + h]) << p << ", " << h;
            ASSERT_EQ(small.returns[p * 60 + h], longer.returns[p * 120 + h]) << p << ", " << h;
            ASSERT_EQ(small.sigma[p * 60 + h], longer.sigma[p * 120 + h]) << p << ", " << h;
        }
    }
    constexpr std::uint64_t kLast = std::numeric_limits<std::uint64_t>::max();
    ve::sim::Simulation zero = run(c, 10, 5, 0);
    ve::sim::Simulation last = run(c, 10, 5, kLast);
    EXPECT_FALSE(same_bits(zero.returns, last.returns));
    EXPECT_FALSE(same_bits(zero.returns, run(c, 10, 5, 42).returns));
}

// A point given by name; every parameter of the layout must be named.
std::vector<double> point(const ve::Spec& spec, const std::map<std::string, double>& values)
{
    std::vector<std::string> names = ve::model::parameter_names(ve::model::layout_of(spec));
    std::vector<double> theta;
    for (const std::string& name : names)
    {
        auto found = values.find(name);
        EXPECT_NE(found, values.end()) << name;
        theta.push_back(found == values.end() ? 0.0 : found->second);
    }
    return theta;
}

void expect_defined(const ve::sim::Simulation& sim)
{
    for (std::size_t i = 0; i < sim.sigma.size(); ++i)
    {
        ASSERT_FALSE(std::isnan(sim.sigma[i])) << i;
        ASSERT_GT(sim.sigma[i], 0.0) << i;
    }
}

TEST(Simulate, ExplosivePathsStayDefined)
{
    Case igarch = fitted(ve::Spec{.model = Model::IGarch});
    ASSERT_FALSE(igarch.fit.sigma.empty());
    expect_defined(run(igarch, 2000, 60));
    ve::Spec spec{.model = Model::ApArch};
    Case aparch;
    aparch.fit.status = ve::FitStatus::Converged;
    aparch.fit.spec = spec;
    aparch.fit.nobs = 100;
    aparch.fit.estimate =
        point(spec, {{"mu", 0.0}, {"omega", 0.01}, {"alpha1", 0.1}, {"beta1", 0.899}, {"gamma1", 0.0}, {"delta", 2.0}});
    aparch.fit.state = {{0.5}, {0.4}, {}, {0.5}};
    expect_defined(run(aparch, 2000, 60));
}

// The path of simulate_path stepped by hand on a state that only grows: what simulate must equal, bit for bit, however
// it keeps its state.
ve::sim::Simulation reference(const Case& c, std::size_t paths, std::size_t horizon, std::uint64_t seed)
{
    ve::sim::Start start = ve::sim::start_of_fit(c.fit);
    ve::sim::Sampler sampler(start.layout.dist, ve::model::shape_parameters_of<double>(start.theta, start.layout));
    ve::sim::Simulation out{paths, horizon, std::vector<double>(paths * horizon), std::vector<double>(paths * horizon)};
    for (std::size_t p = 0; p < paths; ++p)
    {
        ve::sim::Stream stream(seed, p);
        ve::forecast::ForecastState state = start.state;
        for (std::size_t h = 0; h < horizon; ++h)
        {
            ve::forecast::FutureRegressors row{};
            ve::forecast::detail::Inputs in{state, start.layout, start.theta, row, 1, start.moments};
            ve::forecast::detail::Step step = ve::forecast::detail::first_step(in);
            double sigma = std::sqrt(step.sigma2);
            double e = sigma * sampler.draw(stream);
            out.returns[p * horizon + h] = step.mean + e;
            out.sigma[p * horizon + h] = sigma;
            state.residuals.push_back(e);
            state.sigma2.push_back(step.sigma2);
            if (start.layout.model == Model::CsGarch)
            {
                state.permanent.push_back(step.permanent);
            }
            state.deviations.push_back(step.mean + e - step.level);
        }
    }
    return out;
}

TEST(Simulate, EqualsTheGrowingState)
{
    for (const ve::Spec& spec : {ve::Spec{.model = Model::ApArch, .ar = 1, .ma = 1}, ve::Spec{.model = Model::EGarch, .p = 2},
                                 ve::Spec{.model = Model::CsGarch}, ve::Spec{.ar = 1, .ma = 1, .arfima = true}})
    {
        SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(spec.model) << " arfima " << spec.arfima);
        Case c = fitted(spec, 0);
        ve::sim::Simulation expected = reference(c, 3, 300, 17);
        ve::sim::Simulation sim = run(c, 3, 300, 17);
        EXPECT_TRUE(same_bits(sim.returns, expected.returns));
        EXPECT_TRUE(same_bits(sim.sigma, expected.sigma));
    }
}

// The rules read only the last L entries of the state: a long horizon costs O(H) per path, not O(H^2) (eGARCH,
// apARCH and fGARCH transform the whole state at every step). 100 000 steps take well under a second then, and
// about a minute when the state grows without bound.
TEST(Simulate, LongHorizonCostsLinearTime)
{
    Case c = fitted(ve::Spec{.model = Model::ApArch}, 0);
    auto begin = std::chrono::steady_clock::now();
    ve::sim::Simulation sim = run(c, 1, 100000, 3, 1);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
    EXPECT_LT(seconds, 5.0);
    EXPECT_TRUE(std::isfinite(sim.sigma.back()));
}

// A Path stepped by hand with the draws of Stream(seed, 0) is row 0 of simulate.
TEST(Simulate, PathStepMatchesSimulate)
{
    constexpr std::size_t steps = 60;
    for (const ve::Spec& spec : {ve::Spec{.dist = Dist::Std}, ve::Spec{.model = Model::CsGarch}})
    {
        Case c = fitted(spec, steps);
        ve::sim::Simulation sim = run(c, 3, steps, 42, 1);
        ve::sim::Start start = ve::sim::start_of_fit(c.fit);
        ve::sim::Sampler sampler(start.layout.dist, ve::model::shape_parameters_of<double>(start.theta, start.layout));
        ve::sim::Stream stream(42, 0);
        ve::sim::Path path(start);
        path.draw_presample(stream, sampler);
        for (std::size_t h = 0; h < steps; ++h)
        {
            ve::sim::Path::Draw draw = path.step(ve::sim::future_row(c.future, h), sampler.draw(stream));
            EXPECT_EQ(draw.r, sim.returns[h]) << h;
            EXPECT_EQ(draw.sigma, sim.sigma[h]) << h;
        }
    }
}

}
