#include "optim/lbfgs.hpp"

#include "core/cancel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <vector>

namespace
{

using ve::optim::LbfgsOptions;
using ve::optim::LbfgsResult;
using ve::optim::Stop;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();

double rosenbrock(std::span<const double> x, std::span<double> g)
{
    double a = 1.0 - x[0];
    double b = x[1] - x[0] * x[0];
    g[0] = -2.0 * a - 400.0 * x[0] * b;
    g[1] = 200.0 * b;
    return a * a + 100.0 * b * b;
}

double weighted_quadratic(std::span<const double> x, std::span<double> g)
{
    double value = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        double weight = static_cast<double>(i + 1);
        value += weight * x[i] * x[i];
        g[i] = 2.0 * weight * x[i];
    }
    return value;
}

TEST(Lbfgs, MinimizesRosenbrock)
{
    std::vector<double> x{-1.2, 1.0};
    LbfgsResult result = ve::optim::lbfgs_minimize(rosenbrock, x, {});
    EXPECT_EQ(result.stop, Stop::Converged);
    EXPECT_LT(std::abs(x[0] - 1.0), 1e-5);
    EXPECT_LT(std::abs(x[1] - 1.0), 1e-5);
}

TEST(Lbfgs, CountsEvaluations)
{
    int calls = 0;
    auto counted = [&calls](std::span<const double> x, std::span<double> g)
    {
        ++calls;
        return weighted_quadratic(x, g);
    };
    std::vector<double> x(10, 1.0);
    LbfgsResult result = ve::optim::lbfgs_minimize(counted, x, {});
    EXPECT_EQ(result.evaluations, calls);
    EXPECT_GE(result.evaluations, result.iterations + 1);
}

// The likelihoods have at most a few dozen parameters: a memory at least as long as k makes L-BFGS behave like full
// BFGS: 37 iterations on this ill-conditioned quadratic with k = 12, against 135 with a memory of 10.
TEST(Lbfgs, DefaultMemoryCoversTwelveParameters)
{
    auto ill_conditioned = [](std::span<const double> x, std::span<double> g)
    {
        double value = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            const double weight = std::pow(10.0, 3.0 * static_cast<double>(i) / 11.0);
            value += 0.5 * weight * x[i] * x[i];
            g[i] = weight * x[i];
        }
        return value;
    };
    std::vector<double> x(12, 1.0);
    LbfgsResult result = ve::optim::lbfgs_minimize(ill_conditioned, x, {});
    EXPECT_EQ(result.stop, Stop::Converged);
    EXPECT_LE(result.iterations, 45);
}

TEST(Lbfgs, ResultValueIsObjectiveAtReturnedPoint)
{
    std::vector<double> x{-1.2, 1.0};
    std::vector<double> g(2);
    LbfgsResult result = ve::optim::lbfgs_minimize(rosenbrock, x, {});
    EXPECT_EQ(result.f, rosenbrock(x, g));
}

TEST(Lbfgs, MinimizesQuadratic10d)
{
    std::vector<double> x(10, 1.0);
    LbfgsOptions options;
    options.gtol = 1e-10;
    options.ftol = 1e-24;
    LbfgsResult result = ve::optim::lbfgs_minimize(weighted_quadratic, x, options);
    EXPECT_EQ(result.stop, Stop::Converged);
    EXPECT_LE(result.iterations, 50);
    for (double value : x)
    {
        EXPECT_LT(std::abs(value), 1e-8);
    }
}

TEST(Lbfgs, ReportsMaxIter)
{
    std::vector<double> x{-1.2, 1.0};
    LbfgsOptions options;
    options.max_iter = 3;
    LbfgsResult result = ve::optim::lbfgs_minimize(rosenbrock, x, options);
    EXPECT_EQ(result.stop, Stop::MaxIter);
    EXPECT_EQ(result.iterations, 3);
}

TEST(Lbfgs, ConvergesThroughFtolWhenDecreaseIsBelowTolerance)
{
    auto tiny_quadratic = [](std::span<const double> x, std::span<double> g)
    {
        g[0] = 3.0 * x[0];
        return 1.5 * x[0] * x[0];
    };
    std::vector<double> x{1e-8};
    LbfgsOptions options;
    options.gtol = 1e-12;
    LbfgsResult result = ve::optim::lbfgs_minimize(tiny_quadratic, x, options);
    EXPECT_EQ(result.stop, Stop::Converged);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_NE(x[0], 0.0);
}

TEST(Lbfgs, FlatStepWithLargeGradientKeepsIterating)
{
    auto shallow_quadratic = [](std::span<const double> x, std::span<double> g)
    {
        g[0] = 0.02 * (x[0] - 1.0);
        return 0.01 * (x[0] - 1.0) * (x[0] - 1.0);
    };
    std::vector<double> x{0.0};
    LbfgsOptions options;
    options.ftol = 0.1;
    LbfgsResult result = ve::optim::lbfgs_minimize(shallow_quadratic, x, options);
    EXPECT_EQ(result.stop, Stop::Converged);
    EXPECT_GT(result.iterations, 1);
    EXPECT_LT(std::abs(x[0] - 1.0), 1e-6);
}

TEST(Lbfgs, FallbackStepBelowFtolIsNotReportedAsConverged)
{
    auto shallow_kink = [](std::span<const double> x, std::span<double> g)
    {
        double offset = x[0] - 0.3;
        g[0] = offset < 0.0 ? -1e-14 : 1e-14;
        return 1e-14 * std::abs(offset);
    };
    std::vector<double> x{3.0};
    LbfgsOptions options;
    options.gtol = 0.0;
    options.max_iter = 4;
    LbfgsResult result = ve::optim::lbfgs_minimize(shallow_kink, x, options);
    EXPECT_EQ(result.stop, Stop::MaxIter);
    EXPECT_EQ(result.iterations, 4);
}

TEST(Lbfgs, ReportsNonFiniteAtStart)
{
    std::vector<double> x{0.5, 0.5};
    auto not_a_number = [](std::span<const double>, std::span<double> g)
    {
        g[0] = 0.0;
        g[1] = 0.0;
        return kNaN;
    };
    LbfgsResult result = ve::optim::lbfgs_minimize(not_a_number, x, {});
    EXPECT_EQ(result.stop, Stop::NonFinite);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(x[0], 0.5);
    EXPECT_EQ(x[1], 0.5);
}

TEST(Lbfgs, NonFiniteTrialShrinksStepInsteadOfStopping)
{
    int non_finite_calls = 0;
    auto cliff = [&non_finite_calls](std::span<const double> x, std::span<double> g)
    {
        if (x[0] < -1.5)
        {
            ++non_finite_calls;
            g[0] = kNaN;
            return kInfinity;
        }
        double magnitude = std::abs(x[0]);
        double sign = x[0] < 0.0 ? -1.0 : 1.0;
        g[0] = 1.5 * sign * std::sqrt(magnitude);
        return magnitude * std::sqrt(magnitude);
    };
    std::vector<double> x{2.5};
    LbfgsResult result = ve::optim::lbfgs_minimize(cliff, x, {});
    EXPECT_GT(non_finite_calls, 0);
    EXPECT_EQ(result.stop, Stop::Converged);
    EXPECT_LT(std::abs(x[0]), 1e-6);
    EXPECT_TRUE(std::isfinite(result.f));
}

// f is x^2 rounded down to a multiple of 1e-6, so no step from x = 1e-4 decreases it, while the gradient 2x is
// above gtol and within sqrt(gtol): the search has reached the resolution of f.
TEST(Lbfgs, LineSearchFailureWithNearlySmallGradientConverges)
{
    auto quantized = [](std::span<const double> x, std::span<double> g)
    {
        g[0] = 2.0 * x[0];
        return std::floor(x[0] * x[0] / 1e-6) * 1e-6;
    };
    std::vector<double> x{1e-4};
    LbfgsResult result = ve::optim::lbfgs_minimize(quantized, x, {});
    EXPECT_EQ(result.stop, Stop::Converged);
    EXPECT_LE(std::abs(x[0]), 1e-4);
}

// A narrow curved valley (Rosenbrock with a steep wall): every step along its floor decreases f by less than 1e-12
// while the gradient is still far above gtol. Reporting Converged there stops on the ridge, short of the optimum.
TEST(Lbfgs, RidgeIsNotReportedConverged)
{
    auto steep_valley = [](std::span<const double> x, std::span<double> g)
    {
        constexpr double kWall = 1e8;
        double a = 1.0 - x[0];
        double b = x[1] - x[0] * x[0];
        g[0] = -2.0 * a - 4.0 * kWall * x[0] * b;
        g[1] = 2.0 * kWall * b;
        return a * a + kWall * b * b;
    };
    std::vector<double> x{-1.2, 1.0};
    LbfgsOptions options;
    options.max_iter = 5000;
    LbfgsResult result = ve::optim::lbfgs_minimize(steep_valley, x, options);
    std::vector<double> g(2);
    steep_valley(x, g);
    const double scale = std::max(1.0, std::abs(result.f));
    if (result.stop == Stop::Converged)
    {
        EXPECT_LE(std::max(std::abs(g[0]), std::abs(g[1])), options.gtol * scale);
    }
    EXPECT_LT(std::abs(x[0] - 1.0), 1e-4);
}

TEST(Lbfgs, ReportsLineSearchFailedWhenNoOtherPointIsFinite)
{
    auto finite_only_at_start = [](std::span<const double> x, std::span<double> g)
    {
        if (x[0] == 0.0)
        {
            g[0] = 1.0;
            return 0.0;
        }
        g[0] = kNaN;
        return kInfinity;
    };
    std::vector<double> x{0.0};
    LbfgsResult result = ve::optim::lbfgs_minimize(finite_only_at_start, x, {});
    EXPECT_EQ(result.stop, Stop::LineSearchFailed);
    EXPECT_EQ(x[0], 0.0);
    EXPECT_EQ(result.f, 0.0);
}

TEST(Lbfgs, RequestedFlagThrowsCancelled)
{
    ve::CancelFlag flag;
    flag.request();
    LbfgsOptions options;
    options.cancel = &flag;
    std::vector<double> x(10, 1.0);
    EXPECT_THROW(ve::optim::lbfgs_minimize(weighted_quadratic, x, options), ve::Cancelled);
}

TEST(Lbfgs, UnrequestedFlagChangesNothing)
{
    ve::CancelFlag flag;
    LbfgsOptions with;
    with.cancel = &flag;
    std::vector<double> x(10, 1.0);
    std::vector<double> y(10, 1.0);
    LbfgsResult plain = ve::optim::lbfgs_minimize(weighted_quadratic, x, {});
    LbfgsResult flagged = ve::optim::lbfgs_minimize(weighted_quadratic, y, with);
    EXPECT_EQ(flagged.stop, plain.stop);
    EXPECT_EQ(flagged.iterations, plain.iterations);
    EXPECT_EQ(flagged.evaluations, plain.evaluations);
    EXPECT_EQ(flagged.f, plain.f);
    EXPECT_EQ(std::memcmp(x.data(), y.data(), x.size() * sizeof(double)), 0);
}

TEST(Lbfgs, RequestDuringRunStops)
{
    std::vector<double> x(10, 1.0);
    int full = ve::optim::lbfgs_minimize(weighted_quadratic, x, {}).evaluations;
    ve::CancelFlag flag;
    int calls = 0;
    auto objective = [&](std::span<const double> v, std::span<double> g)
    {
        if (++calls == 3)
        {
            flag.request();
        }
        return weighted_quadratic(v, g);
    };
    LbfgsOptions options;
    options.cancel = &flag;
    std::vector<double> y(10, 1.0);
    EXPECT_THROW(ve::optim::lbfgs_minimize(objective, y, options), ve::Cancelled);
    EXPECT_LT(calls, full);
}

// The checkpoint comes after the convergence tests: a start at the minimum returns normally.
TEST(Lbfgs, ConvergedStartReturnsNormally)
{
    ve::CancelFlag flag;
    flag.request();
    LbfgsOptions options;
    options.cancel = &flag;
    std::vector<double> x(10, 0.0);
    LbfgsResult result = ve::optim::lbfgs_minimize(weighted_quadratic, x, options);
    EXPECT_EQ(result.stop, Stop::Converged);
}

}
