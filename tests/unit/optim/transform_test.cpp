#include "optim/transform.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <random>
#include <span>
#include <vector>

namespace
{

using ve::optim::sgarch_chain_gradient;
using ve::optim::sgarch_to_natural;
using ve::optim::sgarch_to_unconstrained;
using ve::optim::Standardization;

const Standardization kIdentity{};

double max_abs_difference(std::span<const double> a, std::span<const double> b)
{
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        result = std::max(result, std::abs(a[i] - b[i]));
    }
    return result;
}

double weighted_sum(std::span<const double> weights, std::span<const double> values)
{
    double total = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        total += weights[i] * values[i];
    }
    return total;
}

TEST(Transform, RoundTrip)
{
    std::array<double, 4> theta{0.05, 0.01, 0.08, 0.9};
    std::array<double, 4> u{};
    std::array<double, 4> back{};
    sgarch_to_unconstrained(theta, u, true, kIdentity);
    sgarch_to_natural(u, back, true, kIdentity);
    EXPECT_LT(max_abs_difference(theta, back), 1e-12);
}

TEST(Transform, RoundTripWithoutMean)
{
    std::array<double, 3> theta{0.01, 0.08, 0.9};
    std::array<double, 3> u{};
    std::array<double, 3> back{};
    sgarch_to_unconstrained(theta, u, false, kIdentity);
    sgarch_to_natural(u, back, false, kIdentity);
    EXPECT_LT(max_abs_difference(theta, back), 1e-12);
}

TEST(Transform, MeanAndVarianceLevelAreStandardized)
{
    std::array<double, 4> theta{-0.37, 0.01, 0.08, 0.9};
    std::array<double, 4> u{};
    std::array<double, 4> back{};
    const Standardization standardization{0.1, 0.5};
    sgarch_to_unconstrained(theta, u, true, standardization);
    EXPECT_DOUBLE_EQ(u[0], (-0.37 - 0.1) / 0.5);
    EXPECT_DOUBLE_EQ(u[1], std::log(0.01 / 0.25));
    sgarch_to_natural(u, back, true, standardization);
    EXPECT_LT(max_abs_difference(theta, back), 1e-12);
}

TEST(Transform, UnconstrainedPointDoesNotDependOnUnits)
{
    std::array<double, 4> theta{0.05, 0.01, 0.08, 0.9};
    std::array<double, 4> scaled{0.05e-4, 0.01e-8, 0.08, 0.9};
    std::array<double, 4> u{};
    std::array<double, 4> u_scaled{};
    sgarch_to_unconstrained(theta, u, true, Standardization{0.04, 0.7});
    sgarch_to_unconstrained(scaled, u_scaled, true, Standardization{0.04e-4, 0.7e-4});
    EXPECT_LT(max_abs_difference(u, u_scaled), 1e-10);
}

TEST(Transform, TinyAlphaAndBetaInvertWithoutInfinity)
{
    std::array<double, 3> theta{0.01, 1e-12, 1e-12};
    std::array<double, 3> u{};
    std::array<double, 3> back{};
    sgarch_to_unconstrained(theta, u, false, kIdentity);
    for (double value : u)
    {
        EXPECT_TRUE(std::isfinite(value));
    }
    sgarch_to_natural(u, back, false, kIdentity);
    EXPECT_LT(std::abs(back[1] - 1e-12), 1e-24);
    EXPECT_LT(std::abs(back[2] - 1e-12), 1e-24);
}

TEST(Transform, ZeroAlphaInvertsToFiniteUnconstrained)
{
    std::array<double, 3> theta{0.01, 0.0, 0.9};
    std::array<double, 3> u{};
    std::array<double, 3> back{};
    sgarch_to_unconstrained(theta, u, false, kIdentity);
    for (double value : u)
    {
        EXPECT_TRUE(std::isfinite(value));
    }
    sgarch_to_natural(u, back, false, kIdentity);
    EXPECT_LT(back[1], 1e-300);
    EXPECT_LT(std::abs(back[2] - 0.9), 1e-12);
}

TEST(Transform, AlwaysStationaryAndPositive)
{
    std::mt19937_64 engine(7);
    std::uniform_real_distribution<double> draw(-40.0, 40.0);
    for (int trial = 0; trial < 1000; ++trial)
    {
        std::array<double, 4> u{draw(engine), draw(engine), draw(engine), draw(engine)};
        std::array<double, 4> theta{};
        sgarch_to_natural(u, theta, true, kIdentity);
        EXPECT_GT(theta[1], 0.0);
        EXPECT_GE(theta[2], 0.0);
        EXPECT_GE(theta[3], 0.0);
        EXPECT_LT(theta[2] + theta[3], 1.0);
    }
}

TEST(Transform, ExtremeUnconstrainedValuesStayFinite)
{
    std::array<double, 3> u{0.0, 800.0, -800.0};
    std::array<double, 3> theta{};
    sgarch_to_natural(u, theta, false, kIdentity);
    EXPECT_EQ(theta[0], 1.0);
    EXPECT_LT(theta[1] + theta[2], 1.0);
    EXPECT_GE(theta[1], 0.0);
    EXPECT_GE(theta[2], 0.0);
    u = {0.0, -800.0, 800.0};
    sgarch_to_natural(u, theta, false, kIdentity);
    EXPECT_EQ(theta[1] + theta[2], 0.0);
}

void expect_chain_gradient_matches_finite_differences(bool include_mean)
{
    const Standardization standardization{0.3, 2.5};
    std::mt19937_64 engine(11);
    std::uniform_real_distribution<double> draw(-2.0, 2.0);
    std::size_t size = include_mean ? 4 : 3;
    std::vector<double> u(size);
    std::vector<double> c(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        u[i] = draw(engine);
        c[i] = draw(engine);
    }
    std::vector<double> analytic(size);
    sgarch_chain_gradient(u, c, analytic, include_mean, standardization);
    std::vector<double> theta(size);
    constexpr double kStep = 1e-6;
    for (std::size_t i = 0; i < size; ++i)
    {
        std::vector<double> shifted = u;
        shifted[i] = u[i] + kStep;
        sgarch_to_natural(shifted, theta, include_mean, standardization);
        double forward = weighted_sum(c, theta);
        shifted[i] = u[i] - kStep;
        sgarch_to_natural(shifted, theta, include_mean, standardization);
        double backward = weighted_sum(c, theta);
        EXPECT_NEAR(analytic[i], (forward - backward) / (2.0 * kStep), 1e-7);
    }
}

TEST(Transform, ChainGradientMatchesFiniteDifferences)
{
    expect_chain_gradient_matches_finite_differences(true);
}

TEST(Transform, ChainGradientMatchesFiniteDifferencesWithoutMean)
{
    expect_chain_gradient_matches_finite_differences(false);
}

}
