#include "ad/dual.hpp"
#include "dist/bessel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>

namespace
{

using ve::dist::log_bessel_k;

double half_order_log_k(double x)
{
    return 0.5 * std::log(std::numbers::pi / (2.0 * x)) - x;
}

void expect_relative(double actual, double expected, double tolerance)
{
    EXPECT_NEAR(actual, expected, tolerance * std::fabs(expected));
}

TEST(Bessel, MatchesClosedForms)
{
    for (double x : {1e-3, 0.5, 3.0, 40.0, 699.0, 701.0, 5000.0})
    {
        double half = half_order_log_k(x);
        double three_halves = half + std::log1p(1.0 / x);
        for (double sign : {1.0, -1.0})
        {
            expect_relative(log_bessel_k(sign * 0.5, x), half, 1e-13);
            expect_relative(log_bessel_k(sign * 1.5, x), three_halves, 1e-13);
        }
    }
    expect_relative(std::exp(log_bessel_k(0.0, 1.0)), 0.42102443824070834, 1e-14);
    expect_relative(std::exp(log_bessel_k(1.0, 1.0)), 0.60190723019723457, 1e-14);
}

TEST(Bessel, AsymptoticJoinsBoost)
{
    constexpr double kGap = 1e-6;
    for (double order : {0.0, 1.0, 2.3, 6.0})
    {
        double jump = log_bessel_k(order, 700.0 + kGap) - log_bessel_k(order, 700.0 - kGap);
        double slope = log_bessel_k(order, ve::ad::variable<1>(700.0, 0)).grad[0];
        EXPECT_NEAR(jump, 2.0 * kGap * slope, 1e-11) << "order " << order;
    }
}

TEST(Bessel, DerivativesMatchFiniteDifferences)
{
    for (double order : {-1.0, 0.3, 2.5})
    {
        for (double x : {0.2, 3.0, 900.0})
        {
            auto dual = log_bessel_k(ve::ad::variable<8>(order, 0), ve::ad::variable<8>(x, 1));
            double h_order = 1e-5 * std::max(1.0, std::fabs(order));
            double h_x = 1e-5 * std::max(1.0, x);
            double d_order = (log_bessel_k(order + h_order, x) - log_bessel_k(order - h_order, x)) / (2.0 * h_order);
            double d_x = (log_bessel_k(order, x + h_x) - log_bessel_k(order, x - h_x)) / (2.0 * h_x);
            EXPECT_EQ(dual.value, log_bessel_k(order, x));
            EXPECT_NEAR(dual.grad[0], d_order, 1e-6 * std::max(1.0, std::fabs(d_order))) << order << ", " << x;
            EXPECT_NEAR(dual.grad[1], d_x, 1e-6 * std::max(1.0, std::fabs(d_x))) << order << ", " << x;

            auto fixed = log_bessel_k(order, ve::ad::variable<8>(x, 1));
            EXPECT_EQ(fixed.value, dual.value);
            EXPECT_EQ(fixed.grad[0], 0.0);
            EXPECT_EQ(fixed.grad[1], dual.grad[1]);
        }
    }
}

TEST(Bessel, NeverThrows)
{
    constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
    constexpr double kInf = std::numeric_limits<double>::infinity();
    double results[] = {log_bessel_k(50.0, 1e-300), log_bessel_k(300.0, 1e-5), log_bessel_k(0.0, 0.0),
                        log_bessel_k(1.0, -1.0),    log_bessel_k(kNan, 1.0),   log_bessel_k(kInf, 1.0),
                        log_bessel_k(1e300, 1.0),   log_bessel_k(1.0, kNan),   log_bessel_k(kNan, 800.0)};
    for (double result : results)
    {
        EXPECT_TRUE(std::isnan(result) || result == std::numeric_limits<double>::infinity()) << result;
    }
    EXPECT_NO_THROW(log_bessel_k(ve::ad::variable<2>(300.0, 0), ve::ad::variable<2>(1e-5, 1)));
}

}
