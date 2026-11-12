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

TEST(Bessel, LargeOrdersAboveTheSeriesStartMatchScipy)
{
    struct Reference
    {
        double order;
        double x;
        double expected;
    };
    // log(scipy.special.kve(order, x)) - x
    const Reference references[] = {
        {150.0, 700.0, -687.0503734075295}, {200.0, 700.0, -674.6878578440724}, {300.0, 700.0, -639.7401042737811},
        {150.0, 900.0, -890.711083541808},  {200.0, 900.0, -881.0554829819462}, {300.0, 900.0, -853.650108907324}};
    for (const Reference& r : references)
    {
        expect_relative(log_bessel_k(r.order, r.x), r.expected, 1e-12);
    }
    double jump = log_bessel_k(200.0, 700.0 + 1e-6) - log_bessel_k(200.0, 700.0 - 1e-6);
    EXPECT_NEAR(jump, 2e-6 * log_bessel_k(200.0, ve::ad::variable<1>(700.0, 0)).grad[0], 1e-11);
}

TEST(Bessel, OrderDerivativeMatchesClosedFormAtHalf)
{
    // d log K_v / dv at v = 1/2 is e^(2x) E1(2x); references from scipy.special.exp1.
    const std::array<std::array<double, 2>, 2> references = {{{0.2, 1.0478280084560063}, {3.0, 0.1452676292338869}}};
    for (const auto& [x, expected] : references)
    {
        auto value = log_bessel_k(ve::ad::variable<2>(0.5, 0), ve::ad::variable<2>(x, 1));
        expect_relative(value.grad[0], expected, 1e-8);
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

// K_v = K_(v-2) + 2 (v - 1) / x K_(v-1), from orders far from 5/2.
double five_halves_recurrence(double order, double x)
{
    double below = std::exp(log_bessel_k(order - 2.0, x));
    double above = std::exp(log_bessel_k(order - 1.0, x));
    return std::log(below + 2.0 * (order - 1.0) / x * above);
}

// Boost takes K_v for v within ~7e-6 x of 5/2 (its band in long double) from a truncated Hankel series,
// exact at 5/2 only; log_bessel_k recurs within ~4.9e-5 x of it.
TEST(Bessel, SmoothAroundFiveHalves)
{
    for (double x : {0.7, 2.0, 10.0, 100.0, 650.0})
    {
        for (double gap : {0.0, 1e-6, -1e-5, 1e-5, 3e-5, 6e-5, 1e-3})
        {
            double order = 2.5 + gap;
            expect_relative(log_bessel_k(order, x), five_halves_recurrence(order, x), 1e-14);
        }
        double step = 2.5e-5;
        double slope =
            (five_halves_recurrence(2.5 + step, x) - five_halves_recurrence(2.5 - step, x)) / (2.0 * step);
        auto dual = log_bessel_k(ve::ad::variable<1>(2.5, 0), ve::ad::Dual<1>{x});
        EXPECT_NEAR(dual.grad[0], slope, 1e-9 * std::max(1.0, std::fabs(slope))) << x;
    }
}

TEST(Bessel, NeverThrows)
{
    constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
    constexpr double kInf = std::numeric_limits<double>::infinity();
    double results[] = {log_bessel_k(50.0, 1e-300), log_bessel_k(300.0, 1e-5), log_bessel_k(0.0, 0.0),
                        log_bessel_k(1.0, -1.0),    log_bessel_k(kNan, 1.0),   log_bessel_k(kInf, 1.0),
                        log_bessel_k(1e300, 1.0),   log_bessel_k(1.0, kNan),   log_bessel_k(kNan, 800.0),
                        log_bessel_k(1e300, kNan), log_bessel_k(1e300, -1.0)};
    for (double result : results)
    {
        EXPECT_TRUE(std::isnan(result) || result == std::numeric_limits<double>::infinity()) << result;
    }
    EXPECT_TRUE(std::isnan(log_bessel_k(1e300, kNan)));
    EXPECT_TRUE(std::isnan(log_bessel_k(1e300, -1.0)));
    EXPECT_NO_THROW(log_bessel_k(ve::ad::variable<2>(300.0, 0), ve::ad::variable<2>(1e-5, 1)));
}

}
