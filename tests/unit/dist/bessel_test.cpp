#include "ad/dual.hpp"
#include "dist/bessel.hpp"
#include "x87_precision.hpp"

#include <algorithm>
#include <array>
#include <boost/math/special_functions/bessel.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <vector>

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
        expect_relative(value.grad[0], expected, 1e-13);
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
            EXPECT_NEAR(fixed.grad[1], dual.grad[1], 1e-13 * std::max(1.0, std::fabs(dual.grad[1])));
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

// log K_|order|(x) in long double, the reference of the slope tests; Boost's K is even in the order.
// The references below are Richardson differences in long double (64 bits of mantissa on x86 GCC). Where long double is
// double (MSVC, Apple Silicon) their rounding, eps |log K| / 1e-4, is as large as the tolerances, so the tests skip.
constexpr bool kHasExtendedPrecision = std::numeric_limits<long double>::digits > std::numeric_limits<double>::digits;

using IgnoreOverflow = boost::math::policies::policy<boost::math::policies::overflow_error<boost::math::policies::ignore_error>>;

long double reference_log_k(long double order, long double x)
{
    return std::log(boost::math::cyl_bessel_k(std::fabs(order), x, IgnoreOverflow()));
}

// Richardson on central differences of the long double reference, accurate to O(step^4).
long double reference_order_slope(double order, double x, long double step)
{
    auto f = [&](long double v) { return reference_log_k(v, x); };
    long double fine = (f(order + step) - f(order - step)) / (2.0L * step);
    long double coarse = (f(order + 2.0L * step) - f(order - 2.0L * step)) / (4.0L * step);
    return (4.0L * fine - coarse) / 3.0L;
}

// d log K_v / dx = -K_(v-1) / K_v - v / x in long double.
long double reference_x_slope(double order, double x)
{
    long double v = std::fabs(static_cast<long double>(order));
    return -std::exp(reference_log_k(v - 1.0L, x) - reference_log_k(v, x)) - v / x;
}

void expect_near_reference(double actual, long double reference, double tolerance, const char* what, double order,
                           double x)
{
    EXPECT_NEAR(actual, static_cast<double>(reference), tolerance * std::max(1.0, std::fabs(static_cast<double>(reference))))
        << what << " at order " << order << ", x " << x;
}

// Orders that skip 5/2, where Boost's long double K switches to a truncated Hankel series that is neither accurate
// nor smooth in v (see SmoothAroundFiveHalves) and so is no reference.
TEST(Bessel, SlopesMatchLongDoubleReference)
{
    if (!kHasExtendedPrecision)
    {
        GTEST_SKIP() << "long double is double here: the reference is not more precise than the slopes";
    }
    for (double order : {-3.3, -1.5, -0.5, 0.0, 0.1, 0.4999, 0.5, 1.0, 2.4, 3.7, 10.0, 50.0, 200.0})
    {
        for (double x : {0.01, 0.1, 1.0, 1.78, 10.0, 31.6, 300.0, 699.0})
        {
            if (!std::isfinite(static_cast<double>(reference_log_k(order, x))))
            {
                continue;
            }
            ve::dist::detail::LogBesselSlopes slopes = ve::dist::detail::log_bessel_k_slopes(order, x);
            expect_near_reference(slopes.order, reference_order_slope(order, x, 1e-4L), 1e-12, "order slope", order, x);
            expect_near_reference(slopes.x, reference_x_slope(order, x), 1e-12, "x slope", order, x);
        }
    }
}

TEST(Bessel, OrderSlopeAtSmallArgument)
{
    if (!kHasExtendedPrecision)
    {
        GTEST_SKIP() << "long double is double here: the reference is not more precise than the slopes";
    }
    for (double x : {1e-8, 1e-6, 1e-4})
    {
        ve::dist::detail::LogBesselSlopes slopes = ve::dist::detail::log_bessel_k_slopes(0.1, x);
        expect_near_reference(slopes.order, reference_order_slope(0.1, x, 1e-6L), 1e-11, "order slope", 0.1, x);
    }
}

TEST(Bessel, OrderSlopeIsOddInTheOrder)
{
    for (double x : {0.3, 5.0})
    {
        EXPECT_EQ(ve::dist::detail::log_bessel_k_slopes(0.0, x).order, 0.0);
        double above = ve::dist::detail::log_bessel_k_slopes(1e-9, x).order;
        double below = ve::dist::detail::log_bessel_k_slopes(-1e-9, x).order;
        EXPECT_EQ(above, -below);
        EXPECT_LT(std::fabs(above), 1e-6);
        EXPECT_EQ(ve::dist::detail::log_bessel_k_slopes(-1.5, x).order, -ve::dist::detail::log_bessel_k_slopes(1.5, x).order);
    }
}

// At x >= 700 the value comes from the Hankel series with a recurrence; the slopes must agree with its differences.
TEST(Bessel, SlopesAtLargeArgumentMatchScaledDifferences)
{
    for (double x : {900.0, 5000.0})
    {
        for (double order : {0.3, 2.5, 40.0})
        {
            double step = 1e-4;
            double d_order = (ve::dist::log_bessel_k_scaled(order + step, x) -
                              ve::dist::log_bessel_k_scaled(order - step, x)) / (2.0 * step);
            double d_x = (ve::dist::log_bessel_k_scaled(order, x + step * x) -
                          ve::dist::log_bessel_k_scaled(order, x - step * x)) / (2.0 * step * x);
            auto scaled = ve::dist::log_bessel_k_scaled(ve::ad::variable<2>(order, 0), ve::ad::variable<2>(x, 1));
            EXPECT_NEAR(scaled.grad[0], d_order, 1e-7 * std::max(1.0, std::fabs(d_order))) << order << ", " << x;
            EXPECT_NEAR(scaled.grad[1], d_x, 1e-7 * std::max(1.0, std::fabs(d_x))) << order << ", " << x;
        }
    }
}

// The trapezoid grid has a step of 0.4 (x^2 + v^2)^(-1/4): below the resolution of exp at x ~ 1e25, where the sum
// stopped never ended. Past the domain of the trapezoid the slopes come from differences of the scaled log K.
TEST(Bessel, SlopesAtHugeArgumentTerminate)
{
    for (double order : {0.0, 0.7, 40.0})
    {
        for (double x : {1e6, 1e8, 1e25, 1e30, 1e150, 1e300})
        {
            const ve::dist::detail::LogBesselSlopes slopes = ve::dist::detail::log_bessel_k_slopes(order, x);
            // K_v(x) ~ sqrt(pi / 2x) e^-x: d log K / dx = -1 - 1/(2x) + O(v^2/x^2), d/dv ~ v / x.
            EXPECT_NEAR(slopes.x, -1.0, 1e-5) << order << ", " << x;
            EXPECT_NEAR(slopes.order, 0.0, 1e-3) << order << ", " << x;
            auto scaled = ve::dist::log_bessel_k_scaled(ve::ad::variable<2>(order, 0), ve::ad::variable<2>(x, 1));
            EXPECT_TRUE(std::isfinite(scaled.grad[0]) && std::isfinite(scaled.grad[1])) << order << ", " << x;
            EXPECT_NEAR(scaled.grad[1], 0.0, 1e-5) << order << ", " << x;
        }
    }
}

// Outside the trapezoid's domain the scaled overload takes its x slope from the ratio of the scaled logs, which has
// not the rounding of x that the unscaled K carries at a large x.
TEST(Bessel, ScaledSlopeOutsideTheTrapezoidDomain)
{
    const double order = 300.0;
    for (double x : {900.0, 5000.0, 1e5})
    {
        auto scaled = ve::dist::log_bessel_k_scaled(ve::ad::variable<2>(order, 0), ve::ad::variable<2>(x, 1));
        EXPECT_NEAR(scaled.grad[1], ve::dist::detail::log_bessel_k_scaled_slope(order, x), 1e-14) << x;
    }
}

TEST(Bessel, FreeOrderDualNeverThrows)
{
    constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
    constexpr double kInf = std::numeric_limits<double>::infinity();
    for (double order : {kNan, 1e5, 0.5})
    {
        for (double x : {kNan, -1.0, 0.0, kInf})
        {
            double plain = log_bessel_k(order, x);
            double scaled_plain = ve::dist::log_bessel_k_scaled(order, x);
            auto dual = ve::dist::log_bessel_k(ve::ad::variable<2>(order, 0), ve::ad::variable<2>(x, 1));
            auto scaled = ve::dist::log_bessel_k_scaled(ve::ad::variable<2>(order, 0), ve::ad::variable<2>(x, 1));
            EXPECT_TRUE(dual.value == plain || (std::isnan(dual.value) && std::isnan(plain))) << order << ", " << x;
            EXPECT_TRUE(scaled.value == scaled_plain || (std::isnan(scaled.value) && std::isnan(scaled_plain)))
                << order << ", " << x;
        }
    }
}

// Boost computes in long double by default, and on MinGW a pool thread runs the x87 unit at another precision
// than the Python main thread, so a fit could change with the thread it runs on.
TEST(Bessel, SameBitsAtAnyX87Precision)
{
    const auto values = [] {
        std::vector<double> out;
        for (double order : {-1.5, 0.3, 1.7, 2.5})
        {
            for (double x : {0.2, 3.0, 40.0})
            {
                out.push_back(log_bessel_k(order, x));
                out.push_back(ve::dist::log_bessel_k_scaled(order, x));
            }
        }
        return out;
    };
    const auto [extended, rounded] = ve::testing::at_both_x87_precisions(values);
    EXPECT_EQ(extended, rounded);
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
