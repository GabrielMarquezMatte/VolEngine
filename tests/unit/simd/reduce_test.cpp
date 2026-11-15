#include "simd/reduce.hpp"

#include "simd/targets.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace
{

constexpr double kMaxUlps = 4.0;

std::vector<double> series(std::size_t n)
{
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        x[i] = std::sin(0.37 * static_cast<double>(i)) * (1.0 + 1e-3 * static_cast<double>(i % 17));
    }
    return x;
}

long double scalar_sum(const std::vector<double>& x)
{
    long double sum = 0.0L;
    for (const double v : x)
    {
        sum += v;
    }
    return sum;
}

// The sum of absolute values bounds the rounding error of any summation order, so the error is measured against it.
double abs_sum(const std::vector<double>& x)
{
    double sum = 0.0;
    for (const double v : x)
    {
        sum += std::fabs(v);
    }
    return sum;
}

TEST(SimdSum, MatchesScalarOnEveryTarget)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        for (const std::size_t n : {0U, 1U, 3U, 7U, 21U, 5000U})
        {
            const auto x = series(n);
            const double expected = static_cast<double>(scalar_sum(x));
            const double bound = kMaxUlps * std::numeric_limits<double>::epsilon() * abs_sum(x);
            EXPECT_NEAR(ve::simd::sum(x), expected, bound) << "n = " << n;
        }
    }
}

// GCC bug 54412: MinGW-w64 may not realign the stack for 32-byte vector spills. The caller's frame size moves the
// stack pointer by 8 bytes at a time, so every alignment of the kernel's entry gets tried.
double sum_at_depth(const std::vector<double>& x, unsigned depth)
{
    volatile char pad[8] = {};
    if (depth == 0)
    {
        return ve::simd::sum(x);
    }
    return sum_at_depth(x, depth - 1) + static_cast<double>(pad[0]);
}

TEST(SimdSum, SurvivesAMillionCallsOnEveryStackAlignment)
{
    const auto x = series(21);
    const double expected = static_cast<double>(scalar_sum(x));
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        double worst = 0.0;
        for (unsigned call = 0; call < 1'000'000U; ++call)
        {
            worst = std::max(worst, std::fabs(sum_at_depth(x, call % 5U) - expected));
        }
        EXPECT_LE(worst, kMaxUlps * std::numeric_limits<double>::epsilon() * abs_sum(x));
    }
}

TEST(SimdDot, MatchesScalarOnEveryTarget)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        for (const std::size_t n : {0U, 1U, 3U, 7U, 21U, 231U, 5000U})
        {
            const auto x = series(n);
            std::vector<double> y(n);
            long double expected = 0.0L;
            double bound = 0.0;
            for (std::size_t i = 0; i < n; ++i)
            {
                y[i] = std::cos(0.11 * static_cast<double>(i)) + 0.5;
                expected += static_cast<long double>(x[i]) * y[i];
                bound += std::fabs(x[i] * y[i]);
            }
            EXPECT_NEAR(ve::simd::dot(x, y), static_cast<double>(expected),
                        kMaxUlps * std::numeric_limits<double>::epsilon() * bound)
                << "n = " << n;
        }
    }
}

}
