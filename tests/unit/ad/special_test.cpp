#include "ad/dual.hpp"
#include "ad/special.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numbers>
#include <thread>
#include <vector>

namespace
{

TEST(Special, DigammaKnownValues)
{
    EXPECT_NEAR(ve::ad::digamma(1.0), -0.57721566490153286, 1e-14);
    EXPECT_NEAR(ve::ad::digamma(0.5), -1.9635100260214235, 1e-13);
    EXPECT_NEAR(ve::ad::digamma(10.0), 2.2517525890667211, 1e-14);
    EXPECT_NEAR(ve::ad::digamma(1e-3), -1000.5755719318103, 1e-9);
}

TEST(Special, LogGammaMatchesKnownValues)
{
    EXPECT_NEAR(ve::ad::log_gamma(0.5), 0.5723649429247001, 1e-15 * 0.5723649429247001);
    EXPECT_EQ(ve::ad::log_gamma(1.0), 0.0);
    EXPECT_NEAR(ve::ad::log_gamma(4.0), std::log(6.0), 1e-15 * std::log(6.0));
    EXPECT_NEAR(ve::ad::log_gamma(100.0), 359.1342053695754, 1e-15 * 359.1342053695754);
}

// std::lgamma writes the global signgam, a data race between the pool's threads; on MinGW the pool threads also run
// the x87 unit at another precision than the main thread.
TEST(Special, LogGammaIsSameOnEveryThread)
{
    std::vector<double> points;
    for (int i = 0; i < 1000; ++i)
    {
        points.push_back(0.1 + 199.9 * i / 999.0);
    }
    std::vector<std::uint64_t> expected;
    for (const double x : points)
    {
        expected.push_back(std::bit_cast<std::uint64_t>(ve::ad::log_gamma(x)));
    }
    std::vector<std::vector<std::uint64_t>> results(8);
    {
        std::vector<std::jthread> threads;
        for (std::vector<std::uint64_t>& result : results)
        {
            threads.emplace_back(
                [&points, &result]
                {
                    for (const double x : points)
                    {
                        result.push_back(std::bit_cast<std::uint64_t>(ve::ad::log_gamma(x)));
                    }
                });
        }
    }
    for (const std::vector<std::uint64_t>& result : results)
    {
        EXPECT_EQ(result, expected);
    }
}

TEST(Special, LgammaDualDerivative)
{
    for (double x : {0.3, 1.0, 4.5, 50.0})
    {
        auto f = ve::ad::log_gamma(ve::ad::variable<1>(x, 0));
        EXPECT_NEAR(f.value, std::lgamma(x), 1e-15 * std::abs(std::lgamma(x)) + 1e-15);
        EXPECT_NEAR(f.grad[0], ve::ad::digamma(x), 1e-14);
    }
}

}
