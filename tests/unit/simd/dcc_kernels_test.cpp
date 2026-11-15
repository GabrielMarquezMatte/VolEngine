#include "simd/dcc_kernels.hpp"

#include "simd/fixtures.hpp"
#include "simd/targets.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace
{

constexpr double kEpsilon = 2.220446049250313e-16;
// The kernels multiply, divide and add in the order of the scalar code, none of them fused, so they agree to the last
// bit on every target; this leaves one ulp of slack.
constexpr double kExactTolerance = kEpsilon;

const std::size_t kDimensions[] = {2, 3, 5, 21};

std::size_t packed_count(std::size_t m)
{
    return m * (m + 1) / 2;
}

std::size_t packed_index(std::size_t i, std::size_t j)
{
    return i * (i + 1) / 2 + j;
}

TEST(DccKernels, MultiplyAddMatchesTheScalarLoopOnEveryTarget)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(3);
        for (const std::size_t m : kDimensions)
        {
            const std::size_t count = packed_count(m);
            const auto weights = stream.vector(count);
            const auto x = stream.vector(count);
            const auto start = stream.vector(count);
            auto acc = start;
            ve::simd::multiply_add(acc, weights, x);
            for (std::size_t k = 0; k < count; ++k)
            {
                const double expected = start[k] + weights[k] * x[k];
                EXPECT_NEAR(acc[k], expected, kExactTolerance * std::fabs(expected)) << "m = " << m << " k = " << k;
            }
        }
    }
}

TEST(DccKernels, AddOuterMatchesTheScalarProductsOnEveryTarget)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(5);
        for (const std::size_t m : kDimensions)
        {
            const auto v = stream.vector(m);
            const auto weights = stream.vector(packed_count(m));
            const auto start = stream.vector(packed_count(m));
            auto acc = start;
            ve::simd::add_outer(acc, weights, v);
            for (std::size_t i = 0; i < m; ++i)
            {
                for (std::size_t j = 0; j <= i; ++j)
                {
                    const std::size_t k = packed_index(i, j);
                    const double expected = start[k] + weights[k] * (v[i] * v[j]);
                    EXPECT_NEAR(acc[k], expected, kExactTolerance * std::fabs(expected))
                        << "m = " << m << " (" << i << ", " << j << ")";
                }
            }
        }
    }
}

TEST(DccKernels, CorrelationFromQMatchesTheScalarNormalisationOnEveryTarget)
{
    for (const auto target : ve::test::runnable_targets())
    {
        const ve::test::ForcedTarget forced(target);
        ve::test::Stream stream(9);
        for (const std::size_t m : kDimensions)
        {
            auto q = stream.vector(packed_count(m));
            for (std::size_t i = 0; i < m; ++i)
            {
                q[packed_index(i, i)] = 1.0 + 0.5 * std::fabs(q[packed_index(i, i)]);
            }
            std::vector<double> scale(m);
            std::vector<double> r(m * m, -7.0);
            ve::simd::correlation_from_q(q, m, scale, r);
            for (std::size_t i = 0; i < m; ++i)
            {
                EXPECT_EQ(r[i * m + i], 1.0);
                for (std::size_t j = 0; j < i; ++j)
                {
                    const double scale_i = 1.0 / std::sqrt(q[packed_index(i, i)]);
                    const double scale_j = 1.0 / std::sqrt(q[packed_index(j, j)]);
                    const double expected = q[packed_index(i, j)] * scale_i * scale_j;
                    EXPECT_NEAR(r[i * m + j], expected, kExactTolerance * std::fabs(expected))
                        << "m = " << m << " (" << i << ", " << j << ")";
                    EXPECT_EQ(r[j * m + i], r[i * m + j]) << "the upper triangle mirrors the lower, exactly";
                }
            }
        }
    }
}

}
