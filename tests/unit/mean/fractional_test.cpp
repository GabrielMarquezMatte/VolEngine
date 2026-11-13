#include "ad/dual.hpp"
#include "mean/fractional.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace
{

// pi = (1, -0.3, -0.105, -0.0595) for d = 0.3.
TEST(Fractional, HandCase)
{
    std::vector<double> e{1.0, 2.0, -1.0, 0.5};
    ve::mean::fractional_difference<double>(e, 0.3);
    std::vector<double> expected{1.0, 1.7, -1.705, 0.5305};
    for (std::size_t t = 0; t < e.size(); ++t)
    {
        EXPECT_NEAR(e[t], expected[t], 1e-15) << "t = " << t;
    }
}

TEST(Fractional, ZeroIsIdentity)
{
    std::vector<double> e{0.12, -0.05, 0.31, 0.02, -0.22, 0.17};
    std::vector<double> before = e;
    ve::mean::fractional_difference<double>(e, 0.0);
    EXPECT_EQ(e, before);
}

// The direct sum, k ascending, one product and one sum per term.
template <class T>
std::vector<T> direct_sum(const std::vector<T>& e, const T& d)
{
    std::vector<T> pi(e.size(), T{1.0});
    for (std::size_t k = 1; k < e.size(); ++k)
    {
        pi[k] = pi[k - 1] * ((static_cast<double>(k) - 1.0) - d) / static_cast<double>(k);
    }
    std::vector<T> out(e.size());
    for (std::size_t t = 0; t < e.size(); ++t)
    {
        T sum = e[t];
        for (std::size_t k = 1; k <= t; ++k)
        {
            sum = sum + pi[k] * e[t - k];
        }
        out[t] = sum;
    }
    return out;
}

std::vector<double> wavy(std::size_t count)
{
    std::vector<double> e(count);
    for (std::size_t t = 0; t < count; ++t)
    {
        e[t] = std::sin(0.7 * static_cast<double>(t)) + 0.1 * std::cos(3.1 * static_cast<double>(t));
    }
    return e;
}

// Same terms in the same order as the direct sum, for lengths around the block size.
TEST(Fractional, DoubleMatchesDirectSumBitForBit)
{
    for (std::size_t count : {1u, 7u, 8u, 9u, 37u, 300u})
    {
        std::vector<double> e = wavy(count);
        std::vector<double> expected = direct_sum(e, 0.37);
        ve::mean::fractional_difference<double>(e, 0.37);
        EXPECT_EQ(e, expected) << "count " << count;
    }
}

// Lane 0 moves the residuals, lane 1 moves d, lane 2 moves both, lane 3 moves neither.
TEST(Fractional, DualMatchesDirectSum)
{
    using D = ve::ad::Dual<4>;
    for (std::size_t count : {1u, 9u, 37u, 300u})
    {
        std::vector<double> base = wavy(count);
        std::vector<D> e(count);
        for (std::size_t t = 0; t < count; ++t)
        {
            e[t] = D{base[t], {1.0 + 0.01 * static_cast<double>(t), 0.0, -0.5, 0.0}};
        }
        D d{0.37, {0.0, 1.0, 0.25, 0.0}};
        std::vector<D> expected = direct_sum(e, d);
        ve::mean::fractional_difference<D>(e, d);
        for (std::size_t t = 0; t < count; ++t)
        {
            EXPECT_EQ(e[t].value, expected[t].value) << "count " << count << " t " << t;
            for (std::size_t j = 0; j < 4; ++j)
            {
                EXPECT_NEAR(e[t].grad[j], expected[t].grad[j], 1e-12 * (1.0 + std::abs(expected[t].grad[j])))
                    << "count " << count << " t " << t << " lane " << j;
            }
            EXPECT_EQ(e[t].grad[3], 0.0);
        }
    }
}

}
