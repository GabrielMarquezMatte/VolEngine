#include "linalg/lu.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace
{

constexpr std::size_t kSize = 4;

// Non-symmetric, with a zero leading entry, so that the factorization must pivot.
const std::array<double, kSize * kSize> kMatrix{
    0.0, 2.0, -1.0, 3.0,   //
    4.0, -1.0, 0.5, 1.0,   //
    -2.0, 3.0, 5.0, -4.0,  //
    1.0, 0.0, 2.0, 7.0,    //
};

TEST(Lu, SolvesANonSymmetricSystem)
{
    std::vector<double> factor(kMatrix.begin(), kMatrix.end());
    std::vector<std::size_t> pivots(kSize);
    ASSERT_TRUE(ve::linalg::lu_factor(factor, pivots, kSize));
    const std::array<double, kSize> expected{1.5, -2.0, 0.25, 3.0};
    std::vector<double> b(kSize, 0.0);
    for (std::size_t i = 0; i < kSize; ++i)
    {
        for (std::size_t j = 0; j < kSize; ++j)
        {
            b[i] += kMatrix[i * kSize + j] * expected[j];
        }
    }
    ve::linalg::lu_solve(factor, pivots, kSize, b);
    for (std::size_t i = 0; i < kSize; ++i)
    {
        EXPECT_NEAR(b[i], expected[i], 1e-14) << i;
    }
}

TEST(Lu, RejectsASingularMatrix)
{
    std::vector<double> singular(kMatrix.begin(), kMatrix.end());
    for (std::size_t j = 0; j < kSize; ++j)
    {
        singular[3 * kSize + j] = singular[j] + singular[kSize + j];
    }
    std::vector<std::size_t> pivots(kSize);
    EXPECT_FALSE(ve::linalg::lu_factor(singular, pivots, kSize));
}

}
