#include "mean/fractional.hpp"

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

}
