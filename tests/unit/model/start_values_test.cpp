#include "model/start_values.hpp"

#include "model/parameter_map.hpp"
#include "simulated_garch.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

namespace
{

using ve::Dist;
using ve::model::is_valid;
using ve::model::Layout;
using ve::model::start_values;

TEST(StartValues, FollowRule)
{
    std::vector<double> r = ve::testing::simulate_garch_series(3, 400);
    double mean = std::accumulate(r.begin(), r.end(), 0.0) / 400.0;
    double variance = 0.0;
    for (double value : r)
    {
        variance += (value - mean) * (value - mean);
    }
    variance /= 400.0;
    Layout layout{true, 1, 1, 2, 3, Dist::Std};
    std::vector<double> theta = start_values(r, layout);
    ASSERT_EQ(theta.size(), layout.count());
    EXPECT_DOUBLE_EQ(theta[layout.mu()], mean);
    EXPECT_EQ(theta[layout.ar_begin()], 0.0);
    EXPECT_EQ(theta[layout.ma_begin()], 0.0);
    EXPECT_NEAR(theta[layout.omega()], variance * 0.05, 1e-15);
    for (std::size_t i = 0; i < 2; ++i)
    {
        EXPECT_DOUBLE_EQ(theta[layout.alpha_begin() + i], 0.025);
    }
    for (std::size_t j = 0; j < 3; ++j)
    {
        EXPECT_DOUBLE_EQ(theta[layout.beta_begin() + j], 0.3);
    }
    EXPECT_EQ(theta[layout.shape()], 8.0);
    layout.dist = Dist::Ged;
    EXPECT_EQ(start_values(r, layout)[layout.shape()], 2.0);
}

TEST(StartValues, AreValidForEveryLayout)
{
    std::vector<double> r = ve::testing::simulate_garch_series(3, 400);
    for (int arch : {1, 2, 6})
    {
        for (int garch : {1, 3, 6})
        {
            for (Dist dist : {Dist::Norm, Dist::Std, Dist::Ged})
            {
                for (bool include_mean : {true, false})
                {
                    Layout layout{include_mean, 2, 1, arch, garch, dist};
                    EXPECT_TRUE(is_valid(start_values(r, layout), layout));
                }
            }
        }
    }
}

}
