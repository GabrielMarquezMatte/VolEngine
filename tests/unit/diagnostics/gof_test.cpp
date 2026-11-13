#include "diagnostics/gof.hpp"
#include "diagnostics/tails.hpp"
#include "diagnostics/diagnostics_series.hpp"

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <vector>

namespace
{

TEST(Gof, MatchesRugarch)
{
    auto series = ve::testing::diagnostics_series();
    auto student = ve::diagnostics::pearson_gof(series.z, ve::Dist::Std, {0.0, 6.0, 0.0}, 4);
    EXPECT_DOUBLE_EQ(student.statistic, 3.0);
    EXPECT_NEAR(student.p_value, 0.39162517627108911, 1e-12 * 0.39162517627108911);
    auto skewed = ve::diagnostics::pearson_gof(series.z, ve::Dist::Sstd, {1.3, 6.0, 0.0}, 5);
    EXPECT_DOUBLE_EQ(skewed.statistic, 4.25);
    EXPECT_NEAR(skewed.p_value, 0.3732280258334989, 1e-12 * 0.3732280258334989);
    auto normal = ve::diagnostics::pearson_gof(series.z, ve::Dist::Norm, {}, 8);
    EXPECT_DOUBLE_EQ(normal.statistic, 12.0);
    EXPECT_NEAR(normal.p_value, 0.10055886850835885, 1e-12 * 0.10055886850835885);
}

TEST(Gof, DropsEmptyTopGroupsLikeRugarch)
{
    std::vector<double> z;
    for (int i = 0; i < 10; ++i)
    {
        z.push_back(-3.0 + static_cast<double>(i) / 9.0);
    }
    auto result = ve::diagnostics::pearson_gof(z, ve::Dist::Norm, {}, 4);
    EXPECT_DOUBLE_EQ(result.statistic, 22.5);
    EXPECT_DOUBLE_EQ(result.p_value, ve::diagnostics::chi_squared_upper(22.5, 3.0));
}

TEST(Gof, PreservesEmptyLowerAndMiddleGroups)
{
    const std::array<double, 4> z{-10.0, -10.0, 10.0, 10.0};
    auto result = ve::diagnostics::pearson_gof(z, ve::Dist::Norm, {}, 4);
    EXPECT_DOUBLE_EQ(result.statistic, 4.0);
    const std::array<double, 4> positive{10.0, 10.0, 10.0, 10.0};
    auto upper = ve::diagnostics::pearson_gof(positive, ve::Dist::Norm, {}, 4);
    EXPECT_DOUBLE_EQ(upper.statistic, 12.0);
}

TEST(Gof, NonFiniteStandardizedResidualsBecomeZero)
{
    auto series = ve::testing::diagnostics_series();
    series.z[3] = std::numeric_limits<double>::quiet_NaN();
    series.z[12] = std::numeric_limits<double>::infinity();
    auto missing = ve::diagnostics::pearson_gof(series.z, ve::Dist::Norm, {}, 8);
    series.z[3] = 0.0;
    series.z[12] = 0.0;
    auto zero = ve::diagnostics::pearson_gof(series.z, ve::Dist::Norm, {}, 8);
    EXPECT_DOUBLE_EQ(missing.statistic, zero.statistic);
    EXPECT_DOUBLE_EQ(missing.p_value, zero.p_value);
}

}
