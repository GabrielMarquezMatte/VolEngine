#include "diagnostics/portmanteau.hpp"
#include "diagnostics/tails.hpp"
#include "diagnostics/diagnostics_series.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>

namespace
{

using namespace ve::diagnostics;

TEST(Portmanteau, LjungBoxMatchesRugarch)
{
    auto series = ve::testing::diagnostics_series();
    auto plain = weighted_ljung_box(series.z, 5, 2, false);
    EXPECT_NEAR(plain.statistic, 41.287943416174535, 1e-12 * 41.287943416174535);
    EXPECT_GT(plain.p_value, 0.0);
    EXPECT_LT(plain.p_value, 1e-12);
    auto squared = weighted_ljung_box(series.z, 5, 0, true);
    EXPECT_NEAR(squared.statistic, 22.269312541169139, 1e-12 * 22.269312541169139);
    EXPECT_NEAR(squared.p_value, 4.5843285358948549e-06, 1e-9 * 4.5843285358948549e-06);
    auto ignored_df = weighted_ljung_box(series.z, 5, 3, true);
    EXPECT_DOUBLE_EQ(ignored_df.statistic, squared.statistic);
    EXPECT_DOUBLE_EQ(ignored_df.p_value, squared.p_value);
    auto first = weighted_ljung_box(series.z, 1, 0, false);
    EXPECT_NEAR(first.statistic, 20.243295814457056, 1e-12 * 20.243295814457056);
    EXPECT_NEAR(first.p_value, 6.8192010626022537e-06, 1e-9 * 6.8192010626022537e-06);
}

TEST(Portmanteau, ArchLmMatchesRugarch)
{
    auto series = ve::testing::diagnostics_series();
    auto first = weighted_arch_lm(series.residuals, series.sigma, 5, 2);
    EXPECT_NEAR(first.statistic, 23.14757005967083, 1e-12 * 23.14757005967083);
    EXPECT_NEAR(first.p_value, 3.4416950271953795e-06, 1e-9 * 3.4416950271953795e-06);
    auto second = weighted_arch_lm(series.residuals, series.sigma, 3, 1);
    EXPECT_NEAR(second.statistic, 16.096717784598464, 1e-12 * 16.096717784598464);
    EXPECT_NEAR(second.p_value, 8.2766564685865518e-05, 1e-9 * 8.2766564685865518e-05);
}

TEST(Portmanteau, NonFiniteAndConstant)
{
    auto series = ve::testing::diagnostics_series();
    series.z[2] = std::numeric_limits<double>::quiet_NaN();
    series.z[7] = std::numeric_limits<double>::infinity();
    auto missing = weighted_ljung_box(series.z, 5, 0, false);
    auto missing_squared = weighted_ljung_box(series.z, 5, 0, true);
    series.z[2] = 0.0;
    series.z[7] = 0.0;
    auto zero = weighted_ljung_box(series.z, 5, 0, false);
    auto zero_squared = weighted_ljung_box(series.z, 5, 0, true);
    EXPECT_DOUBLE_EQ(missing.statistic, zero.statistic);
    EXPECT_DOUBLE_EQ(missing.p_value, zero.p_value);
    EXPECT_DOUBLE_EQ(missing_squared.statistic, zero_squared.statistic);
    EXPECT_DOUBLE_EQ(missing_squared.p_value, zero_squared.p_value);
    const std::array<double, 8> constant{1, 1, 1, 1, 1, 1, 1, 1};
    auto result = weighted_ljung_box(constant, 5, 0, false);
    EXPECT_TRUE(std::isnan(result.statistic));
    EXPECT_TRUE(std::isnan(result.p_value));
    auto arch = weighted_arch_lm(constant, constant, 5, 2);
    EXPECT_TRUE(std::isnan(arch.statistic));
    EXPECT_TRUE(std::isnan(arch.p_value));
}

TEST(Portmanteau, Problems)
{
    EXPECT_TRUE(ljung_box_problem(40, 0, 0, false));
    EXPECT_TRUE(ljung_box_problem(40, 4, 2, false));
    EXPECT_TRUE(ljung_box_problem(5, 5, 0, false));
    EXPECT_TRUE(ljung_box_problem(40, 5, std::numeric_limits<std::size_t>::max(), false));
    EXPECT_FALSE(ljung_box_problem(40, 5, 2, false));
    EXPECT_FALSE(ljung_box_problem(40, 5, std::numeric_limits<std::size_t>::max(), true));
    EXPECT_FALSE(ljung_box_problem(40, 2, 1, false));
    EXPECT_FALSE(ljung_box_problem(40, 3, 1, false));
    EXPECT_FALSE(ljung_box_problem(40, 4, 1, false));
    EXPECT_TRUE(arch_lm_problem(40, 5, 0));
    EXPECT_TRUE(arch_lm_problem(40, 5, 5));
    EXPECT_TRUE(arch_lm_problem(5, 5, 2));
    EXPECT_FALSE(arch_lm_problem(40, 5, 2));
}

TEST(Portmanteau, UpperTailsRetainSmallProbabilities)
{
    EXPECT_GT(chi_squared_upper(100.0, 1.0), 0.0);
    EXPECT_LT(chi_squared_upper(100.0, 1.0), 1e-20);
    EXPECT_DOUBLE_EQ(student_two_sided(0.0, 10.0), 1.0);
    EXPECT_DOUBLE_EQ(student_two_sided(-2.0, 10.0), student_two_sided(2.0, 10.0));
}

}
