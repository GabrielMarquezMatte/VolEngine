#include "diagnostics/sign_bias.hpp"
#include "diagnostics/diagnostics_series.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>

namespace
{

TEST(SignBias, MatchesRugarchRegression)
{
    auto series = ve::testing::diagnostics_series();
    auto result = ve::diagnostics::sign_bias(series.z, series.residuals);
    EXPECT_NEAR(result.sign.statistic, 0.80605064564506934, 1e-10 * 0.80605064564506934);
    EXPECT_NEAR(result.negative.statistic, 0.58096539518493029, 1e-10 * 0.58096539518493029);
    EXPECT_NEAR(result.positive.statistic, 0.44824700499741182, 1e-10 * 0.44824700499741182);
    EXPECT_NEAR(result.sign.p_value, 0.42565663652195873, 1e-10 * 0.42565663652195873);
    EXPECT_NEAR(result.negative.p_value, 0.56498640324614569, 1e-10 * 0.56498640324614569);
    EXPECT_NEAR(result.positive.p_value, 0.6567376139868587, 1e-10 * 0.6567376139868587);
    EXPECT_NEAR(result.joint.statistic, 0.66016010279206094, 1e-10 * 0.66016010279206094);
    EXPECT_NEAR(result.joint.p_value, 0.88253222361606198, 1e-10 * 0.88253222361606198);
}

TEST(SignBias, SingularDesignReturnsNaN)
{
    const std::array<double, 8> constant{1, 1, 1, 1, 1, 1, 1, 1};
    auto result = ve::diagnostics::sign_bias(constant, constant);
    EXPECT_TRUE(std::isnan(result.sign.statistic));
    EXPECT_TRUE(std::isnan(result.negative.statistic));
    EXPECT_TRUE(std::isnan(result.positive.statistic));
    EXPECT_TRUE(std::isnan(result.joint.statistic));
    EXPECT_TRUE(std::isnan(result.joint.p_value));
}

TEST(SignBias, NonFiniteStandardizedResidualsBecomeZero)
{
    auto series = ve::testing::diagnostics_series();
    series.z[3] = std::numeric_limits<double>::quiet_NaN();
    series.z[12] = std::numeric_limits<double>::infinity();
    auto missing = ve::diagnostics::sign_bias(series.z, series.residuals);
    series.z[3] = 0.0;
    series.z[12] = 0.0;
    auto zero = ve::diagnostics::sign_bias(series.z, series.residuals);
    EXPECT_DOUBLE_EQ(missing.sign.statistic, zero.sign.statistic);
    EXPECT_DOUBLE_EQ(missing.negative.statistic, zero.negative.statistic);
    EXPECT_DOUBLE_EQ(missing.positive.statistic, zero.positive.statistic);
    EXPECT_DOUBLE_EQ(missing.joint.statistic, zero.joint.statistic);
}

}
