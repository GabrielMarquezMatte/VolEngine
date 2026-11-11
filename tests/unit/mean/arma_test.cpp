#include "mean/arma.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace
{

TEST(Arma, ResidualsHandCase)
{
    std::vector<double> r{0.2, 0.4, -0.1};
    std::vector<double> ar{0.5};
    std::vector<double> ma{0.3};
    std::vector<double> residuals(3);
    ve::mean::arma_residuals<double>(r, {0.1, ar, ma}, residuals);
    EXPECT_NEAR(residuals[0], 0.1, 1e-14);
    EXPECT_NEAR(residuals[1], 0.3 - 0.5 * 0.1 - 0.3 * 0.1, 1e-14);
    EXPECT_NEAR(residuals[1], 0.22, 1e-14);
    EXPECT_NEAR(residuals[2], -0.416, 1e-14);
}

TEST(Arma, ResidualsBeforeTheArOrderAreCenteredReturns)
{
    std::vector<double> r{1.0, 2.0, 3.0};
    std::vector<double> ar{0.5, 0.25};
    std::vector<double> ma{0.4};
    std::vector<double> residuals(3);
    ve::mean::arma_residuals<double>(r, {0.0, ar, ma}, residuals);
    EXPECT_EQ(residuals[0], 1.0);
    EXPECT_EQ(residuals[1], 2.0);
    EXPECT_EQ(residuals[2], 3.0 - 0.5 * 2.0 - 0.25 * 1.0 - 0.4 * 2.0);
}

TEST(Arma, MaLagsBeforeTheSampleAreZero)
{
    std::vector<double> r{1.0, 2.0, 3.0};
    std::vector<double> ma{0.5, 0.25};
    std::vector<double> residuals(3);
    ve::mean::arma_residuals<double>(r, {0.0, {}, ma}, residuals);
    EXPECT_EQ(residuals[0], 1.0);
    EXPECT_EQ(residuals[1], 1.5);
    EXPECT_EQ(residuals[2], 3.0 - 0.5 * 1.5 - 0.25 * 1.0);
}

TEST(Arma, WithoutLagsResidualsAreCenteredReturns)
{
    std::vector<double> r{0.3, -0.1};
    std::vector<double> residuals(2);
    ve::mean::arma_residuals<double>(r, {0.1, {}, {}}, residuals);
    EXPECT_EQ(residuals[0], 0.3 - 0.1);
    EXPECT_EQ(residuals[1], -0.1 - 0.1);
}

}
