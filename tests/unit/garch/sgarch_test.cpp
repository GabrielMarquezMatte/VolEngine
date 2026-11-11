#include "garch/sgarch.hpp"

#include "ad/dual.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace
{

TEST(SGarch, HeldPresampleThenRecursionMatchesHand)
{
    std::vector<double> residuals{0.1, -0.2, 0.3};
    std::vector<double> alpha{0.1};
    std::vector<double> beta{0.8};
    std::vector<double> sigma2(3);
    double mean_square = 0.14 / 3.0;
    ve::garch::sgarch_variance<double>(residuals, {0.1, alpha, beta}, {mean_square, 1}, sigma2);
    EXPECT_NEAR(sigma2[0], mean_square, 1e-15);
    EXPECT_NEAR(sigma2[1], 0.1 + 0.1 * 0.01 + 0.8 * mean_square, 1e-15);
    EXPECT_NEAR(sigma2[2], 0.1 + 0.1 * 0.04 + 0.8 * sigma2[1], 1e-15);
}

TEST(SGarch, NegativeLagsReadThePresample)
{
    std::vector<double> residuals{0.1, -0.2, 0.3};
    std::vector<double> alpha{0.1, 0.05};
    std::vector<double> beta{0.7};
    std::vector<double> sigma2(3);
    double b = 0.2;
    ve::garch::sgarch_variance<double>(residuals, {0.1, alpha, beta}, {b, 0}, sigma2);
    EXPECT_NEAR(sigma2[0], 0.1 + 0.1 * b + 0.05 * b + 0.7 * b, 1e-15);
    EXPECT_NEAR(sigma2[1], 0.1 + 0.1 * 0.01 + 0.05 * b + 0.7 * sigma2[0], 1e-15);
    EXPECT_NEAR(sigma2[2], 0.1 + 0.1 * 0.04 + 0.05 * 0.01 + 0.7 * sigma2[1], 1e-15);
}

TEST(SGarch, HoldLongerThanTheSeriesFillsEveryVariance)
{
    std::vector<double> residuals{0.1, -0.2};
    std::vector<double> alpha{0.1, 0.1, 0.1};
    std::vector<double> beta{0.2, 0.2, 0.2};
    std::vector<double> sigma2(2);
    ve::garch::sgarch_variance<double>(residuals, {0.1, alpha, beta}, {0.3, 3}, sigma2);
    EXPECT_EQ(sigma2[0], 0.3);
    EXPECT_EQ(sigma2[1], 0.3);
}

TEST(SGarch, DualRecursionCarriesTheGradient)
{
    using D = ve::ad::Dual<3>;
    std::vector<D> residuals{ve::ad::constant<3>(0.1), ve::ad::constant<3>(-0.2)};
    std::vector<D> alpha{ve::ad::variable<3>(0.1, 1)};
    std::vector<D> beta{ve::ad::variable<3>(0.8, 2)};
    std::vector<D> sigma2(2);
    D b = ve::ad::constant<3>(0.2);
    ve::garch::sgarch_variance<D>(residuals, {ve::ad::variable<3>(0.1, 0), alpha, beta}, {b, 0}, sigma2);
    EXPECT_NEAR(sigma2[1].grad[0], 1.0 + 0.8, 1e-15);
    EXPECT_NEAR(sigma2[1].grad[1], 0.01 + 0.8 * 0.2, 1e-15);
    EXPECT_NEAR(sigma2[1].grad[2], sigma2[0].value + 0.8 * 0.2, 1e-15);
}

}
