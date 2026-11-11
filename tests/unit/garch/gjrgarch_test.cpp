#include "garch/gjrgarch.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace
{

TEST(GjrGarch, HandCase)
{
    std::vector<double> residuals{0.1, -0.2, 0.3};
    std::vector<double> alpha{0.05};
    std::vector<double> beta{0.8};
    std::vector<double> gamma{0.1};
    std::vector<double> sigma2(3);
    double mean_square = 0.14 / 3.0;
    ve::garch::gjrgarch_variance<double>(residuals, {0.1, alpha, beta, gamma}, {mean_square, 1}, sigma2);
    EXPECT_NEAR(sigma2[0], mean_square, 1e-14);
    EXPECT_NEAR(sigma2[1], 0.1 + 0.05 * 0.01 + 0.8 * mean_square, 1e-14);
    EXPECT_NEAR(sigma2[2], 0.1 + (0.05 + 0.1) * 0.04 + 0.8 * sigma2[1], 1e-14);
}

TEST(GjrGarch, BackcastPresample)
{
    std::vector<double> residuals{-0.1, 0.2, 0.3};
    std::vector<double> alpha{0.05, 0.03};
    std::vector<double> beta{0.8};
    std::vector<double> gamma{0.1, 0.04};
    std::vector<double> sigma2(3);
    double b = 0.05;
    ve::garch::gjrgarch_variance<double>(residuals, {0.1, alpha, beta, gamma}, {b, 0}, sigma2);
    EXPECT_NEAR(sigma2[0], 0.1 + (0.05 + 0.03 + 0.5 * (0.1 + 0.04) + 0.8) * b, 1e-14);
    double expected = 0.1 + (0.05 + 0.1) * 0.01 + (0.03 + 0.5 * 0.04) * b + 0.8 * sigma2[0];
    EXPECT_NEAR(sigma2[1], expected, 1e-14);
}

}
