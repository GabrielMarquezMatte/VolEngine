#include "garch/csgarch.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace
{

TEST(CsGarch, HandCase)
{
    std::vector<double> squared_residuals{0.01, 0.04, 0.09};
    std::vector<double> alpha{0.05};
    std::vector<double> beta{0.8};
    std::vector<double> sigma2(3);
    std::vector<double> permanent(3);
    double mean_square = 0.14 / 3.0;
    ve::garch::csgarch_variance<double>(squared_residuals, {0.01, alpha, beta, 0.95, 0.03}, {mean_square, 1},
                                        sigma2, permanent);
    double q0 = 0.01 / (1.0 - 0.95);
    double sigma2_0 = mean_square + q0;
    double q1 = 0.01 + 0.95 * q0 + 0.03 * (0.01 - sigma2_0);
    double sigma2_1 = q1 + 0.05 * (0.01 - q0) + 0.8 * (sigma2_0 - q0);
    double q2 = 0.01 + 0.95 * q1 + 0.03 * (0.04 - sigma2_1);
    double sigma2_2 = q2 + 0.05 * (0.04 - q1) + 0.8 * (sigma2_1 - q1);
    EXPECT_NEAR(sigma2[0], sigma2_0, 1e-14);
    EXPECT_NEAR(permanent[0], q0, 1e-14);
    EXPECT_NEAR(sigma2[1], sigma2_1, 1e-14);
    EXPECT_NEAR(sigma2[2], sigma2_2, 1e-14);
}

TEST(CsGarch, BackcastPresample)
{
    std::vector<double> squared_residuals{0.01, 0.04, 0.09};
    std::vector<double> alpha{0.05};
    std::vector<double> beta{0.8};
    std::vector<double> sigma2(3);
    std::vector<double> permanent(3);
    double b = 0.05;
    ve::garch::csgarch_variance<double>(squared_residuals, {0.01, alpha, beta, 0.95, 0.03}, {b, 0}, sigma2,
                                        permanent);
    double q0 = 0.01 + 0.95 * b;
    double q1 = 0.01 + 0.95 * q0 + 0.03 * (0.01 - q0);
    EXPECT_NEAR(sigma2[0], q0, 1e-14);
    EXPECT_NEAR(sigma2[1], q1 + 0.05 * (0.01 - q0) + 0.8 * (q0 - q0), 1e-14);
}

}
