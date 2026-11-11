#include "garch/power_garch.hpp"

#include <cmath>
#include <gtest/gtest.h>
#include <vector>

namespace
{

// sigma^power (|z - eta2| - eta1 (z - eta2))^power for z = residual / sigma.
double shock(double residual, double sigma_power, double eta1, double eta2)
{
    double z = residual / std::pow(sigma_power, 1.0 / 1.5);
    double shift = z - eta2;
    return sigma_power * std::pow(std::abs(shift) - eta1 * shift, 1.5);
}

TEST(PowerGarch, ApArchHandCase)
{
    std::vector<double> residuals{0.1, -0.2, 0.3};
    std::vector<double> alpha{0.1};
    std::vector<double> beta{0.8};
    std::vector<double> gamma{0.3};
    std::vector<double> moments{1.0};
    std::vector<double> sigma2(3);
    double mean_square = 0.14 / 3.0;
    ve::garch::power_garch_variance<double>(residuals, {0.1, alpha, beta, gamma, {}, 1.5, moments},
                                            {mean_square, 1}, sigma2);
    double sigma_power_1 = 0.1 + 0.1 * std::pow(0.1 - 0.3 * 0.1, 1.5) + 0.8 * std::pow(mean_square, 0.75);
    double sigma_power_2 = 0.1 + 0.1 * std::pow(0.2 + 0.3 * 0.2, 1.5) + 0.8 * sigma_power_1;
    EXPECT_NEAR(sigma2[0], mean_square, 1e-14);
    EXPECT_NEAR(sigma2[1], std::pow(sigma_power_1, 2.0 / 1.5), 1e-14);
    EXPECT_NEAR(sigma2[2], std::pow(sigma_power_2, 2.0 / 1.5), 1e-14);
}

TEST(PowerGarch, AllGarchHandCase)
{
    std::vector<double> residuals{0.1, -0.2, 0.3};
    std::vector<double> alpha{0.1};
    std::vector<double> beta{0.8};
    std::vector<double> eta1{0.3};
    std::vector<double> eta2{0.2};
    std::vector<double> moments{1.0};
    std::vector<double> sigma2(3);
    double mean_square = 0.14 / 3.0;
    ve::garch::power_garch_variance<double>(residuals, {0.1, alpha, beta, eta1, eta2, 1.5, moments},
                                            {mean_square, 1}, sigma2);
    double sigma_power_0 = std::pow(mean_square, 0.75);
    double z0 = 0.1 / std::sqrt(mean_square);
    double asymmetry = std::abs(z0 - 0.2) - 0.3 * (z0 - 0.2);
    double sigma_power_1 = 0.1 + 0.1 * sigma_power_0 * std::pow(asymmetry, 1.5) + 0.8 * sigma_power_0;
    double sigma_power_2 = 0.1 + 0.1 * shock(-0.2, sigma_power_1, 0.3, 0.2) + 0.8 * sigma_power_1;
    EXPECT_NEAR(std::pow(sigma2[1], 0.75), sigma_power_1, 1e-14);
    EXPECT_NEAR(std::pow(sigma2[2], 0.75), sigma_power_2, 1e-14);
}

// z_0 = eta2 exactly: the smoothed |z - eta2| is sqrt(1e-6) there, not 0.
TEST(PowerGarch, SmoothedAbsoluteValue)
{
    double mean_square = 0.04;
    double sigma_power_0 = std::pow(mean_square, 0.75);
    std::vector<double> residuals{0.2 * std::sqrt(mean_square), -0.2, 0.3};
    std::vector<double> alpha{0.1};
    std::vector<double> beta{0.8};
    std::vector<double> eta1{0.3};
    std::vector<double> eta2{0.2};
    std::vector<double> moments{1.0};
    std::vector<double> sigma2(3);
    ve::garch::power_garch_variance<double>(residuals, {0.1, alpha, beta, eta1, eta2, 1.5, moments, 1e-6},
                                            {mean_square, 1}, sigma2);
    double sigma_power_1 = 0.1 + 0.1 * sigma_power_0 * std::pow(1e-3, 1.5) + 0.8 * sigma_power_0;
    double z1 = -0.2 / std::pow(sigma_power_1, 1.0 / 1.5);
    double asymmetry = std::sqrt((z1 - 0.2) * (z1 - 0.2) + 1e-6) - 0.3 * (z1 - 0.2);
    double sigma_power_2 = 0.1 + 0.1 * sigma_power_1 * std::pow(asymmetry, 1.5) + 0.8 * sigma_power_1;
    EXPECT_NEAR(std::pow(sigma2[1], 0.75), sigma_power_1, 1e-14);
    EXPECT_NEAR(std::pow(sigma2[2], 0.75), sigma_power_2, 1e-14);
}

TEST(PowerGarch, BackcastPresample)
{
    std::vector<double> residuals{-0.1, 0.2, 0.3};
    std::vector<double> alpha{0.05, 0.03};
    std::vector<double> beta{0.8};
    std::vector<double> eta1{0.3, -0.2};
    std::vector<double> eta2{0.2, 0.1};
    std::vector<double> moments{1.1, 0.9};
    std::vector<double> sigma2(3);
    double b = 0.05;
    ve::garch::power_garch_variance<double>(residuals, {0.1, alpha, beta, eta1, eta2, 1.5, moments}, {b, 0},
                                            sigma2);
    double presample_power = std::pow(b, 0.75);
    double sigma_power_0 = 0.1 + (0.05 * 1.1 + 0.03 * 0.9 + 0.8) * presample_power;
    double sigma_power_1 =
        0.1 + 0.05 * shock(-0.1, sigma_power_0, 0.3, 0.2) + 0.03 * 0.9 * presample_power + 0.8 * sigma_power_0;
    EXPECT_NEAR(std::pow(sigma2[0], 0.75), sigma_power_0, 1e-14);
    EXPECT_NEAR(std::pow(sigma2[1], 0.75), sigma_power_1, 1e-14);
}

}
