#include "garch/egarch.hpp"

#include <cmath>
#include <limits>
#include <gtest/gtest.h>
#include <numbers>
#include <vector>

namespace
{

const double kNormalAbsMoment = std::sqrt(2.0 / std::numbers::pi);

double shock(double z, double alpha, double gamma)
{
    return alpha * z + gamma * (std::abs(z) - kNormalAbsMoment);
}

TEST(EGarch, HandCase)
{
    std::vector<double> residuals{0.1, -0.2, 0.3};
    std::vector<double> alpha{-0.05};
    std::vector<double> beta{0.9};
    std::vector<double> gamma{0.1};
    std::vector<double> sigma2(3);
    double mean_square = 0.14 / 3.0;
    ve::garch::egarch_variance<double>(residuals, {-0.1, alpha, beta, gamma, kNormalAbsMoment}, {mean_square, 1},
                                       sigma2);
    double z0 = 0.1 / std::sqrt(mean_square);
    double log_sigma2_1 = -0.1 + shock(z0, -0.05, 0.1) + 0.9 * std::log(mean_square);
    double z1 = -0.2 / std::sqrt(std::exp(log_sigma2_1));
    double log_sigma2_2 = -0.1 + shock(z1, -0.05, 0.1) + 0.9 * log_sigma2_1;
    EXPECT_NEAR(sigma2[0], mean_square, 1e-14);
    EXPECT_NEAR(std::log(sigma2[1]), log_sigma2_1, 1e-14);
    EXPECT_NEAR(std::log(sigma2[2]), log_sigma2_2, 1e-14);
}

TEST(EGarch, BackcastPresample)
{
    std::vector<double> residuals{0.1, -0.2, 0.3};
    std::vector<double> alpha{-0.05};
    std::vector<double> beta{0.9};
    std::vector<double> gamma{0.1};
    std::vector<double> sigma2(3);
    double b = 0.05;
    ve::garch::egarch_variance<double>(residuals, {-0.1, alpha, beta, gamma, kNormalAbsMoment}, {b, 0}, sigma2);
    double log_sigma2_0 = -0.1 + 0.9 * std::log(b);
    double z0 = 0.1 / std::sqrt(std::exp(log_sigma2_0));
    EXPECT_NEAR(std::log(sigma2[0]), log_sigma2_0, 1e-14);
    EXPECT_NEAR(std::log(sigma2[1]), -0.1 + shock(z0, -0.05, 0.1) + 0.9 * log_sigma2_0, 1e-14);
}

// arch drops a lag of |z| - sqrt(2/pi) before t = 0: |z| reads sqrt(2/pi) there, whatever E|z| is.
TEST(EGarch, BackcastPresampleReadsTheNormalAbsMoment)
{
    std::vector<double> residuals{0.1, -0.2, 0.3};
    std::vector<double> alpha{-0.05, 0.02};
    std::vector<double> beta{0.9};
    std::vector<double> gamma{0.1, 0.04};
    std::vector<double> sigma2(3);
    double b = 0.05;
    double abs_moment = 0.75;
    ve::garch::egarch_variance<double>(residuals, {-0.1, alpha, beta, gamma, abs_moment}, {b, 0}, sigma2);
    double gap = kNormalAbsMoment - abs_moment;
    double log_sigma2_0 = -0.1 + (0.1 + 0.04) * gap + 0.9 * std::log(b);
    double z0 = 0.1 / std::sqrt(std::exp(log_sigma2_0));
    double log_sigma2_1 = -0.1 - 0.05 * z0 + 0.1 * (std::abs(z0) - abs_moment) + 0.04 * gap + 0.9 * log_sigma2_0;
    EXPECT_NEAR(std::log(sigma2[0]), log_sigma2_0, 1e-14);
    EXPECT_NEAR(std::log(sigma2[1]), log_sigma2_1, 1e-14);
}

TEST(EGarch, InterceptShiftEntersEveryStepButTheHeldOnes)
{
    std::vector<double> residuals{0.1, -0.2, 0.3};
    std::vector<double> alpha{-0.05};
    std::vector<double> beta{0.9};
    std::vector<double> gamma{0.1};
    std::vector<double> shift{std::numeric_limits<double>::quiet_NaN(), 0.03, -0.02};
    std::vector<double> sigma2(3);
    double mean_square = 0.14 / 3.0;
    ve::garch::egarch_variance<double>(residuals, {-0.1, alpha, beta, gamma, kNormalAbsMoment, shift},
                                       {mean_square, 1}, sigma2);
    double z0 = 0.1 / std::sqrt(mean_square);
    double log_sigma2_1 = -0.1 + 0.03 + shock(z0, -0.05, 0.1) + 0.9 * std::log(mean_square);
    double z1 = -0.2 / std::sqrt(std::exp(log_sigma2_1));
    double log_sigma2_2 = -0.1 - 0.02 + shock(z1, -0.05, 0.1) + 0.9 * log_sigma2_1;
    EXPECT_NEAR(sigma2[0], mean_square, 1e-15);
    EXPECT_NEAR(std::log(sigma2[1]), log_sigma2_1, 1e-14);
    EXPECT_NEAR(std::log(sigma2[2]), log_sigma2_2, 1e-14);
}

}
