#include "garch/sgarch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::garch::SGarchConfig;
using ve::garch::VarInit;

std::vector<double> simulate_garch_returns(std::size_t count)
{
    std::mt19937_64 engine(42);
    std::normal_distribution<double> shock(0.0, 1.0);
    double mu = 0.05;
    double omega = 0.1;
    double alpha = 0.1;
    double beta = 0.85;
    double sigma2 = omega / (1.0 - alpha - beta);
    std::vector<double> r(count);
    for (double& value : r)
    {
        double residual = std::sqrt(sigma2) * shock(engine);
        value = mu + residual;
        sigma2 = omega + alpha * residual * residual + beta * sigma2;
    }
    return r;
}

std::vector<double> evaluation_theta(const SGarchConfig& config)
{
    if (config.include_mean)
    {
        return {0.03, 0.08, 0.12, 0.8};
    }
    return {0.08, 0.12, 0.8};
}

double loglik_at(std::span<const double> r, std::span<const double> theta, const SGarchConfig& config)
{
    return ve::garch::sgarch_norm_loglik(r, theta, config, {});
}

std::vector<double> central_difference_gradient(std::span<const double> r, const std::vector<double>& theta,
                                                const SGarchConfig& config)
{
    std::vector<double> gradient(theta.size());
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        double step = 1e-6 * std::max(1.0, std::abs(theta[i]));
        std::vector<double> up = theta;
        std::vector<double> down = theta;
        up[i] += step;
        down[i] -= step;
        gradient[i] = (loglik_at(r, up, config) - loglik_at(r, down, config)) / (2.0 * step);
    }
    return gradient;
}

std::string config_name(const ::testing::TestParamInfo<SGarchConfig>& info)
{
    std::string init = info.param.var_init == VarInit::Sample ? "Sample" : "Backcast";
    std::string mean = info.param.include_mean ? "WithMean" : "NoMean";
    return init + mean;
}

class SGarchDerivatives : public ::testing::TestWithParam<SGarchConfig>
{
};

}

TEST(SGarch, SampleInitRecursionMatchesHand)
{
    std::vector<double> r{0.1, -0.2, 0.3};
    std::vector<double> s2(3);
    std::vector<double> th{0.0, 0.1, 0.1, 0.8};
    double ll = ve::garch::sgarch_norm_loglik(r, th, {}, {.sigma2 = s2});
    EXPECT_NEAR(s2[0], 0.14 / 3, 1e-12);
    EXPECT_NEAR(s2[1], 0.1 + 0.1 * 0.01 + 0.8 * (0.14 / 3), 1e-12);
    EXPECT_NEAR(s2[2], 0.1 + 0.1 * 0.04 + 0.8 * s2[1], 1e-12);
    double expect = 0;
    for (int t = 0; t < 3; ++t)
    {
        expect += -0.5 * (std::log(2 * std::numbers::pi) + std::log(s2[t]) + r[t] * r[t] / s2[t]);
    }
    EXPECT_NEAR(ll, expect, 1e-12);
}

TEST(SGarch, BackcastInitMatchesArchFormula)
{
    std::vector<double> r(100);
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        r[t] = 0.5 * std::sin(0.7 * static_cast<double>(t)) + 0.1 * std::cos(1.3 * static_cast<double>(t));
    }
    std::vector<double> s2(100);
    std::vector<double> th{0.05, 0.1, 0.1, 0.8};
    ve::garch::sgarch_norm_loglik(r, th, {.var_init = VarInit::Backcast}, {.sigma2 = s2});
    double weighted_square_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < 75; ++i)
    {
        double weight = std::pow(0.94, static_cast<double>(i));
        double residual = r[i] - 0.05;
        weighted_square_sum += weight * residual * residual;
        weight_sum += weight;
    }
    double b = weighted_square_sum / weight_sum;
    EXPECT_NEAR(s2[0], 0.1 + 0.1 * b + 0.8 * b, 1e-12);
    double first_residual = r[0] - 0.05;
    EXPECT_NEAR(s2[1], 0.1 + 0.1 * first_residual * first_residual + 0.8 * s2[0], 1e-12);
}

TEST(SGarch, BackcastWindowIsWholeSampleWhenShorterThan75)
{
    std::vector<double> r{0.1, -0.2, 0.3};
    std::vector<double> s2(3);
    std::vector<double> th{0.0, 0.1, 0.1, 0.8};
    ve::garch::sgarch_norm_loglik(r, th, {.var_init = VarInit::Backcast}, {.sigma2 = s2});
    double b = (0.01 + 0.94 * 0.04 + 0.94 * 0.94 * 0.09) / (1.0 + 0.94 + 0.94 * 0.94);
    EXPECT_NEAR(s2[0], 0.1 + 0.1 * b + 0.8 * b, 1e-12);
}

TEST(SGarch, NoMeanHasThreeParams)
{
    EXPECT_EQ(ve::garch::sgarch_param_count({.include_mean = false}), 3u);
    EXPECT_EQ(ve::garch::sgarch_param_count({}), 4u);
}

TEST(SGarch, NoMeanUsesRawReturnsAsResiduals)
{
    std::vector<double> r{0.1, -0.2, 0.3};
    std::vector<double> with_mean(3);
    std::vector<double> without_mean(3);
    std::vector<double> zero_mean_theta{0.0, 0.1, 0.1, 0.8};
    std::vector<double> no_mean_theta{0.1, 0.1, 0.8};
    double ll_with = ve::garch::sgarch_norm_loglik(r, zero_mean_theta, {}, {.sigma2 = with_mean});
    double ll_without =
        ve::garch::sgarch_norm_loglik(r, no_mean_theta, {.include_mean = false}, {.sigma2 = without_mean});
    EXPECT_EQ(ll_with, ll_without);
    EXPECT_EQ(with_mean, without_mean);
}

TEST_P(SGarchDerivatives, GradientMatchesCentralDifferences)
{
    SGarchConfig config = GetParam();
    std::vector<double> r = simulate_garch_returns(500);
    std::vector<double> theta = evaluation_theta(config);
    std::vector<double> grad(theta.size());
    double ll = ve::garch::sgarch_norm_loglik(r, theta, config, {.grad = grad});
    EXPECT_EQ(ll, loglik_at(r, theta, config));
    std::vector<double> numeric = central_difference_gradient(r, theta, config);
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        EXPECT_NEAR(grad[i], numeric[i], 1e-6 * std::max(1.0, std::abs(grad[i]))) << "parameter " << i;
    }
}

TEST_P(SGarchDerivatives, ScoresSumToGradient)
{
    SGarchConfig config = GetParam();
    std::vector<double> r = simulate_garch_returns(500);
    std::vector<double> theta = evaluation_theta(config);
    std::size_t k = theta.size();
    std::vector<double> grad(k);
    std::vector<double> scores(r.size() * k);
    ve::garch::sgarch_norm_loglik(r, theta, config, {.grad = grad});
    ve::garch::sgarch_norm_loglik(r, theta, config, {.scores = scores});
    for (std::size_t i = 0; i < k; ++i)
    {
        double sum = 0.0;
        for (std::size_t t = 0; t < r.size(); ++t)
        {
            sum += scores[t * k + i];
        }
        EXPECT_NEAR(sum, grad[i], 1e-9 * std::max(1.0, std::abs(grad[i]))) << "parameter " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(SGarch, SGarchDerivatives,
                         ::testing::Values(SGarchConfig{.include_mean = true, .var_init = VarInit::Sample},
                                           SGarchConfig{.include_mean = false, .var_init = VarInit::Sample},
                                           SGarchConfig{.include_mean = true, .var_init = VarInit::Backcast},
                                           SGarchConfig{.include_mean = false, .var_init = VarInit::Backcast}),
                         config_name);

TEST(SGarch, NonPositiveVarianceGivesNonFinite)
{
    std::vector<double> r{0.1, -0.2, 0.3};
    std::vector<double> th{0.0, -1.0, 0.1, 0.8};
    double ll = ve::garch::sgarch_norm_loglik(r, th, {}, {});
    EXPECT_FALSE(std::isfinite(ll));
}
