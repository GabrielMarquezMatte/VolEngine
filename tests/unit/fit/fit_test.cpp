#include "fit/fit.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <string_view>
#include <vector>

namespace
{

using ve::FitResult;
using ve::FitStatus;
using ve::Spec;

constexpr std::size_t kObservations = 5000;

std::vector<double> simulate_garch_returns(std::size_t count)
{
    std::mt19937_64 engine(7);
    std::normal_distribution<double> shock(0.0, 1.0);
    double mu = 0.05;
    double omega = 0.01;
    double alpha = 0.08;
    double beta = 0.9;
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

const std::vector<double>& simulated()
{
    static const std::vector<double> series = simulate_garch_returns(kObservations);
    return series;
}

FitResult fit_simulated(const Spec& spec)
{
    return ve::fit_one(simulated(), spec, ve::FitConfig{});
}

bool all_nan(const std::vector<double>& values)
{
    return std::ranges::all_of(values, [](double value) { return std::isnan(value); });
}

std::uint64_t bits(double value)
{
    return std::bit_cast<std::uint64_t>(value);
}

TEST(Fit, RecoversSimulatedParameters)
{
    FitResult result = fit_simulated(Spec{});
    ASSERT_EQ(result.status, FitStatus::Converged);
    ASSERT_EQ(result.estimate.size(), 4u);
    EXPECT_NEAR(result.estimate[0], 0.05, 0.03);
    EXPECT_NEAR(result.estimate[2], 0.08, 0.02);
    EXPECT_NEAR(result.estimate[3], 0.9, 0.03);
    EXPECT_TRUE(std::isfinite(result.loglik));
    EXPECT_EQ(result.nobs, kObservations);
}

TEST(Fit, NamesFollowRugarch)
{
    FitResult with_mean = fit_simulated(Spec{});
    std::vector<std::string_view> expected{"mu", "omega", "alpha1", "beta1"};
    EXPECT_EQ(with_mean.names, expected);
    Spec no_mean;
    no_mean.include_mean = false;
    FitResult without_mean = fit_simulated(no_mean);
    std::vector<std::string_view> expected_no_mean{"omega", "alpha1", "beta1"};
    EXPECT_EQ(without_mean.names, expected_no_mean);
    EXPECT_EQ(without_mean.estimate.size(), 3u);
}

TEST(Fit, StandardErrorsPositiveAndRobustDiffers)
{
    FitResult result = fit_simulated(Spec{});
    ASSERT_EQ(result.se_hessian.size(), 4u);
    ASSERT_EQ(result.se_robust.size(), 4u);
    bool any_different = false;
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_GT(result.se_hessian[i], 0.0);
        EXPECT_GT(result.se_robust[i], 0.0);
        any_different = any_different || result.se_hessian[i] != result.se_robust[i];
    }
    EXPECT_TRUE(any_different);
}

TEST(Fit, HessianAndRobustStandardErrorsAgreeWhenModelIsCorrect)
{
    FitResult result = fit_simulated(Spec{});
    ASSERT_EQ(result.status, FitStatus::Converged);
    for (std::size_t i = 0; i < 4; ++i)
    {
        double ratio = result.se_robust[i] / result.se_hessian[i];
        EXPECT_GT(ratio, 0.5) << result.names[i];
        EXPECT_LT(ratio, 2.0) << result.names[i];
    }
}

TEST(Fit, FittedSeriesConsistent)
{
    FitResult result = fit_simulated(Spec{});
    ASSERT_EQ(result.sigma.size(), kObservations);
    ASSERT_EQ(result.residuals.size(), kObservations);
    ASSERT_EQ(result.std_residuals.size(), kObservations);
    ASSERT_EQ(result.fitted.size(), kObservations);
    double mu = result.estimate[0];
    const std::vector<double>& r = simulated();
    for (std::size_t t = 0; t < kObservations; ++t)
    {
        EXPECT_EQ(result.residuals[t], r[t] - mu);
        EXPECT_NEAR(result.std_residuals[t] * result.sigma[t], result.residuals[t], 1e-12);
        EXPECT_EQ(result.fitted[t], mu);
        EXPECT_GT(result.sigma[t], 0.0);
    }
}

TEST(Fit, FittedIsZeroWithoutMean)
{
    Spec no_mean;
    no_mean.include_mean = false;
    FitResult result = fit_simulated(no_mean);
    ASSERT_EQ(result.fitted.size(), kObservations);
    EXPECT_TRUE(std::ranges::all_of(result.fitted, [](double value) { return value == 0.0; }));
    EXPECT_EQ(result.residuals[10], simulated()[10]);
}

TEST(Fit, InsufficientData)
{
    std::vector<double> r{0.1, -0.2, 0.3, -0.1, 0.2};
    FitResult result = ve::fit_one(r, Spec{}, ve::FitConfig{});
    EXPECT_EQ(result.status, FitStatus::InsufficientData);
    EXPECT_EQ(result.nobs, 5u);
    std::vector<std::string_view> expected{"mu", "omega", "alpha1", "beta1"};
    EXPECT_EQ(result.names, expected);
    EXPECT_EQ(result.estimate.size(), 4u);
    EXPECT_TRUE(all_nan(result.estimate));
    EXPECT_TRUE(all_nan(result.se_hessian));
    EXPECT_TRUE(all_nan(result.se_robust));
    EXPECT_TRUE(std::isnan(result.loglik));
    EXPECT_TRUE(result.sigma.empty());
    EXPECT_TRUE(result.residuals.empty());
}

TEST(Fit, EmptySeriesIsInsufficientData)
{
    FitResult result = ve::fit_one(std::vector<double>{}, Spec{}, ve::FitConfig{});
    EXPECT_EQ(result.status, FitStatus::InsufficientData);
    EXPECT_EQ(result.nobs, 0u);
}

TEST(Fit, ConstantSeriesFails)
{
    std::vector<double> r(1000, 0.3);
    FitResult result = ve::fit_one(r, Spec{}, ve::FitConfig{});
    EXPECT_EQ(result.status, FitStatus::Failed);
    EXPECT_EQ(result.nobs, 1000u);
    EXPECT_TRUE(all_nan(result.estimate));
    EXPECT_TRUE(std::isnan(result.loglik));
    EXPECT_TRUE(result.sigma.empty());
}

TEST(Fit, NonFiniteObservationFails)
{
    std::vector<double> r = simulate_garch_returns(200);
    r[50] = std::numeric_limits<double>::quiet_NaN();
    FitResult result = ve::fit_one(r, Spec{}, ve::FitConfig{});
    EXPECT_EQ(result.status, FitStatus::Failed);
    EXPECT_TRUE(all_nan(result.estimate));
    EXPECT_TRUE(result.sigma.empty());
}

TEST(Fit, StandardErrorsAreNaNOrPositiveWhenArchEffectIsAbsent)
{
    std::mt19937_64 engine(3);
    std::normal_distribution<double> shock(0.0, 1.0);
    std::vector<double> r(3000);
    std::ranges::generate(r, [&] { return shock(engine); });
    FitResult result = ve::fit_one(r, Spec{}, ve::FitConfig{});
    ASSERT_TRUE(result.status == FitStatus::Converged || result.status == FitStatus::MaxIter);
    for (std::size_t i = 0; i < result.estimate.size(); ++i)
    {
        EXPECT_TRUE(std::isfinite(result.estimate[i]));
        EXPECT_TRUE(std::isnan(result.se_hessian[i]) || result.se_hessian[i] > 0.0);
        EXPECT_TRUE(std::isnan(result.se_robust[i]) || result.se_robust[i] > 0.0);
    }
}

TEST(Fit, IterationLimitReportsMaxIter)
{
    ve::FitConfig config;
    config.max_iter = 2;
    FitResult result = ve::fit_one(simulated(), Spec{}, config);
    EXPECT_EQ(result.status, FitStatus::MaxIter);
    EXPECT_TRUE(std::isfinite(result.loglik));
    EXPECT_EQ(result.sigma.size(), kObservations);
}

TEST(Fit, Deterministic)
{
    FitResult first = fit_simulated(Spec{});
    FitResult second = fit_simulated(Spec{});
    ASSERT_EQ(first.estimate.size(), second.estimate.size());
    for (std::size_t i = 0; i < first.estimate.size(); ++i)
    {
        EXPECT_EQ(bits(first.estimate[i]), bits(second.estimate[i]));
        EXPECT_EQ(bits(first.se_hessian[i]), bits(second.se_hessian[i]));
        EXPECT_EQ(bits(first.se_robust[i]), bits(second.se_robust[i]));
    }
}

}
