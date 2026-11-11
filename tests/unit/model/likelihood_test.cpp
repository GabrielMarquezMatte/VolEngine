#include "model/likelihood.hpp"

#include "simulated_garch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <numbers>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace
{

using ve::Dist;
using ve::VarInit;
using ve::model::Layout;
using ve::model::loglik;

template <std::size_t N>
using Dual = ve::ad::Dual<N>;

constexpr Layout kSGarch11Norm{
    .include_mean = true, .ar = 0, .ma = 0, .arch = 1, .garch = 1, .dist = Dist::Norm};

double loglik_at(std::span<const double> r, std::span<const double> theta, const Layout& layout,
                 VarInit var_init)
{
    return loglik<double>(r, theta, layout, var_init, {});
}

template <std::size_t N>
std::vector<Dual<N>> seeded(std::span<const double> theta)
{
    std::vector<Dual<N>> result;
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        result.push_back(ve::ad::variable<N>(theta[i], i));
    }
    return result;
}

double arch_backcast(std::span<const double> r, double center)
{
    std::size_t window = std::min<std::size_t>(75, r.size());
    double weighted_square_sum = 0.0;
    double weight_sum = 0.0;
    for (std::size_t i = 0; i < window; ++i)
    {
        double weight = std::pow(0.94, static_cast<double>(i));
        double residual = r[i] - center;
        weighted_square_sum += weight * residual * residual;
        weight_sum += weight;
    }
    return weighted_square_sum / weight_sum;
}

double mean_of(std::span<const double> values)
{
    double sum = 0.0;
    for (double value : values)
    {
        sum += value;
    }
    return sum / static_cast<double>(values.size());
}

std::vector<double> wave_series(std::size_t count)
{
    std::vector<double> r(count);
    for (std::size_t t = 0; t < count; ++t)
    {
        r[t] = 0.5 * std::sin(0.7 * static_cast<double>(t)) + 0.1 * std::cos(1.3 * static_cast<double>(t));
    }
    return r;
}

TEST(Likelihood, PlanOneHandCaseStillHolds)
{
    std::vector<double> r{0.1, -0.2, 0.3};
    std::vector<double> theta{0.0, 0.1, 0.1, 0.8};
    std::vector<double> sigma2(3);
    double ll = loglik<double>(r, theta, kSGarch11Norm, VarInit::Sample, {.sigma2 = sigma2});
    EXPECT_NEAR(sigma2[0], 0.14 / 3, 1e-12);
    EXPECT_NEAR(sigma2[1], 0.1 + 0.1 * 0.01 + 0.8 * (0.14 / 3), 1e-12);
    EXPECT_NEAR(sigma2[2], 0.1 + 0.1 * 0.04 + 0.8 * sigma2[1], 1e-12);
    double expected = 0.0;
    for (std::size_t t = 0; t < 3; ++t)
    {
        expected += -0.5 * (std::log(2 * std::numbers::pi) + std::log(sigma2[t]) + r[t] * r[t] / sigma2[t]);
    }
    EXPECT_NEAR(ll, expected, 1e-12);
}

// Computed once by the analytic sGARCH(1,1)-normal likelihood this template replaced.
TEST(Likelihood, MatchesPlanOneSimulatedValues)
{
    std::vector<double> r = ve::testing::simulate_portable_garch_series(42, 500);
    std::vector<double> theta{0.05, 0.01, 0.08, 0.9};
    double sample = loglik_at(r, theta, kSGarch11Norm, VarInit::Sample);
    double backcast = loglik_at(r, theta, kSGarch11Norm, VarInit::Backcast);
    EXPECT_NEAR(sample, -477.70334775209335, 1e-10 * 477.70);
    EXPECT_NEAR(backcast, -476.61341495603085, 1e-10 * 476.61);
}

TEST(Likelihood, SampleInitHoldsFirstMaxOrderVariances)
{
    Layout layout{.include_mean = true, .ar = 0, .ma = 0, .arch = 2, .garch = 3, .dist = Dist::Norm};
    std::vector<double> r = ve::testing::simulate_garch_series(3, 50);
    std::vector<double> theta{0.04, 0.02, 0.05, 0.03, 0.3, 0.3, 0.2};
    std::vector<double> residuals(r.size());
    std::vector<double> sigma2(r.size());
    loglik<double>(r, theta, layout, VarInit::Sample, {.residuals = residuals, .sigma2 = sigma2});
    std::vector<double> squared(r.size());
    std::ranges::transform(residuals, squared.begin(), [](double e) { return e * e; });
    double mean_square = mean_of(squared);
    for (std::size_t t = 0; t < 3; ++t)
    {
        EXPECT_NEAR(sigma2[t], mean_square, 1e-12) << "t = " << t;
    }
    double expected = 0.02 + 0.05 * squared[2] + 0.03 * squared[1] + 0.3 * sigma2[2] + 0.3 * sigma2[1] +
                      0.2 * sigma2[0];
    EXPECT_NEAR(sigma2[3], expected, 1e-12);
}

TEST(Likelihood, SampleInitHoldCountsTheArmaOrders)
{
    Layout layout{.include_mean = true, .ar = 0, .ma = 3, .arch = 1, .garch = 2, .dist = Dist::Norm};
    std::vector<double> r = ve::testing::simulate_garch_series(3, 50);
    std::vector<double> theta{0.04, 0.2, 0.1, -0.1, 0.02, 0.08, 0.5, 0.3};
    std::vector<double> residuals(r.size());
    std::vector<double> sigma2(r.size());
    loglik<double>(r, theta, layout, VarInit::Sample, {.residuals = residuals, .sigma2 = sigma2});
    std::vector<double> squared(r.size());
    std::ranges::transform(residuals, squared.begin(), [](double e) { return e * e; });
    double mean_square = mean_of(squared);
    for (std::size_t t = 0; t < 3; ++t)
    {
        EXPECT_NEAR(sigma2[t], mean_square, 1e-12) << "t = " << t;
    }
    EXPECT_NEAR(sigma2[3], 0.02 + 0.08 * squared[2] + 0.5 * sigma2[2] + 0.3 * sigma2[1], 1e-12);
}

TEST(Likelihood, BackcastFeedsPresample)
{
    Layout layout{.include_mean = true, .ar = 0, .ma = 0, .arch = 2, .garch = 2, .dist = Dist::Norm};
    std::vector<double> r = ve::testing::simulate_garch_series(3, 120);
    std::vector<double> theta{0.04, 0.02, 0.05, 0.03, 0.5, 0.3};
    std::vector<double> sigma2(r.size());
    loglik<double>(r, theta, layout, VarInit::Backcast, {.sigma2 = sigma2});
    double b = arch_backcast(r, mean_of(r));
    EXPECT_NEAR(sigma2[0], 0.02 + (0.05 + 0.03 + 0.5 + 0.3) * b, 1e-12);
}

TEST(Likelihood, BackcastInitMatchesArchFormula)
{
    std::vector<double> r = wave_series(100);
    std::vector<double> sigma2(100);
    std::vector<double> theta{0.05, 0.1, 0.1, 0.8};
    loglik<double>(r, theta, kSGarch11Norm, VarInit::Backcast, {.sigma2 = sigma2});
    double b = arch_backcast(r, mean_of(r));
    EXPECT_NEAR(sigma2[0], 0.1 + 0.1 * b + 0.8 * b, 1e-12);
    double first_residual = r[0] - 0.05;
    EXPECT_NEAR(sigma2[1], 0.1 + 0.1 * first_residual * first_residual + 0.8 * sigma2[0], 1e-12);
}

TEST(Likelihood, BackcastWindowIsWholeSampleWhenShorterThan75)
{
    Layout layout = kSGarch11Norm;
    layout.include_mean = false;
    std::vector<double> r{0.1, -0.2, 0.3};
    std::vector<double> sigma2(3);
    std::vector<double> theta{0.1, 0.1, 0.8};
    loglik<double>(r, theta, layout, VarInit::Backcast, {.sigma2 = sigma2});
    double b = (0.01 + 0.94 * 0.04 + 0.94 * 0.94 * 0.09) / (1.0 + 0.94 + 0.94 * 0.94);
    EXPECT_NEAR(sigma2[0], 0.1 + 0.1 * b + 0.8 * b, 1e-12);
}

TEST(Likelihood, BackcastIsCenteredOnSampleMeanNotOnMu)
{
    std::vector<double> r{0.4, -0.2, 0.7};
    std::vector<double> low_mu(3);
    std::vector<double> high_mu(3);
    std::vector<double> low_theta{-1.0, 0.1, 0.1, 0.8};
    std::vector<double> high_theta{2.0, 0.1, 0.1, 0.8};
    loglik<double>(r, low_theta, kSGarch11Norm, VarInit::Backcast, {.sigma2 = low_mu});
    loglik<double>(r, high_theta, kSGarch11Norm, VarInit::Backcast, {.sigma2 = high_mu});
    double b = (0.01 + 0.94 * 0.25 + 0.94 * 0.94 * 0.16) / (1.0 + 0.94 + 0.94 * 0.94);
    EXPECT_NEAR(low_mu[0], 0.1 + 0.1 * b + 0.8 * b, 1e-12);
    EXPECT_EQ(low_mu[0], high_mu[0]);
}

TEST(Likelihood, NoMeanUsesRawReturnsAsResiduals)
{
    Layout no_mean = kSGarch11Norm;
    no_mean.include_mean = false;
    std::vector<double> r{0.1, -0.2, 0.3};
    std::vector<double> with_mean(3);
    std::vector<double> without_mean(3);
    std::vector<double> zero_mean_theta{0.0, 0.1, 0.1, 0.8};
    std::vector<double> no_mean_theta{0.1, 0.1, 0.8};
    double ll_with =
        loglik<double>(r, zero_mean_theta, kSGarch11Norm, VarInit::Sample, {.sigma2 = with_mean});
    double ll_without = loglik<double>(r, no_mean_theta, no_mean, VarInit::Sample, {.sigma2 = without_mean});
    EXPECT_EQ(ll_with, ll_without);
    EXPECT_EQ(with_mean, without_mean);
}

TEST(Likelihood, GedZeroResidualHasFiniteGradient)
{
    Layout layout{.include_mean = true, .ar = 0, .ma = 0, .arch = 1, .garch = 1, .dist = Dist::Ged};
    std::vector<double> r = ve::testing::simulate_garch_series(11, 600);
    std::vector<double> theta{0.05, 0.02, 0.08, 0.85, 0.7};
    r[10] = theta[0];
    std::vector<Dual<8>> seeds = seeded<8>(theta);
    std::vector<Dual<8>> residuals(r.size());
    Dual<8> ll = loglik<Dual<8>>(r, seeds, layout, VarInit::Sample, {.residuals = residuals});
    ASSERT_EQ(residuals[10].value, 0.0);
    EXPECT_TRUE(std::isfinite(ll.value));
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        EXPECT_TRUE(std::isfinite(ll.grad[i])) << "parameter " << i;
    }
}

TEST(Likelihood, InvalidDomainGivesNonFinite)
{
    std::vector<double> r{0.1, -0.2, 0.3};
    std::vector<double> negative_omega{0.0, -0.1, 0.1, 0.8};
    EXPECT_FALSE(std::isfinite(loglik_at(r, negative_omega, kSGarch11Norm, VarInit::Sample)));
    Layout student = kSGarch11Norm;
    student.dist = Dist::Std;
    std::vector<double> low_nu{0.0, 0.1, 0.1, 0.8, 1.5};
    EXPECT_FALSE(std::isfinite(loglik_at(r, low_nu, student, VarInit::Sample)));
    Layout ged = kSGarch11Norm;
    ged.dist = Dist::Ged;
    std::vector<double> zero_nu{0.0, 0.1, 0.1, 0.8, 0.0};
    EXPECT_FALSE(std::isfinite(loglik_at(r, zero_nu, ged, VarInit::Sample)));
    std::vector<Dual<8>> zero_nu_dual = seeded<8>(zero_nu);
    EXPECT_FALSE(std::isfinite(loglik<Dual<8>>(r, zero_nu_dual, ged, VarInit::Backcast, {}).value));
}

struct Orders
{
    int arch;
    int garch;
};

struct ArmaOrders
{
    int ar;
    int ma;
};

using GridCase = std::tuple<Orders, ArmaOrders, Dist, VarInit, bool>;

Layout grid_layout(const GridCase& c)
{
    auto [orders, arma, dist, var_init, include_mean] = c;
    return Layout{include_mean, arma.ar, arma.ma, orders.arch, orders.garch, dist};
}

std::vector<double> interior_theta(const Layout& layout)
{
    std::vector<double> theta(layout.count());
    if (layout.include_mean)
    {
        theta[layout.mu()] = 0.04;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(layout.ar); ++i)
    {
        theta[layout.ar_begin() + i] = 0.2 / static_cast<double>(i + 1);
    }
    for (std::size_t j = 0; j < static_cast<std::size_t>(layout.ma); ++j)
    {
        theta[layout.ma_begin() + j] = -0.15 / static_cast<double>(j + 1);
    }
    theta[layout.omega()] = 0.02;
    for (std::size_t i = 0; i < static_cast<std::size_t>(layout.arch); ++i)
    {
        theta[layout.alpha_begin() + i] = 0.08 / layout.arch;
    }
    for (std::size_t j = 0; j < static_cast<std::size_t>(layout.garch); ++j)
    {
        theta[layout.beta_begin() + j] = 0.85 / layout.garch;
    }
    if (layout.has_shape())
    {
        theta[layout.shape()] = layout.dist == Dist::Std ? 6.0 : 1.5;
    }
    return theta;
}

std::string grid_name(const ::testing::TestParamInfo<GridCase>& info)
{
    auto [orders, arma, dist, var_init, include_mean] = info.param;
    std::string dists[] = {"Norm", "Std", "Ged"};
    std::string name = "P" + std::to_string(orders.arch) + "Q" + std::to_string(orders.garch) + "Arma" +
                       std::to_string(arma.ar) + std::to_string(arma.ma) + dists[static_cast<int>(dist)];
    name += var_init == VarInit::Sample ? "Sample" : "Backcast";
    name += include_mean ? "Mean" : "NoMean";
    return name;
}

class LikelihoodGrid : public ::testing::TestWithParam<GridCase>
{
protected:
    std::vector<double> r = ve::testing::simulate_garch_series(11, 600);
    Layout layout = grid_layout(GetParam());
    VarInit var_init = std::get<VarInit>(GetParam());
    std::vector<double> theta = interior_theta(layout);
};

TEST_P(LikelihoodGrid, GradientMatchesFiniteDifferences)
{
    std::vector<Dual<32>> seeds = seeded<32>(theta);
    Dual<32> ll = loglik<Dual<32>>(r, seeds, layout, var_init, {});
    ASSERT_TRUE(std::isfinite(ll.value));
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        double step = 1e-6 * std::max(1.0, std::abs(theta[i]));
        std::vector<double> up = theta;
        std::vector<double> down = theta;
        up[i] += step;
        down[i] -= step;
        double central =
            (loglik_at(r, up, layout, var_init) - loglik_at(r, down, layout, var_init)) / (2.0 * step);
        EXPECT_NEAR(ll.grad[i], central, 1e-6 * std::max(1.0, std::abs(ll.grad[i]))) << "parameter " << i;
    }
}

TEST_P(LikelihoodGrid, TermsSumToTotalAndScoresSumToGradient)
{
    std::vector<Dual<32>> seeds = seeded<32>(theta);
    std::vector<Dual<32>> terms(r.size());
    Dual<32> ll = loglik<Dual<32>>(r, seeds, layout, var_init, {.terms = terms});
    Dual<32> sum = ve::ad::constant<32>(0.0);
    for (const Dual<32>& term : terms)
    {
        sum = sum + term;
    }
    EXPECT_NEAR(sum.value, ll.value, 1e-10 * std::abs(ll.value));
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        EXPECT_NEAR(sum.grad[i], ll.grad[i], 1e-9 * std::max(1.0, std::abs(ll.grad[i]))) << "parameter " << i;
    }
}

TEST_P(LikelihoodGrid, DoubleAndDualAgreeOnValue)
{
    std::vector<Dual<16>> seeds = seeded<16>(theta);
    std::vector<double> sigma2(r.size());
    std::vector<Dual<16>> dual_sigma2(r.size());
    double plain = loglik<double>(r, theta, layout, var_init, {.sigma2 = sigma2});
    Dual<16> dual = loglik<Dual<16>>(r, seeds, layout, var_init, {.sigma2 = dual_sigma2});
    EXPECT_EQ(ve::ad::value_of(dual), plain);
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        ASSERT_EQ(dual_sigma2[t].value, sigma2[t]) << "t = " << t;
    }
}

INSTANTIATE_TEST_SUITE_P(Likelihood, LikelihoodGrid,
                         ::testing::Combine(::testing::Values(Orders{1, 1}, Orders{2, 1}, Orders{1, 2},
                                                              Orders{3, 3}),
                                            ::testing::Values(ArmaOrders{0, 0}, ArmaOrders{1, 1},
                                                              ArmaOrders{2, 0}),
                                            ::testing::Values(Dist::Norm, Dist::Std, Dist::Ged),
                                            ::testing::Values(VarInit::Sample, VarInit::Backcast),
                                            ::testing::Bool()),
                         grid_name);

}
