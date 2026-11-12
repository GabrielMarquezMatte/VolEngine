#include "linalg/least_squares.hpp"
#include "model/data.hpp"
#include "model/likelihood.hpp"
#include "model/likelihood_terms.hpp"
#include "model/parameter_map.hpp"

#include "model/start_values.hpp"
#include "simulated_garch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <gtest/gtest.h>
#include <numbers>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace
{

using ve::Dist;
using ve::Model;
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
    std::vector<double> r = ve::testing::simulate_garch_series(42, 500);
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

TEST(Likelihood, PowerSampleInitHoldsMeanAbsolutePower)
{
    for (Model model : {Model::ApArch, Model::FGarchTGarch})
    {
        SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(model));
        Layout layout = ve::model::layout_of({.model = model, .p = 2, .q = 1, .ar = 3});
        std::vector<double> r = ve::testing::simulate_garch_series(3, 50);
        std::vector<double> theta = ve::model::start_values(r, layout);
        theta[layout.ar_begin()] = 0.2;
        if (layout.free_power)
        {
            theta[layout.power()] = 1.3;
        }
        double power = ve::model::power_of<double>(theta, layout);
        std::vector<double> residuals(r.size());
        std::vector<double> sigma2(r.size());
        loglik<double>(r, theta, layout, VarInit::Sample, {.residuals = residuals, .sigma2 = sigma2});
        std::vector<double> powers(r.size());
        std::ranges::transform(residuals, powers.begin(), [&](double e) { return std::pow(std::abs(e), power); });
        double held = std::pow(mean_of(powers), 2.0 / power);
        for (std::size_t t = 0; t < 3; ++t)
        {
            EXPECT_NEAR(sigma2[t], held, 1e-12) << "t = " << t;
        }
    }
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

TEST(Likelihood, IGarchIgnoresStoredLastBeta)
{
    Layout igarch{.model = Model::IGarch, .include_mean = true, .ar = 0, .ma = 0, .arch = 1, .garch = 2,
                  .dist = Dist::Norm};
    Layout sgarch = igarch;
    sgarch.model = Model::SGarch;
    std::vector<double> r = ve::testing::simulate_garch_series(5, 500);
    std::vector<double> igarch_theta{0.05, 0.02, 0.08, 0.5, 0.123};
    std::vector<double> sgarch_theta{0.05, 0.02, 0.08, 0.5, 1.0 - 0.08 - 0.5};
    std::vector<double> igarch_sigma2(r.size());
    std::vector<double> sgarch_sigma2(r.size());
    double igarch_ll = loglik<double>(r, igarch_theta, igarch, VarInit::Sample, {.sigma2 = igarch_sigma2});
    double sgarch_ll = loglik<double>(r, sgarch_theta, sgarch, VarInit::Sample, {.sigma2 = sgarch_sigma2});
    EXPECT_NEAR(igarch_ll, sgarch_ll, 1e-14 * std::abs(sgarch_ll));
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        ASSERT_NEAR(igarch_sigma2[t], sgarch_sigma2[t], 1e-14 * sgarch_sigma2[t]) << "t = " << t;
    }
    std::vector<Dual<8>> seeds = seeded<8>(igarch_theta);
    Dual<8> ll = loglik<Dual<8>>(r, seeds, igarch, VarInit::Sample, {});
    EXPECT_EQ(ll.grad[igarch.derived()], 0.0);
}

// At sd 0.07 the outlier is z ~ 1e4, so gamma |z| pushes log sigma2 past the largest finite exp.
TEST(Likelihood, EGarchOutlierGivesNonFinite)
{
    Layout layout = ve::model::layout_of({.model = Model::EGarch});
    std::vector<double> r = ve::testing::simulate_garch_series(42, 500);
    std::ranges::transform(r, r.begin(), [](double value) { return 0.1 * value; });
    std::vector<double> theta{0.005, -0.5, 0.05, 0.9, 0.2};
    for (VarInit var_init : {VarInit::Sample, VarInit::Backcast})
    {
        EXPECT_TRUE(std::isfinite(loglik_at(r, theta, layout, var_init)));
    }
    r[100] = 1e3;
    for (VarInit var_init : {VarInit::Sample, VarInit::Backcast})
    {
        EXPECT_FALSE(std::isfinite(loglik_at(r, theta, layout, var_init)));
    }
}

struct ModelPoint
{
    Layout layout;
    std::vector<double> theta;
};

// fGARCH takes |z - eta2| as sqrt((z - eta2)^2 + 1e-6), as rugarch does, so a submodel matches its named model
// only up to that smoothing: by up to 1.3e-6 in sigma2 and 3e-7 in loglik (relative) on this series. A wrong
// parameter mapping moves both by orders of magnitude more.
void expect_same_variance(std::span<const double> r, const ModelPoint& model, const ModelPoint& reference,
                          VarInit var_init)
{
    std::vector<double> sigma2(r.size());
    std::vector<double> reference_sigma2(r.size());
    double ll = loglik<double>(r, model.theta, model.layout, var_init, {.sigma2 = sigma2});
    double reference_ll =
        loglik<double>(r, reference.theta, reference.layout, var_init, {.sigma2 = reference_sigma2});
    ASSERT_TRUE(std::isfinite(reference_ll));
    EXPECT_NEAR(ll, reference_ll, 1e-6 * std::abs(reference_ll));
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        ASSERT_NEAR(sigma2[t], reference_sigma2[t], 1e-5 * reference_sigma2[t]) << "t = " << t;
    }
}

// theta = (mu, omega, alpha, beta, extra..., shape when the distribution has one).
ModelPoint model_point(Model model, Dist dist, std::vector<double> extra)
{
    std::vector<double> theta{0.04, 0.02, 0.08, 0.85};
    theta.insert(theta.end(), extra.begin(), extra.end());
    if (dist != Dist::Norm)
    {
        theta.push_back(dist == Dist::Std ? 6.0 : 1.5);
    }
    return {ve::model::layout_of({.model = model, .dist = dist}), theta};
}

TEST(Likelihood, SubmodelsMatchTheirModels)
{
    std::vector<double> r = ve::testing::simulate_garch_series(42, 500);
    double eta1 = 0.3;
    for (Dist dist : {Dist::Norm, Dist::Std, Dist::Ged})
    {
        for (VarInit var_init : {VarInit::Sample, VarInit::Backcast})
        {
            SCOPED_TRACE(::testing::Message() << "dist " << static_cast<int>(dist) << ", var_init "
                                            << static_cast<int>(var_init));
            expect_same_variance(r, model_point(Model::FGarchGarch, dist, {}),
                                 model_point(Model::SGarch, dist, {}), var_init);
            expect_same_variance(r, model_point(Model::FGarchApArch, dist, {eta1, 1.5}),
                                 model_point(Model::ApArch, dist, {eta1, 1.5}), var_init);
            ModelPoint gjr = model_point(Model::GjrGarch, dist, {4.0 * 0.08 * eta1});
            gjr.theta[2] = 0.08 * (1.0 - eta1) * (1.0 - eta1);
            expect_same_variance(r, model_point(Model::FGarchGjrGarch, dist, {eta1}), gjr, var_init);
        }
    }
}

// E|z|^power is infinite for std with shape <= power, and so is every kappa.
TEST(Likelihood, PowerStdNeedsShapeAbovePower)
{
    std::vector<double> r = ve::testing::simulate_garch_series(42, 500);
    for (VarInit var_init : {VarInit::Sample, VarInit::Backcast})
    {
        ModelPoint aparch = model_point(Model::ApArch, Dist::Std, {0.1, 2.5});
        aparch.theta.back() = 2.4;
        EXPECT_FALSE(std::isfinite(loglik_at(r, aparch.theta, aparch.layout, var_init)));
        aparch.theta.back() = 3.0;
        EXPECT_TRUE(std::isfinite(loglik_at(r, aparch.theta, aparch.layout, var_init)));
        ModelPoint allgarch = model_point(Model::FGarchAllGarch, Dist::Std, {0.1, 0.2, 2.6});
        allgarch.theta.back() = 2.5;
        EXPECT_FALSE(std::isfinite(loglik_at(r, allgarch.theta, allgarch.layout, var_init)));
    }
}

// E|z|^power is infinite for sstd with shape <= power, as for std (the tail is the t one).
TEST(Likelihood, PowerSkewedStudentNeedsShapeAbovePower)
{
    std::vector<double> r = ve::testing::simulate_garch_series(42, 500);
    for (VarInit var_init : {VarInit::Sample, VarInit::Backcast})
    {
        ModelPoint aparch = model_point(Model::ApArch, Dist::Norm, {0.1, 2.5});
        aparch.layout.dist = Dist::Sstd;
        aparch.theta.push_back(0.9);
        aparch.theta.push_back(2.4);
        EXPECT_FALSE(std::isfinite(loglik_at(r, aparch.theta, aparch.layout, var_init)));
        aparch.theta.back() = 3.0;
        EXPECT_TRUE(std::isfinite(loglik_at(r, aparch.theta, aparch.layout, var_init)));
    }
}

struct SkewedCase
{
    Dist dist;
    std::vector<double> parameters;
};

std::vector<SkewedCase> skewed_cases()
{
    return {{Dist::Sstd, {0.8, 6.0}},
            {Dist::Sged, {1.2, 1.5}},
            {Dist::Nig, {-0.3, 1.5}},
            {Dist::Ghyp, {0.3, 2.0, 1.5}},
            {Dist::Jsu, {0.5, 2.0}}};
}

// sGARCH(1,1) without a mean, sample init: sigma2_0 is the mean square and the recursion is the plan 2a one.
TEST(Likelihood, SkewedHandCase)
{
    std::vector<double> r{0.1, -0.2, 0.3};
    double omega = 0.1;
    double alpha = 0.1;
    double beta = 0.8;
    std::vector<double> sigma2(3);
    sigma2[0] = 0.14 / 3;
    sigma2[1] = omega + alpha * r[0] * r[0] + beta * sigma2[0];
    sigma2[2] = omega + alpha * r[1] * r[1] + beta * sigma2[1];
    for (const SkewedCase& c : skewed_cases())
    {
        SCOPED_TRACE(::testing::Message() << "dist " << static_cast<int>(c.dist));
        Layout layout = ve::model::layout_of({.model = Model::SGarch, .dist = c.dist, .include_mean = false});
        std::vector<double> theta{omega, alpha, beta};
        theta.insert(theta.end(), c.parameters.begin(), c.parameters.end());
        ASSERT_EQ(theta.size(), layout.count());
        ve::dist::Density<double> density =
            ve::dist::make_density(c.dist, ve::model::shape_parameters_of<double>(theta, layout));
        double expected = 0.0;
        for (std::size_t t = 0; t < 3; ++t)
        {
            expected += ve::dist::log_density(density, r[t] / std::sqrt(sigma2[t])) - 0.5 * std::log(sigma2[t]);
        }
        double ll = loglik_at(r, theta, layout, VarInit::Sample);
        EXPECT_NEAR(ll, expected, 1e-14 * std::abs(expected));
    }
}

// zeta this small with order 40 overflows the Bessel ratios, so the ghyp cannot be standardized: it is not
// computable, and the likelihood is -inf rather than NaN.
TEST(Likelihood, NonFiniteTermIsInvalid)
{
    std::vector<double> r = ve::testing::simulate_garch_series(42, 50);
    Layout layout = ve::model::layout_of({.model = Model::SGarch, .dist = Dist::Ghyp});
    std::vector<double> theta{0.0, 0.1, 0.1, 0.8, 0.3, 1e-300, 40.0};
    ASSERT_EQ(theta.size(), layout.count());
    for (VarInit var_init : {VarInit::Sample, VarInit::Backcast})
    {
        double ll = 0.0;
        ASSERT_NO_THROW(ll = loglik_at(r, theta, layout, var_init));
        EXPECT_EQ(ll, -std::numeric_limits<double>::infinity());
        std::vector<Dual<8>> seeds = seeded<8>(theta);
        Dual<8> dual = loglik<Dual<8>>(r, seeds, layout, var_init, {});
        EXPECT_EQ(dual.value, -std::numeric_limits<double>::infinity());
    }
}

// Review Focus 2: a sum that is +inf, from a +inf term or from finite terms that overflow, is never an
// improvement.
template <class T>
void expect_infinite_sum_invalid()
{
    std::vector<T> residuals{T{0.1}, T{-0.2}, T{0.3}};
    std::vector<T> sigma2{T{1.0}, T{1.0}, T{1.0}};
    std::vector<T> terms(3);
    auto huge = [](const T&, const T&) { return T{std::numeric_limits<double>::max()}; };
    auto infinite = [](const T&, const T&) { return T{std::numeric_limits<double>::infinity()}; };
    double invalid = -std::numeric_limits<double>::infinity();
    T overflowed = ve::model::sum_of_terms<T>(residuals, sigma2, huge, std::span<T>(terms));
    EXPECT_EQ(ve::ad::value_of(overflowed), invalid);
    EXPECT_EQ(ve::ad::value_of(terms[2]), std::numeric_limits<double>::max());
    T infinite_sum = ve::model::sum_of_terms<T>(residuals, sigma2, infinite, std::span<T>{});
    EXPECT_EQ(ve::ad::value_of(infinite_sum), invalid);
}

TEST(Likelihood, InfiniteSumIsInvalid)
{
    expect_infinite_sum_invalid<double>();
    expect_infinite_sum_invalid<Dual<8>>();
}

// A GARCH(1,1) point with the given distribution parameters; ghlambda = 1.5 for ghyp.
std::vector<double> hyperbolic_theta(const Layout& layout, double rho, double zeta)
{
    std::vector<double> theta(layout.count());
    theta[layout.mu()] = 0.04;
    theta[layout.omega()] = 0.05;
    theta[layout.alpha_begin()] = 0.05;
    theta[layout.beta_begin()] = 0.85;
    if (layout.model == Model::GjrGarch)
    {
        theta[layout.gamma_begin()] = 0.05;
    }
    theta[layout.skew()] = rho;
    theta[layout.shape()] = zeta;
    if (layout.has_ghlambda())
    {
        theta[layout.ghlambda()] = 1.5;
    }
    return theta;
}

struct HyperbolicPoint
{
    double rho;
    double zeta;
    bool computable;
};

// Review Focus 1: the likelihood and the validity rule agree with in_domain on where nig and ghyp are
// computable.
TEST(Likelihood, HyperbolicComputabilityAgrees)
{
    std::vector<double> r = ve::testing::simulate_garch_series(42, 200);
    double past_cap = std::nextafter(ve::dist::kMaxHyperbolicShape, 2e6);
    std::vector<HyperbolicPoint> points{{0.99, 1e9, false},    {1.0 - 1e-12, 1e12, false},
                                        {0.3, past_cap, false}, {0.3, 1e-300, false},
                                        {1.0 - 1e-12, 1e6, true}, {-0.3, 5.0, true}};
    for (Model model : {Model::SGarch, Model::GjrGarch})
    {
        for (Dist dist : {Dist::Nig, Dist::Ghyp})
        {
            Layout layout = ve::model::layout_of({.model = model, .dist = dist});
            for (const HyperbolicPoint& point : points)
            {
                SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(model) << ", dist "
                                                  << static_cast<int>(dist) << ", rho " << point.rho
                                                  << ", zeta " << point.zeta);
                std::vector<double> theta = hyperbolic_theta(layout, point.rho, point.zeta);
                EXPECT_EQ(ve::dist::in_domain(dist, ve::model::shape_parameters_of<double>(theta, layout)),
                          point.computable);
                EXPECT_EQ(ve::model::is_valid(theta, layout), point.computable);
                EXPECT_EQ(std::isfinite(loglik_at(r, theta, layout, VarInit::Sample)), point.computable);
                std::vector<Dual<8>> seeds = seeded<8>(theta);
                Dual<8> dual = loglik<Dual<8>>(r, seeds, layout, VarInit::Sample, {});
                EXPECT_EQ(std::isfinite(dual.value), point.computable);
            }
        }
    }
}

TEST(Likelihood, SkewedDomainGivesNonFinite)
{
    std::vector<double> r{0.1, -0.2, 0.3};
    Layout sstd = ve::model::layout_of({.model = Model::SGarch, .dist = Dist::Sstd, .include_mean = false});
    EXPECT_FALSE(std::isfinite(loglik_at(r, std::vector<double>{0.1, 0.1, 0.8, 0.8, 1.9}, sstd, VarInit::Sample)));
    EXPECT_FALSE(std::isfinite(loglik_at(r, std::vector<double>{0.1, 0.1, 0.8, -0.8, 6.0}, sstd, VarInit::Sample)));
    Layout nig = ve::model::layout_of({.model = Model::SGarch, .dist = Dist::Nig, .include_mean = false});
    EXPECT_FALSE(std::isfinite(loglik_at(r, std::vector<double>{0.1, 0.1, 0.8, 1.0, 1.5}, nig, VarInit::Sample)));
}

constexpr std::array<double, 4> kTargetingSeries{0.1, -0.2, 0.3, 0.05};

Layout targeting_layout(Model model, bool targeting)
{
    return ve::model::layout_of(
        {.model = model, .dist = Dist::Norm, .include_mean = false, .variance_targeting = targeting});
}

double mean_square_of(std::span<const double> values)
{
    double sum = 0.0;
    for (double value : values)
    {
        sum += value * value;
    }
    return sum / static_cast<double>(values.size());
}

// The targeted likelihood equals the plain one at the omega the formula gives, whatever omega the targeted
// theta carries.
void expect_targeting_matches_explicit_omega(const Layout& layout, std::vector<double> theta, double omega)
{
    Layout targeted = targeting_layout(layout.model, true);
    Layout plain = targeting_layout(layout.model, false);
    std::vector<double> given = theta;
    theta[plain.omega()] = omega;
    given[plain.omega()] = 1234.0;
    double expected = loglik_at(kTargetingSeries, theta, plain, VarInit::Sample);
    ASSERT_TRUE(std::isfinite(expected));
    EXPECT_NEAR(loglik_at(kTargetingSeries, given, targeted, VarInit::Sample), expected,
                1e-14 * std::abs(expected));
    EXPECT_NEAR(ve::model::targeted_omega<double>(kTargetingSeries, given, targeted), omega,
                1e-14 * std::abs(omega));
}

TEST(Likelihood, TargetedOmegaHandCase)
{
    double m2 = mean_square_of(kTargetingSeries);
    Layout sgarch = targeting_layout(Model::SGarch, false);
    expect_targeting_matches_explicit_omega(sgarch, {0.0, 0.1, 0.8}, m2 * 0.1);

    Layout egarch = targeting_layout(Model::EGarch, false);
    std::vector<double> egarch_theta(egarch.count());
    egarch_theta[egarch.alpha_begin()] = 0.1;
    egarch_theta[egarch.beta_begin()] = 0.9;
    egarch_theta[egarch.gamma_begin()] = 0.05;
    expect_targeting_matches_explicit_omega(egarch, egarch_theta, std::log(m2) * 0.1);

    double delta = 1.5;
    Layout aparch = targeting_layout(Model::ApArch, false);
    std::vector<double> aparch_theta(aparch.count());
    aparch_theta[aparch.alpha_begin()] = 0.1;
    aparch_theta[aparch.beta_begin()] = 0.8;
    aparch_theta[aparch.power()] = delta;
    double mean_power = 0.0;
    for (double value : kTargetingSeries)
    {
        mean_power += std::pow(std::abs(value), delta) / 4.0;
    }
    double kappa = std::pow(2.0, delta / 2.0) * std::tgamma((delta + 1.0) / 2.0) / std::sqrt(std::numbers::pi);
    expect_targeting_matches_explicit_omega(aparch, aparch_theta, mean_power * (1.0 - 0.1 * kappa - 0.8));
}

TEST(Likelihood, TargetingNearCeilingStaysFinite)
{
    std::vector<double> r = wave_series(200);
    Layout sgarch = targeting_layout(Model::SGarch, true);
    std::vector<double> sgarch_theta{0.0, 0.1, 0.9 - 2e-8};
    double omega = ve::model::targeted_omega<double>(r, sgarch_theta, sgarch);
    EXPECT_GT(omega, 0.0);
    EXPECT_NEAR(omega, mean_square_of(r) * 2e-8, 1e-6 * omega);
    EXPECT_TRUE(std::isfinite(loglik_at(r, sgarch_theta, sgarch, VarInit::Sample)));

    std::vector<double> small(200);
    for (std::size_t t = 0; t < small.size(); ++t)
    {
        small[t] = t % 2 == 0 ? 0.2 : -0.2;
    }
    Layout egarch = targeting_layout(Model::EGarch, true);
    std::vector<double> egarch_theta(egarch.count());
    egarch_theta[egarch.alpha_begin()] = 0.1;
    egarch_theta[egarch.beta_begin()] = 0.9;
    egarch_theta[egarch.gamma_begin()] = 0.05;
    double egarch_omega = ve::model::targeted_omega<double>(small, egarch_theta, egarch);
    EXPECT_NEAR(egarch_omega, std::log(0.04) * 0.1, 1e-12);
    EXPECT_LT(egarch_omega, 0.0);
    EXPECT_TRUE(std::isfinite(loglik_at(small, egarch_theta, egarch, VarInit::Sample)));
}

// Two columns of 6 rows, row-major; with the series of the hand case.
const std::vector<double> kHandSeries{0.12, -0.05, 0.31, 0.02, -0.22, 0.17};
const std::vector<double> kHandRegressors{0.5, -1.0, 1.5, 0.2, -0.7, 0.9, 0.1, -0.4, 0.8, 1.1, -1.3, 0.6};

TEST(Likelihood, MeanRegressorsHandCase)
{
    Layout layout = ve::model::layout_of({.p = 1, .q = 1, .ar = 1, .dist = Dist::Norm, .mxreg = 2});
    ASSERT_EQ(layout.count(), 7u);
    const double mu = 0.03;
    const double phi = 0.4;
    const double omega = 0.02;
    const double alpha = 0.1;
    const double beta = 0.8;
    std::vector<double> theta{mu, phi, 0.5, -0.3, omega, alpha, beta};
    ve::model::Data data{kHandSeries, {kHandRegressors, 2}};
    std::vector<double> residuals(6);
    std::vector<double> sigma2(6);
    double ll = loglik<double>(data, theta, layout, VarInit::Sample, {.residuals = residuals, .sigma2 = sigma2});
    std::vector<double> d(6);
    std::vector<double> e(6);
    for (std::size_t t = 0; t < 6; ++t)
    {
        d[t] = kHandSeries[t] - mu - 0.5 * kHandRegressors[2 * t] + 0.3 * kHandRegressors[2 * t + 1];
        e[t] = t == 0 ? d[t] : d[t] - phi * d[t - 1];
        EXPECT_NEAR(residuals[t], e[t], 1e-14) << "t = " << t;
    }
    double mean_square = 0.0;
    for (double value : e)
    {
        mean_square += value * value / 6.0;
    }
    double expected = 0.0;
    double variance = mean_square;
    for (std::size_t t = 0; t < 6; ++t)
    {
        if (t > 0)
        {
            variance = omega + alpha * e[t - 1] * e[t - 1] + beta * variance;
        }
        EXPECT_NEAR(sigma2[t], variance, 1e-14 * std::max(1.0, variance)) << "t = " << t;
        expected += -0.5 * (std::log(2 * std::numbers::pi) + std::log(variance) + e[t] * e[t] / variance);
    }
    EXPECT_NEAR(ll, expected, 1e-14 * std::abs(expected));
}

std::vector<double> regressor_columns(std::size_t rows, std::size_t columns)
{
    std::vector<double> x(rows * columns);
    for (std::size_t t = 0; t < rows; ++t)
    {
        for (std::size_t k = 0; k < columns; ++k)
        {
            x[t * columns + k] = std::sin(0.37 * static_cast<double>((k + 1) * t) + static_cast<double>(k)) +
                                 0.3 * static_cast<double>(k) * std::cos(0.11 * static_cast<double>(t));
        }
    }
    return x;
}

TEST(Likelihood, BackcastFromOls)
{
    Layout layout = ve::model::layout_of({.p = 1, .q = 1, .dist = Dist::Norm, .mxreg = 2});
    std::vector<double> r = ve::testing::simulate_garch_series(5, 120);
    std::vector<double> x = regressor_columns(r.size(), 2);
    std::vector<double> a;
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        a.insert(a.end(), {1.0, x[2 * t], x[2 * t + 1]});
    }
    std::vector<double> b = ve::linalg::least_squares(a, 3, r);
    std::vector<double> ols_residuals(r.size());
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        ols_residuals[t] = r[t] - b[0] - b[1] * x[2 * t] - b[2] * x[2 * t + 1];
    }
    const double omega = 0.02;
    std::vector<double> theta{0.04, 0.2, -0.1, omega, 0.05, 0.9};
    std::vector<double> sigma2(r.size());
    loglik<double>(ve::model::Data{r, {x, 2}}, theta, layout, VarInit::Backcast, {.sigma2 = sigma2});
    EXPECT_NEAR(sigma2[0], omega + (0.05 + 0.9) * arch_backcast(ols_residuals, 0.0), 1e-14);
}

TEST(Likelihood, BackcastWithConstantRegressorColumnStaysFinite)
{
    Layout layout = ve::model::layout_of({.p = 1, .q = 1, .dist = Dist::Norm, .mxreg = 2});
    std::vector<double> r = ve::testing::simulate_garch_series(5, 120);
    std::vector<double> x(2 * r.size(), 1.0);
    std::vector<double> theta{0.04, 0.2, -0.1, 0.02, 0.05, 0.9};
    double ll = loglik<double>(ve::model::Data{r, {x, 2}}, theta, layout, VarInit::Backcast, {});
    EXPECT_TRUE(std::isfinite(ll));
}

// A variance regressor in [0.5, 1], so that a positive coefficient keeps every sigma2 positive.
std::vector<double> variance_columns(std::size_t rows, std::size_t columns)
{
    std::vector<double> v(rows * columns);
    for (std::size_t t = 0; t < rows; ++t)
    {
        for (std::size_t k = 0; k < columns; ++k)
        {
            double s = std::sin(0.23 * static_cast<double>((k + 1) * t) + static_cast<double>(k));
            v[t * columns + k] = 0.5 + 0.5 * s * s;
        }
    }
    return v;
}

// plain with the coefficients zeta put in the vxreg block.
ModelPoint with_vxreg(const ModelPoint& plain, std::span<const double> zeta)
{
    ModelPoint result = plain;
    result.layout.vxreg = static_cast<int>(zeta.size());
    result.theta.insert(result.theta.begin() + static_cast<std::ptrdiff_t>(plain.layout.vxreg_begin()), zeta.begin(),
                        zeta.end());
    return result;
}

struct ShiftCase
{
    Model model;
    std::vector<double> extra;
};

// A constant regressor v = 0.7 with zeta = 0.03 moves the intercept to omega + 0.03 * 0.7 in the scale of each
// model, and nothing else.
TEST(VarianceRegressors, ConstantRegressorShiftsOmega)
{
    std::vector<double> r = ve::testing::simulate_garch_series(42, 200);
    std::vector<double> v(r.size(), 0.7);
    const double zeta = 0.03;
    const double shift = zeta * 0.7;
    std::vector<ShiftCase> cases{{Model::SGarch, {}},
                                 {Model::IGarch, {}},
                                 {Model::EGarch, {0.05}},
                                 {Model::GjrGarch, {0.05}},
                                 {Model::ApArch, {0.3, 1.5}},
                                 {Model::FGarchAllGarch, {0.3, 0.2, 1.5}}};
    for (const ShiftCase& c : cases)
    {
        for (VarInit var_init : {VarInit::Sample, VarInit::Backcast})
        {
            SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(c.model) << ", var_init "
                                            << static_cast<int>(var_init));
            ModelPoint plain = model_point(c.model, Dist::Norm, c.extra);
            ModelPoint with = with_vxreg(plain, std::vector<double>{zeta});
            ModelPoint moved = plain;
            moved.theta[plain.layout.omega()] += shift;
            std::vector<double> sigma2(r.size());
            std::vector<double> moved_sigma2(r.size());
            double ll = loglik<double>(ve::model::Data{r, {}, {v, 1}}, with.theta, with.layout, var_init,
                                       {.sigma2 = sigma2});
            double moved_ll = loglik<double>(r, moved.theta, moved.layout, var_init, {.sigma2 = moved_sigma2});
            ASSERT_TRUE(std::isfinite(moved_ll));
            EXPECT_EQ(ll, moved_ll);
            for (std::size_t t = 0; t < r.size(); ++t)
            {
                ASSERT_EQ(sigma2[t], moved_sigma2[t]) << "t = " << t;
            }
        }
    }
}

// csGARCH: c_t enters q_t, and the presample q stays at omega / (1 - rho), not at (omega + c) / (1 - rho).
TEST(VarianceRegressors, ConstantRegressorEntersTheCsGarchTrend)
{
    std::vector<double> r = ve::testing::simulate_garch_series(42, 50);
    std::vector<double> v(r.size(), 0.7);
    const double omega = 0.02;
    const double alpha = 0.08;
    const double beta = 0.85;
    const double rho = 0.95;
    const double phi = 0.03;
    const double mu = 0.04;
    const double shift = 0.03 * 0.7;
    ModelPoint plain = model_point(Model::CsGarch, Dist::Norm, {rho, phi});
    ModelPoint with = with_vxreg(plain, std::vector<double>{0.03});
    std::vector<double> residuals(r.size());
    std::vector<double> sigma2(r.size());
    std::vector<double> moved_sigma2(r.size());
    loglik<double>(ve::model::Data{r, {}, {v, 1}}, with.theta, with.layout, VarInit::Sample,
                   {.residuals = residuals, .sigma2 = sigma2});
    ModelPoint moved = plain;
    moved.theta[plain.layout.omega()] += shift;
    loglik<double>(r, moved.theta, moved.layout, VarInit::Sample, {.sigma2 = moved_sigma2});
    double mean_square = 0.0;
    for (double e : residuals)
    {
        mean_square += e * e / static_cast<double>(r.size());
    }
    EXPECT_NEAR(residuals[0], r[0] - mu, 1e-15);
    double q = omega / (1.0 - rho);
    double variance = mean_square + q;
    EXPECT_NEAR(sigma2[0], variance, 1e-14);
    for (std::size_t t = 1; t < r.size(); ++t)
    {
        double square = residuals[t - 1] * residuals[t - 1];
        double next_q = omega + shift + rho * q + phi * (square - variance);
        variance = next_q + alpha * (square - q) + beta * (variance - q);
        q = next_q;
        EXPECT_NEAR(sigma2[t], variance, 1e-14 * std::max(1.0, variance)) << "t = " << t;
    }
    EXPECT_GT(std::abs(sigma2.back() - moved_sigma2.back()), 1e-6);
}

TEST(VarianceRegressors, TimeVaryingHandCase)
{
    const std::vector<double> r{0.12, -0.05, 0.31, 0.02, -0.22};
    std::vector<double> v(5);
    for (std::size_t t = 0; t < v.size(); ++t)
    {
        v[t] = std::abs(std::cos(static_cast<double>(t) / 25.0));
    }
    const double mu = 0.03;
    const double omega = 0.02;
    const double alpha = 0.1;
    const double beta = 0.8;
    const double zeta = 0.05;
    ModelPoint plain = model_point(Model::SGarch, Dist::Norm, {});
    ModelPoint with = with_vxreg(plain, std::vector<double>{zeta});
    with.theta = {mu, omega, alpha, beta, zeta};
    std::vector<double> sigma2(5);
    double ll = loglik<double>(ve::model::Data{r, {}, {v, 1}}, with.theta, with.layout, VarInit::Sample,
                               {.sigma2 = sigma2});
    double mean_square = 0.0;
    for (double value : r)
    {
        mean_square += (value - mu) * (value - mu) / 5.0;
    }
    double expected = 0.0;
    double variance = mean_square;
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        double e = r[t] - mu;
        if (t > 0)
        {
            double previous = r[t - 1] - mu;
            variance = omega + zeta * v[t] + alpha * previous * previous + beta * variance;
        }
        EXPECT_NEAR(sigma2[t], variance, 1e-14 * std::max(1.0, variance)) << "t = " << t;
        expected += -0.5 * (std::log(2 * std::numbers::pi) + std::log(variance) + e * e / variance);
    }
    EXPECT_NEAR(ll, expected, 1e-14 * std::abs(expected));
}

// Under variance targeting, omega = target - sum_k zeta_k mean(v_k): the targeted likelihood is the plain one at that
// omega.
TEST(VarianceRegressors, TargetingSubtractsRegressorMeans)
{
    std::vector<double> r = ve::testing::simulate_garch_series(42, 200);
    std::vector<double> v = variance_columns(r.size(), 1);
    const double zeta = 0.03;
    const double v_mean = mean_of(v);
    for (Model model : {Model::SGarch, Model::EGarch})
    {
        SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(model));
        std::vector<double> extra = model == Model::EGarch ? std::vector<double>{0.05} : std::vector<double>{};
        ModelPoint with = with_vxreg(model_point(model, Dist::Norm, extra), std::vector<double>{zeta});
        ModelPoint targeted = with;
        targeted.layout.variance_targeting = true;
        ve::model::Data data{r, {}, {v, 1}};
        std::vector<double> residuals(r.size());
        std::vector<double> sigma2(r.size());
        double ll = loglik<double>(data, targeted.theta, targeted.layout, VarInit::Sample,
                                   {.residuals = residuals, .sigma2 = sigma2});
        double m2 = mean_square_of(residuals);
        double target = model == Model::EGarch ? std::log(m2) * (1.0 - 0.85) : m2 * (1.0 - 0.08 - 0.85);
        double omega = target - zeta * v_mean;
        EXPECT_NEAR(ve::model::targeted_omega<double>(residuals, targeted.theta, targeted.layout, data.variance),
                    omega, 1e-15);
        ModelPoint explicit_omega = with;
        explicit_omega.theta[with.layout.omega()] = omega;
        std::vector<double> reference_sigma2(r.size());
        double reference = loglik<double>(data, explicit_omega.theta, explicit_omega.layout, VarInit::Sample,
                                          {.sigma2 = reference_sigma2});
        ASSERT_TRUE(std::isfinite(reference));
        EXPECT_NEAR(ll, reference, 1e-14 * std::abs(reference));
        for (std::size_t t = 0; t < r.size(); ++t)
        {
            ASSERT_NEAR(sigma2[t], reference_sigma2[t], 1e-14 * reference_sigma2[t]) << "t = " << t;
        }
    }
}

TEST(VarianceRegressors, NonPositiveVarianceIsInvalid)
{
    std::vector<double> r = ve::testing::simulate_garch_series(42, 200);
    std::vector<double> v = variance_columns(r.size(), 1);
    ModelPoint with = with_vxreg(model_point(Model::SGarch, Dist::Norm, {}), std::vector<double>{-10.0});
    double ll = 0.0;
    EXPECT_NO_THROW(ll = loglik<double>(ve::model::Data{r, {}, {v, 1}}, with.theta, with.layout, VarInit::Sample, {}));
    EXPECT_EQ(ll, -std::numeric_limits<double>::infinity());
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

using GridCase = std::tuple<Model, Orders, ArmaOrders, Dist, VarInit, bool, bool>;

Layout grid_layout(const GridCase& c)
{
    auto [model, orders, arma, dist, var_init, include_mean, targeting] = c;
    return ve::model::layout_of({.model = model,
                                 .p = orders.arch,
                                 .q = orders.garch,
                                 .ar = arma.ar,
                                 .ma = arma.ma,
                                 .dist = dist,
                                 .include_mean = include_mean,
                                 .variance_targeting = targeting});
}

// An sGARCH point, kept from before the variants.
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

// theta = to_natural(to_unconstrained(start_values) + 0.1 N(0, 1)).
std::vector<double> drawn_theta(std::span<const double> r, const Layout& layout)
{
    std::mt19937_64 engine(13);
    ve::testing::PortableNormal draw{1.0};
    ve::model::Standardization standardization = ve::model::standardization_of(r);
    std::vector<double> u(layout.count());
    ve::model::to_unconstrained(ve::model::start_values(r, layout), u, layout, standardization);
    std::ranges::transform(u, u.begin(), [&](double value) { return value + 0.1 * draw(engine); });
    std::vector<double> theta(layout.count());
    ve::model::to_natural<double>(u, theta, layout, standardization);
    return theta;
}

std::vector<double> grid_theta(std::span<const double> r, const Layout& layout)
{
    if (layout.model == Model::SGarch && !layout.has_skew())
    {
        return interior_theta(layout);
    }
    return drawn_theta(r, layout);
}

// r of 6 points, ARMA(1,1), one mean regressor, sGARCH(1,1)-norm and ARCH-in-mean, sample init: sigma2_t, m_t, d_t and
// e_t one after the other from the rule, the presample from e0 (the residuals without the archm term).
TEST(Likelihood, ArchInMeanHandCase)
{
    const double mu = 0.03;
    const double phi = 0.4;
    const double theta_ma = -0.3;
    const double xi = 0.7;
    const double delta = 0.5;
    const double omega = 0.02;
    const double alpha = 0.1;
    const double beta = 0.8;
    const std::vector<double> x{0.5, -1.0, 1.5, 0.2, -0.7, 0.9};
    for (int power : {1, 2})
    {
        SCOPED_TRACE(::testing::Message() << "archm " << power);
        Layout layout = ve::model::layout_of({.p = 1, .q = 1, .ar = 1, .ma = 1, .dist = Dist::Norm, .mxreg = 1,
                                              .archm = power});
        ASSERT_EQ(layout.count(), 8u);
        std::vector<double> theta{mu, phi, theta_ma, xi, delta, omega, alpha, beta};
        std::vector<double> residuals(6);
        std::vector<double> sigma2(6);
        double ll = loglik<double>(ve::model::Data{kHandSeries, {x, 1}}, theta, layout, VarInit::Sample,
                                   {.residuals = residuals, .sigma2 = sigma2});
        std::vector<double> e0(6);
        double mean_square = 0.0;
        for (std::size_t t = 0; t < 6; ++t)
        {
            double d0 = kHandSeries[t] - mu - delta * x[t];
            double previous_d0 = t == 0 ? 0.0 : kHandSeries[t - 1] - mu - delta * x[t - 1];
            e0[t] = t == 0 ? d0 : d0 - phi * previous_d0 - theta_ma * e0[t - 1];
            mean_square += e0[t] * e0[t] / 6.0;
        }
        std::vector<double> d(6);
        std::vector<double> e(6);
        double expected = 0.0;
        double variance = mean_square;
        for (std::size_t t = 0; t < 6; ++t)
        {
            if (t > 0)
            {
                variance = omega + alpha * e[t - 1] * e[t - 1] + beta * variance;
            }
            double sigma_power = power == 1 ? std::sqrt(variance) : variance;
            d[t] = kHandSeries[t] - (mu + delta * x[t] + xi * sigma_power);
            e[t] = t == 0 ? d[t] : d[t] - phi * d[t - 1] - theta_ma * e[t - 1];
            EXPECT_NEAR(sigma2[t], variance, 1e-14 * std::max(1.0, variance)) << "t = " << t;
            EXPECT_NEAR(residuals[t], e[t], 1e-14) << "t = " << t;
            expected += -0.5 * (std::log(2 * std::numbers::pi) + std::log(variance) + e[t] * e[t] / variance);
        }
        EXPECT_NEAR(ll, expected, 1e-14 * std::abs(expected));
    }
}

// theta of the plain layout with the archm slot added, at xi.
std::vector<double> with_archm(const Layout& plain, std::span<const double> theta, double xi)
{
    std::vector<double> result(theta.begin(), theta.end());
    result.insert(result.begin() + static_cast<std::ptrdiff_t>(plain.archm_index()), xi);
    return result;
}

TEST(Likelihood, ArchInMeanZeroIsPlain)
{
    std::vector<double> r = ve::testing::simulate_garch_series(11, 300);
    for (Model model : {Model::SGarch, Model::IGarch, Model::EGarch, Model::GjrGarch, Model::ApArch, Model::CsGarch,
                        Model::FGarchAllGarch})
    {
        for (ArmaOrders arma : {ArmaOrders{0, 0}, ArmaOrders{1, 1}})
        {
            for (VarInit var_init : {VarInit::Sample, VarInit::Backcast})
            {
                for (int power : {1, 2})
                {
                    SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(model) << ", arma "
                                                    << arma.ar << arma.ma << ", var_init "
                                                    << static_cast<int>(var_init) << ", archm " << power);
                    Layout plain = ve::model::layout_of({.model = model, .ar = arma.ar, .ma = arma.ma});
                    Layout archm = ve::model::layout_of({.model = model, .ar = arma.ar, .ma = arma.ma,
                                                         .archm = power});
                    std::vector<double> theta = drawn_theta(r, plain);
                    std::vector<double> theta_archm = with_archm(plain, theta, 0.0);
                    std::vector<double> residuals(r.size());
                    std::vector<double> sigma2(r.size());
                    std::vector<double> archm_residuals(r.size());
                    std::vector<double> archm_sigma2(r.size());
                    double expected = loglik<double>(r, theta, plain, var_init,
                                                     {.residuals = residuals, .sigma2 = sigma2});
                    double ll = loglik<double>(r, theta_archm, archm, var_init,
                                               {.residuals = archm_residuals, .sigma2 = archm_sigma2});
                    ASSERT_TRUE(std::isfinite(expected));
                    EXPECT_EQ(ll, expected);
                    EXPECT_EQ(archm_residuals, residuals);
                    EXPECT_EQ(archm_sigma2, sigma2);
                }
            }
        }
    }
}

TEST(Likelihood, ArchInMeanNonPositiveVarianceIsInvalid)
{
    std::vector<double> r = ve::testing::simulate_garch_series(42, 200);
    std::vector<double> v = variance_columns(r.size(), 1);
    Layout layout = ve::model::layout_of({.vxreg = 1, .archm = 1});
    std::vector<double> theta{0.04, 0.1, 0.02, 0.08, 0.85, -10.0};
    double ll = 0.0;
    EXPECT_NO_THROW(ll = loglik<double>(ve::model::Data{r, {}, {v, 1}}, theta, layout, VarInit::Sample, {}));
    EXPECT_EQ(ll, -std::numeric_limits<double>::infinity());
}

// y_t = sum_{k=0}^{t} pi_k e_{t-k}, pi_0 = 1, pi_k = pi_{k-1} (k - 1 - d) / k.
std::vector<double> fractionally_differenced(std::span<const double> e, double d)
{
    std::vector<double> pi(e.size());
    pi[0] = 1.0;
    for (std::size_t k = 1; k < pi.size(); ++k)
    {
        pi[k] = pi[k - 1] * (static_cast<double>(k) - 1.0 - d) / static_cast<double>(k);
    }
    std::vector<double> y(e.size());
    for (std::size_t t = 0; t < e.size(); ++t)
    {
        for (std::size_t k = 0; k <= t; ++k)
        {
            y[t] += pi[k] * e[t - k];
        }
    }
    return y;
}

// r of 6 points, ARMA(1,0), sGARCH(1,1)-norm, ARFIMA d = 0.3, sample init: e by the mean rule, y by the fractional
// difference, one held variance at the mean of y^2.
TEST(Likelihood, ArfimaHandCase)
{
    const double mu = 0.03;
    const double phi = 0.4;
    const double d = 0.3;
    const double omega = 0.02;
    const double alpha = 0.1;
    const double beta = 0.8;
    Layout layout = ve::model::layout_of({.p = 1, .q = 1, .ar = 1, .dist = Dist::Norm, .arfima = true});
    ASSERT_EQ(layout.count(), 6u);
    std::vector<double> theta{mu, phi, d, omega, alpha, beta};
    std::vector<double> residuals(6);
    std::vector<double> sigma2(6);
    double ll = loglik<double>(kHandSeries, theta, layout, VarInit::Sample, {.residuals = residuals, .sigma2 = sigma2});
    std::vector<double> e(6);
    for (std::size_t t = 0; t < 6; ++t)
    {
        e[t] = t == 0 ? kHandSeries[t] - mu : kHandSeries[t] - mu - phi * (kHandSeries[t - 1] - mu);
    }
    std::vector<double> y = fractionally_differenced(e, d);
    double expected = 0.0;
    double variance = mean_square_of(y);
    for (std::size_t t = 0; t < 6; ++t)
    {
        if (t > 0)
        {
            variance = omega + alpha * y[t - 1] * y[t - 1] + beta * variance;
        }
        EXPECT_NEAR(residuals[t], y[t], 1e-14) << "t = " << t;
        EXPECT_NEAR(sigma2[t], variance, 1e-14 * std::max(1.0, variance)) << "t = " << t;
        expected += -0.5 * (std::log(2 * std::numbers::pi) + std::log(variance) + y[t] * y[t] / variance);
    }
    EXPECT_NEAR(ll, expected, 1e-14 * std::abs(expected));
}

TEST(Likelihood, ArfimaZeroIsPlain)
{
    std::vector<double> r = ve::testing::simulate_garch_series(11, 300);
    std::vector<double> x = regressor_columns(r.size(), 2);
    ve::model::Data data{r, {x, 2}};
    for (Model model : {Model::SGarch, Model::EGarch, Model::ApArch, Model::CsGarch})
    {
        for (VarInit var_init : {VarInit::Sample, VarInit::Backcast})
        {
            SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(model) << ", var_init "
                                            << static_cast<int>(var_init));
            Layout plain = ve::model::layout_of({.model = model, .ar = 1, .ma = 1, .mxreg = 2});
            Layout arfima = ve::model::layout_of({.model = model, .ar = 1, .ma = 1, .mxreg = 2, .arfima = true});
            std::vector<double> theta = drawn_theta(r, ve::model::layout_of({.model = model, .ar = 1, .ma = 1}));
            theta.insert(theta.begin() + static_cast<std::ptrdiff_t>(plain.mxreg_begin()), {0.3, -0.2});
            std::vector<double> theta_arfima = theta;
            theta_arfima.insert(theta_arfima.begin() + static_cast<std::ptrdiff_t>(arfima.arfima_index()), 0.0);
            std::vector<double> residuals(r.size());
            std::vector<double> arfima_residuals(r.size());
            double expected = loglik<double>(data, theta, plain, var_init, {.residuals = residuals});
            double ll = loglik<double>(data, theta_arfima, arfima, var_init, {.residuals = arfima_residuals});
            ASSERT_TRUE(std::isfinite(expected));
            EXPECT_EQ(ll, expected);
            EXPECT_EQ(arfima_residuals, residuals);
        }
    }
}

TEST(Likelihood, ArfimaTargetsDifferencedResiduals)
{
    std::vector<double> r = ve::testing::simulate_garch_series(11, 300);
    Layout layout = ve::model::layout_of({.ar = 1, .variance_targeting = true, .arfima = true});
    std::vector<double> theta{0.04, 0.2, 0.3, 1234.0, 0.08, 0.85};
    std::vector<double> residuals(r.size());
    loglik<double>(r, theta, layout, VarInit::Sample, {.residuals = residuals});
    double omega = mean_square_of(residuals) * (1.0 - 0.08 - 0.85);
    EXPECT_NEAR(ve::model::targeted_omega_at(ve::model::Data{r}, theta, layout), omega, 1e-15 * omega);
}

std::string grid_name(const ::testing::TestParamInfo<GridCase>& info)
{
    auto [model, orders, arma, dist, var_init, include_mean, targeting] = info.param;
    std::string models[] = {"SGarch", "EGarch", "GjrGarch", "ApArch", "IGarch", "CsGarch", "FGarchGarch",
                            "FGarchTGarch", "FGarchAvGarch", "FGarchNGarch", "FGarchNaGarch", "FGarchApArch",
                            "FGarchGjrGarch", "FGarchAllGarch"};
    std::string dists[] = {"Norm", "Std", "Ged", "Sstd", "Sged", "Nig", "Ghyp", "Jsu"};
    std::string name = models[static_cast<int>(model)] + "P" + std::to_string(orders.arch) + "Q" +
                       std::to_string(orders.garch) + "Arma" + std::to_string(arma.ar) + std::to_string(arma.ma) +
                       dists[static_cast<int>(dist)];
    name += var_init == VarInit::Sample ? "Sample" : "Backcast";
    name += include_mean ? "Mean" : "NoMean";
    name += targeting ? "Targeted" : "";
    return name;
}

class LikelihoodGrid : public ::testing::TestWithParam<GridCase>
{
protected:
    std::vector<double> r = ve::testing::simulate_garch_series(11, 600);
    Layout layout = grid_layout(GetParam());
    VarInit var_init = std::get<VarInit>(GetParam());
    std::vector<double> theta = grid_theta(r, layout);

    double central_difference(std::size_t index, double step) const
    {
        std::vector<double> up = theta;
        std::vector<double> down = theta;
        up[index] += step;
        down[index] -= step;
        return (loglik_at(r, up, layout, var_init) - loglik_at(r, down, layout, var_init)) / (2.0 * step);
    }
};

// Richardson extrapolation of central differences at step and 2 step, accurate to O(step^4). A small omega curves
// the likelihood enough to put one central difference past the tolerance; a smaller step meets the rounding noise
// of the likelihood, and a larger one crosses the kinks of |z - eta2| in the power-1 models. The ghyp lambda
// enters only as a Bessel order, where log K is smooth but the rounding of Boost's K, which computes in double
// where long double is double (Apple Silicon, MSVC), reaches 1e-6 of the gradient at that step: it takes the
// step of 1e-3 that the engine takes.
TEST_P(LikelihoodGrid, GradientMatchesFiniteDifferences)
{
    std::vector<Dual<32>> seeds = seeded<32>(theta);
    Dual<32> ll = loglik<Dual<32>>(r, seeds, layout, var_init, {});
    ASSERT_TRUE(std::isfinite(ll.value));
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        bool bessel_order = layout.has_ghlambda() && i == layout.ghlambda();
        double step = (bessel_order ? 1e-3 : 3e-6) * std::max(1.0, std::abs(theta[i]));
        double fine = central_difference(i, step);
        double coarse = central_difference(i, 2.0 * step);
        double extrapolated = (4.0 * fine - coarse) / 3.0;
        EXPECT_NEAR(ll.grad[i], extrapolated, 1e-6 * std::max(1.0, std::abs(ll.grad[i]))) << "parameter " << i;
    }
    if (layout.variance_targeting)
    {
        EXPECT_EQ(ll.grad[layout.omega()], 0.0);
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
                         ::testing::Combine(::testing::Values(Model::SGarch),
                                            ::testing::Values(Orders{1, 1}, Orders{2, 1}, Orders{1, 2},
                                                              Orders{3, 3}),
                                            ::testing::Values(ArmaOrders{0, 0}, ArmaOrders{1, 1},
                                                              ArmaOrders{2, 0}),
                                            ::testing::Values(Dist::Norm, Dist::Std, Dist::Ged),
                                            ::testing::Values(VarInit::Sample, VarInit::Backcast),
                                            ::testing::Bool(), ::testing::Values(false)),
                         grid_name);

INSTANTIATE_TEST_SUITE_P(Variants, LikelihoodGrid,
                         ::testing::Combine(::testing::Values(Model::EGarch, Model::GjrGarch, Model::ApArch,
                                                              Model::IGarch, Model::CsGarch, Model::FGarchGarch,
                                                              Model::FGarchTGarch, Model::FGarchAvGarch,
                                                              Model::FGarchNGarch, Model::FGarchNaGarch,
                                                              Model::FGarchApArch, Model::FGarchGjrGarch,
                                                              Model::FGarchAllGarch),
                                            ::testing::Values(Orders{1, 1}, Orders{2, 2}),
                                            ::testing::Values(ArmaOrders{0, 0}, ArmaOrders{1, 1}),
                                            ::testing::Values(Dist::Norm, Dist::Std, Dist::Ged),
                                            ::testing::Values(VarInit::Sample, VarInit::Backcast),
                                            ::testing::Values(true), ::testing::Values(false)),
                         grid_name);

INSTANTIATE_TEST_SUITE_P(Skewed, LikelihoodGrid,
                         ::testing::Combine(::testing::Values(Model::SGarch, Model::EGarch, Model::GjrGarch,
                                                              Model::ApArch, Model::FGarchAllGarch),
                                            ::testing::Values(Orders{1, 1}),
                                            ::testing::Values(ArmaOrders{0, 0}, ArmaOrders{1, 1}),
                                            ::testing::Values(Dist::Sstd, Dist::Sged, Dist::Nig, Dist::Ghyp,
                                                              Dist::Jsu),
                                            ::testing::Values(VarInit::Sample, VarInit::Backcast),
                                            ::testing::Values(true), ::testing::Values(false)),
                         grid_name);

INSTANTIATE_TEST_SUITE_P(Targeted, LikelihoodGrid,
                         ::testing::Combine(::testing::Values(Model::SGarch, Model::EGarch, Model::GjrGarch,
                                                              Model::ApArch, Model::FGarchAllGarch),
                                            ::testing::Values(Orders{1, 1}, Orders{2, 2}),
                                            ::testing::Values(ArmaOrders{0, 0}, ArmaOrders{1, 1}),
                                            ::testing::Values(Dist::Norm, Dist::Std, Dist::Sstd),
                                            ::testing::Values(VarInit::Sample, VarInit::Backcast),
                                            ::testing::Values(true), ::testing::Values(true)),
                         grid_name);

using RegressorCase = std::tuple<Model, ArmaOrders, Dist, VarInit, int, int, bool, int, bool>;

class MeanRegressorGrid : public ::testing::TestWithParam<RegressorCase>
{
protected:
    static constexpr std::size_t kLength = 400;
    std::vector<double> r = ve::testing::simulate_garch_series(11, kLength);
    Layout layout = layout_of_case();
    std::vector<double> x = regressor_columns(kLength, static_cast<std::size_t>(layout.mxreg));
    std::vector<double> v = variance_columns(kLength, static_cast<std::size_t>(layout.vxreg));
    ve::model::Data data{r, {x, static_cast<std::size_t>(layout.mxreg)}, {v, static_cast<std::size_t>(layout.vxreg)}};
    VarInit var_init = std::get<VarInit>(GetParam());
    std::vector<double> theta = theta_of_case();

    Layout layout_of_case() const
    {
        auto [model, arma, dist, init, mxreg, vxreg, targeting, archm, arfima] = GetParam();
        return ve::model::layout_of({.model = model,
                                     .ar = arma.ar,
                                     .ma = arma.ma,
                                     .dist = dist,
                                     .variance_targeting = targeting,
                                     .mxreg = mxreg,
                                     .vxreg = vxreg,
                                     .archm = archm,
                                     .arfima = arfima});
    }

    // The point of the same grid without regressors, with 0.3 in the arfima slot, 0.15 in the archm slot, the
    // coefficients 0.3, -0.2 put in the mxreg block and 0.01 in the vxreg block.
    std::vector<double> theta_of_case() const
    {
        Layout plain = layout;
        plain.mxreg = 0;
        plain.vxreg = 0;
        plain.archm = 0;
        plain.arfima = false;
        std::vector<double> base = grid_theta(r, plain);
        std::vector<double> result(base.begin(), base.begin() + static_cast<std::ptrdiff_t>(plain.mxreg_begin()));
        if (layout.has_arfima())
        {
            result.push_back(0.3);
        }
        if (layout.has_archm())
        {
            result.push_back(0.15);
        }
        for (int k = 0; k < layout.mxreg; ++k)
        {
            result.push_back(k == 0 ? 0.3 : -0.2);
        }
        result.insert(result.end(), base.begin() + static_cast<std::ptrdiff_t>(plain.mxreg_begin()),
                      base.begin() + static_cast<std::ptrdiff_t>(plain.vxreg_begin()));
        result.insert(result.end(), static_cast<std::size_t>(layout.vxreg), 0.01);
        result.insert(result.end(), base.begin() + static_cast<std::ptrdiff_t>(plain.vxreg_begin()), base.end());
        return result;
    }

    double central_difference(std::size_t index, double step) const
    {
        std::vector<double> up = theta;
        std::vector<double> down = theta;
        up[index] += step;
        down[index] -= step;
        return (loglik<double>(data, up, layout, var_init, {}) - loglik<double>(data, down, layout, var_init, {})) /
               (2.0 * step);
    }
};

TEST_P(MeanRegressorGrid, GradientMatchesFiniteDifferences)
{
    std::vector<Dual<32>> seeds = seeded<32>(theta);
    Dual<32> ll = loglik<Dual<32>>(data, seeds, layout, var_init, {});
    ASSERT_TRUE(std::isfinite(ll.value));
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        double step = 3e-6 * std::max(1.0, std::abs(theta[i]));
        double extrapolated = (4.0 * central_difference(i, step) - central_difference(i, 2.0 * step)) / 3.0;
        EXPECT_NEAR(ll.grad[i], extrapolated, 1e-6 * std::max(1.0, std::abs(ll.grad[i]))) << "parameter " << i;
    }
}

TEST_P(MeanRegressorGrid, TermsSumToTotalAndScoresSumToGradient)
{
    std::vector<Dual<32>> seeds = seeded<32>(theta);
    std::vector<Dual<32>> terms(r.size());
    Dual<32> ll = loglik<Dual<32>>(data, seeds, layout, var_init, {.terms = terms});
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

TEST_P(MeanRegressorGrid, DoubleAndDualAgreeOnValue)
{
    std::vector<Dual<16>> seeds = seeded<16>(theta);
    std::vector<double> sigma2(r.size());
    std::vector<Dual<16>> dual_sigma2(r.size());
    double plain = loglik<double>(data, theta, layout, var_init, {.sigma2 = sigma2});
    Dual<16> dual = loglik<Dual<16>>(data, seeds, layout, var_init, {.sigma2 = dual_sigma2});
    EXPECT_EQ(ve::ad::value_of(dual), plain);
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        ASSERT_EQ(dual_sigma2[t].value, sigma2[t]) << "t = " << t;
    }
}

std::string regressor_name(const ::testing::TestParamInfo<RegressorCase>& info)
{
    auto [model, arma, dist, var_init, mxreg, vxreg, targeting, archm, arfima] = info.param;
    std::string models[] = {"SGarch", "EGarch", "GjrGarch", "ApArch", "IGarch", "CsGarch"};
    std::string name = models[static_cast<int>(model)] + "Arma" + std::to_string(arma.ar) + std::to_string(arma.ma);
    name += dist == Dist::Norm ? "Norm" : "Std";
    name += var_init == VarInit::Sample ? "Sample" : "Backcast";
    name += "Mxreg" + std::to_string(mxreg) + "Vxreg" + std::to_string(vxreg) + "Archm" + std::to_string(archm);
    return name + (targeting ? "Targeted" : "") + (arfima ? "Arfima" : "");
}

INSTANTIATE_TEST_SUITE_P(Regressors, MeanRegressorGrid,
                         ::testing::Combine(::testing::Values(Model::SGarch, Model::EGarch, Model::ApArch,
                                                              Model::CsGarch),
                                            ::testing::Values(ArmaOrders{0, 0}, ArmaOrders{1, 1}),
                                            ::testing::Values(Dist::Norm, Dist::Std),
                                            ::testing::Values(VarInit::Sample, VarInit::Backcast),
                                            ::testing::Values(2), ::testing::Values(0, 1), ::testing::Values(false),
                                            ::testing::Values(0), ::testing::Values(false)),
                         regressor_name);

INSTANTIATE_TEST_SUITE_P(VarianceOnly, MeanRegressorGrid,
                         ::testing::Combine(::testing::Values(Model::SGarch, Model::EGarch, Model::ApArch,
                                                              Model::CsGarch),
                                            ::testing::Values(ArmaOrders{0, 0}, ArmaOrders{1, 1}),
                                            ::testing::Values(Dist::Norm, Dist::Std),
                                            ::testing::Values(VarInit::Sample, VarInit::Backcast),
                                            ::testing::Values(0), ::testing::Values(1), ::testing::Values(false),
                                            ::testing::Values(0), ::testing::Values(false)),
                         regressor_name);

INSTANTIATE_TEST_SUITE_P(TargetedRegressors, MeanRegressorGrid,
                         ::testing::Combine(::testing::Values(Model::SGarch, Model::EGarch, Model::ApArch),
                                            ::testing::Values(ArmaOrders{0, 0}, ArmaOrders{1, 1}),
                                            ::testing::Values(Dist::Norm, Dist::Std),
                                            ::testing::Values(VarInit::Sample, VarInit::Backcast),
                                            ::testing::Values(0, 2), ::testing::Values(1), ::testing::Values(true),
                                            ::testing::Values(0), ::testing::Values(false)),
                         regressor_name);

INSTANTIATE_TEST_SUITE_P(ArchInMean, MeanRegressorGrid,
                         ::testing::Combine(::testing::Values(Model::SGarch, Model::EGarch, Model::GjrGarch,
                                                              Model::ApArch, Model::CsGarch),
                                            ::testing::Values(ArmaOrders{0, 0}, ArmaOrders{1, 1}),
                                            ::testing::Values(Dist::Norm),
                                            ::testing::Values(VarInit::Sample, VarInit::Backcast),
                                            ::testing::Values(0, 2), ::testing::Values(0, 1),
                                            ::testing::Values(false), ::testing::Values(1, 2),
                                            ::testing::Values(false)),
                         regressor_name);

INSTANTIATE_TEST_SUITE_P(Arfima, MeanRegressorGrid,
                         ::testing::Combine(::testing::Values(Model::SGarch, Model::EGarch, Model::ApArch,
                                                              Model::CsGarch),
                                            ::testing::Values(ArmaOrders{0, 0}, ArmaOrders{1, 1}),
                                            ::testing::Values(Dist::Norm),
                                            ::testing::Values(VarInit::Sample, VarInit::Backcast),
                                            ::testing::Values(0, 2), ::testing::Values(0), ::testing::Values(false),
                                            ::testing::Values(0), ::testing::Values(true)),
                         regressor_name);

}
