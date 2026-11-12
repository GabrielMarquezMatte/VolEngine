#include "model/start_values.hpp"

#include "dist/moments.hpp"
#include "linalg/least_squares.hpp"
#include "model/data.hpp"
#include "model/parameter_map.hpp"
#include "simulated_garch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <numeric>
#include <span>
#include <vector>

namespace
{

using ve::Dist;
using ve::model::is_valid;
using ve::model::Layout;
using ve::model::start_values;

TEST(StartValues, FollowRule)
{
    std::vector<double> r = ve::testing::simulate_garch_series(3, 400);
    double mean = std::accumulate(r.begin(), r.end(), 0.0) / 400.0;
    double variance = 0.0;
    for (double value : r)
    {
        variance += (value - mean) * (value - mean);
    }
    variance /= 400.0;
    Layout layout{.include_mean = true, .ar = 1, .ma = 1, .arch = 2, .garch = 3, .dist = Dist::Std};
    std::vector<double> theta = start_values(r, layout);
    ASSERT_EQ(theta.size(), layout.count());
    EXPECT_DOUBLE_EQ(theta[layout.mu()], mean);
    EXPECT_EQ(theta[layout.ar_begin()], 0.0);
    EXPECT_EQ(theta[layout.ma_begin()], 0.0);
    EXPECT_NEAR(theta[layout.omega()], variance * 0.05, 1e-15);
    for (std::size_t i = 0; i < 2; ++i)
    {
        EXPECT_DOUBLE_EQ(theta[layout.alpha_begin() + i], 0.025);
    }
    for (std::size_t j = 0; j < 3; ++j)
    {
        EXPECT_DOUBLE_EQ(theta[layout.beta_begin() + j], 0.3);
    }
    EXPECT_EQ(theta[layout.shape()], 8.0);
    layout.dist = Dist::Ged;
    EXPECT_EQ(start_values(r, layout)[layout.shape()], 2.0);
}

TEST(StartValues, AreValidForEveryLayout)
{
    std::vector<double> r = ve::testing::simulate_garch_series(3, 400);
    for (int arch : {1, 2, 6})
    {
        for (int garch : {1, 3, 6})
        {
            for (Dist dist : {Dist::Norm, Dist::Std, Dist::Ged})
            {
                for (bool include_mean : {true, false})
                {
                    Layout layout{
                        .include_mean = include_mean, .ar = 2, .ma = 1, .arch = arch, .garch = garch, .dist = dist};
                    EXPECT_TRUE(is_valid(start_values(r, layout), layout));
                }
            }
        }
    }
}

}

namespace
{

using ve::Model;

struct Sample
{
    std::vector<double> r = ve::testing::simulate_garch_series(3, 400);
    double variance = 0.0;
};

Sample sample()
{
    Sample s;
    double mean = std::accumulate(s.r.begin(), s.r.end(), 0.0) / 400.0;
    for (double value : s.r)
    {
        s.variance += (value - mean) * (value - mean);
    }
    s.variance /= 400.0;
    return s;
}

void expect_block(std::span<const double> theta, std::size_t begin, int count, double value)
{
    for (std::size_t i = 0; i < static_cast<std::size_t>(count); ++i)
    {
        EXPECT_NEAR(theta[begin + i], value, 1e-15) << "entry " << begin + i;
    }
}

Layout model_layout(Model model, Dist dist)
{
    return ve::model::layout_of({.model = model, .p = 2, .q = 3, .ar = 1, .ma = 1, .dist = dist});
}

TEST(StartValues, FollowRuleEGarchGjrIGarchCs)
{
    Sample s = sample();
    Layout egarch = model_layout(Model::EGarch, Dist::Norm);
    std::vector<double> theta = start_values(s.r, egarch);
    EXPECT_NEAR(theta[egarch.omega()], 0.1 * std::log(s.variance), 1e-15);
    expect_block(theta, egarch.alpha_begin(), 2, 0.0);
    expect_block(theta, egarch.beta_begin(), 3, 0.3);
    expect_block(theta, egarch.gamma_begin(), 2, 0.05);
    Layout gjr = model_layout(Model::GjrGarch, Dist::Norm);
    theta = start_values(s.r, gjr);
    EXPECT_NEAR(theta[gjr.omega()], 0.05 * s.variance, 1e-15);
    expect_block(theta, gjr.alpha_begin(), 2, 0.015);
    expect_block(theta, gjr.beta_begin(), 3, 0.3);
    expect_block(theta, gjr.gamma_begin(), 2, 0.02);
    Layout igarch = model_layout(Model::IGarch, Dist::Norm);
    theta = start_values(s.r, igarch);
    EXPECT_NEAR(theta[igarch.omega()], 0.01 * s.variance, 1e-15);
    expect_block(theta, igarch.alpha_begin(), 2, 0.05);
    expect_block(theta, igarch.beta_begin(), 3, 0.3);
    Layout cs = model_layout(Model::CsGarch, Dist::Norm);
    theta = start_values(s.r, cs);
    EXPECT_NEAR(theta[cs.omega()], 0.02 * s.variance, 1e-15);
    expect_block(theta, cs.alpha_begin(), 2, 0.025);
    expect_block(theta, cs.beta_begin(), 3, 0.3);
    EXPECT_EQ(theta[cs.eta1_begin()], 0.98);
    EXPECT_EQ(theta[cs.eta2_begin()], 0.03);
}

TEST(StartValues, FollowRulePowerFamily)
{
    Sample s = sample();
    for (Model model : {Model::ApArch, Model::FGarchGarch, Model::FGarchTGarch, Model::FGarchAvGarch,
                        Model::FGarchNGarch, Model::FGarchNaGarch, Model::FGarchApArch, Model::FGarchGjrGarch,
                        Model::FGarchAllGarch})
    {
        for (Dist dist : {Dist::Norm, Dist::Std, Dist::Ged})
        {
            Layout layout = model_layout(model, dist);
            SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(model) << ", dist "
                                            << static_cast<int>(dist));
            std::vector<double> theta = start_values(s.r, layout);
            double power = layout.fixed_power;
            if (layout.free_power)
            {
                power = 2.0;
                EXPECT_EQ(theta[layout.power()], 2.0);
            }
            expect_block(theta, layout.gamma_begin(), layout.gamma, 0.0);
            expect_block(theta, layout.eta1_begin(), layout.eta1, 0.0);
            expect_block(theta, layout.eta2_begin(), layout.eta2, 0.0);
            EXPECT_NEAR(theta[layout.omega()], 0.05 * std::pow(s.variance, power / 2.0), 1e-15);
            expect_block(theta, layout.beta_begin(), 3, 0.3);
            double kappa = ve::dist::abs_moment(dist, {.shape = theta[layout.count() - 1]}, power);
            if (dist == Dist::Norm)
            {
                kappa = ve::dist::abs_moment(dist, {}, power);
            }
            expect_block(theta, layout.alpha_begin(), 2, 0.05 / (2.0 * kappa));
            if (power == 2.0)
            {
                expect_block(theta, layout.alpha_begin(), 2, 0.025);
            }
        }
    }
}

TEST(StartValues, AreValidForEveryModelLayout)
{
    std::vector<double> r = ve::testing::simulate_garch_series(3, 400);
    for (int model = 0; model < 14; ++model)
    {
        for (int order : {1, 2, 6})
        {
            for (Dist dist : {Dist::Norm, Dist::Std, Dist::Ged})
            {
                Layout layout = ve::model::layout_of(
                    {.model = static_cast<Model>(model), .p = order, .q = 7 - order, .ar = 2, .ma = 1, .dist = dist});
                EXPECT_TRUE(is_valid(start_values(r, layout), layout))
                    << "model " << model << ", p " << order << ", dist " << static_cast<int>(dist);
            }
        }
    }
}

}

namespace
{

constexpr Dist kSkewedDists[] = {Dist::Sstd, Dist::Sged, Dist::Nig, Dist::Ghyp, Dist::Jsu};

struct DistributionStart
{
    Dist dist;
    double skew;
    double shape;
    double ghlambda;
};

TEST(StartValues, Distributions)
{
    std::vector<double> r = ve::testing::simulate_garch_series(3, 400);
    std::vector<DistributionStart> starts{{Dist::Sstd, 1.0, 8.0, 0.0},
                                          {Dist::Sged, 1.0, 2.0, 0.0},
                                          {Dist::Nig, 0.0, 2.0, 0.0},
                                          {Dist::Ghyp, 0.0, 2.0, -0.5},
                                          {Dist::Jsu, 0.0, 2.0, 0.0}};
    for (const DistributionStart& start : starts)
    {
        Layout layout = model_layout(Model::SGarch, start.dist);
        std::vector<double> theta = start_values(r, layout);
        EXPECT_NEAR(theta[layout.skew()], start.skew, 1e-15) << "dist " << static_cast<int>(start.dist);
        EXPECT_NEAR(theta[layout.shape()], start.shape, 1e-15) << "dist " << static_cast<int>(start.dist);
        if (layout.has_ghlambda())
        {
            EXPECT_NEAR(theta[layout.ghlambda()], start.ghlambda, 1e-15);
        }
    }
    for (int model = 0; model < 14; ++model)
    {
        for (int order : {1, 2, 6})
        {
            for (Dist dist : kSkewedDists)
            {
                auto spec_model = static_cast<Model>(model);
                Layout layout = ve::model::layout_of(
                    {.model = spec_model, .p = order, .q = 7 - order, .ar = 2, .ma = 1, .dist = dist});
                EXPECT_TRUE(is_valid(start_values(r, layout), layout))
                    << "model " << model << ", p " << order << ", dist " << static_cast<int>(dist);
            }
        }
    }
}

}

namespace
{

using ve::Spec;
using ve::model::layout_of;
using ve::model::to_unconstrained;

constexpr double kCeiling = 1.0 - 1e-8;

std::vector<double> start_of(const Spec& spec)
{
    return start_values(sample().r, layout_of(spec));
}

TEST(StartValues, PlainSpecsUnchangedByTheFixedPath)
{
    Sample s = sample();
    Layout plain = layout_of({.model = Model::SGarch, .p = 1, .q = 1, .dist = Dist::Std});
    std::vector<double> theta = start_values(s.r, plain);
    EXPECT_EQ(theta[plain.alpha_begin()], 0.05);
    EXPECT_EQ(theta[plain.beta_begin()], 0.9);
    EXPECT_EQ(theta[plain.shape()], 8.0);
    Layout cs = layout_of({.model = Model::CsGarch, .p = 2, .q = 6});
    theta = start_values(s.r, cs);
    EXPECT_EQ(theta[cs.eta1_begin()], 0.98);
    EXPECT_EQ(theta[cs.eta2_begin()], 0.03);
}

TEST(StartValues, UserStartAndFixedReplaceDefaults)
{
    Sample s = sample();
    Spec spec{.model = Model::SGarch,
              .dist = Dist::Std,
              .fixed = {{"mu", 0.0}},
              .start = {{"beta1", 0.8}, {"shape", 5.0}}};
    Layout layout = layout_of(spec);
    std::vector<double> theta = start_values(s.r, layout);
    EXPECT_EQ(theta[layout.mu()], 0.0);
    EXPECT_NEAR(theta[layout.omega()], 0.05 * s.variance, 1e-15);
    EXPECT_EQ(theta[layout.alpha_begin()], 0.05);
    EXPECT_EQ(theta[layout.beta_begin()], 0.8);
    EXPECT_EQ(theta[layout.shape()], 5.0);
    Layout ap = layout_of({.model = Model::ApArch, .dist = Dist::Ged, .fixed = {{"delta", 1.5}}});
    theta = start_values(s.r, ap);
    double kappa = ve::model::shock_moment<double>(theta, ap, 0);
    EXPECT_EQ(theta[ap.power()], 1.5);
    EXPECT_NEAR(theta[ap.alpha_begin()], 0.05 / kappa, 1e-15);
    EXPECT_NEAR(theta[ap.omega()], 0.05 * std::pow(s.variance, 0.75), 1e-15);
}

TEST(StartValues, DefaultWeightsRescaleIntoBudget)
{
    Layout fixed = layout_of({.model = Model::SGarch, .fixed = {{"alpha1", 0.3}}});
    std::vector<double> theta = start_of({.model = Model::SGarch, .fixed = {{"alpha1", 0.3}}});
    double scale = std::min(1.0, 0.95 * (kCeiling - 0.3) / 0.9);
    EXPECT_NEAR(theta[fixed.beta_begin()], 0.9 * scale, 1e-15);
    EXPECT_TRUE(is_valid(theta, fixed));
    theta = start_of({.model = Model::SGarch, .start = {{"alpha1", 0.03}}});
    EXPECT_EQ(theta[fixed.beta_begin()], 0.9);
}

TEST(StartValues, IGarchFixedAlphaLeavesADerivedBeta)
{
    Spec spec{.model = Model::IGarch, .p = 1, .q = 2, .fixed = {{"alpha1", 0.95}}};
    Layout layout = layout_of(spec);
    std::vector<double> theta = start_of(spec);
    double scale = 0.95 * 0.05 / 0.45;
    EXPECT_NEAR(theta[layout.beta_begin()], 0.45 * scale, 1e-15);
    EXPECT_GE(theta[layout.beta_begin() + 1], 0.0);
    EXPECT_NEAR(theta[layout.beta_begin() + 1], 1.0 - 0.95 - theta[layout.beta_begin()], 1e-15);
    EXPECT_TRUE(is_valid(theta, layout));
}

TEST(StartValues, GjrFixedGammaRaisesTheDefaultAlpha)
{
    Spec spec{.model = Model::GjrGarch, .fixed = {{"gamma1", -0.05}}};
    Layout layout = layout_of(spec);
    std::vector<double> theta = start_of(spec);
    double gamma_term = 0.05 - 0.5 * 0.05;
    double scale = std::min(1.0, 0.95 * (kCeiling - gamma_term) / (0.03 + 0.9));
    EXPECT_NEAR(theta[layout.alpha_begin()], 0.05 + 0.03 * scale, 1e-15);
    EXPECT_EQ(theta[layout.gamma_begin()], -0.05);
    EXPECT_TRUE(is_valid(theta, layout));
}

TEST(StartValues, CsGarchRescalesBelowRhoAndKeepsPhiInReach)
{
    Spec spec{.model = Model::CsGarch, .fixed = {{"alpha1", 0.9}}};
    Layout layout = layout_of(spec);
    std::vector<double> theta = start_of(spec);
    EXPECT_NEAR(theta[layout.beta_begin()], 0.9 * 0.95 * (0.98 - 0.9) / 0.9, 1e-15);
    EXPECT_EQ(theta[layout.eta2_begin()], std::min(0.03, 0.5 * theta[layout.beta_begin()]));
    EXPECT_TRUE(is_valid(theta, layout));
    spec = {.model = Model::CsGarch, .fixed = {{"alpha1", 0.97}}};
    theta = start_of(spec);
    EXPECT_LT(theta[layout.eta2_begin()], 0.03);
    EXPECT_TRUE(is_valid(theta, layout_of(spec)));
}

void expect_rescaled_under_given_rho(const Spec& spec)
{
    Layout layout = layout_of(spec);
    std::vector<double> theta = start_of(spec);
    double scale = std::min(1.0, 0.95 * 0.9 / 0.95);
    EXPECT_EQ(theta[layout.eta1_begin()], 0.9);
    EXPECT_NEAR(theta[layout.alpha_begin()], 0.05 * scale, 1e-15);
    EXPECT_NEAR(theta[layout.beta_begin()], 0.9 * scale, 1e-15);
    EXPECT_EQ(theta[layout.eta2_begin()], std::min(0.03, 0.5 * theta[layout.beta_begin()]));
    EXPECT_TRUE(is_valid(theta, layout));
}

TEST(StartValues, CsGarchRescalesDefaultWeightsUnderAGivenRho)
{
    expect_rescaled_under_given_rho({.model = Model::CsGarch, .fixed = {{"eta11", 0.9}}});
    expect_rescaled_under_given_rho({.model = Model::CsGarch, .start = {{"eta11", 0.9}}});
}

TEST(StartValues, InvalidCombinationStaysInvalid)
{
    Spec both{.model = Model::SGarch, .fixed = {{"alpha1", 0.6}, {"beta1", 0.5}}};
    EXPECT_FALSE(is_valid(start_of(both), layout_of(both)));
    Spec over{.model = Model::SGarch, .fixed = {{"alpha1", 1.2}}};
    EXPECT_FALSE(is_valid(start_of(over), layout_of(over)));
}

TEST(StartValues, WarmStartFromEstimate)
{
    Spec igarch{.model = Model::IGarch};
    Layout layout = layout_of(igarch);
    igarch.start = {{"mu", 0.01}, {"omega", 0.002}, {"alpha1", 0.12}, {"beta1", 0.88}};
    std::vector<double> theta = start_of(igarch);
    EXPECT_EQ(theta[layout.mu()], 0.01);
    EXPECT_EQ(theta[layout.omega()], 0.002);
    EXPECT_EQ(theta[layout.alpha_begin()], 0.12);
    EXPECT_NEAR(theta[layout.beta_begin()], 1.0 - 0.12, 1e-15);
    EXPECT_TRUE(is_valid(theta, layout));
    Spec tiny{.model = Model::SGarch, .start = {{"alpha1", 1e-12}}};
    Layout plain = layout_of(tiny);
    theta = start_of(tiny);
    EXPECT_TRUE(is_valid(theta, plain));
    std::vector<double> u(plain.count());
    to_unconstrained(theta, u, plain, {});
    EXPECT_TRUE(std::ranges::all_of(u, [](double value) { return std::isfinite(value); }));
}


// Two regressor columns, row-major.
std::vector<double> mean_regressors(std::size_t rows)
{
    std::vector<double> x(rows * 2);
    for (std::size_t t = 0; t < rows; ++t)
    {
        x[2 * t] = std::sin(0.37 * static_cast<double>(t));
        x[2 * t + 1] = std::cos(0.11 * static_cast<double>(t)) + 0.2 * std::sin(1.3 * static_cast<double>(t));
    }
    return x;
}

TEST(StartValues, OlsStart)
{
    std::vector<double> r = ve::testing::simulate_garch_series(5, 150);
    std::vector<double> x = mean_regressors(r.size());
    Layout layout = ve::model::layout_of({.mxreg = 2, .vxreg = 1, .archm = 1});
    ve::model::Data data{r, {x, 2}, {x, 2}};
    std::vector<double> a;
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        a.insert(a.end(), {1.0, x[2 * t], x[2 * t + 1]});
    }
    std::vector<double> b = ve::linalg::least_squares(a, 3, r);
    std::vector<double> theta = start_values(data, layout);
    EXPECT_TRUE(is_valid(theta, layout));
    EXPECT_NEAR(theta[layout.mu()], b[0], 1e-14);
    EXPECT_NEAR(theta[layout.mxreg_begin()], b[1], 1e-14);
    EXPECT_NEAR(theta[layout.mxreg_begin() + 1], b[2], 1e-14);
    EXPECT_EQ(theta[layout.archm_index()], 0.0);
    EXPECT_EQ(theta[layout.vxreg_begin()], 0.0);
    Layout plain = ve::model::layout_of({});
    EXPECT_EQ(start_values(r, plain), start_values(ve::model::Data{r}, plain));
}

TEST(StartValues, OlsStartWithoutMean)
{
    std::vector<double> r = ve::testing::simulate_garch_series(6, 100);
    std::vector<double> x = mean_regressors(r.size());
    Layout layout = ve::model::layout_of({.include_mean = false, .mxreg = 2});
    std::vector<double> b = ve::linalg::least_squares(x, 2, r);
    std::vector<double> theta = start_values(ve::model::Data{r, {x, 2}}, layout);
    EXPECT_NEAR(theta[layout.mxreg_begin()], b[0], 1e-14);
    EXPECT_NEAR(theta[layout.mxreg_begin() + 1], b[1], 1e-14);
}

TEST(StartValues, CollinearRegressorsStayFinite)
{
    std::vector<double> r = ve::testing::simulate_garch_series(7, 100);
    std::vector<double> x(r.size() * 3);
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        x[3 * t] = 1.0;
        x[3 * t + 1] = 1.0;
        x[3 * t + 2] = std::sin(0.3 * static_cast<double>(t));
    }
    Layout layout = ve::model::layout_of({.mxreg = 3});
    std::vector<double> theta = start_values(ve::model::Data{r, {x, 3}}, layout);
    for (double value : theta)
    {
        EXPECT_TRUE(std::isfinite(value));
    }
    EXPECT_TRUE(is_valid(theta, layout));
}

TEST(StartValues, FixedRegressorLeavesTheLeastSquaresToTheOthers)
{
    std::vector<double> r = ve::testing::simulate_garch_series(8, 150);
    std::vector<double> x = mean_regressors(r.size());
    Layout layout = ve::model::layout_of({.fixed = {{"mxreg1", 0.5}}, .mxreg = 2});
    std::vector<double> a;
    std::vector<double> y(r.size());
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        a.insert(a.end(), {1.0, x[2 * t + 1]});
        y[t] = r[t] - 0.5 * x[2 * t];
    }
    std::vector<double> b = ve::linalg::least_squares(a, 2, y);
    std::vector<double> theta = start_values(ve::model::Data{r, {x, 2}}, layout);
    EXPECT_EQ(theta[layout.mxreg_begin()], 0.5);
    EXPECT_NEAR(theta[layout.mu()], b[0], 1e-12);
    EXPECT_NEAR(theta[layout.mxreg_begin() + 1], b[1], 1e-12);
    EXPECT_TRUE(is_valid(theta, layout));
}

TEST(StartValues, ArfimaStart)
{
    std::vector<double> r = ve::testing::simulate_garch_series(9, 150);
    Layout layout = ve::model::layout_of({.ar = 1, .arfima = true});
    std::vector<double> theta = start_values(r, layout);
    EXPECT_EQ(theta[layout.arfima_index()], 0.25);
    EXPECT_TRUE(is_valid(theta, layout));
    std::vector<double> plain = start_values(r, ve::model::layout_of({.ar = 1}));
    theta.erase(theta.begin() + static_cast<std::ptrdiff_t>(layout.arfima_index()));
    EXPECT_EQ(theta, plain);
    Layout started = ve::model::layout_of({.ar = 1, .start = {{"arfima", 0.1}}, .arfima = true});
    EXPECT_EQ(start_values(r, started)[started.arfima_index()], 0.1);
    Layout fixed = ve::model::layout_of({.ar = 1, .fixed = {{"arfima", 0.2}}, .arfima = true});
    EXPECT_EQ(start_values(r, fixed)[fixed.arfima_index()], 0.2);
}

}
