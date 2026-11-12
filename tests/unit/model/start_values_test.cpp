#include "model/start_values.hpp"

#include "dist/moments.hpp"
#include "model/parameter_map.hpp"
#include "simulated_garch.hpp"

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
