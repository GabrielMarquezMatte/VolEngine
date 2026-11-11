#include "model/parameter_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <random>
#include <span>
#include <tuple>
#include <vector>

namespace
{

using ve::Dist;
using ve::model::is_valid;
using ve::model::Layout;
using ve::model::Standardization;
using ve::model::to_natural;
using ve::model::to_unconstrained;

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

using GridCase = std::tuple<Orders, ArmaOrders, Dist, bool>;

Layout grid_layout(const GridCase& c)
{
    auto [orders, arma, dist, include_mean] = c;
    return Layout{.include_mean = include_mean,
                  .ar = arma.ar,
                  .ma = arma.ma,
                  .arch = orders.arch,
                  .garch = orders.garch,
                  .dist = dist};
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

std::vector<GridCase> grid()
{
    std::vector<GridCase> cases;
    for (Orders orders : {Orders{1, 1}, Orders{2, 1}, Orders{1, 2}, Orders{3, 3}})
    {
        for (ArmaOrders arma : {ArmaOrders{0, 0}, ArmaOrders{1, 1}, ArmaOrders{2, 0}})
        {
            for (Dist dist : {Dist::Norm, Dist::Std, Dist::Ged})
            {
                for (bool include_mean : {true, false})
                {
                    cases.emplace_back(orders, arma, dist, include_mean);
                }
            }
        }
    }
    return cases;
}

double logistic(double x)
{
    return 1.0 / (1.0 + std::exp(-x));
}

TEST(ParameterMap, OneOneMatchesPlanOneTransform)
{
    Layout layout{.include_mean = true, .ar = 0, .ma = 0, .arch = 1, .garch = 1, .dist = Dist::Norm};
    std::vector<double> u{0.2, -1.0, 2.0, 0.5};
    std::vector<double> theta(4);
    to_natural<double>(u, theta, layout, Standardization{0.01, 1.3});
    double persistence = logistic(2.0) * (1.0 - 1e-8);
    EXPECT_NEAR(theta[0], 0.01 + 1.3 * 0.2, 1e-15);
    EXPECT_NEAR(theta[1], 1.69 * std::exp(-1.0), 1e-15);
    EXPECT_NEAR(theta[2], persistence * logistic(0.5), 1e-15);
    EXPECT_NEAR(theta[3], persistence * (1.0 - logistic(0.5)), 1e-15);
}

TEST(ParameterMap, RoundTripAllShapes)
{
    const Standardization standardization{0.03, 0.7};
    for (const GridCase& c : grid())
    {
        Layout layout = grid_layout(c);
        std::vector<double> theta = interior_theta(layout);
        std::vector<double> u(theta.size());
        std::vector<double> back(theta.size());
        to_unconstrained(theta, u, layout, standardization);
        to_natural<double>(u, back, layout, standardization);
        for (std::size_t i = 0; i < theta.size(); ++i)
        {
            EXPECT_NEAR(back[i], theta[i], 1e-12 * std::max(1.0, std::abs(theta[i])))
                << "parameter " << i << " of layout with " << theta.size() << " parameters";
        }
    }
}

TEST(ParameterMap, AlwaysValid)
{
    std::mt19937_64 engine(5);
    std::normal_distribution<double> draw(0.0, 3.0);
    for (const GridCase& c : grid())
    {
        Layout layout = grid_layout(c);
        std::vector<double> u(layout.count());
        std::vector<double> theta(layout.count());
        for (int trial = 0; trial < 2000; ++trial)
        {
            std::ranges::generate(u, [&] { return draw(engine); });
            to_natural<double>(u, theta, layout, Standardization{0.01, 1.3});
            ASSERT_TRUE(is_valid(theta, layout)) << "trial " << trial;
        }
    }
}

TEST(ParameterMap, TinyWeightsInvertFinite)
{
    Layout layout{.include_mean = false, .ar = 0, .ma = 0, .arch = 2, .garch = 1, .dist = Dist::Norm};
    std::vector<double> theta{0.01, 0.05, 1e-14, 0.0};
    std::vector<double> u(4);
    to_unconstrained(theta, u, layout, Standardization{});
    for (double value : u)
    {
        EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(ParameterMap, IsValidRejectsEachViolation)
{
    Layout layout{.include_mean = false, .ar = 0, .ma = 0, .arch = 1, .garch = 1, .dist = Dist::Std};
    EXPECT_TRUE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 6.0}, layout));
    EXPECT_FALSE(is_valid(std::vector<double>{0.0, 0.05, 0.9, 6.0}, layout));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, -0.05, 0.9, 6.0}, layout));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 0.1, 0.9, 6.0}, layout));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 2.0}, layout));
}

TEST(ParameterMap, DualMatchesDouble)
{
    using Dual32 = ve::ad::Dual<32>;
    const Standardization standardization{0.02, 1.7};
    std::mt19937_64 engine(9);
    std::normal_distribution<double> draw(0.0, 2.0);
    for (const GridCase& c : grid())
    {
        Layout layout = grid_layout(c);
        std::vector<double> u(layout.count());
        std::ranges::generate(u, [&] { return draw(engine); });
        std::vector<Dual32> seeds;
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            seeds.push_back(ve::ad::variable<32>(u[i], i));
        }
        std::vector<Dual32> dual_theta(u.size());
        std::vector<double> theta(u.size());
        to_natural<Dual32>(seeds, dual_theta, layout, standardization);
        to_natural<double>(u, theta, layout, standardization);
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            ASSERT_EQ(ve::ad::value_of(dual_theta[i]), theta[i]) << "parameter " << i;
        }
    }
}

TEST(ParameterMap, DualJacobianMatchesFiniteDifferences)
{
    using Dual32 = ve::ad::Dual<32>;
    const Standardization standardization{0.02, 1.7};
    Layout layout{.include_mean = true, .ar = 1, .ma = 1, .arch = 2, .garch = 2, .dist = Dist::Std};
    std::vector<double> u{0.1, 0.2, -0.3, -0.4, 0.5, 0.6, -0.2, 0.3, 0.7};
    std::vector<Dual32> seeds;
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        seeds.push_back(ve::ad::variable<32>(u[i], i));
    }
    std::vector<Dual32> dual_theta(u.size());
    to_natural<Dual32>(seeds, dual_theta, layout, standardization);
    constexpr double kStep = 1e-6;
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        std::vector<double> up = u;
        std::vector<double> down = u;
        up[i] += kStep;
        down[i] -= kStep;
        std::vector<double> theta_up(u.size());
        std::vector<double> theta_down(u.size());
        to_natural<double>(up, theta_up, layout, standardization);
        to_natural<double>(down, theta_down, layout, standardization);
        for (std::size_t j = 0; j < u.size(); ++j)
        {
            EXPECT_NEAR(dual_theta[j].grad[i], (theta_up[j] - theta_down[j]) / (2.0 * kStep), 1e-7)
                << "d theta" << j << " / d u" << i;
        }
    }
}

TEST(ParameterMap, ExtremeUnconstrainedValuesStayValid)
{
    Layout layout{.include_mean = false, .ar = 0, .ma = 0, .arch = 1, .garch = 1, .dist = Dist::Norm};
    std::vector<double> theta(3);
    to_natural<double>(std::vector<double>{0.0, 800.0, -800.0}, theta, layout, Standardization{});
    EXPECT_EQ(theta[0], 1.0);
    EXPECT_LT(theta[1] + theta[2], 1.0);
    to_natural<double>(std::vector<double>{0.0, -800.0, 800.0}, theta, layout, Standardization{});
    EXPECT_EQ(theta[1] + theta[2], 0.0);
}

}
