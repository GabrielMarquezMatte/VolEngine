#include "model/parameter_map.hpp"

#include "dist/densities.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <numbers>
#include <random>
#include <span>
#include <string>
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

namespace
{

using ve::Model;
using ve::model::on_boundary;
using ve::model::power_of;
using ve::model::shape_parameters_of;
using ve::model::shock_moment;
using Indices = std::vector<std::size_t>;

constexpr double kCeiling = 1.0 - 1e-8;

constexpr Model kAllModels[] = {Model::SGarch,        Model::EGarch,         Model::GjrGarch,
                                Model::ApArch,        Model::IGarch,         Model::CsGarch,
                                Model::FGarchGarch,   Model::FGarchTGarch,   Model::FGarchAvGarch,
                                Model::FGarchNGarch,  Model::FGarchNaGarch,  Model::FGarchApArch,
                                Model::FGarchGjrGarch, Model::FGarchAllGarch};

std::vector<Layout> model_layouts()
{
    std::vector<Layout> layouts;
    for (Model model : kAllModels)
    {
        for (Orders orders : {Orders{1, 1}, Orders{2, 2}})
        {
            for (ArmaOrders arma : {ArmaOrders{0, 0}, ArmaOrders{1, 1}})
            {
                for (Dist dist : {Dist::Norm, Dist::Std, Dist::Ged})
                {
                    layouts.push_back(ve::model::layout_of({.model = model,
                                                            .p = orders.arch,
                                                            .q = orders.garch,
                                                            .ar = arma.ar,
                                                            .ma = arma.ma,
                                                            .dist = dist}));
                }
            }
        }
    }
    return layouts;
}

void fill_block(std::vector<double>& theta, std::size_t begin, int count, double value)
{
    std::fill_n(theta.begin() + static_cast<std::ptrdiff_t>(begin), count, value);
}

double sum_of(std::span<const double> values)
{
    double sum = 0.0;
    for (double value : values)
    {
        sum += value;
    }
    return sum;
}

// alpha = 0.08 / p, beta = 0.85 / q, gamma = 0.1 / p, eta1 = 0.3 (rho = 0.95 in csGARCH), eta2 = 0.2 (phi = 0.03
// in csGARCH), power 1.5; eGARCH omega = -0.1 and alpha = -0.05.
std::vector<double> model_interior_theta(const Layout& layout)
{
    std::vector<double> theta = interior_theta(layout);
    if (layout.model == Model::EGarch)
    {
        theta[layout.omega()] = -0.1;
        fill_block(theta, layout.alpha_begin(), layout.arch, -0.05);
    }
    fill_block(theta, layout.gamma_begin(), layout.gamma, 0.1 / layout.arch);
    bool cs = layout.model == Model::CsGarch;
    fill_block(theta, layout.eta1_begin(), layout.eta1, cs ? 0.95 : 0.3);
    fill_block(theta, layout.eta2_begin(), layout.eta2, cs ? 0.03 : 0.2);
    if (layout.free_power)
    {
        theta[layout.power()] = 1.5;
    }
    if (layout.has_derived())
    {
        std::span<const double> others(theta.data() + layout.alpha_begin(), layout.derived() - layout.alpha_begin());
        theta[layout.derived()] = 1.0 - sum_of(others);
    }
    return theta;
}

std::string name_of(const Layout& layout)
{
    return "model " + std::to_string(static_cast<int>(layout.model)) + ", count " + std::to_string(layout.count()) +
           ", dist " + std::to_string(static_cast<int>(layout.dist));
}

TEST(ParameterMap, SGarchUnchanged)
{
    Layout plain{.include_mean = true, .ar = 0, .ma = 0, .arch = 1, .garch = 1, .dist = Dist::Norm};
    std::vector<double> u{0.2, -1.0, 2.0, 0.5};
    std::vector<double> theta(4);
    std::vector<double> expected(4);
    to_natural<double>(u, theta, ve::model::layout_of({}), Standardization{0.01, 1.3});
    to_natural<double>(u, expected, plain, Standardization{0.01, 1.3});
    EXPECT_EQ(theta, expected);
}

TEST(ParameterMap, RoundTripAllModels)
{
    const Standardization standardization{0.03, 0.7};
    for (const Layout& layout : model_layouts())
    {
        SCOPED_TRACE(name_of(layout));
        std::vector<double> theta = model_interior_theta(layout);
        ASSERT_TRUE(is_valid(theta, layout));
        std::vector<double> u(theta.size());
        std::vector<double> back(theta.size());
        to_unconstrained(theta, u, layout, standardization);
        to_natural<double>(u, back, layout, standardization);
        for (std::size_t i = 0; i < theta.size(); ++i)
        {
            EXPECT_NEAR(back[i], theta[i], 1e-12 * std::max(1.0, std::abs(theta[i]))) << "parameter " << i;
        }
    }
}

// log E(|z| - eta1 z)^power of a ged, from its closed form.
double ged_log_moment(const Layout& layout, std::span<const double> theta, std::size_t i)
{
    double nu = shape_parameters_of(theta, layout).shape;
    double power = power_of(theta, layout);
    double eta1 = 0.0;
    if (layout.asymmetry_count() > 0)
    {
        eta1 = theta[layout.asymmetry_begin() + i];
    }
    double log_abs_moment = power * ve::dist::ged_log_scale(nu) + (power / nu) * std::numbers::ln2 +
                            std::lgamma((power + 1.0) / nu) - std::lgamma(1.0 / nu);
    return log_abs_moment + std::log(0.5 * (std::pow(1.0 - eta1, power) + std::pow(1.0 + eta1, power)));
}

// kappa itself is past the double range: +inf at an extreme power, or, at a ged shape near 0 with power != 2
// (log kappa ~ -0.2 / nu), below the smallest normal double, where alpha = w / kappa overflows.
bool moment_past_double_range(const Layout& layout, std::span<const double> theta, std::size_t i)
{
    double kappa = shock_moment<double>(theta, layout, i);
    if (std::isinf(kappa))
    {
        return true;
    }
    double log_smallest = std::log(std::numeric_limits<double>::min());
    return layout.dist == Dist::Ged && ged_log_moment(layout, theta, i) < log_smallest;
}

// The eta2 models take kappa by quadrature, which returns NaN or 0 where the closed form is still finite: a std
// shape near the power, a ged shape near 0, an extreme power.
bool quadrature_out_of_range(const Layout& layout, std::span<const double> theta, std::size_t i)
{
    double kappa = shock_moment<double>(theta, layout, i);
    return layout.eta2 > 0 && !(std::isfinite(kappa) && kappa > 0.0);
}

// Outside the domain whatever the map does: std with shape <= power, where kappa is infinite, or a kappa that is
// not a positive double.
bool moment_out_of_range(const Layout& layout, std::span<const double> theta)
{
    if (!layout.is_power_family())
    {
        return false;
    }
    if (layout.dist == Dist::Std && shape_parameters_of(theta, layout).shape <= power_of(theta, layout))
    {
        return true;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(layout.arch); ++i)
    {
        if (moment_past_double_range(layout, theta, i) || quadrature_out_of_range(layout, theta, i))
        {
            return true;
        }
    }
    return false;
}

TEST(ParameterMap, AlwaysValidAllModels)
{
    std::mt19937_64 engine(5);
    std::normal_distribution<double> draw(0.0, 3.0);
    for (const Layout& layout : model_layouts())
    {
        std::vector<double> u(layout.count());
        std::vector<double> theta(layout.count());
        for (int trial = 0; trial < 2000; ++trial)
        {
            std::ranges::generate(u, [&] { return draw(engine); });
            to_natural<double>(u, theta, layout, Standardization{0.01, 1.3});
            ASSERT_TRUE(is_valid(theta, layout) || moment_out_of_range(layout, theta))
                << name_of(layout) << ", trial " << trial;
        }
    }
}

TEST(ParameterMap, DualMatchesDoubleAllModels)
{
    using Dual42 = ve::ad::Dual<42>;
    const Standardization standardization{0.02, 1.7};
    std::mt19937_64 engine(9);
    std::normal_distribution<double> draw(0.0, 1.0);
    for (const Layout& layout : model_layouts())
    {
        std::vector<double> u(layout.count());
        std::ranges::generate(u, [&] { return draw(engine); });
        std::vector<Dual42> seeds;
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            seeds.push_back(ve::ad::variable<42>(u[i], i));
        }
        std::vector<Dual42> dual_theta(u.size());
        std::vector<double> theta(u.size());
        to_natural<Dual42>(seeds, dual_theta, layout, standardization);
        to_natural<double>(u, theta, layout, standardization);
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            ASSERT_EQ(ve::ad::value_of(dual_theta[i]), theta[i]) << name_of(layout) << ", parameter " << i;
        }
    }
}

TEST(ParameterMap, EGarchOmegaShiftsWithScale)
{
    Layout layout = ve::model::layout_of({.model = Model::EGarch, .p = 2, .q = 2});
    std::vector<double> u{0.1, -0.4, 0.3, -0.2, 1.5, 0.4, 0.2, -0.1};
    std::vector<double> theta(u.size());
    for (double scale : {0.01, 1.0, 37.0})
    {
        to_natural<double>(u, theta, layout, Standardization{0.0, scale});
        double beta_sum = theta[layout.beta_begin()] + theta[layout.beta_begin() + 1];
        EXPECT_NEAR(theta[layout.omega()], u[layout.omega()] + (1.0 - beta_sum) * std::log(scale * scale), 1e-15);
        EXPECT_EQ(theta[layout.alpha_begin()], u[layout.alpha_begin()]);
        EXPECT_EQ(theta[layout.gamma_begin() + 1], u[layout.gamma_begin() + 1]);
    }
}

TEST(ParameterMap, IGarchWeightsSumToOne)
{
    std::mt19937_64 engine(7);
    std::normal_distribution<double> draw(0.0, 2.0);
    for (Orders orders : {Orders{1, 1}, Orders{2, 3}, Orders{6, 6}})
    {
        Layout layout = ve::model::layout_of({.model = Model::IGarch, .p = orders.arch, .q = orders.garch});
        std::vector<double> u(layout.count());
        std::vector<double> theta(layout.count());
        std::size_t weight_count = layout.derived() + 1 - layout.alpha_begin();
        for (int trial = 0; trial < 100; ++trial)
        {
            std::ranges::generate(u, [&] { return draw(engine); });
            to_natural<double>(u, theta, layout, Standardization{});
            std::span<const double> weights(theta.data() + layout.alpha_begin(), weight_count);
            EXPECT_LE(std::abs(sum_of(weights) - 1.0), 1e-15);
            double derived = 1.0;
            for (double weight : weights.first(weight_count - 1))
            {
                derived -= weight;
            }
            EXPECT_EQ(theta[layout.derived()], derived);
        }
    }
}

TEST(ParameterMap, PowerAlphaCarriesMoment)
{
    std::mt19937_64 engine(3);
    std::normal_distribution<double> draw(0.0, 1.0);
    for (Model model : {Model::ApArch, Model::FGarchAllGarch})
    {
        for (Dist dist : {Dist::Norm, Dist::Std, Dist::Ged})
        {
            Layout layout = ve::model::layout_of({.model = model, .p = 2, .q = 2, .dist = dist});
            std::vector<double> u(layout.count());
            std::vector<double> theta(layout.count());
            std::ranges::generate(u, [&] { return draw(engine); });
            to_natural<double>(u, theta, layout, Standardization{});
            double persistence = theta[layout.beta_begin()] + theta[layout.beta_begin() + 1];
            for (std::size_t i = 0; i < 2; ++i)
            {
                persistence += theta[layout.alpha_begin() + i] * shock_moment<double>(theta, layout, i);
            }
            EXPECT_NEAR(persistence, logistic(u[layout.alpha_begin()]) * kCeiling, 1e-14) << name_of(layout);
        }
    }
}

TEST(ParameterMap, CsGarchOrdering)
{
    std::mt19937_64 engine(11);
    std::normal_distribution<double> draw(0.0, 3.0);
    Layout layout = ve::model::layout_of({.model = Model::CsGarch, .p = 2, .q = 2, .dist = Dist::Std});
    std::vector<double> u(layout.count());
    std::vector<double> theta(layout.count());
    for (int trial = 0; trial < 2000; ++trial)
    {
        std::ranges::generate(u, [&] { return draw(engine); });
        to_natural<double>(u, theta, layout, Standardization{0.0, 2.0});
        double beta_sum = theta[layout.beta_begin()] + theta[layout.beta_begin() + 1];
        double persistence = theta[layout.alpha_begin()] + theta[layout.alpha_begin() + 1] + beta_sum;
        double rho = theta[layout.eta1_begin()];
        double phi = theta[layout.eta2_begin()];
        ASSERT_LT(persistence, rho) << "trial " << trial;
        ASSERT_LT(rho, kCeiling) << "trial " << trial;
        ASSERT_GT(phi, 0.0) << "trial " << trial;
        ASSERT_LT(phi, beta_sum) << "trial " << trial;
    }
}

void expect_finite_unconstrained(std::span<const double> theta, const Layout& layout)
{
    ASSERT_TRUE(is_valid(theta, layout));
    std::vector<double> u(theta.size());
    to_unconstrained(theta, u, layout, Standardization{});
    for (double value : u)
    {
        EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(ParameterMap, TinyWeightsInvertFiniteAllModels)
{
    // gjr (omega, alpha1, beta1, gamma1) with alpha1 + gamma1 = 0.
    expect_finite_unconstrained(std::vector<double>{0.01, 0.05, 0.9, -0.05},
                                ve::model::layout_of({.model = Model::GjrGarch, .include_mean = false}));
    // cs (omega, alpha1, beta1, rho, phi) with phi = 0.
    expect_finite_unconstrained(std::vector<double>{0.01, 0.05, 0.9, 0.98, 0.0},
                                ve::model::layout_of({.model = Model::CsGarch, .include_mean = false}));
    // ALLGARCH (omega, alpha1, beta1, eta11, eta21, lambda) with beta1 = 0.
    expect_finite_unconstrained(std::vector<double>{0.01, 0.05, 0.0, 0.3, 0.2, 1.5},
                                ve::model::layout_of({.model = Model::FGarchAllGarch, .include_mean = false}));
}

Indices boundary_of(const Layout& layout, std::span<const double> theta)
{
    Indices marked;
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        if (on_boundary(layout, theta, i))
        {
            marked.push_back(i);
        }
    }
    return marked;
}

TEST(ParameterMap, BoundaryRules)
{
    // gjr (omega, alpha1, alpha2, beta1, gamma1, gamma2)
    Layout gjr = ve::model::layout_of({.model = Model::GjrGarch, .p = 2, .include_mean = false});
    EXPECT_EQ(boundary_of(gjr, std::vector<double>{0.01, 1e-9, 0.05, 0.8, 0.1, 0.1}), Indices{1});
    EXPECT_EQ(boundary_of(gjr, std::vector<double>{0.01, 0.05, 0.05, 0.8, -0.05 + 1e-9, 0.1}), Indices{4});
    EXPECT_EQ(boundary_of(gjr, std::vector<double>{0.01, 0.05, 0.05, 0.8, -0.04, 0.1}), Indices{});
    // ALLGARCH (omega, alpha1, beta1, eta11, eta21, lambda)
    Layout all = ve::model::layout_of({.model = Model::FGarchAllGarch, .include_mean = false});
    EXPECT_EQ(boundary_of(all, std::vector<double>{0.01, 0.05, 0.9, 1.0 - 1e-9, 0.2, 1.5}), Indices{3});
    EXPECT_EQ(boundary_of(all, std::vector<double>{0.01, 0.05, 0.9, -1.0 + 1e-9, -5.0, 1.5}), Indices{3});
    EXPECT_EQ(boundary_of(all, std::vector<double>{0.01, 0.05, 0.9, 0.3, 0.0, 1e-9}), Indices{});
    // apARCH (omega, alpha1, beta1, gamma1, delta)
    Layout aparch = ve::model::layout_of({.model = Model::ApArch, .include_mean = false});
    EXPECT_EQ(boundary_of(aparch, std::vector<double>{0.01, 0.05, 0.9, -1.0 + 1e-9, 1.5}), Indices{3});
    // eGARCH (omega, alpha1, beta1, gamma1): alpha and gamma are free reals.
    Layout egarch = ve::model::layout_of({.model = Model::EGarch, .include_mean = false});
    EXPECT_EQ(boundary_of(egarch, std::vector<double>{-0.1, 0.0, 0.9, 0.0}), Indices{});
    // cs (omega, alpha1, beta1, rho, phi)
    Layout cs = ve::model::layout_of({.model = Model::CsGarch, .include_mean = false});
    EXPECT_EQ(boundary_of(cs, std::vector<double>{0.01, 0.05, 0.9, 0.98, 1e-9}), Indices{4});
    EXPECT_EQ(boundary_of(cs, std::vector<double>{0.01, 0.05, 0.9, 0.95 + 1e-9, 0.03}), Indices{3});
    // iGARCH (mu, omega, alpha1, beta1, beta2): the derived beta2, always.
    Layout igarch = ve::model::layout_of({.model = Model::IGarch, .q = 2});
    EXPECT_EQ(boundary_of(igarch, std::vector<double>{0.0, 0.01, 0.1, 0.5, 0.4}), Indices{4});
    // std sGARCH (omega, alpha1, beta1, shape)
    Layout student = ve::model::layout_of({.dist = Dist::Std, .include_mean = false});
    EXPECT_EQ(boundary_of(student, std::vector<double>{0.01, 0.05, 0.9, 1e9}), Indices{3});
    EXPECT_EQ(boundary_of(student, std::vector<double>{0.01, 1e-9, 0.9, 8.0}), Indices{1});
}

TEST(ParameterMap, IsValidRejectsVariantViolations)
{
    Layout gjr = ve::model::layout_of({.model = Model::GjrGarch, .include_mean = false});
    EXPECT_TRUE(is_valid(std::vector<double>{0.01, 0.05, 0.85, 0.1}, gjr));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 0.05, 0.85, -0.06}, gjr));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 0.05, 0.85, 0.21}, gjr));
    Layout egarch = ve::model::layout_of({.model = Model::EGarch, .include_mean = false});
    EXPECT_TRUE(is_valid(std::vector<double>{-0.1, -0.3, 0.9, 0.2}, egarch));
    EXPECT_FALSE(is_valid(std::vector<double>{-0.1, -0.3, 1.0, 0.2}, egarch));
    Layout igarch = ve::model::layout_of({.model = Model::IGarch, .include_mean = false});
    EXPECT_TRUE(is_valid(std::vector<double>{0.01, 1.0, 123.0}, igarch));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 1.1, -0.1}, igarch));
    Layout aparch = ve::model::layout_of({.model = Model::ApArch, .dist = Dist::Std, .include_mean = false});
    EXPECT_TRUE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 0.3, 1.5, 6.0}, aparch));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 1.0, 1.5, 6.0}, aparch));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 0.3, 0.0, 6.0}, aparch));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 0.3, 6.5, 6.0}, aparch));
    Layout cs = ve::model::layout_of({.model = Model::CsGarch, .include_mean = false});
    EXPECT_TRUE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 0.98, 0.03}, cs));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 0.94, 0.03}, cs));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 0.98, 0.91}, cs));
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 1.0, 0.03}, cs));
}

}
