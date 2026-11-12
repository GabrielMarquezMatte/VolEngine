#include "model/parameter_map.hpp"

#include "dist/densities.hpp"
#include "dist/moments.hpp"
#include "model/data.hpp"
#include "model/likelihood.hpp"
#include "model/start_values.hpp"
#include "model/weight_terms.hpp"
#include "simulated_garch.hpp"

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
#include <utility>
#include <vector>

namespace
{

using ve::Dist;
using ve::model::is_valid;
using ve::model::Layout;
using ve::model::Standardization;
using ve::model::to_natural;
using ve::model::to_unconstrained;

// Bit for bit, where two NaN agree: a point outside the computable domain maps to NaN in both.
bool same_value(double a, double b)
{
    return (std::isnan(a) && std::isnan(b)) || a == b;
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
    ve::testing::PortableNormal draw{3.0};
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
    ve::testing::PortableNormal draw{2.0};
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
            ASSERT_TRUE(same_value(ve::ad::value_of(dual_theta[i]), theta[i])) << "parameter " << i;
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

// The eta2 models and the skewed distributions take kappa by quadrature, which returns NaN or 0 where the
// closed form is still finite: a std shape near the power, a ged shape near 0, an extreme power; under the
// skewed distributions also an sstd shape near 2 or a jsu shape near 0.
bool quadrature_out_of_range(const Layout& layout, std::span<const double> theta, std::size_t i)
{
    double kappa = shock_moment<double>(theta, layout, i);
    return (layout.eta2 > 0 || layout.has_skew()) && !(std::isfinite(kappa) && kappa > 0.0);
}

// Outside the domain whatever the map does: std or sstd with shape <= power, where kappa is infinite, or a
// kappa that is not a positive double.
bool moment_out_of_range(const Layout& layout, std::span<const double> theta)
{
    if (!layout.is_power_family())
    {
        return false;
    }
    bool student_tail = layout.dist == Dist::Std || layout.dist == Dist::Sstd;
    if (student_tail && shape_parameters_of(theta, layout).shape <= power_of(theta, layout))
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
    ve::testing::PortableNormal draw{3.0};
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
    using DualMax = ve::ad::Dual<ve::kMaxParameterCount>;
    const Standardization standardization{0.02, 1.7};
    std::mt19937_64 engine(9);
    ve::testing::PortableNormal draw{1.0};
    for (const Layout& layout : model_layouts())
    {
        std::vector<double> u(layout.count());
        std::ranges::generate(u, [&] { return draw(engine); });
        std::vector<DualMax> seeds;
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            seeds.push_back(ve::ad::variable<ve::kMaxParameterCount>(u[i], i));
        }
        std::vector<DualMax> dual_theta(u.size());
        std::vector<double> theta(u.size());
        to_natural<DualMax>(seeds, dual_theta, layout, standardization);
        to_natural<double>(u, theta, layout, standardization);
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            ASSERT_TRUE(same_value(ve::ad::value_of(dual_theta[i]), theta[i]))
                << name_of(layout) << ", parameter " << i;
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
    ve::testing::PortableNormal draw{2.0};
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
    ve::testing::PortableNormal draw{1.0};
    for (Model model : {Model::ApArch, Model::FGarchAllGarch})
    {
        for (Dist dist : {Dist::Norm, Dist::Std, Dist::Ged})
        {
            Layout layout = ve::model::layout_of({.model = model, .p = 2, .q = 2, .dist = dist});
            std::vector<double> u(layout.count());
            std::vector<double> theta(layout.count());
            std::ranges::generate(u, [&] { return draw(engine); });
            if (dist == Dist::Std)
            {
                // A std shape at or below the power has no moment; nu = 2 + lambda keeps it above.
                u[layout.shape()] = u[layout.power()];
            }
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
    ve::testing::PortableNormal draw{3.0};
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

namespace
{

using ve::ad::Dual;
using ve::dist::ShapeParameters;

constexpr Dist kSkewedDists[] = {Dist::Sstd, Dist::Sged, Dist::Nig, Dist::Ghyp, Dist::Jsu};

// The gjrGARCH(1,1) weights of plan 2b, operation for operation: P = logistic(v) (1 - 1e-8) with v >= 0, the
// softmax of (u_beta, u_gamma, 0), then alpha = 2 w_alpha and gamma = 2 (w_gamma - w_alpha).
template <class T>
std::vector<T> two_b_gjr_weights(std::span<const T> u)
{
    using std::exp;
    T persistence = (1.0 / (1.0 + exp(-u[1]))) * kCeiling;
    double peak = std::max({0.0, ve::ad::value_of(u[2]), ve::ad::value_of(u[3])});
    std::vector<T> weights{exp(u[2] - peak), exp(u[3] - peak), exp(T{} - peak)};
    T total{};
    for (const T& weight : weights)
    {
        total = total + weight;
    }
    for (T& weight : weights)
    {
        weight = persistence * (weight / total);
    }
    return {2.0 * weights[0], weights[1], 2.0 * (weights[2] - weights[0])};
}

TEST(ParameterMap, SymmetricUnchanged)
{
    // std gjrGARCH (omega, alpha1, beta1, gamma1, shape): pi = 0.5 exactly gives the 2b weights bit for bit.
    Layout layout =
        ve::model::layout_of({.model = Model::GjrGarch, .dist = Dist::Std, .include_mean = false});
    std::vector<double> u{-1.0, 1.5, 0.8, -0.4, 0.7};
    std::vector<Dual<8>> seeds;
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        seeds.push_back(ve::ad::variable<8>(u[i], i));
    }
    std::vector<double> theta(u.size());
    std::vector<Dual<8>> dual_theta(u.size());
    to_natural<double>(u, theta, layout, Standardization{});
    to_natural<Dual<8>>(seeds, dual_theta, layout, Standardization{});
    std::vector<double> expected = two_b_gjr_weights<double>(u);
    std::vector<Dual<8>> dual_expected = two_b_gjr_weights<Dual<8>>(seeds);
    for (std::size_t k = 0; k < 3; ++k)
    {
        EXPECT_EQ(theta[1 + k], expected[k]) << "weight " << k;
        EXPECT_EQ(dual_theta[1 + k].value, expected[k]) << "weight " << k;
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            EXPECT_EQ(dual_theta[1 + k].grad[i], dual_expected[k].grad[i])
                << "d weight " << k << " / d u" << i;
        }
    }
}

// skew, shape, ghlambda: sstd (0.8, 6), sged (1.2, 1.5), nig (-0.3, 1.5), ghyp (0.3, 2, 1.5), jsu (0.5, 2).
ShapeParameters<double> interior_parameters(Dist dist)
{
    switch (dist)
    {
        case Dist::Sstd:
            return {0.8, 6.0, 0.0};
        case Dist::Sged:
            return {1.2, 1.5, 0.0};
        case Dist::Nig:
            return {-0.3, 1.5, 0.0};
        case Dist::Ghyp:
            return {0.3, 2.0, 1.5};
        case Dist::Jsu:
            return {0.5, 2.0, 0.0};
        case Dist::Norm:
        case Dist::Std:
        case Dist::Ged:
            break;
    }
    return {};
}

void set_parameters(std::vector<double>& theta, const Layout& layout,
                    const ShapeParameters<double>& parameters)
{
    if (layout.has_skew())
    {
        theta[layout.skew()] = parameters.skew;
    }
    if (layout.has_shape())
    {
        theta[layout.shape()] = parameters.shape;
    }
    if (layout.has_ghlambda())
    {
        theta[layout.ghlambda()] = parameters.ghlambda;
    }
}

// The 14 models at (1,1), ARMA (0,0) and (1,1), under the five skewed distributions.
std::vector<Layout> skewed_layouts()
{
    std::vector<Layout> layouts;
    for (Model model : kAllModels)
    {
        for (ArmaOrders arma : {ArmaOrders{0, 0}, ArmaOrders{1, 1}})
        {
            for (Dist dist : kSkewedDists)
            {
                layouts.push_back(
                    ve::model::layout_of({.model = model, .ar = arma.ar, .ma = arma.ma, .dist = dist}));
            }
        }
    }
    return layouts;
}

void expect_round_trip(std::span<const double> theta, const Layout& layout)
{
    const Standardization standardization{0.03, 0.7};
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

TEST(ParameterMap, RoundTripAllDistributions)
{
    for (const Layout& layout : skewed_layouts())
    {
        SCOPED_TRACE(name_of(layout));
        std::vector<double> theta = model_interior_theta(layout);
        set_parameters(theta, layout, interior_parameters(layout.dist));
        expect_round_trip(theta, layout);
    }
    for (Model model : {Model::GjrGarch, Model::ApArch})
    {
        for (Dist dist : {Dist::Sstd, Dist::Sged})
        {
            for (double skew : {0.05, 20.0})
            {
                Layout layout = ve::model::layout_of({.model = model, .dist = dist});
                SCOPED_TRACE(name_of(layout) + ", skew " + std::to_string(skew));
                std::vector<double> theta = model_interior_theta(layout);
                set_parameters(theta, layout, {skew, interior_parameters(dist).shape, 0.0});
                expect_round_trip(theta, layout);
            }
        }
    }
}

// A density that is not finite at z = 0 (jsu with a shape near 0), or, in gjrGARCH, a pi = P(z < 0) within
// 1e-8 of 0 or 1, which cannot split alpha + gamma.
bool shock_out_of_range(const Layout& layout, std::span<const double> theta)
{
    ShapeParameters<double> parameters = shape_parameters_of(theta, layout);
    double at_zero = ve::dist::log_density(ve::dist::make_density(layout.dist, parameters), 0.0);
    if (!std::isfinite(at_zero))
    {
        return true;
    }
    if (layout.model != Model::GjrGarch)
    {
        return false;
    }
    double pi = ve::dist::probability_below_zero(layout.dist, parameters);
    return !(pi >= 1e-8 && pi <= 1.0 - 1e-8);
}

// Outside the computable domain whatever the map does: some kappa not a positive finite double, pi within
// 1e-8 of 0 or 1, or a density that is not finite at 0.
bool outside_computable_domain(const Layout& layout, std::span<const double> theta)
{
    return moment_out_of_range(layout, theta) || (layout.has_skew() && shock_out_of_range(layout, theta));
}

// The invalid theta among `trials` draws of u ~ N(0, 3); each must be outside the computable domain, and its
// likelihood must not be finite.
int exempt_draws(const Layout& layout, std::mt19937_64& engine, int trials, std::span<const double> r)
{
    ve::testing::PortableNormal draw{3.0};
    std::vector<double> u(layout.count());
    std::vector<double> theta(layout.count());
    int exempt = 0;
    for (int trial = 0; trial < trials; ++trial)
    {
        std::ranges::generate(u, [&] { return draw(engine); });
        to_natural<double>(u, theta, layout, Standardization{0.01, 1.3});
        if (is_valid(theta, layout))
        {
            continue;
        }
        ++exempt;
        double ll = ve::model::loglik<double>(r, theta, layout, ve::VarInit::Sample, {});
        if (!outside_computable_domain(layout, theta) || std::isfinite(ll))
        {
            ADD_FAILURE() << name_of(layout) << ", trial " << trial << ", loglik " << ll;
            return exempt;
        }
    }
    return exempt;
}

// The 14 models at (1,1) under the five skewed distributions; ARMA plays no part in validity. The layouts
// that read kappa or pi take both by quadrature, in to_natural and in is_valid, and get fewer draws. Up to
// ~1/3 of the draws fall outside the computable domain (sstd with shape <= a free power, jsu with a shape
// below ~0.06, an extreme free power); the 40% cap fails a map that invalidates everything.
TEST(ParameterMap, AlwaysValidSkewed)
{
    std::mt19937_64 engine(5);
    std::vector<double> r = ve::testing::simulate_garch_series(42, 50);
    for (Model model : kAllModels)
    {
        for (Dist dist : kSkewedDists)
        {
            Layout layout = ve::model::layout_of({.model = model, .dist = dist});
            bool quadrature = layout.is_power_family() || model == Model::GjrGarch;
            int trials = quadrature ? 1000 : 2000;
            int exempt = exempt_draws(layout, engine, trials, r);
            EXPECT_LT(exempt, 2 * trials / 5) << name_of(layout);
        }
    }
}

TEST(ParameterMap, GjrPersistenceUsesProbabilityBelowZero)
{
    Layout layout = ve::model::layout_of({.model = Model::GjrGarch, .p = 2, .q = 2, .dist = Dist::Sstd});
    std::mt19937_64 engine(3);
    ve::testing::PortableNormal draw{1.0};
    std::vector<double> u(layout.count());
    std::ranges::generate(u, [&] { return draw(engine); });
    u[layout.skew()] = std::log(0.7);
    u[layout.shape()] = std::log(6.0 - 2.0);
    std::vector<double> theta(layout.count());
    to_natural<double>(u, theta, layout, Standardization{});
    double pi = ve::dist::probability_below_zero(Dist::Sstd, shape_parameters_of<double>(theta, layout));
    ASSERT_LT(pi, 0.49);
    double persistence = 0.0;
    for (std::size_t i = 0; i < 2; ++i)
    {
        persistence += theta[layout.alpha_begin() + i] + theta[layout.beta_begin() + i] +
                       pi * theta[layout.gamma_begin() + i];
    }
    EXPECT_NEAR(persistence, logistic(u[layout.alpha_begin()]) * kCeiling, 1e-14);
    EXPECT_TRUE(is_valid(theta, layout));
}

// gjrGARCH reads alpha back as w_alpha / (1 - pi) and gamma as w_gamma / pi - alpha, so a pi within 1e-8 of
// 0 or 1 makes theta invalid even with finite weights. The sged here is past what the quadrature resolves.
TEST(ParameterMap, GjrNeedsProbabilityBelowZeroInsideUnitInterval)
{
    // sged gjrGARCH (omega, alpha1, beta1, gamma1, skew, shape)
    Layout layout =
        ve::model::layout_of({.model = Model::GjrGarch, .dist = Dist::Sged, .include_mean = false});
    std::vector<double> sharp{0.01, 0.05, 0.85, 0.1, 19.48, 0.000646};
    double pi = ve::dist::probability_below_zero(Dist::Sged, shape_parameters_of<double>(sharp, layout));
    EXPECT_EQ(is_valid(sharp, layout), pi > 0.0 && pi < 1.0) << "pi " << pi;
    EXPECT_TRUE(is_valid(std::vector<double>{0.01, 0.05, 0.85, 0.1, 19.48, 1.5}, layout));
}

// alpha_i is read back as w / kappa_i, so a kappa that is not a positive finite double makes theta invalid:
// here the jsu constant c overflows to 0 at tau = 0.03, and kappa is NaN.
TEST(ParameterMap, PowerFamilyNeedsPositiveFiniteMoment)
{
    // jsu apARCH (omega, alpha1, beta1, gamma1, delta, skew, shape)
    Layout layout = ve::model::layout_of({.model = Model::ApArch, .dist = Dist::Jsu, .include_mean = false});
    EXPECT_FALSE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 0.0, 2.0, 0.5, 0.03}, layout));
    EXPECT_TRUE(is_valid(std::vector<double>{0.01, 0.05, 0.9, 0.0, 2.0, 0.5, 2.0}, layout));
}

// jsu with skew 2.89 and shape 0.063 puts pi = P(z < 0) within ~1e-13 of 1, where 1 - pi keeps only a few
// digits: alpha = w / (1 - pi) came out ~1e12 and gamma ~ -1e12, which is_valid rejected while the likelihood
// stayed finite. Such a pi is not computable: the mapped theta is NaN.
TEST(ParameterMap, GjrProbabilityNearOneIsNotComputable)
{
    Layout layout = ve::model::layout_of({.model = Model::GjrGarch, .dist = Dist::Jsu});
    std::vector<double> u{0x1.aba8dd8d08dfap+0,  -0x1.080551346a65fp+2, 0x1.3d39091c43106p+3,
                          -0x1.d08f906568f92p+1, 0x1.96d8429f4afe8p+1,  0x1.71b5b122fc5a0p+1,
                          -0x1.623dbce211174p+1};
    std::vector<double> theta(layout.count());
    to_natural<double>(u, theta, layout, Standardization{0.01, 1.3});
    double pi = ve::dist::probability_below_zero(Dist::Jsu, ve::model::shape_parameters_of<double>(theta, layout));
    ASSERT_GT(pi, 1.0 - 1e-8);
    EXPECT_FALSE(is_valid(theta, layout));
    EXPECT_TRUE(std::isnan(theta[layout.alpha_begin()]));
    EXPECT_FALSE(std::isfinite(ve::model::loglik<double>(ve::testing::simulate_garch_series(3, 200), theta,
                                                         layout, ve::VarInit::Sample, {})));
}

// gjrGARCH, apARCH and fGARCH-ALLGARCH read pi or kappa, taken by quadrature under the skewed distributions.
std::vector<Layout> skewed_quadrature_layouts()
{
    std::vector<Layout> layouts;
    for (Model model : {Model::GjrGarch, Model::ApArch, Model::FGarchAllGarch})
    {
        for (Dist dist : kSkewedDists)
        {
            layouts.push_back(ve::model::layout_of({.model = model, .ar = 1, .ma = 1, .dist = dist}));
        }
    }
    return layouts;
}

TEST(ParameterMap, DualMatchesDoubleSkewed)
{
    using DualMax = ve::ad::Dual<ve::kMaxParameterCount>;
    const Standardization standardization{0.02, 1.7};
    std::mt19937_64 engine(9);
    ve::testing::PortableNormal draw{1.0};
    for (const Layout& layout : skewed_quadrature_layouts())
    {
        std::vector<double> u(layout.count());
        std::ranges::generate(u, [&] { return draw(engine); });
        std::vector<DualMax> seeds;
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            seeds.push_back(ve::ad::variable<ve::kMaxParameterCount>(u[i], i));
        }
        std::vector<DualMax> dual_theta(u.size());
        std::vector<double> theta(u.size());
        to_natural<DualMax>(seeds, dual_theta, layout, standardization);
        to_natural<double>(u, theta, layout, standardization);
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            ASSERT_TRUE(same_value(ve::ad::value_of(dual_theta[i]), theta[i]))
                << name_of(layout) << ", parameter " << i;
        }
    }
}

void expect_jacobian_matches(const Layout& layout, std::span<const double> u,
                             const Standardization& standardization = Standardization{0.02, 1.7})
{
    using Dual16 = ve::ad::Dual<16>;
    std::vector<Dual16> seeds;
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        seeds.push_back(ve::ad::variable<16>(u[i], i));
    }
    std::vector<Dual16> dual_theta(u.size());
    to_natural<Dual16>(seeds, dual_theta, layout, standardization);
    constexpr double kStep = 1e-6;
    std::vector<double> up(u.begin(), u.end());
    std::vector<double> down(u.begin(), u.end());
    std::vector<double> theta_up(u.size());
    std::vector<double> theta_down(u.size());
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        up[i] += kStep;
        down[i] -= kStep;
        to_natural<double>(up, theta_up, layout, standardization);
        to_natural<double>(down, theta_down, layout, standardization);
        up[i] = u[i];
        down[i] = u[i];
        for (std::size_t j = 0; j < u.size(); ++j)
        {
            double difference = (theta_up[j] - theta_down[j]) / (2.0 * kStep);
            double gradient = dual_theta[j].grad[i];
            EXPECT_NEAR(gradient, difference, 1e-7 * std::max(1.0, std::abs(gradient)))
                << name_of(layout) << ", d theta" << j << " / d u" << i;
        }
    }
}

TEST(ParameterMap, DualJacobianMatchesFiniteDifferencesSkewed)
{
    std::mt19937_64 engine(17);
    ve::testing::PortableNormal draw{0.5};
    for (const Layout& layout : skewed_quadrature_layouts())
    {
        std::vector<double> u(layout.count());
        std::ranges::generate(u, [&] { return draw(engine); });
        expect_jacobian_matches(layout, u);
    }
}

TEST(ParameterMap, DistributionBoundaries)
{
    // nig sGARCH (omega, alpha1, beta1, skew, shape)
    Layout nig = ve::model::layout_of({.dist = Dist::Nig, .include_mean = false});
    EXPECT_EQ(boundary_of(nig, std::vector<double>{0.01, 0.05, 0.9, 1.0 - 1e-9, 1.5}), Indices{3});
    EXPECT_EQ(boundary_of(nig, std::vector<double>{0.01, 0.05, 0.9, -1.0 + 1e-9, 1.5}), Indices{3});
    EXPECT_EQ(boundary_of(nig, std::vector<double>{0.01, 0.05, 0.9, 0.5, 1e9}), Indices{4});
    EXPECT_EQ(boundary_of(nig, std::vector<double>{0.01, 0.05, 0.9, 0.5, 1.5}), Indices{});
    // sstd and sged sGARCH (omega, alpha1, beta1, skew, shape): sged has no shape boundary.
    Layout sstd = ve::model::layout_of({.dist = Dist::Sstd, .include_mean = false});
    EXPECT_EQ(boundary_of(sstd, std::vector<double>{0.01, 0.05, 0.9, 0.8, 1e9}), Indices{4});
    EXPECT_EQ(boundary_of(sstd, std::vector<double>{0.01, 0.05, 0.9, 1e9, 6.0}), Indices{});
    Layout sged = ve::model::layout_of({.dist = Dist::Sged, .include_mean = false});
    EXPECT_EQ(boundary_of(sged, std::vector<double>{0.01, 0.05, 0.9, 0.8, 1e9}), Indices{});
    // ghyp sGARCH (omega, alpha1, beta1, skew, shape, ghlambda)
    Layout ghyp = ve::model::layout_of({.dist = Dist::Ghyp, .include_mean = false});
    EXPECT_EQ(boundary_of(ghyp, std::vector<double>{0.01, 0.05, 0.9, 0.3, 1e9, 1.5}), (Indices{4, 5}));
    EXPECT_EQ(boundary_of(ghyp, std::vector<double>{0.01, 0.05, 0.9, 1.0 - 1e-9, 2.0, 1.5}), Indices{3});
    EXPECT_EQ(boundary_of(ghyp, std::vector<double>{0.01, 0.05, 0.9, 0.3, 2.0, 1.5}), Indices{});
    // jsu sGARCH (omega, alpha1, beta1, skew, shape)
    Layout jsu = ve::model::layout_of({.dist = Dist::Jsu, .include_mean = false});
    EXPECT_EQ(boundary_of(jsu, std::vector<double>{0.01, 0.05, 0.9, 0.5, 1e9}), Indices{4});
    EXPECT_EQ(boundary_of(jsu, std::vector<double>{0.01, 0.05, 0.9, 1e9, 2.0}), Indices{});
}

}

namespace
{

using ve::Spec;
using ve::model::layout_of;
using ve::model::persistence;
using ve::model::WeightTerms;

Layout fixed_layout(Spec spec, std::vector<ve::ParameterValue> fixed)
{
    spec.fixed = std::move(fixed);
    return layout_of(spec);
}

std::vector<double> spec_interior_theta(const Layout& layout)
{
    std::vector<double> theta = model_interior_theta(layout);
    set_parameters(theta, layout, interior_parameters(layout.dist));
    return theta;
}

struct FixedCase
{
    Layout layout;
    std::vector<double> theta;
    std::size_t index;
};

// The spec with each theta_i in turn fixed at its interior value, where spec_problem accepts it.
void append_fixed_cases(std::vector<FixedCase>& cases, Spec spec)
{
    Layout plain = layout_of(spec);
    std::vector<double> theta = spec_interior_theta(plain);
    std::vector<std::string> names = ve::model::parameter_names(plain);
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        spec.fixed = {{names[index], theta[index]}};
        if (ve::model::spec_problem(spec))
        {
            continue;
        }
        cases.push_back({layout_of(spec), theta, index});
    }
}

std::vector<FixedCase> fixed_cases()
{
    std::vector<FixedCase> cases;
    for (Model model : kAllModels)
    {
        for (Orders orders : {Orders{1, 1}, Orders{2, 2}})
        {
            for (ArmaOrders arma : {ArmaOrders{0, 0}, ArmaOrders{1, 1}})
            {
                for (Dist dist : {Dist::Norm, Dist::Sstd, Dist::Ghyp})
                {
                    append_fixed_cases(cases, {.model = model,
                                               .p = orders.arch,
                                               .q = orders.garch,
                                               .ar = arma.ar,
                                               .ma = arma.ma,
                                               .dist = dist});
                }
            }
        }
    }
    return cases;
}

std::string name_of(const FixedCase& c)
{
    return name_of(c.layout) + ", fixed " + std::to_string(c.index);
}

// Start values change where the optimizer starts, never the map.
TEST(ParameterMap, PlainSpecsUnchanged)
{
    const Standardization standardization{0.02, 1.7};
    std::mt19937_64 engine(9);
    ve::testing::PortableNormal draw{1.0};
    for (const Layout& layout : model_layouts())
    {
        Layout started = layout;
        for (std::size_t i = 0; i < layout.count(); ++i)
        {
            started.started.set(i, layout.is_estimated(i));
            started.given[i] = 0.123;
        }
        std::vector<double> u(layout.count());
        std::ranges::generate(u, [&] { return draw(engine); });
        std::vector<double> theta(u.size());
        std::vector<double> expected(u.size());
        to_natural<double>(u, theta, started, standardization);
        to_natural<double>(u, expected, layout, standardization);
        EXPECT_EQ(theta, expected) << name_of(layout);
        std::vector<double> interior = model_interior_theta(layout);
        to_unconstrained(interior, theta, started, standardization);
        to_unconstrained(interior, expected, layout, standardization);
        EXPECT_EQ(theta, expected) << name_of(layout);
    }
}

// natural_from_terms inverts weight_terms on every entry it writes, and leaves the others alone.
void expect_terms_round_trip(std::span<const double> theta, const Layout& layout)
{
    std::vector<double> rebuilt(theta.begin(), theta.end());
    ve::model::natural_from_terms(ve::model::weight_terms(theta, layout), rebuilt, layout);
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        EXPECT_NEAR(rebuilt[i], theta[i], 1e-12 * std::max(1.0, std::abs(theta[i]))) << "parameter " << i;
    }
}

TEST(ParameterMap, FixedEntriesKeepTheirValue)
{
    std::mt19937_64 engine(13);
    ve::testing::PortableNormal draw{1.0};
    for (const FixedCase& c : fixed_cases())
    {
        SCOPED_TRACE(name_of(c));
        std::vector<double> u(c.theta.size());
        std::ranges::generate(u, [&] { return draw(engine); });
        std::vector<double> mapped(u.size());
        to_natural<double>(u, mapped, c.layout, Standardization{0.03, 0.7});
        EXPECT_EQ(mapped[c.index], c.theta[c.index]);
        expect_round_trip(c.theta, c.layout);
        expect_terms_round_trip(c.theta, c.layout);
    }
}

// theta = to_natural(u) for 200 u ~ N(0, 3) drawn from mt19937_64(7).
std::vector<std::vector<double>> budget_draws(const Layout& layout)
{
    std::mt19937_64 engine(7);
    ve::testing::PortableNormal draw{3.0};
    std::vector<std::vector<double>> draws;
    std::vector<double> u(layout.count());
    for (int trial = 0; trial < 200; ++trial)
    {
        std::ranges::generate(u, [&] { return draw(engine); });
        std::vector<double>& theta = draws.emplace_back(layout.count());
        to_natural<double>(u, theta, layout, Standardization{});
    }
    return draws;
}

void expect_gjr_budgets()
{
    // sstd gjrGARCH (mu, omega, alpha1, beta1, gamma1, skew, shape)
    Layout gamma = fixed_layout({.model = Model::GjrGarch, .dist = Dist::Sstd},
                                {{"gamma1", -0.05}, {"skew", 0.8}, {"shape", 6.0}});
    double pi = ve::dist::probability_below_zero(Dist::Sstd, ShapeParameters<double>{0.8, 6.0, 0.0});
    for (const std::vector<double>& theta : budget_draws(gamma))
    {
        EXPECT_GE(theta[2], 0.05);
        EXPECT_LT(theta[2] + theta[3] + pi * theta[4], kCeiling);
    }
    Layout alpha = fixed_layout({.model = Model::GjrGarch, .dist = Dist::Sstd}, {{"alpha1", 0.02}});
    for (const std::vector<double>& theta : budget_draws(alpha))
    {
        double below = ve::dist::probability_below_zero(Dist::Sstd, shape_parameters_of<double>(theta, alpha));
        if (!(below >= 1e-8 && below <= 1.0 - 1e-8))
        {
            continue;
        }
        EXPECT_EQ(theta[2], 0.02);
        EXPECT_GE(theta[2] + theta[4], 0.0);
    }
}

TEST(ParameterMap, FixedWeightsLeaveTheirBudget)
{
    // sGARCH (mu, omega, alpha1, beta1)
    for (const std::vector<double>& theta : budget_draws(fixed_layout({}, {{"alpha1", 0.3}})))
    {
        EXPECT_LT(theta[2] + theta[3], kCeiling);
        EXPECT_GE(theta[3], 0.0);
    }
    expect_gjr_budgets();
    // csGARCH (mu, omega, alpha1, beta1, eta11, eta21)
    for (const std::vector<double>& theta : budget_draws(fixed_layout({.model = Model::CsGarch}, {{"eta11", 0.95}})))
    {
        EXPECT_LT(theta[2] + theta[3], 0.95);
        EXPECT_EQ(theta[4], 0.95);
    }
    // iGARCH (mu, omega, alpha1, beta1, beta2)
    Layout igarch = fixed_layout({.model = Model::IGarch, .q = 2}, {{"alpha1", 0.2}});
    for (const std::vector<double>& theta : budget_draws(igarch))
    {
        EXPECT_NEAR(theta[2] + theta[3] + theta[4], 1.0, 1e-15);
        EXPECT_GE(theta[4], 0.0);
    }
}

TEST(ParameterMap, FixedBeyondBudgetIsNaN)
{
    // sGARCH (mu, omega, alpha1, beta1): alpha1 + beta1 >= 1 whatever omega is.
    Layout both = fixed_layout({}, {{"alpha1", 0.6}, {"beta1", 0.5}});
    std::vector<double> theta(4);
    to_natural<double>(std::vector<double>{0.1, -2.0, 0.3, 0.4}, theta, both, Standardization{});
    EXPECT_FALSE(is_valid(theta, both));
    for (const std::vector<double>& draw : budget_draws(fixed_layout({}, {{"alpha1", 0.6}})))
    {
        EXPECT_LT(draw[3], 0.4);
    }
    // sGARCH(2,1) (mu, omega, alpha1, alpha2, beta1): no budget is left for alpha2.
    Layout over = fixed_layout({.p = 2}, {{"alpha1", 0.6}, {"beta1", 0.5}});
    std::vector<double> wide(5);
    to_natural<double>(std::vector<double>{0.1, -2.0, 0.3, 0.4, 0.5}, wide, over, Standardization{});
    EXPECT_TRUE(std::isnan(wide[3]));
    EXPECT_FALSE(is_valid(wide, over));
}

// The spec with one parameter, drawn by picker, fixed at its interior value.
Layout one_drawn_fixed(Spec spec, std::mt19937_64& picker)
{
    Layout plain = layout_of(spec);
    std::vector<double> theta = spec_interior_theta(plain);
    std::vector<std::string> names = ve::model::parameter_names(plain);
    for (;;)
    {
        std::size_t index = picker() % names.size();
        spec.fixed = {{names[index], theta[index]}};
        if (!ve::model::spec_problem(spec))
        {
            return layout_of(spec);
        }
    }
}

// The ceiling less the terms of the fixed positions: what the free weights share.
double budget_left(std::span<const double> theta, const Layout& layout)
{
    WeightTerms terms = ve::model::weight_terms(theta, layout);
    std::size_t begin = layout.model == Model::EGarch ? layout.beta_begin() : layout.alpha_begin();
    double budget = terms.ceiling;
    for (std::size_t j = 0; j < terms.count; ++j)
    {
        if (layout.is_fixed(begin + j))
        {
            budget -= terms.terms[j];
        }
    }
    return budget;
}

TEST(ParameterMap, AlwaysValidWithFixed)
{
    std::mt19937_64 engine(5);
    std::mt19937_64 picker(11);
    ve::testing::PortableNormal draw{3.0};
    for (Model model : kAllModels)
    {
        for (Dist dist : {Dist::Norm, Dist::Sstd})
        {
            Layout layout = one_drawn_fixed({.model = model, .dist = dist}, picker);
            std::vector<double> u(layout.count());
            std::vector<double> theta(layout.count());
            for (int trial = 0; trial < 2000; ++trial)
            {
                std::ranges::generate(u, [&] { return draw(engine); });
                to_natural<double>(u, theta, layout, Standardization{0.01, 1.3});
                bool exempt = !(budget_left(theta, layout) > 0.0) || outside_computable_domain(layout, theta);
                ASSERT_TRUE(is_valid(theta, layout) || exempt)
                    << name_of(layout) << ", fixed " << layout.fixed.to_string() << ", trial " << trial;
            }
        }
    }
}

TEST(ParameterMap, DualMatchesDoubleWithFixed)
{
    using Dual16 = ve::ad::Dual<16>;
    const Standardization standardization{0.02, 1.7};
    std::mt19937_64 engine(9);
    ve::testing::PortableNormal draw{1.0};
    for (const FixedCase& c : fixed_cases())
    {
        std::vector<double> u(c.theta.size());
        std::ranges::generate(u, [&] { return draw(engine); });
        std::vector<Dual16> seeds;
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            seeds.push_back(ve::ad::variable<16>(u[i], i));
        }
        std::vector<Dual16> dual_theta(u.size());
        std::vector<double> theta(u.size());
        to_natural<Dual16>(seeds, dual_theta, c.layout, standardization);
        to_natural<double>(u, theta, c.layout, standardization);
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            ASSERT_TRUE(same_value(ve::ad::value_of(dual_theta[i]), theta[i])) << name_of(c) << ", parameter " << i;
        }
        EXPECT_EQ(dual_theta[c.index].grad, Dual16{}.grad) << name_of(c);
    }
}

// sum alpha + sum beta, with pi gamma_i in gjrGARCH and alpha_i kappa_i in apARCH and fGARCH; sum beta in eGARCH.
double expected_persistence(std::span<const double> theta, const Layout& layout)
{
    double sum = sum_of(theta.subspan(layout.beta_begin(), static_cast<std::size_t>(layout.garch)));
    if (layout.model == Model::EGarch)
    {
        return sum;
    }
    double pi = ve::dist::probability_below_zero(layout.dist, shape_parameters_of(theta, layout));
    for (std::size_t i = 0; i < static_cast<std::size_t>(layout.arch); ++i)
    {
        double term = theta[layout.alpha_begin() + i];
        if (layout.model == Model::GjrGarch)
        {
            term += pi * theta[layout.gamma_begin() + i];
        }
        if (layout.is_power_family())
        {
            term *= shock_moment<double>(theta, layout, i);
        }
        sum += term;
    }
    return sum;
}

void expect_persistence_gradient(const Layout& layout, std::span<const double> theta)
{
    using Dual16 = ve::ad::Dual<16>;
    std::vector<Dual16> seeds;
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        seeds.push_back(ve::ad::variable<16>(theta[i], i));
    }
    Dual16 dual = persistence<Dual16>(seeds, layout);
    constexpr double kStep = 1e-6;
    std::vector<double> up(theta.begin(), theta.end());
    std::vector<double> down(theta.begin(), theta.end());
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        up[i] += kStep;
        down[i] -= kStep;
        double difference = (persistence<double>(up, layout) - persistence<double>(down, layout)) / (2.0 * kStep);
        up[i] = theta[i];
        down[i] = theta[i];
        EXPECT_NEAR(dual.grad[i], difference, 1e-7 * std::max(1.0, std::abs(difference)))
            << name_of(layout) << ", d P / d theta" << i;
    }
}

TEST(ParameterMap, PersistenceMatchesWeights)
{
    for (const Layout& layout : model_layouts())
    {
        std::vector<double> theta = model_interior_theta(layout);
        double value = persistence<double>(theta, layout);
        WeightTerms terms = ve::model::weight_terms(theta, layout);
        EXPECT_NEAR(value, sum_of(std::span<const double>(terms.terms).first(terms.count)), 1e-15)
            << name_of(layout);
        EXPECT_NEAR(value, expected_persistence(theta, layout), 1e-15) << name_of(layout);
        expect_persistence_gradient(layout, theta);
    }
}

TEST(ParameterMap, GjrFixedGammaBoundary)
{
    // gjrGARCH (omega, alpha1, beta1, gamma1): alpha1 >= -gamma1 = 0.05.
    Layout layout = fixed_layout({.model = Model::GjrGarch, .include_mean = false}, {{"gamma1", -0.05}});
    EXPECT_TRUE(on_boundary(layout, std::vector<double>{0.01, 0.05 + 1e-9, 0.9, -0.05}, 1));
    EXPECT_FALSE(on_boundary(layout, std::vector<double>{0.01, 0.06, 0.9, -0.05}, 1));
}

TEST(ParameterMap, VarianceTargetingIgnoresOmega)
{
    // sGARCH (omega, alpha1, beta1)
    Layout layout = layout_of({.include_mean = false, .variance_targeting = true});
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(is_valid(std::vector<double>{kNaN, 0.05, 0.9}, layout));
    EXPECT_TRUE(is_valid(std::vector<double>{-1.0, 0.05, 0.9}, layout));
    EXPECT_FALSE(is_valid(std::vector<double>{-1.0, 0.05, 0.95}, layout));
    std::vector<double> u{7.0, 7.0, 7.0};
    to_unconstrained(std::vector<double>{kNaN, 0.05, 0.9}, u, layout, Standardization{});
    EXPECT_EQ(u[0], 0.0);
    EXPECT_TRUE(std::isfinite(u[1]) && std::isfinite(u[2]));
}


// archm 0.3, mxreg 0.5 and -0.2, vxreg 0.01 on top of the model interior.
Layout regressor_layout(Model model, int archm, int mxreg, int vxreg)
{
    return ve::model::layout_of({.model = model, .dist = Dist::Norm, .mxreg = mxreg, .vxreg = vxreg, .archm = archm});
}

std::vector<double> regressor_interior_theta(const Layout& layout)
{
    std::vector<double> theta = model_interior_theta(layout);
    if (layout.has_archm())
    {
        theta[layout.archm_index()] = 0.3;
    }
    for (std::size_t k = 0; k < static_cast<std::size_t>(layout.mxreg); ++k)
    {
        theta[layout.mxreg_begin() + k] = k == 0 ? 0.5 : -0.2;
    }
    for (std::size_t k = 0; k < static_cast<std::size_t>(layout.vxreg); ++k)
    {
        theta[layout.vxreg_begin() + k] = 0.01;
    }
    return theta;
}

Standardization regressor_standardization(double factor, double regressor_factor)
{
    Standardization standardization{0.03 * factor, 0.7 * factor};
    standardization.mean_regressor_scale = {1.3 * regressor_factor, 0.4 * regressor_factor, 1.0, 1.0,
                                            1.0,                    1.0,                    1.0, 1.0};
    standardization.variance_regressor_scale = {2.1 * regressor_factor, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    standardization.mean_regressor_center = {0.8 * regressor_factor, -1.5 * regressor_factor};
    return standardization;
}

TEST(ParameterMap, RegressorRoundTrip)
{
    const Standardization standardization = regressor_standardization(1.0, 1.0);
    for (Model model : kAllModels)
    {
        for (int archm : {0, 1, 2})
        {
            for (int mxreg : {0, 2})
            {
                for (int vxreg : {0, 1})
                {
                    Layout layout = regressor_layout(model, archm, mxreg, vxreg);
                    SCOPED_TRACE(name_of(layout) + ", archm " + std::to_string(archm) + ", mxreg " +
                                 std::to_string(mxreg) + ", vxreg " + std::to_string(vxreg));
                    std::vector<double> theta = regressor_interior_theta(layout);
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
        }
    }
}

// The scale S of the variance intercept for a series scale s.
double intercept_scale_of(const Layout& layout, double scale, double power)
{
    if (layout.model == Model::EGarch)
    {
        return 1.0;
    }
    return layout.is_power_family() ? std::pow(scale, power) : scale * scale;
}

// r x 100 and X, V x 1000: theta scaled by 100/1000 (mxreg), 1 (archm p = 1) or 1/100 (p = 2) and S/1000 (vxreg)
// gives the same u.
TEST(ParameterMap, RegressorScalesMakeMapUnitFree)
{
    const Standardization base = regressor_standardization(1.0, 1.0);
    const Standardization scaled = regressor_standardization(100.0, 1000.0);
    for (Model model : kAllModels)
    {
        for (int archm : {0, 1, 2})
        {
            Layout layout = regressor_layout(model, archm, 2, 1);
            SCOPED_TRACE(name_of(layout) + ", archm " + std::to_string(archm));
            std::vector<double> theta = regressor_interior_theta(layout);
            double power = power_of<double>(theta, layout);
            std::vector<double> units = theta;
            if (archm > 0)
            {
                units[layout.archm_index()] *= archm == 1 ? 1.0 : 0.01;
            }
            for (std::size_t k = 0; k < 2; ++k)
            {
                units[layout.mxreg_begin() + k] *= 100.0 / 1000.0;
            }
            double ratio = intercept_scale_of(layout, 100.0, power) / intercept_scale_of(layout, 1.0, power);
            units[layout.vxreg_begin()] *= ratio / 1000.0;
            std::vector<double> u(theta.size());
            std::vector<double> u_scaled(theta.size());
            to_unconstrained(theta, u, layout, base);
            to_unconstrained(units, u_scaled, layout, scaled);
            std::vector<std::size_t> indices{layout.mxreg_begin(), layout.mxreg_begin() + 1, layout.vxreg_begin()};
            if (archm > 0)
            {
                indices.push_back(layout.archm_index());
            }
            for (std::size_t i : indices)
            {
                EXPECT_NEAR(u_scaled[i], u[i], 1e-12 * std::max(1.0, std::abs(u[i]))) << "parameter " << i;
            }
        }
    }
}

TEST(ParameterMap, NaturalScaleOfEachLocation)
{
    Layout layout = regressor_layout(Model::ApArch, 2, 2, 1);
    std::vector<double> theta = regressor_interior_theta(layout);
    const Standardization standardization = regressor_standardization(1.0, 1.0);
    double s = standardization.scale;
    EXPECT_EQ(ve::model::natural_scale(layout, standardization, theta, layout.mu()), s);
    EXPECT_EQ(ve::model::natural_scale(layout, standardization, theta, layout.archm_index()), 1.0 / s);
    EXPECT_EQ(ve::model::natural_scale(layout, standardization, theta, layout.mxreg_begin() + 1), s / 0.4);
    double expected = std::pow(s, power_of<double>(theta, layout)) / 2.1;
    EXPECT_NEAR(ve::model::natural_scale(layout, standardization, theta, layout.vxreg_begin()), expected, 1e-15);
    layout.archm = 1;
    EXPECT_EQ(ve::model::natural_scale(layout, standardization, theta, layout.archm_index()), 1.0);
}

// mu, mxreg and the apARCH vxreg, whose scale reads the natural delta, depend on other entries of u.
TEST(ParameterMap, DualJacobianMatchesFiniteDifferencesWithRegressors)
{
    Layout layout = regressor_layout(Model::ApArch, 2, 2, 1);
    std::vector<double> u(layout.count());
    to_unconstrained(regressor_interior_theta(layout), u, layout, regressor_standardization(1.0, 1.0));
    expect_jacobian_matches(layout, u, regressor_standardization(1.0, 1.0));
}

// A shift c of column k moves mu by -mxreg_k c and leaves u unchanged.
TEST(ParameterMap, ShiftedRegressorMovesMuAlone)
{
    Layout layout = regressor_layout(Model::SGarch, 0, 2, 0);
    std::vector<double> theta = regressor_interior_theta(layout);
    Standardization standardization = regressor_standardization(1.0, 1.0);
    std::vector<double> u(theta.size());
    to_unconstrained(theta, u, layout, standardization);
    standardization.mean_regressor_center[1] += 100.0;
    theta[layout.mu()] -= theta[layout.mxreg_begin() + 1] * 100.0;
    std::vector<double> shifted(theta.size());
    to_unconstrained(theta, shifted, layout, standardization);
    EXPECT_NEAR(shifted[layout.mu()], u[layout.mu()], 1e-12);
}

TEST(ParameterMap, FixedRegressorsKeepTheirValue)
{
    Layout layout = ve::model::layout_of(
        {.fixed = {{"mu", 0.05}, {"mxreg1", 0.7}, {"vxreg1", -0.02}}, .mxreg = 2, .vxreg = 1, .archm = 2});
    const Standardization standardization = regressor_standardization(1.0, 1.0);
    std::vector<double> u(layout.count(), 0.4);
    std::vector<double> theta(layout.count());
    to_natural<double>(u, theta, layout, standardization);
    EXPECT_EQ(theta[layout.mu()], 0.05);
    EXPECT_EQ(theta[layout.mxreg_begin()], 0.7);
    EXPECT_EQ(theta[layout.vxreg_begin()], -0.02);
    EXPECT_NEAR(theta[layout.mxreg_begin() + 1], 0.4 * 0.7 / 0.4, 1e-15);
    EXPECT_NEAR(theta[layout.archm_index()], 0.4 / 0.7, 1e-15);
    std::vector<double> back(layout.count());
    to_unconstrained(theta, back, layout, standardization);
    EXPECT_EQ(back[layout.mxreg_begin()], 0.0);
    EXPECT_EQ(back[layout.vxreg_begin()], 0.0);
    EXPECT_NEAR(back[layout.mxreg_begin() + 1], 0.4, 1e-14);
}

TEST(ParameterMap, ConstantColumnScaleIsOne)
{
    std::vector<double> r = ve::testing::simulate_garch_series(4, 50);
    std::vector<double> x(r.size() * 2);
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        x[2 * t] = 1.0;
        x[2 * t + 1] = 0.1;
    }
    Standardization constant = ve::model::standardization_of(ve::model::Data{r, {x, 2}, {x, 2}});
    for (std::size_t k = 0; k < 2; ++k)
    {
        EXPECT_EQ(constant.mean_regressor_scale[k], 1.0);
        EXPECT_EQ(constant.variance_regressor_scale[k], 1.0);
    }
    EXPECT_EQ(constant.mean_regressor_scale[2], 1.0);
    for (std::size_t t = 0; t < r.size(); ++t)
    {
        x[2 * t + 1] = static_cast<double>(t % 2);
    }
    Standardization varying = ve::model::standardization_of(ve::model::Data{r, {x, 2}});
    EXPECT_EQ(varying.mean_regressor_scale[1], 0.5);
    EXPECT_EQ(varying.variance_regressor_scale[0], 1.0);
    EXPECT_EQ(constant.mean_regressor_center[0], 1.0);
    EXPECT_DOUBLE_EQ(varying.mean_regressor_center[1], 0.5);
    Standardization plain = ve::model::standardization_of(r);
    EXPECT_EQ(plain.mean_regressor_center[0], 0.0);
    EXPECT_EQ(plain.location, varying.location);
    EXPECT_EQ(plain.scale, varying.scale);
}

}
