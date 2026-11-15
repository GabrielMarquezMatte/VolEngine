#include "model/likelihood/adjoint.hpp"

#include "ad/dual.hpp"
#include "model/data.hpp"
#include "model/layout.hpp"
#include "model/likelihood.hpp"
#include "simulated_garch.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
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
using DualMax = ve::ad::Dual<ve::kMaxParameterCount>;

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

using AdjointCase = std::tuple<Model, Orders, ArmaOrders, Dist, VarInit, bool>;

Layout case_layout(const AdjointCase& c)
{
    auto [model, orders, arma, dist, var_init, include_mean] = c;
    return ve::model::layout_of({.model = model,
                                 .p = orders.arch,
                                 .q = orders.garch,
                                 .ar = arma.ar,
                                 .ma = arma.ma,
                                 .dist = dist,
                                 .include_mean = include_mean});
}

void fill(std::vector<double>& theta, std::size_t begin, int count, double value)
{
    std::fill_n(theta.begin() + static_cast<std::ptrdiff_t>(begin), count, value);
}

// An interior point: sum alpha 0.05, sum beta `beta_sum`; eGARCH alpha -0.05 and gamma 0.1 split over the lags,
// gjrGARCH gamma 0.05 per lag.
std::vector<double> point(const Layout& layout, double beta_sum, double shape)
{
    std::vector<double> theta(layout.count(), 0.0);
    if (layout.include_mean)
    {
        theta[layout.mu()] = 0.04;
    }
    for (int i = 0; i < layout.ar; ++i)
    {
        theta[layout.ar_begin() + static_cast<std::size_t>(i)] = 0.2 / (i + 1);
    }
    for (int j = 0; j < layout.ma; ++j)
    {
        theta[layout.ma_begin() + static_cast<std::size_t>(j)] = -0.15 / (j + 1);
    }
    theta[layout.omega()] = 0.02;
    double alpha = layout.model == Model::EGarch ? -0.05 : 0.05;
    double gamma = layout.model == Model::EGarch ? 0.1 / layout.arch : 0.05;
    fill(theta, layout.alpha_begin(), layout.arch, alpha / layout.arch);
    fill(theta, layout.beta_begin(), layout.garch, beta_sum / layout.garch);
    fill(theta, layout.gamma_begin(), layout.gamma, gamma);
    if (layout.has_shape())
    {
        theta[layout.shape()] = shape;
    }
    return theta;
}

// The interior point, persistence near 1, and the edges of the shape: std nu 2.5 and 30 (the Stirling branch of its
// constant), ged nu 0.8 and 3.
std::vector<std::vector<double>> points(const Layout& layout)
{
    double interior_shape = layout.dist == Dist::Ged ? 1.5 : 8.0;
    std::vector<std::vector<double>> all{point(layout, 0.9, interior_shape), point(layout, 0.949, interior_shape)};
    if (layout.dist == Dist::Std)
    {
        all.push_back(point(layout, 0.9, 2.5));
        all.push_back(point(layout, 0.9, 30.0));
    }
    if (layout.dist == Dist::Ged)
    {
        all.push_back(point(layout, 0.9, 0.8));
        all.push_back(point(layout, 0.9, 3.0));
    }
    return all;
}

DualMax dual_loglik(std::span<const double> r, std::span<const double> theta, const Layout& layout,
                    VarInit var_init)
{
    std::vector<DualMax> seeds(theta.size());
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        seeds[i] = ve::ad::variable<ve::kMaxParameterCount>(theta[i], i);
    }
    return ve::model::loglik<DualMax>(r, seeds, layout, var_init, {});
}

// The adjoint's value is the double likelihood bit for bit; its gradient is the Dual's within 1e-10 of the largest
// partial.
void expect_matches_dual(std::span<const double> r, std::span<const double> theta, const Layout& layout,
                         VarInit var_init)
{
    ASSERT_TRUE(ve::model::has_adjoint(layout, var_init));
    std::vector<double> grad(theta.size());
    ve::model::LoglikAdjoint adjoint;
    double value = adjoint.evaluate(ve::model::Data{r}, theta, layout, var_init, grad);
    double plain = ve::model::loglik<double>(r, theta, layout, var_init, {});
    EXPECT_EQ(std::bit_cast<std::uint64_t>(value), std::bit_cast<std::uint64_t>(plain));
    ASSERT_TRUE(std::isfinite(value));
    DualMax dual = dual_loglik(r, theta, layout, var_init);
    double scale = 1.0;
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        scale = std::max(scale, std::abs(dual.grad[i]));
    }
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        EXPECT_NEAR(grad[i], dual.grad[i], 1e-10 * scale) << "parameter " << i;
    }
}

std::string case_name(const ::testing::TestParamInfo<AdjointCase>& info)
{
    auto [model, orders, arma, dist, var_init, include_mean] = info.param;
    std::string models[] = {"SGarch", "EGarch", "GjrGarch", "ApArch", "IGarch"};
    std::string dists[] = {"Norm", "Std", "Ged"};
    return models[static_cast<int>(model)] + "P" + std::to_string(orders.arch) + "Q" +
           std::to_string(orders.garch) + "Arma" + std::to_string(arma.ar) + std::to_string(arma.ma) +
           dists[static_cast<int>(dist)] + (var_init == VarInit::Sample ? "Sample" : "Backcast") +
           (include_mean ? "Mean" : "NoMean");
}

class LoglikAdjointGrid : public ::testing::TestWithParam<AdjointCase>
{
protected:
    std::vector<double> r = ve::testing::simulate_garch_series(11, 600);
    Layout layout = case_layout(GetParam());
    VarInit var_init = std::get<VarInit>(GetParam());
};

TEST_P(LoglikAdjointGrid, MatchesDual)
{
    for (const std::vector<double>& theta : points(layout))
    {
        SCOPED_TRACE(::testing::Message() << "shape " << (layout.has_shape() ? theta[layout.shape()] : 0.0)
                                          << " beta1 " << theta[layout.beta_begin()]);
        expect_matches_dual(r, theta, layout, var_init);
    }
}

INSTANTIATE_TEST_SUITE_P(PhaseOne, LoglikAdjointGrid,
                         ::testing::Combine(::testing::Values(Model::SGarch, Model::IGarch),
                                            ::testing::Values(Orders{1, 1}, Orders{2, 1}, Orders{1, 2},
                                                              Orders{3, 3}),
                                            ::testing::Values(ArmaOrders{0, 0}, ArmaOrders{1, 1},
                                                              ArmaOrders{2, 2}),
                                            ::testing::Values(Dist::Norm, Dist::Std, Dist::Ged),
                                            ::testing::Values(VarInit::Sample, VarInit::Backcast),
                                            ::testing::Bool()),
                         case_name);

// T = 3 with sGARCH(3,3)-ARMA(2,2): held = 3 >= T, so every sigma2 is the presample.
TEST(LoglikAdjoint, SeriesShorterThanPresample)
{
    std::vector<double> r{0.4, -1.1, 0.7};
    Layout layout = ve::model::layout_of({.p = 3, .q = 3, .ar = 2, .ma = 2, .dist = Dist::Std});
    expect_matches_dual(r, point(layout, 0.9, 8.0), layout, VarInit::Sample);
}

// A fixed beta changes which parameters the optimizer moves, not the gradient over theta.
TEST(LoglikAdjoint, FixedParameterDoesNotChangeGradient)
{
    std::vector<double> r = ve::testing::simulate_garch_series(4, 300);
    Layout free = ve::model::layout_of({.dist = Dist::Std});
    Layout fixed = ve::model::layout_of({.dist = Dist::Std, .fixed = {{"beta1", 0.9}}});
    std::vector<double> theta = point(free, 0.9, 8.0);
    std::vector<double> free_grad(theta.size());
    std::vector<double> fixed_grad(theta.size());
    ve::model::LoglikAdjoint adjoint;
    adjoint.evaluate(ve::model::Data{r}, theta, free, VarInit::Sample, free_grad);
    adjoint.evaluate(ve::model::Data{r}, theta, fixed, VarInit::Sample, fixed_grad);
    EXPECT_EQ(free_grad, fixed_grad);
    expect_matches_dual(r, theta, fixed, VarInit::Sample);
}

TEST(LoglikAdjoint, OutsideDomainIsNotFinite)
{
    std::vector<double> r = ve::testing::simulate_garch_series(4, 300);
    Layout layout = ve::model::layout_of({.dist = Dist::Std});
    std::vector<double> theta = point(layout, 0.9, 8.0);
    theta[layout.omega()] = -1.0;
    std::vector<double> grad(theta.size());
    ve::model::LoglikAdjoint adjoint;
    double value = adjoint.evaluate(ve::model::Data{r}, theta, layout, VarInit::Sample, grad);
    EXPECT_FALSE(std::isfinite(value));
    EXPECT_EQ(std::bit_cast<std::uint64_t>(value),
              std::bit_cast<std::uint64_t>(ve::model::loglik<double>(r, theta, layout, VarInit::Sample, {})));
}

// A return of exactly zero with ARMA(0,0) and no mean is e_t = 0, the kink of |e| in the ged: the adjoint follows the
// Dual's conventions there, finite where it is finite.
TEST(LoglikAdjoint, ZeroResidualWithGedMatchesDual)
{
    std::vector<double> r = ve::testing::simulate_garch_series(4, 300);
    r[100] = 0.0;
    for (double nu : {0.8, 1.5})
    {
        SCOPED_TRACE(::testing::Message() << "nu " << nu);
        Layout layout = ve::model::layout_of({.dist = Dist::Ged, .include_mean = false});
        std::vector<double> theta = point(layout, 0.9, nu);
        std::vector<double> grad(theta.size());
        ve::model::LoglikAdjoint adjoint;
        adjoint.evaluate(ve::model::Data{r}, theta, layout, VarInit::Sample, grad);
        DualMax dual = dual_loglik(r, theta, layout, VarInit::Sample);
        for (std::size_t i = 0; i < theta.size(); ++i)
        {
            ASSERT_EQ(std::isfinite(grad[i]), std::isfinite(dual.grad[i])) << "parameter " << i;
        }
        expect_matches_dual(r, theta, layout, VarInit::Sample);
    }
}

bool covered(const ve::Spec& spec, VarInit var_init = VarInit::Sample)
{
    return ve::model::has_adjoint(ve::model::layout_of(spec), var_init);
}

TEST(LoglikAdjoint, CoversPhaseOneOnly)
{
    for (Model model : {Model::SGarch, Model::IGarch})
    {
        for (Dist dist : {Dist::Norm, Dist::Std, Dist::Ged})
        {
            EXPECT_TRUE(covered({.model = model, .ar = 1, .ma = 1, .dist = dist}, VarInit::Sample));
            EXPECT_TRUE(covered({.model = model, .dist = dist, .include_mean = false}, VarInit::Backcast));
        }
    }
    for (Model model : {Model::ApArch, Model::CsGarch, Model::FGarchAllGarch, Model::GjrGarch, Model::EGarch})
    {
        EXPECT_FALSE(covered({.model = model})) << static_cast<int>(model);
    }
    for (Dist dist : {Dist::Sstd, Dist::Sged, Dist::Nig, Dist::Ghyp, Dist::Jsu})
    {
        EXPECT_FALSE(covered({.dist = dist})) << static_cast<int>(dist);
    }
    EXPECT_FALSE(covered({.mxreg = 1}));
    EXPECT_FALSE(covered({.vxreg = 1}));
    EXPECT_FALSE(covered({.archm = 1}));
    EXPECT_FALSE(covered({.arfima = true}));
    EXPECT_FALSE(covered({.variance_targeting = true}));
}

}
