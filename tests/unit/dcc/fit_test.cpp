#include "dcc/fit.hpp"

#include "ad/dual.hpp"
#include "core/spec.hpp"
#include "dcc/likelihood.hpp"
#include "dcc/panel.hpp"
#include "dcc/parameter_map.hpp"
#include "dcc/spec.hpp"
#include "dcc3_panel.hpp"
#include "fit/fit.hpp"
#include "fit/fit_result.hpp"
#include "linalg/cholesky.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::FitConfig;
using ve::FitStatus;
using ve::dcc::Convention;
using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccSpec;
using ve::dcc::StandardizedPanel;

std::shared_ptr<const StandardizedPanel> dcc3()
{
    static const auto panel = std::make_shared<const StandardizedPanel>(ve::testing::dcc3_panel());
    return panel;
}

TEST(DccFit, AdccDefaultStartRespectsPanelAsymmetry)
{
    auto panel = std::make_shared<StandardizedPanel>();
    panel->t = 500;
    panel->m = 5;
    panel->sigma.assign(panel->t * panel->m, 1.0);
    std::mt19937_64 random(42);
    std::normal_distribution<double> normal;
    panel->z.resize(panel->sigma.size());
    for (double& value : panel->z)
    {
        value = normal(random);
    }
    ve::dcc::refresh_moments(*panel);
    ASSERT_GT(panel->delta, 1.0);
    DccSpec spec;
    spec.model = DccModel::ADcc;
    const auto fit = ve::dcc::fit_dcc(panel, spec, FitConfig{});
    EXPECT_TRUE(std::isfinite(fit.loglik));
    EXPECT_NE(fit.r, nullptr);
    spec.start = {{"dccg1", 1.0}};
    const auto invalid = ve::dcc::fit_dcc(panel, spec, FitConfig{});
    EXPECT_EQ(invalid.status, FitStatus::Failed);
    EXPECT_TRUE(std::isnan(invalid.loglik));
}

DccSpec spec_of(DccModel model, DccDist dist, Convention convention)
{
    DccSpec spec;
    spec.model = model;
    spec.dist = dist;
    spec.convention = convention;
    if (model == DccModel::FDcc)
    {
        spec.groups = {1, 1, 2};
    }
    return spec;
}

// rmgarch's estimates on dcc3 (the notes, section 10), in the VolEngine order, and its reported loglik.
struct Oracle
{
    DccModel model;
    std::vector<double> estimate;
    double loglik;
};

const std::array<Oracle, 3> kOracles{{
    {DccModel::Dcc, {0.0431630001835819, 0.9142701826310855}, -6082.94087772754},
    {DccModel::ADcc, {0.04243321505, 0.91398307950, 0.00240804736}, -6082.91639426},
    {DccModel::FDcc, {0.19004678480, 0.24108698852, 0.96463050567, 0.94296237137}, -6082.56656677},
}};

void expect_relative(std::span<const double> actual, std::span<const double> expected, double tolerance)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(actual[i], expected[i], tolerance * std::fabs(expected[i])) << "parameter " << i;
    }
}

TEST(DccFit, MatchesRmgarchEstimates)
{
    const std::shared_ptr<const StandardizedPanel> panel = dcc3();
    for (const Oracle& oracle : {kOracles[0], kOracles[2]})
    {
        SCOPED_TRACE(static_cast<int>(oracle.model));
        const DccSpec spec = spec_of(oracle.model, DccDist::MvNorm, Convention::Rmgarch);
        const DccFitResult fit = ve::dcc::fit_dcc(panel, spec, FitConfig{});
        ASSERT_EQ(fit.status, FitStatus::Converged);
        const double ours = ve::dcc::objective<double>(*panel, spec, fit.estimate);
        const double theirs = ve::dcc::objective<double>(*panel, spec, oracle.estimate);
        EXPECT_LE(ours, theirs + 1e-7);
        expect_relative(fit.estimate, oracle.estimate, 1e-3);
        EXPECT_NEAR(fit.loglik, oracle.loglik, 1e-7 * std::fabs(oracle.loglik));
        EXPECT_EQ(fit.names, ve::dcc::parameter_names(ve::dcc::layout_of(spec, 3)));
        EXPECT_EQ(fit.nobs, ve::testing::kDcc3Rows);
        EXPECT_EQ(fit.series, ve::testing::kDcc3Series);
        EXPECT_EQ(fit.panel, panel);
        ASSERT_NE(fit.r, nullptr);
        EXPECT_EQ(fit.r->size(), ve::testing::kDcc3Rows * 9);
        EXPECT_EQ(fit.se_hessian.size(), fit.estimate.size());
        EXPECT_TRUE(std::isnan(fit.se_robust[0]));
    }
}

// A weight below it sits on its lower bound 0, which the reparametrisation approaches without reaching.
constexpr double kOnLowerBound = 1e-4;

// The optimality conditions of objective / T, what the optimiser minimises (per observation, as the univariate
// tolerances): a zero gradient in the interior and a gradient >= 0 at a weight on its lower bound. Returns which
// coordinates sit on the bound.
std::vector<bool> expect_optimal(const StandardizedPanel& panel, const DccSpec& spec, const DccFitResult& fit)
{
    using Number = ve::ad::Dual<8>;
    std::vector<Number> theta;
    for (std::size_t i = 0; i < fit.estimate.size(); ++i)
    {
        theta.push_back(ve::ad::variable<8>(fit.estimate[i], i));
    }
    const Number value = ve::dcc::objective<Number>(panel, spec, theta);
    std::vector<bool> on_bound(fit.estimate.size());
    for (std::size_t i = 0; i < fit.estimate.size(); ++i)
    {
        const double gradient = value.grad[i] / static_cast<double>(panel.t);
        on_bound[i] = fit.estimate[i] < kOnLowerBound;
        if (on_bound[i])
        {
            EXPECT_GT(gradient, 0.0) << fit.names[i];
            continue;
        }
        EXPECT_NEAR(gradient, 0.0, 1e-4) << fit.names[i];
    }
    return on_bound;
}

TEST(DccFit, StandardIsAtItsOptimum)
{
    const std::shared_ptr<const StandardizedPanel> panel = dcc3();
    for (const Oracle& oracle : kOracles)
    {
        SCOPED_TRACE(static_cast<int>(oracle.model));
        const DccSpec spec = spec_of(oracle.model, DccDist::MvNorm, Convention::Standard);
        const DccFitResult fit = ve::dcc::fit_dcc(panel, spec, FitConfig{});
        ASSERT_EQ(fit.status, FitStatus::Converged);
        const std::vector<bool> on_bound = expect_optimal(*panel, spec, fit);
        EXPECT_NEAR(fit.loglik, ve::dcc::reported_loglik(*panel, spec, fit.estimate), 1e-9);
        for (std::size_t i = 0; i < fit.estimate.size(); ++i)
        {
            // Only the aDCC g leaves rmgarch's neighbourhood: see StandardADccGammaIsOnItsBound.
            const bool adcc_gamma = oracle.model == DccModel::ADcc && fit.names[i] == "dccg1";
            EXPECT_EQ(on_bound[i], adcc_gamma) << fit.names[i];
            if (!adcc_gamma)
            {
                EXPECT_NEAR(fit.estimate[i], oracle.estimate[i], 5e-2 * oracle.estimate[i]) << fit.names[i];
            }
        }
    }
}

// The uncentered Nbar of the papers (delta 0.589 against rmgarch's 0.393 on dcc3) moves the aDCC g of rmgarch,
// 0.0024 (se 0.0104), to its bound: there the Standard objective is lower than at rmgarch's g with a and b
// re-estimated, while the Rmgarch convention finds rmgarch's interior g.
TEST(DccFit, StandardADccGammaIsOnItsBound)
{
    const std::shared_ptr<const StandardizedPanel> panel = dcc3();
    const Oracle& oracle = kOracles[1];
    const DccSpec spec = spec_of(DccModel::ADcc, DccDist::MvNorm, Convention::Standard);
    const DccFitResult free = ve::dcc::fit_dcc(panel, spec, FitConfig{});
    ASSERT_EQ(free.status, FitStatus::Converged);
    DccSpec at_rmgarch = spec;
    at_rmgarch.fixed = {{"dccg1", oracle.estimate[2]}};
    const DccFitResult fixed = ve::dcc::fit_dcc(panel, at_rmgarch, FitConfig{});
    ASSERT_EQ(fixed.status, FitStatus::Converged);
    const std::vector<double> fixed_point{fixed.estimate[0], fixed.estimate[1], oracle.estimate[2]};
    EXPECT_GT(ve::dcc::objective<double>(*panel, spec, fixed_point) -
                  ve::dcc::objective<double>(*panel, spec, free.estimate),
              1e-3);
    const DccSpec rmgarch = spec_of(DccModel::ADcc, DccDist::MvNorm, Convention::Rmgarch);
    const DccFitResult interior = ve::dcc::fit_dcc(panel, rmgarch, FitConfig{});
    ASSERT_EQ(interior.status, FitStatus::Converged);
    expect_relative(interior.estimate, oracle.estimate, 5e-3);
}

constexpr std::size_t kSimulatedSeries = 4;
constexpr std::size_t kSimulatedRows = 4000;

void correlation_of(std::span<const double> q, std::span<double> r)
{
    for (std::size_t i = 0; i < kSimulatedSeries; ++i)
    {
        for (std::size_t j = 0; j < kSimulatedSeries; ++j)
        {
            const std::size_t k = i * kSimulatedSeries + j;
            r[k] = q[k] / std::sqrt(q[i * kSimulatedSeries + i] * q[j * kSimulatedSeries + j]);
        }
    }
}

// z = L e with L the Cholesky factor of r, which is overwritten.
void correlated_draw(std::span<double> r, std::mt19937_64& engine, std::span<double> z)
{
    std::normal_distribution<double> normal;
    std::array<double, kSimulatedSeries> e{};
    for (double& x : e)
    {
        x = normal(engine);
    }
    ASSERT_TRUE(ve::linalg::cholesky_factor(r, kSimulatedSeries));
    for (std::size_t i = 0; i < kSimulatedSeries; ++i)
    {
        z[i] = 0.0;
        for (std::size_t j = 0; j <= i; ++j)
        {
            z[i] += r[i * kSimulatedSeries + j] * e[j];
        }
    }
}

// DCC(a, b) mvnorm with Q_0 = Qbar = rbar, as columns of standardized returns.
std::vector<std::vector<double>> simulate_dcc(double a, double b)
{
    const std::array<double, 16> rbar{1.0, 0.5, 0.3, 0.2, 0.5, 1.0, 0.4, 0.3, 0.3, 0.4, 1.0, 0.5, 0.2, 0.3, 0.5, 1.0};
    std::mt19937_64 engine(42);
    std::array<double, 16> q = rbar;
    std::array<double, 16> r{};
    std::array<double, kSimulatedSeries> z{};
    std::vector<std::vector<double>> columns(kSimulatedSeries, std::vector<double>(kSimulatedRows));
    for (std::size_t t = 0; t < kSimulatedRows; ++t)
    {
        correlation_of(q, r);
        correlated_draw(r, engine, z);
        for (std::size_t k = 0; k < q.size(); ++k)
        {
            q[k] = (1.0 - a - b) * rbar[k] + a * z[k / kSimulatedSeries] * z[k % kSimulatedSeries] + b * q[k];
        }
        for (std::size_t i = 0; i < kSimulatedSeries; ++i)
        {
            columns[i][t] = z[i];
        }
    }
    return columns;
}

// Univariate fits fixed at mu = 0 and sigma = 1, so that z is the simulated column.
std::shared_ptr<const StandardizedPanel> simulated_panel(double a, double b)
{
    const std::vector<std::vector<double>> columns = simulate_dcc(a, b);
    ve::Spec unit;
    unit.fixed = {{"mu", 0.0}, {"omega", 1.0}, {"alpha1", 0.0}, {"beta1", 0.0}};
    std::vector<ve::FitResult> fits;
    for (const std::vector<double>& column : columns)
    {
        fits.push_back(ve::fit_one(column, unit, FitConfig{}));
    }
    const std::vector<const ve::FitResult*> pointers{&fits[0], &fits[1], &fits[2], &fits[3]};
    EXPECT_FALSE(ve::dcc::panel_problem(pointers).has_value());
    return std::make_shared<const StandardizedPanel>(ve::dcc::standardize(pointers));
}

TEST(DccFit, RecoversSimulatedParameters)
{
    const std::shared_ptr<const StandardizedPanel> panel = simulated_panel(0.06, 0.9);
    EXPECT_EQ(panel->z[5], simulate_dcc(0.06, 0.9)[1][1]);
    const DccFitResult fit = ve::dcc::fit_dcc(panel, DccSpec{}, FitConfig{});
    ASSERT_EQ(fit.status, FitStatus::Converged);
    EXPECT_NEAR(fit.estimate[0], 0.06, 0.02);
    EXPECT_NEAR(fit.estimate[1], 0.9, 0.02);
}

DccSpec fixed_shape_spec()
{
    DccSpec spec = spec_of(DccModel::Dcc, DccDist::MvT, Convention::Rmgarch);
    spec.fixed = {{"mshape", 8.0}};
    return spec;
}

TEST(DccFit, FixedAndStart)
{
    const std::shared_ptr<const StandardizedPanel> panel = dcc3();
    const std::vector<double> rmgarch{0.04016082, 0.91135622};
    const DccSpec spec = fixed_shape_spec();
    const DccFitResult fit = ve::dcc::fit_dcc(panel, spec, FitConfig{});
    ASSERT_EQ(fit.status, FitStatus::Converged);
    EXPECT_EQ(fit.names, (std::vector<std::string>{"dcca1", "dccb1"}));
    EXPECT_NEAR(fit.loglik, -6132.56175655, 1e-6 * 6132.56175655);
    expect_relative(fit.estimate, rmgarch, 1e-3);
    const ve::dcc::DccLayout layout = ve::dcc::layout_of(spec, 3);
    const std::vector<double> full = ve::dcc::full_parameters(layout, fit.estimate);
    EXPECT_EQ(full, (std::vector<double>{fit.estimate[0], fit.estimate[1], 8.0}));
    EXPECT_EQ(ve::dcc::estimated_part(layout, full), fit.estimate);

    DccSpec started = spec;
    started.start = {{"dcca1", 0.04}, {"dccb1", 0.91}, {"mshape", 20.0}};
    const DccFitResult from_start = ve::dcc::fit_dcc(panel, started, FitConfig{});
    ASSERT_EQ(from_start.status, FitStatus::Converged);
    expect_relative(from_start.estimate, rmgarch, 1e-3);
    EXPECT_EQ(ve::dcc::start_values(ve::dcc::layout_of(started, 3)), (std::vector<double>{0.04, 0.91, 8.0}));

    DccSpec all_fixed = spec;
    all_fixed.fixed = {{"dcca1", rmgarch[0]}, {"dccb1", rmgarch[1]}, {"mshape", 8.0}};
    const DccFitResult evaluated = ve::dcc::fit_dcc(panel, all_fixed, FitConfig{});
    EXPECT_EQ(evaluated.status, FitStatus::Converged);
    EXPECT_TRUE(evaluated.names.empty());
    const std::vector<double> point{rmgarch[0], rmgarch[1], 8.0};
    EXPECT_EQ(evaluated.loglik, ve::dcc::reported_loglik(*panel, all_fixed, point));
    EXPECT_NEAR(evaluated.loglik, -6132.56175655, 1e-6 * 6132.56175655);
}

TEST(DccFit, FailsOutsideTheDomain)
{
    DccSpec spec = fixed_shape_spec();
    spec.fixed.push_back({"dccb1", 1.2});
    const DccFitResult fit = ve::dcc::fit_dcc(dcc3(), spec, FitConfig{});
    EXPECT_EQ(fit.status, FitStatus::Failed);
    ASSERT_EQ(fit.estimate.size(), 1U);
    EXPECT_TRUE(std::isnan(fit.estimate[0]));
    EXPECT_TRUE(std::isnan(fit.loglik));
    EXPECT_EQ(fit.r, nullptr);
}

}
