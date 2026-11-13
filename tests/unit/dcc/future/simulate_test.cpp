#include "dcc/future/simulate.hpp"

#include "core/context.hpp"
#include "dcc/copula/copula3_panel.hpp"
#include "dcc/dcc3_panel.hpp"
#include "dcc/fit.hpp"
#include "dcc/future/forecast.hpp"
#include "dcc/future/mixed_panel.hpp"
#include "dcc/spec.hpp"
#include "dist/cdf.hpp"
#include "fit/fit_result.hpp"
#include "model/layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace
{

using ve::FitConfig;
using ve::FitResult;
using ve::dcc::DccCopula;
using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccOptions;
using ve::dcc::DccSpec;
using ve::dcc::future::DccForecast;
using ve::dcc::future::DccRequest;
using ve::dcc::future::DccSimulation;
using ve::dcc::future::SeriesRegressors;

constexpr std::size_t kSeries = 3;
constexpr double kStandardErrors = 5.0;

std::vector<DccFitResult> run_batch(const std::vector<FitResult>& fits, const std::vector<DccSpec>& specs)
{
    ve::Context context;
    std::vector<const FitResult*> pointers;
    for (const FitResult& fit : fits)
    {
        pointers.push_back(&fit);
    }
    const DccOptions options{FitConfig{}, false};
    EXPECT_FALSE(ve::dcc::check_dcc_batch(pointers, specs, options).has_value());
    return ve::dcc::fit_dcc_batch(context, pointers, specs, options);
}

const std::vector<FitResult>& dcc3_fits()
{
    static const std::vector<FitResult> fits = ve::testing::dcc3_fits();
    return fits;
}

const std::vector<FitResult>& copula3_fits()
{
    static const std::vector<FitResult> fits = ve::testing::copula3_fits();
    return fits;
}

DccSpec dcc_spec(DccModel model, DccDist dist = DccDist::MvNorm, DccCopula copula = DccCopula::None)
{
    DccSpec spec;
    spec.model = model;
    spec.dist = dist;
    spec.copula = copula;
    if (dist == DccDist::MvT)
    {
        spec.fixed = {{"mshape", 8.0}};
    }
    return spec;
}

DccSimulation simulate(const DccFitResult& fit, std::size_t horizon, std::size_t paths, std::uint64_t seed,
                       int threads = 4, bool correlations = false, const SeriesRegressors& future = {})
{
    ve::Context context;
    context.set_threads(threads);
    DccRequest request{horizon, paths, seed, correlations, future};
    if (request.future.empty())
    {
        request.future.resize(fit.series);
    }
    EXPECT_FALSE(ve::dcc::future::forecast_problem(fit, horizon, request.future).has_value());
    return ve::dcc::future::simulate_dcc(context, fit, request);
}

void expect_same(const DccSimulation& a, const DccSimulation& b)
{
    EXPECT_EQ(a.returns, b.returns);
    EXPECT_EQ(a.sigma, b.sigma);
    EXPECT_EQ(a.r, b.r);
    EXPECT_EQ(a.h, b.h);
}

TEST(DccSimulate, SameForAnyThreadCount)
{
    const std::vector<DccFitResult> plain =
        run_batch(dcc3_fits(), {dcc_spec(DccModel::Dcc, DccDist::MvT), dcc_spec(DccModel::ADcc),
                                dcc_spec(DccModel::Dcc, DccDist::MvLaplace)});
    const std::vector<DccFitResult> copula = run_batch(
        copula3_fits(), {dcc_spec(DccModel::Dcc, DccDist::MvT, DccCopula::Parametric),
                         dcc_spec(DccModel::Dcc, DccDist::MvNorm, DccCopula::Empirical)});
    std::vector<const DccFitResult*> fits;
    for (const DccFitResult& fit : plain)
    {
        fits.push_back(&fit);
    }
    for (const DccFitResult& fit : copula)
    {
        fits.push_back(&fit);
    }
    for (const DccFitResult* fit : fits)
    {
        SCOPED_TRACE(static_cast<int>(fit->spec.dist));
        const DccSimulation one = simulate(*fit, 12, 50, 7, 1, true);
        expect_same(one, simulate(*fit, 12, 50, 7, 3, true));
        expect_same(one, simulate(*fit, 12, 50, 7, 16, true));
    }
}

TEST(DccSimulate, Prefixes)
{
    const std::vector<DccFitResult> fits =
        run_batch(copula3_fits(), {dcc_spec(DccModel::ADcc, DccDist::MvT, DccCopula::Parametric)});
    const DccSimulation wide = simulate(fits[0], 20, 40, 11, 4, true);
    const DccSimulation small = simulate(fits[0], 8, 10, 11, 4, true);
    for (std::size_t p = 0; p < small.paths; ++p)
    {
        for (std::size_t h = 0; h < small.horizon; ++h)
        {
            for (std::size_t i = 0; i < kSeries; ++i)
            {
                const std::size_t a = (p * small.horizon + h) * kSeries + i;
                const std::size_t b = (p * wide.horizon + h) * kSeries + i;
                ASSERT_EQ(small.returns[a], wide.returns[b]) << p << " " << h;
                ASSERT_EQ(small.sigma[a], wide.sigma[b]) << p << " " << h;
            }
            for (std::size_t e = 0; e < kSeries * kSeries; ++e)
            {
                ASSERT_EQ(small.r[(p * small.horizon + h) * 9 + e], wide.r[(p * wide.horizon + h) * 9 + e]);
            }
        }
    }
}

// The mean of f over the n samples lies within kStandardErrors of target (+ slack).
void expect_mean(const std::vector<double>& samples, double target, double slack, const std::string& what)
{
    double sum = 0.0;
    double squares = 0.0;
    for (double value : samples)
    {
        sum += value;
        squares += value * value;
    }
    const auto n = static_cast<double>(samples.size());
    const double mean = sum / n;
    const double error = std::sqrt(std::max(squares / n - mean * mean, 0.0) / n);
    EXPECT_NEAR(mean, target, kStandardErrors * error + slack) << what;
}

TEST(DccSimulate, FirstStepShocks)
{
    constexpr std::size_t paths = 400000;
    const std::vector<DccFitResult> fits = run_batch(
        dcc3_fits(), {dcc_spec(DccModel::Dcc), dcc_spec(DccModel::Dcc, DccDist::MvT),
                      dcc_spec(DccModel::Dcc, DccDist::MvLaplace)});
    const SeriesRegressors none(kSeries);
    for (const DccFitResult& fit : fits)
    {
        SCOPED_TRACE(static_cast<int>(fit.spec.dist));
        const DccForecast forecast = ve::dcc::future::forecast_dcc(fit, 1, none);
        const DccSimulation sim = simulate(fit, 1, paths, 3);
        for (std::size_t i = 0; i < kSeries; ++i)
        {
            for (std::size_t p = 0; p < paths; ++p)
            {
                ASSERT_EQ(sim.sigma[p * kSeries + i], forecast.sigma[i]);
            }
        }
        for (std::size_t i = 0; i < kSeries; ++i)
        {
            for (std::size_t j = 0; j <= i; ++j)
            {
                std::vector<double> products(paths);
                for (std::size_t p = 0; p < paths; ++p)
                {
                    const double zi = (sim.returns[p * kSeries + i] - forecast.mean[i]) / forecast.sigma[i];
                    const double zj = (sim.returns[p * kSeries + j] - forecast.mean[j]) / forecast.sigma[j];
                    products[p] = zi * zj;
                }
                expect_mean(products, forecast.r[i * kSeries + j], 1e-12, "R " + std::to_string(i) + std::to_string(j));
            }
        }
    }
}

TEST(DccSimulate, MeansFollowTheForecast)
{
    constexpr std::size_t paths = 20000;
    constexpr std::size_t horizon = 20;
    const SeriesRegressors none(kSeries);
    const std::vector<DccFitResult> fits =
        run_batch(dcc3_fits(), {dcc_spec(DccModel::Dcc), dcc_spec(DccModel::ADcc)});
    for (const DccFitResult& fit : fits)
    {
        SCOPED_TRACE(static_cast<int>(fit.spec.model));
        const DccForecast forecast = ve::dcc::future::forecast_dcc(fit, horizon, none);
        const DccSimulation sim = simulate(fit, horizon, paths, 5, 4, true);
        for (std::size_t h : {std::size_t{0}, std::size_t{4}, std::size_t{19}})
        {
            for (std::size_t i = 0; i < kSeries; ++i)
            {
                std::vector<double> variance(paths);
                for (std::size_t p = 0; p < paths; ++p)
                {
                    const double sigma = sim.sigma[(p * horizon + h) * kSeries + i];
                    variance[p] = sigma * sigma;
                }
                const double target = forecast.sigma[h * kSeries + i] * forecast.sigma[h * kSeries + i];
                expect_mean(variance, target, 1e-12 * target, "sigma2 " + std::to_string(h) + " " + std::to_string(i));
            }
            for (std::size_t e : {std::size_t{1}, std::size_t{2}, std::size_t{5}})
            {
                std::vector<double> correlation(paths);
                for (std::size_t p = 0; p < paths; ++p)
                {
                    correlation[p] = sim.r[(p * horizon + h) * 9 + e];
                }
                expect_mean(correlation, forecast.r[h * 9 + e], 0.01, "R " + std::to_string(h) + " " + std::to_string(e));
            }
        }
    }
}

// u = F_i(z) of the standardized shocks of the first step: uniform for the parametric margins, and every z one of
// the z of the sample for the empirical ones.
TEST(DccSimulate, CopulaMargins)
{
    constexpr std::size_t paths = 100000;
    const std::vector<DccFitResult> fits = run_batch(
        copula3_fits(), {dcc_spec(DccModel::Dcc, DccDist::MvNorm, DccCopula::Parametric),
                         dcc_spec(DccModel::Dcc, DccDist::MvNorm, DccCopula::Empirical)});
    const SeriesRegressors none(kSeries);
    for (const DccFitResult& fit : fits)
    {
        const DccForecast forecast = ve::dcc::future::forecast_dcc(fit, 1, none);
        const DccSimulation sim = simulate(fit, 1, paths, 9);
        for (std::size_t i = 0; i < kSeries; ++i)
        {
            const ve::model::Layout layout = ve::model::layout_of(copula3_fits()[i].spec);
            const auto parameters = ve::model::shape_parameters_of<double>(copula3_fits()[i].estimate, layout);
            std::vector<double> sample_z;
            for (std::size_t t = 0; t < fit.panel->t; ++t)
            {
                sample_z.push_back(fit.panel->z[t * kSeries + i]);
            }
            std::ranges::sort(sample_z);
            std::vector<double> uniforms(paths);
            std::vector<double> centred(paths);
            for (std::size_t p = 0; p < paths; ++p)
            {
                const double z = (sim.returns[p * kSeries + i] - forecast.mean[i]) / forecast.sigma[i];
                uniforms[p] = ve::dist::cdf(layout.dist, parameters, z);
                centred[p] = (uniforms[p] - 0.5) * (uniforms[p] - 0.5);
                if (fit.spec.copula == DccCopula::Empirical)
                {
                    const auto above = std::ranges::lower_bound(sample_z, z - 1e-9);
                    ASSERT_NE(above, sample_z.end());
                    ASSERT_NEAR(*above, z, 2e-9) << p;
                }
            }
            if (fit.spec.copula == DccCopula::Parametric)
            {
                expect_mean(uniforms, 0.5, 0.0, "u mean " + std::to_string(i));
                expect_mean(centred, 1.0 / 12.0, 0.0, "u variance " + std::to_string(i));
            }
        }
    }
}

TEST(DccSimulate, CorrelationsOff)
{
    const std::vector<DccFitResult> fits = run_batch(dcc3_fits(), {dcc_spec(DccModel::ADcc)});
    const DccSimulation with = simulate(fits[0], 15, 30, 2, 4, true);
    const DccSimulation without = simulate(fits[0], 15, 30, 2, 4, false);
    EXPECT_TRUE(without.r.empty());
    EXPECT_TRUE(without.h.empty());
    EXPECT_EQ(with.r.size(), 30U * 15U * 9U);
    EXPECT_EQ(with.returns, without.returns);
    EXPECT_EQ(with.sigma, without.sigma);
}

TEST(DccSimulate, ArfimaAndComponentSeries)
{
    constexpr std::size_t horizon = 30;
    const ve::testing::MixedPanel& panel = ve::testing::mixed_panel();
    const std::vector<DccFitResult> fits = run_batch(panel.fits, {dcc_spec(DccModel::Dcc, DccDist::MvT)});
    const DccSimulation sim = simulate(fits[0], horizon, 100, 4, 4, true, ve::testing::mixed_future(panel, horizon));
    for (const std::vector<double>* values : {&sim.returns, &sim.sigma, &sim.r, &sim.h})
    {
        ASSERT_FALSE(values->empty());
        EXPECT_TRUE(std::ranges::all_of(*values, [](double v) { return std::isfinite(v); }));
    }
}

}
