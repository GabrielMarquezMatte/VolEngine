#include "dcc/future/forecast.hpp"

#include "core/context.hpp"
#include "dcc/dcc3_panel.hpp"
#include "dcc/fit.hpp"
#include "dcc/future/state.hpp"
#include "dcc/spec.hpp"
#include "fit/fit.hpp"
#include "fit/fit_result.hpp"
#include "forecast/forecast.hpp"
#include "model/data.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::FitConfig;
using ve::FitResult;
using ve::FitStatus;
using ve::Model;
using ve::dcc::Convention;
using ve::dcc::DccCopula;
using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccOptions;
using ve::dcc::DccSpec;
using ve::dcc::future::DccForecast;
using ve::dcc::future::SeriesRegressors;

constexpr std::size_t kSeries = ve::testing::kDcc3Series;
constexpr std::size_t kHorizon = 10;

std::vector<const FitResult*> pointers_of(const std::vector<FitResult>& fits)
{
    std::vector<const FitResult*> pointers;
    for (const FitResult& fit : fits)
    {
        pointers.push_back(&fit);
    }
    return pointers;
}

std::vector<DccFitResult> run_batch(const std::vector<FitResult>& fits, const std::vector<DccSpec>& specs)
{
    ve::Context context;
    const DccOptions options{FitConfig{}, false};
    const std::vector<const FitResult*> pointers = pointers_of(fits);
    EXPECT_FALSE(ve::dcc::check_dcc_batch(pointers, specs, options).has_value());
    return ve::dcc::fit_dcc_batch(context, pointers, specs, options);
}

const std::vector<FitResult>& dcc3_fits()
{
    static const std::vector<FitResult> fits = ve::testing::dcc3_fits();
    return fits;
}

DccSpec dcc_spec(DccModel model)
{
    DccSpec spec;
    spec.model = model;
    if (model == DccModel::FDcc)
    {
        spec.groups = {1, 1, 2};
    }
    return spec;
}

// diag(Qbar)^-1/2 Qbar diag(Qbar)^-1/2 of the packed Qbar, as the recursion evaluates it.
std::vector<double> bar_of(const DccFitResult& fit)
{
    const std::size_t m = fit.series;
    std::vector<double> scale(m);
    auto at = [&](std::size_t i, std::size_t j) { return fit.end.qbar[i * (i + 1) / 2 + j]; };
    for (std::size_t i = 0; i < m; ++i)
    {
        scale[i] = 1.0 / std::sqrt(at(i, i));
    }
    std::vector<double> r(m * m);
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < m; ++j)
        {
            r[i * m + j] = i == j ? 1.0 : (i > j ? at(i, j) : at(j, i)) * scale[i] * scale[j];
        }
    }
    return r;
}

// The decay k per element: a + b of the DCC (the aDCC g does not enter), a_i a_j + b_i b_j per group of the FDCC.
std::vector<double> decay_of(const DccFitResult& fit)
{
    const std::size_t m = fit.series;
    const std::vector<double>& theta = fit.end.theta;
    std::vector<double> k(m * m, theta[0] + theta[1]);
    if (fit.spec.model != DccModel::FDcc)
    {
        return k;
    }
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < m; ++j)
        {
            const auto gi = static_cast<std::size_t>(fit.spec.groups[i] - 1);
            const auto gj = static_cast<std::size_t>(fit.spec.groups[j] - 1);
            k[i * m + j] = theta[gi] * theta[gj] + theta[2 + gi] * theta[2 + gj];
        }
    }
    return k;
}

void expect_formula(const DccFitResult& fit, const DccForecast& forecast)
{
    const std::size_t m = fit.series;
    const std::vector<double> bar = bar_of(fit);
    const std::vector<double> k = decay_of(fit);
    ve::dcc::future::CorrelationStepper stepper(fit.end, fit.spec, m);
    const std::span<const double> first = stepper.next();
    for (std::size_t e = 0; e < m * m; ++e)
    {
        EXPECT_EQ(forecast.r[e], first[e]) << e;
    }
    for (std::size_t h = 1; h < forecast.horizon; ++h)
    {
        for (std::size_t e = 0; e < m * m; ++e)
        {
            const double weight = std::pow(k[e], static_cast<double>(h));
            const double expected = e / m == e % m ? 1.0 : (1.0 - weight) * bar[e] + weight * first[e];
            EXPECT_NEAR(forecast.r[h * m * m + e], expected, 1e-15) << "h " << h + 1 << " entry " << e;
        }
    }
    for (std::size_t k_index = 0; k_index < forecast.h.size(); ++k_index)
    {
        const std::size_t h = k_index / (m * m);
        const double expected = forecast.sigma[h * m + (k_index / m) % m] * forecast.r[k_index] *
                                forecast.sigma[h * m + k_index % m];
        EXPECT_NEAR(forecast.h[k_index], expected, 1e-15) << k_index;
    }
}

TEST(DccForecast, Formula)
{
    const SeriesRegressors none(kSeries);
    const std::vector<DccFitResult> results =
        run_batch(dcc3_fits(), {dcc_spec(DccModel::Dcc), dcc_spec(DccModel::ADcc), dcc_spec(DccModel::FDcc)});
    for (const DccFitResult& fit : results)
    {
        SCOPED_TRACE(static_cast<int>(fit.spec.model));
        ASSERT_FALSE(ve::dcc::future::forecast_problem(fit, kHorizon, none).has_value());
        const DccForecast forecast = ve::dcc::future::forecast_dcc(fit, kHorizon, none);
        ASSERT_EQ(forecast.r.size(), kHorizon * kSeries * kSeries);
        expect_formula(fit, forecast);
    }
}

// One group of the FDCC with weights a = 0.2^2 and b = (sqrt 0.9)^2 is the DCC(1, 1) with a = 0.04 and b = 0.9.
TEST(DccForecast, OneGroupFdccIsDcc)
{
    DccSpec fdcc;
    fdcc.model = DccModel::FDcc;
    fdcc.groups = {1, 1, 1};
    fdcc.fixed = {{"fdcca1", 0.2}, {"fdccb1", std::sqrt(0.9)}};
    DccSpec dcc;
    dcc.fixed = {{"dcca1", 0.04}, {"dccb1", 0.9}};
    const std::vector<DccFitResult> results = run_batch(dcc3_fits(), {fdcc, dcc});
    const SeriesRegressors none(kSeries);
    const DccForecast a = ve::dcc::future::forecast_dcc(results[0], kHorizon, none);
    const DccForecast b = ve::dcc::future::forecast_dcc(results[1], kHorizon, none);
    for (std::size_t e = 0; e < a.r.size(); ++e)
    {
        EXPECT_NEAR(a.r[e], b.r[e], 1e-14) << e;
    }
}

// A panel of an ARFIMA series, a series with a mean regressor and a csGARCH series on the 1500 rows of dcc3.
struct MixedPanel
{
    std::vector<FitResult> fits;
    std::vector<double> x;
    SeriesRegressors future;
};

const MixedPanel& mixed_panel()
{
    static const MixedPanel panel = []
    {
        MixedPanel out;
        const std::vector<std::vector<double>> columns = ve::testing::read_dcc3();
        const std::size_t rows = ve::testing::kDcc3Rows;
        out.x.resize(rows + kHorizon);
        for (std::size_t t = 0; t < out.x.size(); ++t)
        {
            out.x[t] = 0.1 * std::sin(0.3 * static_cast<double>(t));
        }
        out.fits.push_back(ve::fit_one(columns[0], ve::Spec{.ar = 1, .ma = 1, .arfima = true}, FitConfig{}));
        ve::model::Data with_mean{columns[1], {std::span<const double>(out.x).first(rows), 1}, {}};
        out.fits.push_back(ve::fit_one(with_mean, ve::Spec{.mxreg = 1}, FitConfig{}));
        out.fits.push_back(ve::fit_one(columns[2], ve::Spec{.model = Model::CsGarch}, FitConfig{}));
        out.future.resize(kSeries);
        out.future[1].mean = {std::span<const double>(out.x).last(kHorizon), 1};
        return out;
    }();
    return panel;
}

TEST(DccForecast, SeriesAreTheirOwnForecast)
{
    const MixedPanel& panel = mixed_panel();
    const std::vector<DccFitResult> results = run_batch(panel.fits, {dcc_spec(DccModel::Dcc)});
    ASSERT_FALSE(ve::dcc::future::forecast_problem(results[0], kHorizon, panel.future).has_value());
    const DccForecast forecast = ve::dcc::future::forecast_dcc(results[0], kHorizon, panel.future);
    for (std::size_t i = 0; i < kSeries; ++i)
    {
        ASSERT_FALSE(panel.fits[i].state.residuals.empty()) << i;
        const ve::forecast::Forecast own = ve::forecast::forecast(panel.fits[i], kHorizon, panel.future[i]);
        for (std::size_t h = 0; h < kHorizon; ++h)
        {
            EXPECT_EQ(forecast.sigma[h * kSeries + i], own.sigma[h]) << i << " " << h;
            EXPECT_EQ(forecast.mean[h * kSeries + i], own.mean[h]) << i << " " << h;
        }
    }
}

TEST(DccForecast, Copula)
{
    DccSpec parametric = dcc_spec(DccModel::Dcc);
    parametric.dist = DccDist::MvT;
    parametric.copula = DccCopula::Parametric;
    parametric.fixed = {{"mshape", 8.0}};
    DccSpec empirical = dcc_spec(DccModel::Dcc);
    empirical.copula = DccCopula::Empirical;
    DccSpec fixed_static;
    fixed_static.model = DccModel::Static;
    fixed_static.copula = DccCopula::Parametric;
    fixed_static.static_method = ve::dcc::StaticMethod::Kendall;
    const std::vector<DccFitResult> results = run_batch(dcc3_fits(), {parametric, empirical, fixed_static});
    const SeriesRegressors none(kSeries);
    for (std::size_t k = 0; k < 2; ++k)
    {
        SCOPED_TRACE(k);
        const DccForecast forecast = ve::dcc::future::forecast_dcc(results[k], kHorizon, none);
        expect_formula(results[k], forecast);
    }
    const DccFitResult& constant = results[2];
    const DccForecast forecast = ve::dcc::future::forecast_dcc(constant, kHorizon, none);
    for (std::size_t h = 0; h < kHorizon; ++h)
    {
        for (std::size_t e = 0; e < kSeries * kSeries; ++e)
        {
            EXPECT_EQ(forecast.r[h * kSeries * kSeries + e], constant.end.static_r[e]) << h << " " << e;
        }
    }
}

TEST(DccForecast, Problems)
{
    const SeriesRegressors none(kSeries);
    DccSpec outside = dcc_spec(DccModel::Dcc);
    outside.dist = DccDist::MvT;
    outside.fixed = {{"mshape", 8.0}, {"dccb1", 1.2}};
    const std::vector<DccFitResult> failed = run_batch(dcc3_fits(), {outside, dcc_spec(DccModel::Dcc)});
    ASSERT_EQ(failed[0].status, FitStatus::Failed);
    const std::optional<std::string> no_point = ve::dcc::future::forecast_problem(failed[0], kHorizon, none);
    ASSERT_TRUE(no_point.has_value());
    EXPECT_NE(no_point->find("no finite point"), std::string::npos);

    // A line search that fails keeps its finite point: the same forecast as the converged original.
    DccFitResult kept = failed[1];
    kept.status = FitStatus::Failed;
    ASSERT_FALSE(ve::dcc::future::forecast_problem(kept, kHorizon, none).has_value());
    const DccForecast original = ve::dcc::future::forecast_dcc(failed[1], kHorizon, none);
    const DccForecast copy = ve::dcc::future::forecast_dcc(kept, kHorizon, none);
    EXPECT_EQ(original.r, copy.r);
    EXPECT_EQ(original.sigma, copy.sigma);

    const std::optional<std::string> few = ve::dcc::future::forecast_problem(failed[1], kHorizon, SeriesRegressors(2));
    ASSERT_TRUE(few.has_value());
    EXPECT_NE(few->find("series"), std::string::npos);
    EXPECT_TRUE(ve::dcc::future::forecast_problem(failed[1], 0, none).has_value());

    const std::vector<DccFitResult> mixed = run_batch(mixed_panel().fits, {dcc_spec(DccModel::Dcc)});
    const std::optional<std::string> missing = ve::dcc::future::forecast_problem(mixed[0], kHorizon, none);
    ASSERT_TRUE(missing.has_value());
    EXPECT_NE(missing->find("series 2"), std::string::npos);
    SeriesRegressors short_rows = mixed_panel().future;
    short_rows[1].mean.values = short_rows[1].mean.values.first(kHorizon - 1);
    const std::optional<std::string> short_problem = ve::dcc::future::forecast_problem(mixed[0], kHorizon, short_rows);
    ASSERT_TRUE(short_problem.has_value());
    EXPECT_NE(short_problem->find("series 2"), std::string::npos);
    SeriesRegressors extra = mixed_panel().future;
    extra[0].mean = mixed_panel().future[1].mean;
    const std::optional<std::string> extra_problem = ve::dcc::future::forecast_problem(mixed[0], kHorizon, extra);
    ASSERT_TRUE(extra_problem.has_value());
    EXPECT_NE(extra_problem->find("series 1"), std::string::npos);

    const DccFitResult alone = ve::dcc::fit_dcc(std::make_shared<const ve::dcc::StandardizedPanel>(
                                                    ve::testing::dcc3_panel()),
                                                dcc_spec(DccModel::Dcc), FitConfig{});
    EXPECT_TRUE(ve::dcc::future::forecast_problem(alone, kHorizon, none).has_value());
}

}
