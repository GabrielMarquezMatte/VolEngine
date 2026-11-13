#include "dcc/future/state.hpp"

#include "core/context.hpp"
#include "dcc/fit.hpp"
#include "dcc/parameter_map.hpp"
#include "dcc/spec.hpp"
#include "dcc/dcc3_panel.hpp"
#include "fit/fit_result.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <vector>

namespace
{

using ve::FitConfig;
using ve::FitResult;
using ve::FitStatus;
using ve::dcc::Convention;
using ve::dcc::DccCopula;
using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccOptions;
using ve::dcc::DccSpec;
using ve::dcc::StandardizedPanel;

constexpr std::size_t kSeries = ve::testing::kDcc3Series;

std::shared_ptr<const StandardizedPanel> dcc3()
{
    static const auto panel = std::make_shared<const StandardizedPanel>(ve::testing::dcc3_panel());
    return panel;
}

DccSpec spec_of(DccModel model, Convention convention, int p = 1)
{
    DccSpec spec;
    spec.model = model;
    spec.convention = convention;
    spec.p = p;
    if (model == DccModel::FDcc)
    {
        spec.groups = {1, 1, 2};
    }
    return spec;
}

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

TEST(DccEnd, StepperContinuesTheFit)
{
    const std::shared_ptr<const StandardizedPanel> panel = dcc3();
    const std::size_t rows = panel->t;
    for (Convention convention : {Convention::Standard, Convention::Rmgarch})
    {
        for (const DccSpec& spec : {spec_of(DccModel::Dcc, convention), spec_of(DccModel::Dcc, convention, 2),
                                    spec_of(DccModel::ADcc, convention), spec_of(DccModel::FDcc, convention)})
        {
            SCOPED_TRACE(::testing::Message() << "model " << static_cast<int>(spec.model) << " p " << spec.p
                                              << " convention " << static_cast<int>(convention));
            const DccFitResult fit = ve::dcc::fit_dcc(panel, spec, FitConfig{});
            ASSERT_NE(fit.r, nullptr);
            const ve::dcc::DccLayout layout = ve::dcc::layout_of(spec, kSeries);
            const std::vector<double> theta = ve::dcc::full_parameters(layout, fit.estimate);
            const ve::dcc::future::DccEnd end = ve::dcc::future::dcc_end(*panel, spec, theta, rows - 5);
            ASSERT_FALSE(end.theta.empty());
            ve::dcc::future::CorrelationStepper stepper(end, spec, kSeries);
            for (std::size_t k = 0; k < 5; ++k)
            {
                const std::size_t t = rows - 5 + k;
                const std::span<const double> next = stepper.next();
                for (std::size_t e = 0; e < kSeries * kSeries; ++e)
                {
                    ASSERT_EQ(next[e], (*fit.r)[t * kSeries * kSeries + e]) << "t " << t << " entry " << e;
                }
                stepper.push(std::span<const double>(panel->z).subspan(t * kSeries, kSeries));
            }
        }
    }
}

TEST(DccEnd, FilledByTheBatch)
{
    const std::vector<FitResult> fits = ve::testing::dcc3_fits();
    DccSpec dcc = spec_of(DccModel::Dcc, Convention::Standard);
    DccSpec copula = spec_of(DccModel::Dcc, Convention::Standard);
    copula.dist = DccDist::MvT;
    copula.copula = DccCopula::Parametric;
    copula.fixed.push_back({"mshape", 8.0});
    DccSpec fixed_static;
    fixed_static.model = DccModel::Static;
    fixed_static.copula = DccCopula::Parametric;
    const std::vector<DccFitResult> results = run_batch(fits, {dcc, copula, fixed_static});
    ASSERT_EQ(results.size(), 3U);
    ASSERT_NE(results[0].series_end, nullptr);
    EXPECT_EQ(results[0].series_end, results[1].series_end);
    EXPECT_EQ(results[0].series_end, results[2].series_end);
    ASSERT_EQ(results[0].series_end->size(), kSeries);
    for (std::size_t i = 0; i < kSeries; ++i)
    {
        const ve::dcc::future::SeriesEnd& end = (*results[0].series_end)[i];
        EXPECT_EQ(end.estimate, fits[i].estimate);
        EXPECT_EQ(end.state.residuals, fits[i].state.residuals);
        EXPECT_EQ(end.state.sigma2, fits[i].state.sigma2);
        EXPECT_EQ(end.state.deviations, fits[i].state.deviations);
        EXPECT_EQ(end.spec.model, fits[i].spec.model);
    }
    for (std::size_t k = 0; k < 2; ++k)
    {
        const DccFitResult& result = results[k];
        const ve::dcc::DccLayout layout = ve::dcc::layout_of(result.spec, kSeries);
        EXPECT_EQ(result.end.theta, ve::dcc::full_parameters(layout, result.estimate));
        ve::dcc::future::CorrelationStepper stepper(result.end, result.spec, kSeries);
        const std::span<const double> next = stepper.next();
        for (std::size_t i = 0; i < kSeries; ++i)
        {
            EXPECT_EQ(next[i * kSeries + i], 1.0);
            for (std::size_t j = 0; j < kSeries; ++j)
            {
                EXPECT_TRUE(std::isfinite(next[i * kSeries + j])) << k;
            }
        }
    }
    const DccFitResult& constant = results[2];
    ASSERT_EQ(constant.end.static_r.size(), kSeries * kSeries);
    for (std::size_t e = 0; e < kSeries * kSeries; ++e)
    {
        EXPECT_EQ(constant.end.static_r[e], (*constant.r)[e]);
    }
    ve::dcc::future::CorrelationStepper stepper(constant.end, constant.spec, kSeries);
    EXPECT_EQ(stepper.next()[1], constant.end.static_r[1]);
}

TEST(DccEnd, NoFiniteEstimate)
{
    const std::vector<FitResult> fits = ve::testing::dcc3_fits();
    DccSpec outside = spec_of(DccModel::Dcc, Convention::Standard);
    outside.dist = DccDist::MvT;
    outside.fixed = {{"mshape", 8.0}, {"dccb1", 1.2}};
    DccSpec all_fixed = spec_of(DccModel::Dcc, Convention::Standard);
    all_fixed.fixed = {{"dcca1", 0.04}, {"dccb1", 0.9}};
    const std::vector<DccFitResult> results = run_batch(fits, {outside, all_fixed});
    EXPECT_EQ(results[0].status, FitStatus::Failed);
    EXPECT_EQ(results[0].r, nullptr);
    EXPECT_TRUE(results[0].end.theta.empty());
    EXPECT_EQ(results[0].series_end, nullptr);
    EXPECT_TRUE(results[1].estimate.empty());
    ASSERT_NE(results[1].r, nullptr);
    ASSERT_EQ(results[1].end.theta.size(), 2U);
    EXPECT_EQ(results[1].end.theta[0], 0.04);
    EXPECT_EQ(results[1].end.theta[1], 0.9);
}

// end does not alter what the fit reports: its r is the one the recursion gave.
TEST(DccEnd, NumbersUnchanged)
{
    const std::vector<FitResult> fits = ve::testing::dcc3_fits();
    const DccSpec spec = spec_of(DccModel::Dcc, Convention::Standard);
    const std::vector<DccFitResult> batch = run_batch(fits, {spec});
    const DccFitResult alone = ve::dcc::fit_dcc(dcc3(), spec, FitConfig{});
    EXPECT_EQ(batch[0].loglik, alone.loglik);
    EXPECT_EQ(batch[0].estimate, alone.estimate);
    ASSERT_NE(batch[0].r, nullptr);
    EXPECT_EQ(*batch[0].r, *alone.r);
    EXPECT_EQ(alone.series_end, nullptr);
    EXPECT_TRUE(alone.end.theta.empty());
}

}
