#include "dcc/fit.hpp"

#include "core/context.hpp"
#include "core/spec.hpp"
#include "dcc/likelihood.hpp"
#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "dcc3_panel.hpp"
#include "fit/fit.hpp"
#include "fit/fit_result.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::FitConfig;
using ve::FitResult;
using ve::FitStatus;
using ve::dcc::Convention;
using ve::dcc::Criterion;
using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccOptions;
using ve::dcc::DccSpec;
using ve::dcc::StandardizedPanel;

std::vector<const FitResult*> pointers_of(const std::vector<FitResult>& fits)
{
    std::vector<const FitResult*> pointers;
    for (const FitResult& fit : fits)
    {
        pointers.push_back(&fit);
    }
    return pointers;
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

std::vector<DccFitResult> run_batch(ve::Context& context, const std::vector<FitResult>& fits,
                                    const std::vector<DccSpec>& specs, const FitConfig& config = FitConfig{})
{
    const std::vector<const FitResult*> pointers = pointers_of(fits);
    EXPECT_FALSE(ve::dcc::check_dcc_batch(pointers, specs).has_value());
    return ve::dcc::fit_dcc_batch(context, pointers, specs, DccOptions{config, false});
}

// rmgarch's estimates on dcc3 (the notes, section 10) and its reported loglik.
struct Oracle
{
    DccModel model;
    std::vector<double> estimate;
    double loglik;
    std::size_t parameters;
    std::array<double, 4> criteria;
};

const std::array<Oracle, 2> kOracles{{
    {DccModel::Dcc,
     {0.0431630001835819, 0.9142701826310855},
     -6082.94087772754,
     17,
     {8.13325450364, 8.19347100136, 8.13300143181, 8.15568733467}},
    {DccModel::FDcc,
     {0.19004678480, 0.24108698852, 0.96463050567, 0.94296237137},
     -6082.56656677,
     19,
     {8.13542208903, 8.20272288060, 0.0, 0.0}},
}};

constexpr std::array<Criterion, 4> kCriteria{Criterion::Aic, Criterion::Bic, Criterion::Sic, Criterion::Hqic};

// The dcc3 fits are fixed at rugarch's estimates, so they estimate nothing and the batch counts no univariate
// parameter. The oracle counts the 12 of the free fits.
constexpr std::size_t kUnivariateParameters = 12;

TEST(DccBatch, CriteriaMatchRmgarch)
{
    ve::Context context;
    const std::vector<FitResult> fits = ve::testing::dcc3_fits();
    const std::vector<DccSpec> specs{spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Rmgarch),
                                     spec_of(DccModel::FDcc, DccDist::MvNorm, Convention::Rmgarch)};
    const std::vector<DccFitResult> results = run_batch(context, fits, specs);
    ASSERT_EQ(results.size(), 2U);
    for (std::size_t i = 0; i < results.size(); ++i)
    {
        const Oracle& oracle = kOracles[i];
        SCOPED_TRACE(i);
        EXPECT_EQ(results[i].univariate_parameters, 0U);
        DccFitResult at_rmgarch = results[i];
        at_rmgarch.univariate_parameters = kUnivariateParameters;
        at_rmgarch.estimate = oracle.estimate;
        at_rmgarch.loglik = ve::dcc::reported_loglik(*at_rmgarch.panel, at_rmgarch.spec, oracle.estimate);
        EXPECT_NEAR(at_rmgarch.loglik, oracle.loglik, 1e-9 * std::fabs(oracle.loglik));
        const std::size_t count = ve::testing::kDcc3Rows;
        const double total = static_cast<double>(count);
        const double k = static_cast<double>(oracle.parameters);
        const std::size_t checked = i == 0 ? 4 : 2;
        for (std::size_t c = 0; c < checked; ++c)
        {
            const double expected = oracle.criteria[c];
            EXPECT_NEAR(ve::dcc::dcc_criterion(at_rmgarch, kCriteria[c], true), expected, 1e-10 * expected) << c;
            const double scaled = expected * total;
            EXPECT_NEAR(ve::dcc::dcc_criterion(at_rmgarch, kCriteria[c], false), scaled, 1e-10 * scaled) << c;
        }
        const double aic = (-2.0 * at_rmgarch.loglik + 2.0 * k) / total;
        EXPECT_DOUBLE_EQ(ve::dcc::dcc_criterion(at_rmgarch, Criterion::Aic, true), aic);
    }
}

TEST(DccBatch, CriteriaAreNaNWithoutALoglik)
{
    DccFitResult fit;
    fit.nobs = 1500;
    fit.series = 3;
    EXPECT_TRUE(std::isnan(ve::dcc::dcc_criterion(fit, Criterion::Aic, true)));
    EXPECT_TRUE(std::isnan(ve::dcc::dcc_criterion(fit, Criterion::Sic, false)));
}

TEST(DccBatch, CountsAndKeepsTheUnivariateFits)
{
    ve::Context context;
    const std::vector<std::vector<double>> columns = ve::testing::read_dcc3();
    std::vector<FitResult> fits;
    for (const std::vector<double>& column : columns)
    {
        fits.push_back(ve::fit_one(column, ve::Spec{}, FitConfig{}));
    }
    const std::vector<DccFitResult> results =
        run_batch(context, fits, {spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Standard)});
    ASSERT_EQ(results.size(), 1U);
    std::vector<double> concatenated;
    for (const FitResult& fit : fits)
    {
        concatenated.insert(concatenated.end(), fit.estimate.begin(), fit.estimate.end());
    }
    EXPECT_EQ(concatenated.size(), kUnivariateParameters);
    EXPECT_EQ(results[0].univariate_parameters, kUnivariateParameters);
    EXPECT_EQ(results[0].univariate_estimates, concatenated);
}

bool same_bits(std::span<const double> a, std::span<const double> b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::bit_cast<uint64_t>(a[i]) != std::bit_cast<uint64_t>(b[i]))
        {
            return false;
        }
    }
    return true;
}

void expect_same(const DccFitResult& actual, const DccFitResult& expected)
{
    EXPECT_EQ(actual.status, expected.status);
    EXPECT_EQ(actual.names, expected.names);
    EXPECT_TRUE(same_bits(actual.estimate, expected.estimate));
    EXPECT_EQ(std::bit_cast<uint64_t>(actual.loglik), std::bit_cast<uint64_t>(expected.loglik));
    ASSERT_EQ(actual.r == nullptr, expected.r == nullptr);
    if (actual.r != nullptr)
    {
        EXPECT_TRUE(same_bits(*actual.r, *expected.r));
    }
}

std::vector<DccSpec> grid()
{
    std::vector<DccSpec> specs;
    for (const DccModel model : {DccModel::Dcc, DccModel::ADcc, DccModel::FDcc})
    {
        for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT, DccDist::MvLaplace})
        {
            specs.push_back(spec_of(model, dist, Convention::Standard));
        }
    }
    return specs;
}

// Fitted one by one on a panel of their own, the way a caller without the batch would.
std::vector<DccFitResult> fit_alone(const std::vector<FitResult>& fits, const std::vector<DccSpec>& specs,
                                    const FitConfig& config)
{
    const auto panel = std::make_shared<const StandardizedPanel>(ve::dcc::standardize(pointers_of(fits)));
    std::vector<DccFitResult> results;
    for (const DccSpec& spec : specs)
    {
        results.push_back(ve::dcc::fit_dcc(panel, spec, config));
    }
    return results;
}

void expect_batches_match_alone(const std::vector<FitResult>& fits, const std::vector<DccSpec>& specs,
                                const FitConfig& config = FitConfig{})
{
    const std::vector<DccFitResult> alone = fit_alone(fits, specs, config);
    for (const int threads : {1, 3, 16})
    {
        SCOPED_TRACE(threads);
        ve::Context context;
        context.set_threads(threads);
        const std::vector<DccFitResult> results = run_batch(context, fits, specs, config);
        ASSERT_EQ(results.size(), specs.size());
        for (std::size_t i = 0; i < specs.size(); ++i)
        {
            SCOPED_TRACE(i);
            expect_same(results[i], alone[i]);
        }
    }
}

TEST(DccBatch, SameAsAloneForAnyThreadCount)
{
    expect_batches_match_alone(ve::testing::dcc3_fits(), grid());
}

std::vector<FitResult> independent_pair()
{
    std::mt19937_64 engine(7);
    std::normal_distribution<double> normal;
    std::vector<FitResult> fits;
    ve::Spec unit;
    unit.fixed = {{"mu", 0.0}, {"omega", 1.0}, {"alpha1", 0.0}, {"beta1", 0.0}};
    for (int series = 0; series < 2; ++series)
    {
        std::vector<double> column(500);
        for (double& x : column)
        {
            x = normal(engine);
        }
        fits.push_back(ve::fit_one(column, unit, FitConfig{}));
    }
    return fits;
}

std::vector<DccSpec> hard_specs()
{
    DccSpec adcc = spec_of(DccModel::ADcc, DccDist::MvNorm, Convention::Standard);
    adcc.p = 2;
    DccSpec adcc_t = adcc;
    adcc_t.dist = DccDist::MvT;
    DccSpec dcc = spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Standard);
    dcc.q = 2;
    DccSpec outside = spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Standard);
    outside.fixed = {{"dccb1", 1.2}};
    return {adcc, adcc_t, dcc, spec_of(DccModel::Dcc, DccDist::MvLaplace, Convention::Standard), outside};
}

// Nearly uncorrelated columns, where the aDCC(2, 1) and DCC(1, 2) specs converge all the same, and a spec fixed
// outside its domain, which fails: the batch reports the status each spec has alone and leaves the others unchanged.
// The iteration limit below covers the specs that stop before the optimum.
TEST(DccBatch, SpecsThatDoNotConvergeKeepTheirStatus)
{
    const std::vector<FitResult> fits = independent_pair();
    const std::vector<DccSpec> specs = hard_specs();
    expect_batches_match_alone(fits, specs);
    ve::Context context;
    const std::vector<DccFitResult> results = run_batch(context, fits, specs);
    EXPECT_EQ(results.back().status, FitStatus::Failed);
    EXPECT_EQ(results.back().r, nullptr);
}

// An iteration limit that no spec reaches the optimum within: every status is MaxIter, and still bit-identical.
TEST(DccBatch, SpecsStoppedAtTheIterationLimitKeepTheirStatus)
{
    const std::vector<FitResult> fits = ve::testing::dcc3_fits();
    FitConfig config;
    config.max_iter = 2;
    expect_batches_match_alone(fits, grid(), config);
    ve::Context context;
    const std::vector<DccFitResult> results = run_batch(context, fits, grid(), config);
    for (const DccFitResult& result : results)
    {
        EXPECT_EQ(result.status, FitStatus::MaxIter);
        EXPECT_NE(result.r, nullptr);
    }
}

TEST(DccBatch, ReportsWhyABatchCannotRun)
{
    const std::vector<FitResult> fits = ve::testing::dcc3_fits();
    const std::vector<const FitResult*> pointers = pointers_of(fits);
    EXPECT_TRUE(ve::dcc::check_dcc_batch(pointers, {}).has_value());
    DccSpec groups = spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Standard);
    groups.groups = {1, 1, 2};
    const std::optional<std::string> problem = ve::dcc::check_dcc_batch(
        pointers, std::vector<DccSpec>{spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Standard), groups});
    ASSERT_TRUE(problem.has_value());
    EXPECT_NE(problem->find("spec 2"), std::string::npos) << *problem;
    const std::vector<const FitResult*> one{pointers[0]};
    const std::optional<std::string> panel = ve::dcc::check_dcc_batch(one, std::vector<DccSpec>{DccSpec{}});
    ASSERT_TRUE(panel.has_value());
    EXPECT_NE(panel->find("series"), std::string::npos) << *panel;
}

TEST(DccBatch, CovarianceFromSigma)
{
    ve::Context context;
    std::vector<DccFitResult> results;
    std::vector<double> sigma;
    {
        std::vector<FitResult> fits = ve::testing::dcc3_fits();
        results = run_batch(context, fits, grid());
        for (std::size_t t = 0; t < ve::testing::kDcc3Rows; ++t)
        {
            for (const FitResult& fit : fits)
            {
                sigma.push_back(fit.sigma[t]);
            }
        }
        fits.clear();
    }
    ASSERT_EQ(results.size(), 9U);
    for (const DccFitResult& result : results)
    {
        EXPECT_EQ(result.panel, results[0].panel);
        ASSERT_NE(result.r, nullptr);
    }
    constexpr std::size_t m = ve::testing::kDcc3Series;
    const DccFitResult& fit = results[0];
    const std::vector<double> h = ve::dcc::covariance_path(fit);
    ASSERT_EQ(h.size(), ve::testing::kDcc3Rows * m * m);
    for (std::size_t t = 0; t < ve::testing::kDcc3Rows; ++t)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j < m; ++j)
            {
                const std::size_t k = t * m * m + i * m + j;
                const double expected = sigma[t * m + i] * (*fit.r)[k] * sigma[t * m + j];
                ASSERT_NEAR(h[k], expected, 1e-15 * std::fabs(expected)) << t << ' ' << i << ' ' << j;
            }
        }
    }
    EXPECT_EQ(ve::dcc::covariance_path(fit), h);
    EXPECT_TRUE(ve::dcc::covariance_path(DccFitResult{}).empty());
}

}
