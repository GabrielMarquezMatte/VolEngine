#include "fixture_data.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <vector>
#include <volengine/volengine.hpp>

namespace
{

// The rugarch oracle of case t5000/sgarch11_norm (tests/reference/fixtures/rugarch.json), fitted with tol 1e-7 as in
// python/tests/oracles/test_oracles.py: loglik to 1e-4 relative, parameters to 1e-3 relative.
constexpr double kOracleLoglik = -5058.359626076076;
constexpr double kOracleParams[] = {
    0.045689656391604716, 0.010692733778622033, 0.07516720858429206, 0.9030746410118694};
constexpr std::size_t kRows = 1500;

const std::vector<std::vector<double>>& sgarch_columns()
{
    static const auto columns = ve::testing::read_csv_columns("sgarch11_norm.csv");
    return columns;
}

// The first kRows of every column, row-major (C order).
std::vector<double> sgarch_matrix()
{
    std::vector<double> matrix;
    for (std::size_t t = 0; t < kRows; ++t)
    {
        for (const auto& column : sgarch_columns())
        {
            matrix.push_back(column[t]);
        }
    }
    return matrix;
}

ve::Context make_context()
{
    return ve::Context::create(2).value();
}

TEST(Spec, ValidateRejectsOrderZero)
{
    const auto result = ve::Spec{.p = 0}.validate();
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().status, ve::Status::invalid_argument);
    EXPECT_FALSE(result.error().message.empty());
}

TEST(Spec, ValidateAcceptsDefaults)
{
    EXPECT_TRUE(ve::Spec{}.validate().has_value());
}

TEST(Fit, SingleSeriesMatchesRugarch)
{
    const auto context = make_context();
    const auto fit = context.fit(ve::Spec{}, sgarch_columns()[0], ve::FitOptions{.tol = 1e-7});
    ASSERT_TRUE(fit.has_value());
    EXPECT_EQ(fit->status(), ve::FitStatus::converged);
    EXPECT_EQ(fit->nobs().value(), 5000U);
    const auto names = fit->param_names().value();
    ASSERT_EQ(names.size(), 4U);
    EXPECT_EQ(names[0], "mu");
    EXPECT_EQ(names[3], "beta1");
    EXPECT_NEAR(fit->loglik().value() / kOracleLoglik, 1.0, 1e-4);
    const auto estimates = fit->estimates().value();
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(estimates[i] / kOracleParams[i], 1.0, 1e-3) << names[i];
    }
}

TEST(Fit, PanelFitIndexesSpecMajor)
{
    const auto context = make_context();
    const std::size_t series = sgarch_columns().size();
    const auto matrix = sgarch_matrix();
    const auto panel = ve::Panel::create(matrix, kRows, series).value();
    const std::vector<ve::Spec> specs{ve::Spec{}, ve::Spec{.dist = ve::Dist::student}};
    const auto fits = context.fit(specs, panel);
    ASSERT_TRUE(fits.has_value());
    ASSERT_EQ(fits->size(), 2 * series);
    EXPECT_EQ((*fits)[1 * series + 2].param_names().value().back(), "shape");
    EXPECT_EQ((*fits)[2].param_names().value().back(), "beta1");
}

TEST(Batch, JobsSelectSubset)
{
    const auto context = make_context();
    const auto matrix = sgarch_matrix();
    const auto panel = ve::Panel::create(matrix, kRows, sgarch_columns().size()).value();
    const std::vector<ve::Spec> specs{ve::Spec{}, ve::Spec{.dist = ve::Dist::student}};
    auto batch = ve::Batch::create().value();
    ASSERT_TRUE(batch.set_specs(specs).has_value());
    ASSERT_TRUE(batch.set_panel(panel).has_value());
    const std::vector<std::size_t> spec_index{0, 1};
    const std::vector<std::size_t> series_index{2, 0};
    ASSERT_TRUE(batch.set_jobs(spec_index, series_index).has_value());
    EXPECT_EQ(batch.job_count().value(), 2U);
    const auto fits = context.fit(batch);
    ASSERT_TRUE(fits.has_value());
    EXPECT_EQ(fits->size(), 2U);
}

TEST(Batch, SpecRejectedByTheSetterReturnsNoFits)
{
    const auto context = make_context();
    const auto fits = context.fit(ve::Spec{.p = 0}, sgarch_columns()[0]);
    ASSERT_FALSE(fits.has_value());
    EXPECT_EQ(fits.error().status, ve::Status::invalid_argument);
    EXPECT_FALSE(fits.error().message.empty());
}

// A name the spec does not have is only found when the batch validates its specs: the message carries the spec index.
TEST(Batch, SpecRejectedByTheBatchNamesTheSpec)
{
    const auto context = make_context();
    const auto fits = context.fit(ve::Spec{.fixed = {{"nosuchparam", 1.0}}}, sgarch_columns()[0]);
    ASSERT_FALSE(fits.has_value());
    EXPECT_EQ(fits.error().status, ve::Status::invalid_argument);
    EXPECT_TRUE(fits.error().message.starts_with("spec 0:")) << fits.error().message;
}

TEST(Fit, ShortSeriesIsStatusNotError)
{
    const auto context = make_context();
    const auto fit = context.fit(ve::Spec{}, std::vector<double>{0.1, -0.2, 0.3, 0.0, 0.1});
    ASSERT_TRUE(fit.has_value());
    EXPECT_EQ(fit->status(), ve::FitStatus::insufficient_data);
}

TEST(Fit, NaNInSeriesIsError)
{
    const auto context = make_context();
    auto series = sgarch_columns()[0];
    series[10] = std::numeric_limits<double>::quiet_NaN();
    const auto fit = context.fit(ve::Spec{}, series);
    ASSERT_FALSE(fit.has_value());
    EXPECT_EQ(fit.error().status, ve::Status::invalid_argument);
}

TEST(Panel, EmptyPanelIsAcceptedButCannotBeFitted)
{
    const auto context = make_context();
    const auto panel = ve::Panel::create({}, 0, 0);
    ASSERT_TRUE(panel.has_value());
    const std::vector<ve::Spec> specs{ve::Spec{}};
    EXPECT_FALSE(context.fit(specs, *panel).has_value());
}

TEST(Panel, SpanSmallerThanTheShapeIsError)
{
    const std::vector<double> data(10, 0.0);
    const auto panel = ve::Panel::create(data, 4, 4);
    ASSERT_FALSE(panel.has_value());
    EXPECT_EQ(panel.error().status, ve::Status::invalid_argument);
}

TEST(Panel, StridedPanelEqualsContiguous)
{
    const auto context = make_context();
    const auto& columns = sgarch_columns();
    std::vector<double> c_order;
    std::vector<double> fortran_order;
    for (std::size_t t = 0; t < kRows; ++t)
    {
        c_order.push_back(columns[0][t]);
        c_order.push_back(columns[1][t]);
    }
    fortran_order.insert(fortran_order.end(), columns[0].begin(), columns[0].begin() + kRows);
    fortran_order.insert(fortran_order.end(), columns[1].begin(), columns[1].begin() + kRows);
    const auto contiguous = ve::Panel::create(c_order, kRows, 2).value();
    const auto strided =
        ve::Panel::create_strided(fortran_order, kRows, 2, 1, static_cast<std::ptrdiff_t>(kRows)).value();
    const std::vector<ve::Spec> specs{ve::Spec{}};
    const auto a = context.fit(specs, contiguous).value();
    const auto b = context.fit(specs, strided).value();
    EXPECT_EQ(a[1].loglik().value(), b[1].loglik().value());
}

// The remaining oracle numbers of t5000/sgarch11_norm: Hessian and QML robust standard errors (5e-2 relative) and the
// per-observation criteria (1e-6 absolute), as in python/tests/oracles/test_oracles.py.
constexpr double kOracleSe[] = {0.008663366651859546, 0.002228460618358319, 0.00810505000455375, 0.010657614107829137};
constexpr double kOracleRobustSe[] = {
    0.008677832549151743, 0.0022377402070646816, 0.008353834488907993, 0.010696037257650498};
constexpr double kOracleAic = 2.02494385043043;
constexpr double kOracleBic = 2.0301576049835632;
constexpr double kOracleHqic = 2.02677118938873;
constexpr double kOracleSic = 2.0249425717941274;

ve::Fit fit_oracle_series(const ve::FitOptions& options)
{
    const auto context = make_context();
    return context.fit(ve::Spec{}, sgarch_columns()[0], options).value();
}

TEST(FitResults, StdErrorsMatchRugarch)
{
    const auto fit = fit_oracle_series(ve::FitOptions{.tol = 1e-7});
    const auto hessian = fit.std_errors(ve::SeKind::hessian).value();
    const auto robust = fit.std_errors(ve::SeKind::robust).value();
    ASSERT_EQ(hessian.size(), 4U);
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(hessian[i] / kOracleSe[i], 1.0, 5e-2) << i;
        EXPECT_NEAR(robust[i] / kOracleRobustSe[i], 1.0, 5e-2) << i;
    }
}

TEST(FitResults, TAndPValuesFollowStdErrors)
{
    const auto fit = fit_oracle_series(ve::FitOptions{.tol = 1e-7});
    const auto estimates = fit.estimates().value();
    const auto errors = fit.std_errors(ve::SeKind::hessian).value();
    const auto t = fit.t_values(ve::SeKind::hessian).value();
    const auto p = fit.p_values(ve::SeKind::hessian).value();
    for (std::size_t i = 0; i < estimates.size(); ++i)
    {
        EXPECT_NEAR(t[i], estimates[i] / errors[i], 1e-9 * std::abs(t[i])) << i;
        EXPECT_GE(p[i], 0.0);
        EXPECT_LE(p[i], 1.0);
    }
}

TEST(FitResults, CriteriaMatchRugarchAndScale)
{
    const auto fit = fit_oracle_series(ve::FitOptions{.tol = 1e-7});
    EXPECT_NEAR(fit.aic().value(), kOracleAic, 1e-6);
    EXPECT_NEAR(fit.bic().value(), kOracleBic, 1e-6);
    EXPECT_NEAR(fit.hqic().value(), kOracleHqic, 1e-6);
    EXPECT_NEAR(fit.sic().value(), kOracleSic, 1e-6);
    const double nobs = static_cast<double>(fit.nobs().value());
    EXPECT_NEAR(fit.aic(ve::IcScale::total).value(), fit.aic(ve::IcScale::per_obs).value() * nobs, 1e-6);
    EXPECT_GT(fit.bic().value(), fit.aic().value());
}

TEST(FitResults, CriterionByKindMatchesTheShortcuts)
{
    const auto fit = fit_oracle_series(ve::FitOptions{.tol = 1e-7});
    EXPECT_EQ(fit.criterion(ve::Criterion::aic).value(), fit.aic().value());
    EXPECT_EQ(fit.criterion(ve::Criterion::bic, ve::IcScale::total).value(), fit.bic(ve::IcScale::total).value());
    EXPECT_EQ(fit.criterion(ve::Criterion::hqic).value(), fit.hqic().value());
    EXPECT_EQ(fit.criterion(ve::Criterion::sic).value(), fit.sic().value());
    EXPECT_FALSE(fit.criterion(static_cast<ve::Criterion>(9)).has_value());
}

TEST(FitResults, SeriesViewsHaveNobsLength)
{
    const auto fit = fit_oracle_series(ve::FitOptions{});
    const auto nobs = fit.nobs().value();
    const auto sigma = fit.sigma().value();
    EXPECT_EQ(sigma.size(), nobs);
    EXPECT_EQ(fit.residuals().value().size(), nobs);
    EXPECT_EQ(fit.std_residuals().value().size(), nobs);
    EXPECT_EQ(fit.fitted().value().size(), nobs);
    EXPECT_GT(*std::min_element(sigma.begin(), sigma.end()), 0.0);
}

TEST(FitResults, StandardErrorsOffMakesInferenceFailButNotTheEstimates)
{
    const auto on = fit_oracle_series(ve::FitOptions{.tol = 1e-7});
    const auto off = fit_oracle_series(ve::FitOptions{.tol = 1e-7, .standard_errors = false});
    for (const auto kind : {ve::SeKind::hessian, ve::SeKind::robust})
    {
        const auto errors = off.std_errors(kind);
        ASSERT_FALSE(errors.has_value());
        EXPECT_EQ(errors.error().status, ve::Status::invalid_argument);
        EXPECT_FALSE(off.t_values(kind).has_value());
        EXPECT_FALSE(off.p_values(kind).has_value());
    }
    EXPECT_FALSE(off.nyblom().has_value());
    // The estimates are bit for bit equal; the reported log-likelihood differs at the 13th digit (see the ABI findings).
    EXPECT_NEAR(off.loglik().value() / on.loglik().value(), 1.0, 1e-11);
    EXPECT_EQ(off.estimates().value(), on.estimates().value());
}

} // namespace
