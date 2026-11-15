#include "fixture_data.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <string_view>
#include <vector>
#include <volengine/volengine.hpp>

namespace
{

// The rmgarch panel dcc3 (tests/reference/data/dcc3.csv, columns A, B, C): the univariate sGARCH(1,1) fits start at
// the oracle's estimates and stop there (tol = 1), as in python/tests/oracles/test_dcc_oracles.py.
constexpr double kUnivariate[3][4] = {
    {-0.008920131111225086, 0.019478881309471997, 0.07282124034137887, 0.9065714955528129},
    {-0.02778242404762519, 0.061441088509364916, 0.084080533476537, 0.8462651839828331},
    {-0.009972193379566455, 0.025655072442689313, 0.050370071713741006, 0.9338721906860878},
};
constexpr const char* kNames[4] = {"mu", "omega", "alpha1", "beta1"};

std::vector<ve::Fit> univariate_fits(const ve::Context& context)
{
    const auto columns = ve::testing::read_csv_columns("dcc3.csv");
    std::vector<ve::Fit> fits;
    for (std::size_t i = 0; i < 3; ++i)
    {
        ve::NamedValues start;
        for (std::size_t k = 0; k < 4; ++k)
        {
            start.emplace_back(kNames[k], kUnivariate[i][k]);
        }
        fits.push_back(context.fit(ve::Spec{.start = start}, columns[i], ve::FitOptions{.tol = 1.0}).value());
    }
    return fits;
}

TEST(Dcc, MatchesRmgarch)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = univariate_fits(context);
    const auto fit = context.fit_dcc(ve::DccSpec{}, fits, ve::DccOptions{.standard_errors = false});
    ASSERT_TRUE(fit.has_value());
    EXPECT_EQ(fit->status(), ve::FitStatus::converged);
    const auto names = fit->param_names().value();
    ASSERT_EQ(names.size(), 2U);
    EXPECT_EQ(names[0], "dcca1");
    EXPECT_EQ(names[1], "dccb1");
    // rmgarch: dcca1 0.04316409231673219, dccb1 0.914269645079578 (5% relative, the Python test's limit), llh -6082.94.
    const auto estimates = fit->estimates().value();
    EXPECT_NEAR(estimates[0] / 0.04316409231673219, 1.0, 5e-2);
    EXPECT_NEAR(estimates[1] / 0.914269645079578, 1.0, 5e-2);
    EXPECT_NEAR(fit->loglik().value() / -6082.940870895449, 1.0, 1e-3);
}

TEST(Dcc, CorrelationIsTimeByMByMRowMajor)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = univariate_fits(context);
    const auto fit = context.fit_dcc(ve::DccSpec{}, fits, ve::DccOptions{.standard_errors = false}).value();
    const std::size_t nobs = fit.nobs().value();
    EXPECT_EQ(fit.series().value(), 3U);
    const auto correlation = fit.correlation().value();
    ASSERT_EQ(correlation.size(), nobs * 9);
    for (std::size_t t = 0; t < nobs; t += 97)
    {
        const double* r = correlation.data() + t * 9;
        for (std::size_t i = 0; i < 3; ++i)
        {
            EXPECT_NEAR(r[i * 3 + i], 1.0, 1e-12);
            EXPECT_NEAR(r[i * 3 + (i + 1) % 3], r[((i + 1) % 3) * 3 + i], 1e-12);
        }
    }
}

TEST(Dcc, CovarianceDiagonalIsTheUnivariateVariance)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = univariate_fits(context);
    const auto fit = context.fit_dcc(ve::DccSpec{}, fits, ve::DccOptions{.standard_errors = false}).value();
    const auto covariance = fit.covariance().value();
    const auto sigma = fits[1].sigma().value();
    ASSERT_EQ(covariance.size(), fit.nobs().value() * 9);
    for (std::size_t t = 0; t < sigma.size(); t += 97)
    {
        EXPECT_NEAR(covariance[t * 9 + 4], sigma[t] * sigma[t], 1e-9 * sigma[t] * sigma[t]);
    }
}

TEST(Dcc, GridKeepsSpecOrder)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = univariate_fits(context);
    const std::vector<ve::DccSpec> grid{
        ve::DccSpec{.model = ve::DccModel::dcc},
        ve::DccSpec{.model = ve::DccModel::adcc},
        ve::DccSpec{.model = ve::DccModel::fdcc, .groups = {1, 1, 2}},
    };
    const auto result = context.fit_dcc(grid, fits, ve::DccOptions{.standard_errors = false});
    ASSERT_TRUE(result.has_value());
    ASSERT_EQ(result->size(), 3U);
    EXPECT_EQ((*result)[0].param_names().value()[0], "dcca1");
    const auto adcc = (*result)[1].param_names().value();
    EXPECT_NE(std::find(adcc.begin(), adcc.end(), std::string_view("dccg1")), adcc.end());
    EXPECT_EQ((*result)[2].param_names().value()[0], "fdcca1");
}

TEST(Dcc, StandardErrorsFromFitsMatchTheOption)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = univariate_fits(context);
    const auto without = context.fit_dcc(ve::DccSpec{}, fits, ve::DccOptions{.standard_errors = false}).value();
    const auto with = context.fit_dcc(ve::DccSpec{}, fits).value();
    const auto computed = context.standard_errors(without, fits).value();
    for (const auto kind : {ve::SeKind::hessian, ve::SeKind::robust})
    {
        const auto expected = with.std_errors(kind).value();
        const auto actual = computed.std_errors(kind).value();
        ASSERT_EQ(actual.size(), expected.size());
        for (std::size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_NEAR(actual[i] / expected[i], 1.0, 1e-8);
            EXPECT_GT(actual[i], 0.0);
        }
    }
    EXPECT_EQ(computed.nobs().value(), without.nobs().value());
}

TEST(Dcc, SingleSeriesIsRejected)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = univariate_fits(context);
    const auto fit = context.fit_dcc(ve::DccSpec{}, std::span<const ve::Fit>(fits.data(), 1));
    ASSERT_FALSE(fit.has_value());
    EXPECT_EQ(fit.error().status, ve::Status::invalid_argument);
}

// Unlike a univariate Fit, a DccFit without standard errors answers with NaN, not with an error.
TEST(Dcc, StandardErrorsOffGiveNaN)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = univariate_fits(context);
    const auto fit = context.fit_dcc(ve::DccSpec{}, fits, ve::DccOptions{.standard_errors = false}).value();
    for (const auto kind : {ve::SeKind::hessian, ve::SeKind::robust})
    {
        for (const auto& values : {fit.std_errors(kind), fit.t_values(kind), fit.p_values(kind)})
        {
            ASSERT_TRUE(values.has_value());
            EXPECT_TRUE(std::all_of(values->begin(), values->end(), [](double v) { return std::isnan(v); }));
        }
    }
}

TEST(Dcc, CopulaAndStaticModelsConverge)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = univariate_fits(context);
    const auto copula = context.fit_dcc(
        ve::DccSpec{.dist = ve::DccDist::mvt, .copula = ve::Copula::parametric}, fits,
        ve::DccOptions{.standard_errors = false});
    ASSERT_TRUE(copula.has_value()) << copula.error().message;
    EXPECT_EQ(copula->status(), ve::FitStatus::converged);
    const auto fixed = context.fit_dcc(
        ve::DccSpec{.model = ve::DccModel::static_,
            .copula = ve::Copula::parametric,
            .static_method = ve::StaticMethod::kendall},
        fits, ve::DccOptions{.standard_errors = false});
    ASSERT_TRUE(fixed.has_value()) << fixed.error().message;
    EXPECT_EQ(fixed->status(), ve::FitStatus::converged);
}

// The criteria count the first stage and the correlation intercept too: 3 x 4 univariate parameters, 2 DCC parameters
// and the 3 free entries of Qbar, 17 in all, not the 2 of param_count().
TEST(Dcc, CriterionCountsTheFirstStageAndTheIntercept)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = univariate_fits(context);
    const auto fit = context.fit_dcc(ve::DccSpec{}, fits, ve::DccOptions{.standard_errors = false}).value();
    const double n = static_cast<double>(fit.nobs().value());
    const double loglik = fit.loglik().value();
    const auto aic = fit.criterion(ve::DccCriterion::aic, ve::IcScale::total).value();
    const auto bic = fit.criterion(ve::DccCriterion::bic, ve::IcScale::total).value();
    EXPECT_NEAR(aic, -2.0 * loglik + 2.0 * 17.0, 1e-6 * std::abs(aic));
    EXPECT_NEAR(bic, -2.0 * loglik + 17.0 * std::log(n), 1e-6 * std::abs(bic));
    EXPECT_NEAR(fit.criterion(ve::DccCriterion::aic).value(), aic / n, 1e-9 * std::abs(aic));
    EXPECT_EQ(fit.param_count().value(), 2U);
}

} // namespace
