#include "fixture_data.hpp"

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <vector>
#include <volengine/volengine.hpp>

namespace
{

// rmgarch's DCC forecast of dcc3 (tests/reference/fixtures/rmgarch_dcc_forecast.json, case dcc_mvnorm): the DCC fixed
// at rmgarch's estimates, 10 steps ahead, R and H to 1e-8 absolute and sigma to 1e-8 relative, as in
// tests/unit/dcc/future/forecast_oracle_test.cpp.
constexpr std::size_t kAhead = 10;
const ve::NamedValues kRmgarchDcc{{"dcca1", 0.04316409231673219}, {"dccb1", 0.914269645079578}};

ve::DccFit fixed_dcc(const ve::Context& context, const std::vector<ve::Fit>& fits)
{
    return context
        .fit_dcc(ve::DccSpec{.fixed = kRmgarchDcc}, fits, ve::DccOptions{.standard_errors = false})
        .value();
}

TEST(DccForecast, MatchesRmgarch)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = ve::testing::dcc3_univariate_fits(context);
    const auto forecast = fixed_dcc(context, fits).forecast(kAhead);
    ASSERT_TRUE(forecast.has_value());
    const auto correlation = forecast->correlation().value();
    const auto covariance = forecast->covariance().value();
    const auto sigma = forecast->sigma().value();
    ASSERT_EQ(correlation.size(), kAhead * 9);
    EXPECT_NEAR(correlation[1], 0.6795950881128057, 1e-8);
    EXPECT_NEAR(correlation[2], 0.47693709938012757, 1e-8);
    EXPECT_NEAR(correlation[5], 0.5835730145898013, 1e-8);
    const double* last_r = correlation.data() + 9 * 9;
    EXPECT_NEAR(last_r[1], 0.6132015310781206, 1e-8);
    EXPECT_NEAR(last_r[5], 0.504742182335091, 1e-8);
    const double* last_h = covariance.data() + 9 * 9;
    EXPECT_NEAR(last_h[0], 1.0390578186853268, 1e-8);
    EXPECT_NEAR(last_h[4], 1.199263549722033, 1e-8);
    EXPECT_NEAR(last_h[8], 2.672959638655696, 1e-8);
    EXPECT_NEAR(sigma[0] / 1.0287828757189434, 1.0, 1e-8);
    EXPECT_NEAR(sigma[9 * 3 + 2] / 1.6349188477278302, 1.0, 1e-8);
}

TEST(DccForecast, ShapesAndHorizon)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = ve::testing::dcc3_univariate_fits(context);
    const auto forecast = fixed_dcc(context, fits).forecast(4).value();
    EXPECT_EQ(forecast.horizon().value(), 4U);
    EXPECT_EQ(forecast.series().value(), 3U);
    EXPECT_EQ(forecast.correlation().value().size(), 4U * 9U);
    EXPECT_EQ(forecast.covariance().value().size(), 4U * 9U);
    EXPECT_EQ(forecast.sigma().value().size(), 4U * 3U);
    EXPECT_EQ(forecast.mean().value().size(), 4U * 3U);
    EXPECT_EQ(fixed_dcc(context, fits).forecast(0).error().status, ve::Status::invalid_argument);
}

TEST(DccForecast, OutlivesTheUnivariateFitsAndTheDccFit)
{
    const auto context = ve::Context::create(1).value();
    auto fits = std::make_unique<std::vector<ve::Fit>>(ve::testing::dcc3_univariate_fits(context));
    auto dcc = std::make_unique<ve::DccFit>(fixed_dcc(context, *fits));
    fits.reset();
    const auto forecast = dcc->forecast(kAhead);
    ASSERT_TRUE(forecast.has_value()) << forecast.error().message;
    const std::vector<double> before(forecast->sigma().value().begin(), forecast->sigma().value().end());
    dcc.reset();
    const auto after = forecast->sigma().value();
    EXPECT_TRUE(std::equal(before.begin(), before.end(), after.begin()));
}

TEST(DccForecast, FutureWithAnotherSeriesCountIsError)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = ve::testing::dcc3_univariate_fits(context);
    const auto future = ve::DccFuture::create(2, 5).value();
    const auto forecast = fixed_dcc(context, fits).forecast(5, &future);
    ASSERT_FALSE(forecast.has_value());
    EXPECT_EQ(forecast.error().status, ve::Status::invalid_argument);
    EXPECT_FALSE(forecast.error().message.empty());
}

TEST(DccForecast, FutureWithoutRegressorsForecastsLikeNone)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = ve::testing::dcc3_univariate_fits(context);
    const auto dcc = fixed_dcc(context, fits);
    auto future = ve::DccFuture::create(3, 5).value();
    ASSERT_TRUE(future.set(0, nullptr, nullptr).has_value());
    const auto with = dcc.forecast(5, &future).value();
    const auto without = dcc.forecast(5).value();
    EXPECT_TRUE(std::equal(with.sigma().value().begin(), with.sigma().value().end(), without.sigma().value().begin()));
}

TEST(DccSimulation, IsReproducibleAndShaped)
{
    const auto context = ve::Context::create(2).value();
    const auto fits = ve::testing::dcc3_univariate_fits(context);
    const auto dcc = fixed_dcc(context, fits);
    const ve::SimOptions options{.horizon = 6, .paths = 20, .seed = 3};
    const auto first = context.simulate(dcc, options).value();
    const auto second = context.simulate(dcc, options).value();
    EXPECT_EQ(first.paths().value(), 20U);
    EXPECT_EQ(first.horizon().value(), 6U);
    EXPECT_EQ(first.series().value(), 3U);
    ASSERT_EQ(first.returns().value().size(), 20U * 6U * 3U);
    EXPECT_EQ(first.sigma().value().size(), 20U * 6U * 3U);
    EXPECT_TRUE(std::equal(first.returns().value().begin(), first.returns().value().end(), second.returns().value().begin()));
}

TEST(DccSimulation, CorrelationsOnlyWhenAsked)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = ve::testing::dcc3_univariate_fits(context);
    const auto dcc = fixed_dcc(context, fits);
    const auto plain = context.simulate(dcc, ve::SimOptions{.horizon = 4, .paths = 5, .seed = 1}).value();
    const auto refused = plain.correlation();
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().status, ve::Status::invalid_argument);
    EXPECT_FALSE(plain.covariance().has_value());
    const auto kept = context.simulate(dcc, ve::SimOptions{.horizon = 4, .paths = 5, .seed = 1, .correlations = true}).value();
    EXPECT_EQ(kept.correlation().value().size(), 5U * 4U * 9U);
    EXPECT_EQ(kept.covariance().value().size(), 5U * 4U * 9U);
}

TEST(DccSimulation, RefusesRegressorsInTheOptions)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = ve::testing::dcc3_univariate_fits(context);
    const auto dcc = fixed_dcc(context, fits);
    const std::vector<double> values(4, 0.1);
    const auto panel = ve::Panel::create(values, 4, 1).value();
    const auto simulation =
        context.simulate(dcc, ve::SimOptions{.horizon = 4, .paths = 5, .seed = 1, .mxreg = &panel});
    ASSERT_FALSE(simulation.has_value());
    EXPECT_EQ(simulation.error().status, ve::Status::invalid_argument);
}

TEST(DccSimulation, CumulativeMatchesTheFullOutput)
{
    const auto context = ve::Context::create(2).value();
    const auto fits = ve::testing::dcc3_univariate_fits(context);
    const auto dcc = fixed_dcc(context, fits);
    const std::vector<std::size_t> checkpoints{2, 6};
    const auto full = context.simulate(dcc, ve::SimOptions{.horizon = 6, .paths = 20, .seed = 3}).value();
    const auto cumulative =
        context.simulate(dcc, ve::SimOptions{.horizon = 6, .paths = 20, .seed = 3, .checkpoints = checkpoints}).value();
    const auto returns = full.returns().value();
    const auto sums = cumulative.cumulative().value();
    ASSERT_EQ(sums.size(), 20U * 2U * 3U);
    for (std::size_t p = 0; p < 20; ++p)
    {
        for (std::size_t i = 0; i < 3; ++i)
        {
            double sum = 0.0;
            std::size_t h = 0;
            for (std::size_t j = 0; j < 2; ++j)
            {
                for (; h < checkpoints[j]; ++h)
                {
                    sum += returns[(p * 6 + h) * 3 + i];
                }
                EXPECT_EQ(std::bit_cast<std::uint64_t>(sums[(p * 2 + j) * 3 + i]), std::bit_cast<std::uint64_t>(sum));
            }
        }
    }
    EXPECT_TRUE(std::ranges::equal(cumulative.checkpoints().value(), checkpoints));
    const auto refused = cumulative.returns();
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().status, ve::Status::invalid_argument);
}

TEST(DccForecast, CumulativeCovarianceShape)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = ve::testing::dcc3_univariate_fits(context);
    const auto dcc = fixed_dcc(context, fits);
    const auto forecast = dcc.forecast(5).value();
    const auto covariance = forecast.covariance().value();
    const auto cumulative = forecast.cumulative_covariance().value();
    ASSERT_EQ(cumulative.size(), 5U * 3U * 3U);
    EXPECT_EQ(forecast.cumulative_mean().value().size(), 5U * 3U);
    for (std::size_t i = 0; i < 3; ++i)
    {
        EXPECT_EQ(cumulative[i * 3 + i], covariance[i * 3 + i]);
    }
}
} // namespace
