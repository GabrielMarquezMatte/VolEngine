#include "fixture_data.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <span>
#include <vector>
#include <volengine/volengine.hpp>

namespace
{

// rugarch's forecast of case t5000/sgarch11_norm (tests/reference/fixtures/rugarch_forecast.json): the model fixed at
// rugarch's point on the first 4975 observations, 25 steps ahead, sigma to 1e-8 relative as in
// python/tests/oracles/test_forecast_oracles.py.
const ve::NamedValues kRugarchParams{
    {"mu", 0.045689656391604716},
    {"omega", 0.010692733778622033},
    {"alpha1", 0.07516720858429206},
    {"beta1", 0.9030746410118694},
};
constexpr std::size_t kObservations = 4975;
constexpr std::size_t kHorizon = 25;

std::vector<double> series()
{
    static const auto columns = ve::testing::read_csv_columns("sgarch11_norm.csv");
    return {columns[0].begin(), columns[0].begin() + kObservations};
}

ve::Fit fixed_fit(const ve::Context& context)
{
    return context.fit(ve::Spec{.fixed = kRugarchParams}, series()).value();
}

TEST(Forecast, MatchesRugarch)
{
    const auto context = ve::Context::create(1).value();
    const auto fit = fixed_fit(context);
    ASSERT_EQ(fit.status(), ve::FitStatus::converged);
    const auto forecast = fit.forecast(kHorizon);
    ASSERT_TRUE(forecast.has_value());
    EXPECT_EQ(forecast->horizon().value(), kHorizon);
    const auto sigma = forecast->sigma().value();
    const auto mean = forecast->mean().value();
    ASSERT_EQ(sigma.size(), kHorizon);
    EXPECT_NEAR(sigma[0] / 0.6087651095077266, 1.0, 1e-8);
    EXPECT_NEAR(sigma[4] / 0.6170686780840836, 1.0, 1e-8);
    EXPECT_NEAR(sigma[24] / 0.648199856948522, 1.0, 1e-8);
    EXPECT_NEAR(mean[0], 0.045689656391604716, 1e-8);
    EXPECT_NEAR(mean[24], 0.045689656391604716, 1e-8);
}

TEST(Forecast, OutlivesTheFit)
{
    const auto context = ve::Context::create(1).value();
    auto fit = std::make_unique<ve::Fit>(fixed_fit(context));
    const auto forecast = fit->forecast(kHorizon).value();
    const std::vector<double> before(forecast.sigma().value().begin(), forecast.sigma().value().end());
    fit.reset();
    const auto after = forecast.sigma().value();
    ASSERT_EQ(after.size(), before.size());
    EXPECT_TRUE(std::equal(before.begin(), before.end(), after.begin()));
}

TEST(Forecast, ZeroHorizonIsError)
{
    const auto context = ve::Context::create(1).value();
    const auto forecast = fixed_fit(context).forecast(0);
    ASSERT_FALSE(forecast.has_value());
    EXPECT_EQ(forecast.error().status, ve::Status::invalid_argument);
}

TEST(Simulation, FitIsReproducibleAndPrefixStable)
{
    const auto context = ve::Context::create(2).value();
    const auto fit = fixed_fit(context);
    const auto fifty = context.simulate(fit, ve::SimOptions{.horizon = 20, .paths = 50, .seed = 7}).value();
    const auto again = context.simulate(fit, ve::SimOptions{.horizon = 20, .paths = 50, .seed = 7}).value();
    const auto ten = context.simulate(fit, ve::SimOptions{.horizon = 20, .paths = 10, .seed = 7}).value();
    ASSERT_EQ(fifty.returns().value().size(), 50U * 20U);
    EXPECT_TRUE(std::equal(fifty.returns().value().begin(), fifty.returns().value().end(), again.returns().value().begin()));
    EXPECT_TRUE(std::equal(ten.returns().value().begin(), ten.returns().value().end(), fifty.returns().value().begin()));
    EXPECT_EQ(fifty.paths().value(), 50U);
    EXPECT_EQ(fifty.horizon().value(), 20U);
    EXPECT_EQ(fifty.sigma().value().size(), 50U * 20U);
}

TEST(Simulation, OutlivesFitAndContext)
{
    auto context = std::make_unique<ve::Context>(ve::Context::create(1).value());
    auto fit = std::make_unique<ve::Fit>(fixed_fit(*context));
    const auto simulation = context->simulate(*fit, ve::SimOptions{.horizon = 5, .paths = 3, .seed = 1}).value();
    const std::vector<double> before(simulation.returns().value().begin(), simulation.returns().value().end());
    fit.reset();
    context.reset();
    const auto after = simulation.returns().value();
    EXPECT_TRUE(std::equal(before.begin(), before.end(), after.begin()));
}

TEST(Simulation, SpecNeedsAllParametersFixed)
{
    const auto context = ve::Context::create(1).value();
    const auto free_spec = context.simulate(ve::Spec{}, ve::SimOptions{.horizon = 20, .paths = 5, .seed = 1});
    ASSERT_FALSE(free_spec.has_value());
    EXPECT_EQ(free_spec.error().status, ve::Status::invalid_argument);
    EXPECT_FALSE(free_spec.error().message.empty());

    const auto fixed = context.simulate(
        ve::Spec{.fixed = kRugarchParams}, ve::SimOptions{.horizon = 20, .paths = 5, .seed = 1});
    ASSERT_TRUE(fixed.has_value());
    EXPECT_EQ(fixed->returns().value().size(), 100U);
}

TEST(Simulation, RegressorsMustMatchTheSpec)
{
    const auto context = ve::Context::create(1).value();
    const std::vector<double> values(20, 0.1);
    const auto panel = ve::Panel::create(values, 20, 1).value();
    const auto simulation = context.simulate(
        ve::Spec{.fixed = kRugarchParams}, ve::SimOptions{.horizon = 20, .paths = 5, .seed = 1, .mxreg = &panel});
    ASSERT_FALSE(simulation.has_value());
    EXPECT_EQ(simulation.error().status, ve::Status::invalid_argument);
}

// The left-to-right sum of the first checkpoints[j] steps of every path of a full simulation, as bits.
std::vector<std::uint64_t> left_sums(std::span<const double> returns, std::size_t paths, std::size_t horizon,
                                     std::span<const std::size_t> checkpoints)
{
    std::vector<std::uint64_t> sums;
    for (std::size_t p = 0; p < paths; ++p)
    {
        double sum = 0.0;
        std::size_t h = 0;
        for (const std::size_t k : checkpoints)
        {
            for (; h < k; ++h)
            {
                sum += returns[p * horizon + h];
            }
            sums.push_back(std::bit_cast<std::uint64_t>(sum));
        }
    }
    return sums;
}

TEST(Simulation, CumulativeMatchesTheFullOutput)
{
    const auto context = ve::Context::create(2).value();
    const auto fit = fixed_fit(context);
    const std::vector<std::size_t> checkpoints{1, 5, 20};
    const auto full = context.simulate(fit, ve::SimOptions{.horizon = 20, .paths = 30, .seed = 7}).value();
    const auto cumulative =
        context.simulate(fit, ve::SimOptions{.horizon = 20, .paths = 30, .seed = 7, .checkpoints = checkpoints}).value();
    const auto sums = cumulative.cumulative().value();
    std::vector<std::uint64_t> bits;
    for (const double sum : sums)
    {
        bits.push_back(std::bit_cast<std::uint64_t>(sum));
    }
    EXPECT_EQ(bits, left_sums(full.returns().value(), 30, 20, checkpoints));
    const auto listed = cumulative.checkpoints().value();
    EXPECT_TRUE(std::ranges::equal(listed, checkpoints));
    const auto refused = cumulative.returns();
    ASSERT_FALSE(refused.has_value());
    EXPECT_EQ(refused.error().status, ve::Status::invalid_argument);
    EXPECT_TRUE(full.checkpoints().value().empty());
}
} // namespace
