#include "../fixture_data.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <vector>
#include <volengine/volengine.hpp>

namespace
{

std::vector<double> sample_returns()
{
    std::vector<double> data(200 * 2);
    for (std::size_t k = 0; k < data.size(); ++k)
    {
        data[k] = 0.02 * std::sin(0.7 * static_cast<double>(k));
    }
    return data;
}

const std::vector<double> kWeights{0.6, 0.4};

TEST(Risk, EmpiricalMatchesCapi)
{
    const auto data = sample_returns();
    const auto panel = ve::Panel::create(data, 200, 2).value();
    double var = 0.0;
    double es = 0.0;
    ASSERT_EQ(ve_risk_empirical(panel.native(), kWeights.data(), VE_RETURNS_SIMPLE, 0.95, &var, &es), VE_OK);
    const auto measure = ve::risk::empirical(panel, kWeights, ve::risk::ReturnKind::simple, 0.95).value();
    EXPECT_EQ(measure.var, var);
    EXPECT_EQ(measure.es, es);
}

TEST(Risk, ParametricMatchesCapi)
{
    double var = 0.0;
    double es = 0.0;
    constexpr double nan = std::numeric_limits<double>::quiet_NaN();
    ASSERT_EQ(ve_risk_parametric(VE_DIST_SSTD, 0.9, 7.0, nan, 0.0005, 0.015, 0.99, &var, &es), VE_OK);
    const ve::risk::Distribution distribution{.dist = ve::Dist::sstd, .skew = 0.9, .shape = 7.0};
    const auto measure = ve::risk::parametric(distribution, 0.0005, 0.015, 0.99).value();
    EXPECT_EQ(measure.var, var);
    EXPECT_EQ(measure.es, es);
}

TEST(Risk, WrongWeightCountIsAnError)
{
    const auto data = sample_returns();
    const auto panel = ve::Panel::create(data, 200, 2).value();
    const std::vector<double> weights{1.0};
    const auto measure = ve::risk::empirical(panel, weights, ve::risk::ReturnKind::log, 0.99);
    ASSERT_FALSE(measure.has_value());
    EXPECT_EQ(measure.error().status, ve::Status::invalid_argument);
}

TEST(Risk, MissingShapeIsAnError)
{
    const auto measure = ve::risk::parametric(ve::risk::Distribution{.dist = ve::Dist::student}, 0.0, 0.01, 0.99);
    ASSERT_FALSE(measure.has_value());
    EXPECT_EQ(measure.error().status, ve::Status::invalid_argument);
}

// The cumulative log returns at checkpoint k of a DCC simulation are paths x K x m; the panel of checkpoint k reads
// them in place with a row stride of K m from offset k m.
TEST(Risk, CumulativeDccPathsByStrides)
{
    const auto context = ve::Context::create(1).value();
    const auto fits = ve::testing::dcc3_univariate_fits(context);
    const auto dcc = context.fit_dcc(ve::DccSpec{}, fits, ve::DccOptions{.standard_errors = false}).value();
    const auto paths =
        context.simulate(dcc, ve::SimOptions{.horizon = 10, .paths = 400, .seed = 2, .checkpoints = {1, 5, 10}}).value();
    const auto sums = paths.cumulative().value();
    constexpr std::size_t kPaths = 400;
    constexpr std::size_t kCheckpoints = 3;
    constexpr std::size_t kSeries = 3;
    constexpr std::size_t kLast = 2;
    const auto panel = ve::Panel::create_strided(sums.subspan(kLast * kSeries), kPaths, kSeries,
                                                 static_cast<std::ptrdiff_t>(kCheckpoints * kSeries), 1)
                           .value();
    std::vector<double> copy;
    for (std::size_t path = 0; path < kPaths; ++path)
    {
        for (std::size_t j = 0; j < kSeries; ++j)
        {
            copy.push_back(sums[(path * kCheckpoints + kLast) * kSeries + j]);
        }
    }
    const auto contiguous = ve::Panel::create(copy, kPaths, kSeries).value();
    const std::vector<double> weights{0.5, 0.3, 0.2};
    const auto strided = ve::risk::empirical(panel, weights, ve::risk::ReturnKind::log, 0.99).value();
    const auto expected = ve::risk::empirical(contiguous, weights, ve::risk::ReturnKind::log, 0.99).value();
    EXPECT_EQ(strided.var, expected.var);
    EXPECT_EQ(strided.es, expected.es);
}

}
