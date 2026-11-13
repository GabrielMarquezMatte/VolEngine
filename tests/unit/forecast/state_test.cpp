#include "fit/fit.hpp"
#include "simulated_garch.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <vector>

namespace
{

std::vector<double> last(std::span<const double> values, std::size_t count)
{
    std::span<const double> tail = values.last(count);
    return {tail.begin(), tail.end()};
}

TEST(ForecastState, TailsOfTheFit)
{
    std::vector<double> r = ve::testing::simulate_garch_series(3, 1000);
    ve::FitResult fit = ve::fit_one(r, ve::Spec{.p = 2, .q = 1, .ar = 1, .ma = 1}, ve::FitConfig{});
    ASSERT_FALSE(fit.sigma.empty());
    EXPECT_EQ(fit.state.residuals, last(fit.residuals, 2));
    ASSERT_EQ(fit.state.sigma2.size(), 2u);
    for (std::size_t i = 0; i < 2; ++i)
    {
        double sigma = fit.sigma[r.size() - 2 + i];
        EXPECT_NEAR(fit.state.sigma2[i], sigma * sigma, 1e-15 * sigma * sigma) << "i = " << i;
    }
    ASSERT_EQ(fit.state.deviations.size(), 2u);
    EXPECT_EQ(fit.state.deviations[1], r.back() - fit.estimate[0]);
    EXPECT_TRUE(fit.state.permanent.empty());
}

TEST(ForecastState, CsGarchKeepsTheTrend)
{
    std::vector<double> r = ve::testing::simulate_garch_series(3, 1000);
    ve::FitResult fit = ve::fit_one(r, ve::Spec{.model = ve::Model::CsGarch}, ve::FitConfig{});
    ASSERT_FALSE(fit.sigma.empty());
    ASSERT_EQ(fit.state.permanent.size(), 1u);
    EXPECT_GT(fit.state.permanent[0], 0.0);
    EXPECT_EQ(fit.state.residuals.size(), 1u);
}

TEST(ForecastState, ArfimaKeepsEveryDeviation)
{
    std::vector<double> r = ve::testing::simulate_garch_series(3, 1000);
    ve::FitResult fit = ve::fit_one(r, ve::Spec{.arfima = true}, ve::FitConfig{});
    ASSERT_FALSE(fit.sigma.empty());
    ASSERT_EQ(fit.state.deviations.size(), r.size());
    EXPECT_EQ(fit.state.deviations[0], r[0] - fit.estimate[0]);
    EXPECT_EQ(fit.state.residuals.size(), 1u);
}

TEST(ForecastState, EmptyWithoutFinitePoint)
{
    std::vector<double> r(200, 0.01);
    ve::FitResult fit = ve::fit_one(r, ve::Spec{}, ve::FitConfig{});
    EXPECT_EQ(fit.status, ve::FitStatus::Failed);
    EXPECT_TRUE(fit.sigma.empty());
    EXPECT_TRUE(fit.state.residuals.empty());
    EXPECT_TRUE(fit.state.sigma2.empty());
    EXPECT_TRUE(fit.state.deviations.empty());
    EXPECT_TRUE(fit.state.permanent.empty());
}

}
