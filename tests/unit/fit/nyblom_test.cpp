#include "fit/fit.hpp"
#include "fit/nyblom.hpp"
#include "model/layout.hpp"
#include "simulated_garch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

namespace
{

std::vector<double> reference_scores(std::size_t columns)
{
    std::array<double, 40> e{};
    std::array<double, 40> squared{};
    for (std::size_t i = 0; i < e.size(); ++i)
    {
        const double t = static_cast<double>(i + 1);
        const double z = std::sin(0.7 * t) + 0.3 * std::cos(1.9 * t);
        e[i] = z * (1.0 + 0.2 * std::sin(0.3 * t));
        squared[i] = z * z;
    }
    const double e_mean = std::accumulate(e.begin(), e.end(), 0.0) / 40.0;
    const double squared_mean = std::accumulate(squared.begin(), squared.end(), 0.0) / 40.0;
    std::vector<double> scores(40 * columns, 17.0);
    for (std::size_t i = 0; i < e.size(); ++i)
    {
        scores[i * columns] = e[i] - e_mean;
        scores[i * columns + 1] = squared[i] - squared_mean;
    }
    return scores;
}

TEST(Nyblom, MatchesFormula)
{
    const std::array<bool, 2> estimated{true, true};
    const ve::fit::Nyblom result = ve::fit::nyblom_statistics(reference_scores(2), 2, estimated);
    ASSERT_EQ(result.individual.size(), 2u);
    EXPECT_NEAR(result.individual[0], 0.078595967479814427, 1e-13);
    EXPECT_NEAR(result.individual[1], 0.01440865819792329, 1e-14);
    EXPECT_NEAR(result.joint, 0.093175052912939635, 1e-13);
    EXPECT_EQ(result.estimated, 2u);
    const std::array<bool, 3> mask{true, true, false};
    const ve::fit::Nyblom excluded = ve::fit::nyblom_statistics(reference_scores(3), 3, mask);
    EXPECT_EQ(excluded.individual[0], result.individual[0]);
    EXPECT_EQ(excluded.individual[1], result.individual[1]);
    EXPECT_EQ(excluded.joint, result.joint);
    EXPECT_TRUE(std::isnan(excluded.individual[2]));
    EXPECT_EQ(excluded.estimated, 2u);
}

TEST(Nyblom, SingularJointIsNaN)
{
    const std::array<bool, 2> estimated{true, true};
    const std::array<double, 8> scores{1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0};
    const ve::fit::Nyblom result = ve::fit::nyblom_statistics(scores, 2, estimated);
    EXPECT_TRUE(std::isnan(result.joint));
    EXPECT_TRUE(std::isfinite(result.individual[0]));
    EXPECT_EQ(result.individual[0], result.individual[1]);
    const std::array<double, 8> zero_column{1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0};
    const ve::fit::Nyblom zero = ve::fit::nyblom_statistics(zero_column, 2, estimated);
    EXPECT_TRUE(std::isfinite(zero.individual[0]));
    EXPECT_TRUE(std::isnan(zero.individual[1]));
    EXPECT_TRUE(std::isnan(zero.joint));
}

TEST(Nyblom, CriticalValues)
{
    const std::array<double, 3> one{0.353, 0.470, 0.748};
    const std::array<double, 3> twenty{4.22, 4.52, 5.13};
    EXPECT_EQ(ve::fit::nyblom_critical(1), one);
    EXPECT_EQ(ve::fit::nyblom_critical(20), twenty);
    EXPECT_FALSE(ve::fit::nyblom_critical(0));
    EXPECT_FALSE(ve::fit::nyblom_critical(21));
}

TEST(Nyblom, DuplicateNonIntegerScoresAreSingular)
{
    std::vector<double> scores = reference_scores(2);
    for (std::size_t offset = 0; offset < scores.size(); offset += 2)
    {
        scores[offset + 1] = scores[offset];
    }
    const std::array<bool, 2> estimated{true, true};
    const ve::fit::Nyblom result = ve::fit::nyblom_statistics(scores, 2, estimated);
    EXPECT_TRUE(std::isnan(result.joint));
    EXPECT_TRUE(std::isfinite(result.individual[0]));
    EXPECT_EQ(result.individual[0], result.individual[1]);
}

void expect_estimated_statistics(const ve::FitResult& fit)
{
    ASSERT_TRUE(std::isfinite(fit.loglik));
    const ve::model::Layout layout = ve::model::layout_of(fit.spec);
    ASSERT_EQ(fit.nyblom.individual.size(), layout.count());
    EXPECT_EQ(fit.nyblom.estimated, layout.estimated_count());
    for (std::size_t i = 0; i < layout.count(); ++i)
    {
        EXPECT_EQ(std::isfinite(fit.nyblom.individual[i]), layout.is_estimated(i)) << fit.names[i];
    }
}

TEST(Nyblom, StoredByTheFit)
{
    const std::vector<double> series = ve::testing::simulate_garch_series(7, 1000);
    ve::Spec spec;
    spec.ar = 1;
    spec.ma = 1;
    const ve::FitResult fit = ve::fit_one(series, spec, ve::FitConfig{});
    expect_estimated_statistics(fit);
    EXPECT_TRUE(std::isfinite(fit.nyblom.joint));
    spec.variance_targeting = true;
    const ve::FitResult targeted = ve::fit_one(series, spec, ve::FitConfig{});
    expect_estimated_statistics(targeted);
    EXPECT_TRUE(std::isnan(targeted.nyblom.individual[ve::model::layout_of(spec).omega()]));
    spec = ve::Spec{};
    spec.model = ve::Model::IGarch;
    const ve::FitResult integrated = ve::fit_one(series, spec, ve::FitConfig{});
    expect_estimated_statistics(integrated);
    EXPECT_TRUE(std::isnan(integrated.nyblom.individual[ve::model::layout_of(spec).derived()]));
}

TEST(Nyblom, FixedParametersAndNoFinitePoint)
{
    const std::vector<double> series = ve::testing::simulate_garch_series(7, 1000);
    ve::Spec spec;
    spec.fixed = {{"mu", 0.05}};
    expect_estimated_statistics(ve::fit_one(series, spec, ve::FitConfig{}));
    spec.fixed = {{"mu", 0.05}, {"omega", 0.01}, {"alpha1", 0.08}, {"beta1", 0.9}};
    const ve::FitResult fixed = ve::fit_one(series, spec, ve::FitConfig{});
    expect_estimated_statistics(fixed);
    EXPECT_EQ(fixed.nyblom.estimated, 0u);
    EXPECT_TRUE(std::isnan(fixed.nyblom.joint));
    const ve::FitResult empty = ve::fit_one(std::vector<double>{}, ve::Spec{}, ve::FitConfig{});
    EXPECT_TRUE(empty.nyblom.individual.empty());
    EXPECT_EQ(empty.nyblom.estimated, 0u);
    EXPECT_TRUE(std::isnan(empty.nyblom.joint));
}

}
