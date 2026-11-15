#include "fixture_data.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>
#include <volengine/volengine.hpp>

namespace
{

// rugarch's residual diagnostics of t5000/sgarch11_norm (tests/reference/fixtures/rugarch_diagnostics.json), evaluated
// at rugarch's parameters (every parameter fixed), with the lags python/tests/oracles/test_diagnostics_oracles.py uses:
// statistics to 1e-7 relative, p-values to 1e-7 absolute (the GOF statistic to 1e-12).
const ve::NamedValues kRugarchParams{
    {"mu", 0.045689656391604716},
    {"omega", 0.010692733778622033},
    {"alpha1", 0.07516720858429206},
    {"beta1", 0.9030746410118694},
};

const std::vector<double>& series()
{
    static const auto columns = ve::testing::read_csv_columns("sgarch11_norm.csv");
    return columns[0];
}

ve::Fit fixed_fit()
{
    const auto context = ve::Context::create(1).value();
    return context.fit(ve::Spec{.fixed = kRugarchParams}, series()).value();
}

void expect_test(const ve::Test& actual, double statistic, double p_value, double relative)
{
    EXPECT_NEAR(actual.statistic, statistic, relative * std::abs(statistic));
    EXPECT_NEAR(actual.p_value, p_value, 1e-7);
}

TEST(Diagnostics, LjungBoxMatchesRugarch)
{
    const auto fit = fixed_fit();
    ASSERT_EQ(fit.status(), ve::FitStatus::converged);
    expect_test(fit.ljung_box(1).value(), 1.8415324291061732, 0.1747714433569032, 1e-7);
    expect_test(fit.ljung_box(2).value(), 2.02833194544891, 0.2581070863157935, 1e-7);
    expect_test(fit.ljung_box(5).value(), 3.0240968522740386, 0.40266535950697957, 1e-7);
    expect_test(fit.ljung_box(1, true).value(), 1.45845725259777, 0.22717574757243253, 1e-7);
    expect_test(fit.ljung_box(5, true).value(), 3.3096403717676224, 0.35340590830657215, 1e-7);
    expect_test(fit.ljung_box(9, true).value(), 6.80556187600853, 0.21599828334866544, 1e-7);
}

TEST(Diagnostics, ArchLmMatchesRugarch)
{
    const auto fit = fixed_fit();
    expect_test(fit.arch_lm(3, 2).value(), 0.7751659959198688, 0.37862363943847177, 1e-7);
    expect_test(fit.arch_lm(5, 2).value(), 3.758862220060667, 0.19702656530983376, 1e-7);
    expect_test(fit.arch_lm(7, 2).value(), 5.618181002026672, 0.16907626444593482, 1e-7);
}

TEST(Diagnostics, SignBiasMatchesRugarch)
{
    const auto bias = fixed_fit().sign_bias().value();
    expect_test(bias.sign, 0.7050465642787572, 0.48081413280546714, 1e-7);
    expect_test(bias.negative, 0.1592833217771207, 0.8734521001414853, 1e-7);
    expect_test(bias.positive, 0.32918412421066406, 0.7420303232862089, 1e-7);
    expect_test(bias.joint, 1.5736845674748814, 0.6653712766012059, 1e-7);
}

TEST(Diagnostics, GofMatchesRugarch)
{
    const auto fit = fixed_fit();
    const auto twenty = fit.gof(20).value();
    EXPECT_NEAR(twenty.statistic, 5.76, 1e-12);
    EXPECT_NEAR(twenty.p_value, 0.9984382200696095, 1e-7);
    EXPECT_NEAR(fit.gof(50).value().statistic, 25.58, 1e-12);
    EXPECT_NEAR(fit.gof().value().statistic, 5.76, 1e-12);
}

TEST(Diagnostics, ArgumentsOutOfRangeAreErrors)
{
    const auto fit = fixed_fit();
    const std::size_t nobs = fit.nobs().value();
    const auto lag = fit.ljung_box(nobs);
    ASSERT_FALSE(lag.has_value());
    EXPECT_EQ(lag.error().status, ve::Status::invalid_argument);
    EXPECT_FALSE(fit.arch_lm(2, 2).has_value());
    EXPECT_FALSE(fit.gof(1).has_value());
}

TEST(Diagnostics, NyblomMatchesRugarch)
{
    const auto context = ve::Context::create(1).value();
    const auto fit = context.fit(ve::Spec{.start = kRugarchParams}, series(), ve::FitOptions{.tol = 1.0}).value();
    const auto nyblom = fit.nyblom().value();
    const std::array<double, 4> expected{0.08539854412892842, 0.06676932423364436, 0.07602988996291062,
        0.06831382521675768};
    ASSERT_EQ(nyblom.individual.size(), 4U);
    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(nyblom.individual[i] / expected[i], 1.0, 1e-3) << i;
        EXPECT_EQ(nyblom.estimated[i], 1);
    }
    EXPECT_NEAR(nyblom.joint / 0.43290034680310535, 1.0, 1e-3);
}

TEST(Diagnostics, NyblomCriticalValuesFollowTheCTable)
{
    const auto one = ve::nyblom_critical(1).value();
    EXPECT_EQ(one, (std::array<double, 3>{0.353, 0.470, 0.748}));
    const auto twenty = ve::nyblom_critical(20).value();
    EXPECT_EQ(twenty, (std::array<double, 3>{4.22, 4.52, 5.13}));
    EXPECT_FALSE(ve::nyblom_critical(0).has_value());
    EXPECT_FALSE(ve::nyblom_critical(21).has_value());
}

} // namespace
