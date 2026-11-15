#include "dcc/copula/static_fit.hpp"

#include "copula3_panel.hpp"
#include "dcc/copula/fit.hpp"
#include "dcc/copula/margins.hpp"
#include "dcc/copula/scores.hpp"
#include "dcc/fit.hpp"
#include "dcc/spec.hpp"
#include "fit/fit.hpp"
#include "fit/fit_result.hpp"
#include "linalg/cholesky.hpp"
#include "simulated_garch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <numbers>
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
using ve::dcc::DccCopula;
using ve::dcc::DccDist;
using ve::dcc::DccFitResult;
using ve::dcc::DccModel;
using ve::dcc::DccSpec;
using ve::dcc::StandardizedPanel;
using ve::dcc::StaticMethod;
using ve::dcc::copula::CopulaInputs;

DccSpec static_spec(DccDist dist, DccCopula copula, StaticMethod method, Convention convention = Convention::Standard)
{
    DccSpec spec;
    spec.model = DccModel::Static;
    spec.dist = dist;
    spec.copula = copula;
    spec.static_method = method;
    spec.convention = convention;
    return spec;
}

CopulaInputs inputs_of(const std::vector<FitResult>& fits, const std::vector<DccSpec>& specs)
{
    std::vector<const FitResult*> pointers;
    for (const FitResult& fit : fits)
    {
        pointers.push_back(&fit);
    }
    return ve::dcc::copula::copula_inputs(pointers, specs);
}

TEST(CopulaStatic, FreeParametersGiveCorrelations)
{
    // rmgarch's static_mvt_ml_param C and its R21, R31, R32 (plan 4b notes, section 7).
    const std::vector<double> c{0.594217920400736, 0.436936698871691, 0.277810026573495};
    const std::vector<double> r = ve::dcc::copula::correlation_from_free(c, 3);
    EXPECT_NEAR(r[3], 0.510836135301799, 1e-12);
    EXPECT_NEAR(r[6], 0.388010185709091, 1e-12);
    EXPECT_NEAR(r[7], 0.410293857644785, 1e-12);
    EXPECT_EQ(ve::dcc::copula::correlation_from_free(std::vector<double>(3, 0.0), 3),
              (std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1}));

    std::mt19937_64 engine(7);
    std::normal_distribution<double> normal(0.0, 1.0);
    const std::size_t m = 5;
    std::vector<double> x(m * (m - 1) / 2);
    for (double& value : x)
    {
        value = normal(engine);
    }
    std::vector<double> weights(m * m);
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            weights[i * m + j] = normal(engine);
            weights[j * m + i] = weights[i * m + j];
        }
    }
    const std::vector<double> rx = ve::dcc::copula::correlation_from_free(x, m);
    for (std::size_t i = 0; i < m; ++i)
    {
        EXPECT_NEAR(rx[i * m + i], 1.0, 1e-15);
    }
    std::vector<double> factor = rx;
    EXPECT_TRUE(ve::linalg::cholesky_factor(std::span<double>(factor), m));
    // f(x) = sum_ij W_ij R_ij(x), whose dl/dR is W.
    auto f = [&](const std::vector<double>& at)
    {
        const std::vector<double> rr = ve::dcc::copula::correlation_from_free(at, m);
        double sum = 0.0;
        for (std::size_t k = 0; k < rr.size(); ++k)
        {
            sum += weights[k] * rr[k];
        }
        return sum;
    };
    std::vector<double> gradient(x.size());
    ve::dcc::copula::free_gradient(x, weights, gradient, m);
    for (std::size_t k = 0; k < x.size(); ++k)
    {
        std::vector<double> up = x;
        std::vector<double> down = x;
        up[k] += 1e-6;
        down[k] -= 1e-6;
        const double numeric = (f(up) - f(down)) / 2e-6;
        EXPECT_NEAR(gradient[k], numeric, 1e-7 * std::max(1.0, std::abs(numeric))) << k;
    }
}

TEST(CopulaStatic, AnalyticGradient)
{
    const std::vector<FitResult> fits = ve::testing::copula3_fits();
    const DccSpec spec = static_spec(DccDist::MvT, DccCopula::Parametric, StaticMethod::Ml);
    const CopulaInputs inputs = inputs_of(fits, {spec});
    const std::vector<double> r0 = ve::dcc::copula::correlation_from_free(
        std::vector<double>{0.594217920400736, 0.436936698871691, 0.277810026573495}, 3);
    for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT})
    {
        const ve::dcc::copula::CopulaPanel scores =
            ve::dcc::copula::copula_panel(inputs.uniforms_of(spec), *inputs.univariate, dist, 6.0);
        std::vector<double> gradient(9);
        ve::dcc::copula::static_terms(*scores.panel, r0, dist, 6.0, gradient);
        for (const auto& [i, j] : std::vector<std::array<std::size_t, 2>>{{1, 0}, {2, 0}, {2, 1}})
        {
            std::vector<double> up = r0;
            std::vector<double> down = r0;
            up[i * 3 + j] += 1e-6;
            up[j * 3 + i] += 1e-6;
            down[i * 3 + j] -= 1e-6;
            down[j * 3 + i] -= 1e-6;
            const double numeric = (ve::dcc::copula::static_terms(*scores.panel, up, dist, 6.0, {}) -
                                    ve::dcc::copula::static_terms(*scores.panel, down, dist, 6.0, {})) /
                                   2e-6;
            EXPECT_NEAR(gradient[i * 3 + j] + gradient[j * 3 + i], numeric, 1e-6 * std::abs(numeric));
        }
    }
}

TEST(CopulaStatic, KendallFormula)
{
    // Three series, six rows, with a tie in series 1.
    const std::vector<std::array<double, 3>> rows{{0.1, 0.5, 0.9}, {0.4, 0.2, 0.3}, {0.2, 0.5, 0.1},
                                                  {0.9, 0.8, 0.4}, {0.6, 0.1, 0.7}, {0.3, 0.6, 0.2}};
    std::vector<double> u;
    for (const auto& row : rows)
    {
        u.insert(u.end(), row.begin(), row.end());
    }
    const std::optional<std::vector<double>> r = ve::dcc::copula::kendall_correlation(u, 6, 3);
    ASSERT_TRUE(r.has_value());
    auto tau_b = [&](std::size_t a, std::size_t b)
    {
        double concordant = 0.0;
        double discordant = 0.0;
        double ties_a = 0.0;
        double ties_b = 0.0;
        for (std::size_t s = 0; s < 6; ++s)
        {
            for (std::size_t t = s + 1; t < 6; ++t)
            {
                const double da = rows[s][a] - rows[t][a];
                const double db = rows[s][b] - rows[t][b];
                concordant += da * db > 0 ? 1.0 : 0.0;
                discordant += da * db < 0 ? 1.0 : 0.0;
                ties_a += da == 0 ? 1.0 : 0.0;
                ties_b += db == 0 ? 1.0 : 0.0;
            }
        }
        return (concordant - discordant) / std::sqrt((15.0 - ties_a) * (15.0 - ties_b));
    };
    for (const auto& [i, j] : std::vector<std::array<std::size_t, 2>>{{1, 0}, {2, 0}, {2, 1}})
    {
        EXPECT_NEAR((*r)[i * 3 + j], std::sin(std::numbers::pi * tau_b(i, j) / 2.0), 1e-15);
        EXPECT_EQ((*r)[i * 3 + j], (*r)[j * 3 + i]);
    }
}

// The pair-count tau-b the library used before Knight's algorithm, the reference of the bit-for-bit guard below.
double pair_count_tau_b(const std::vector<double>& u, std::size_t t, std::size_t m, std::size_t a, std::size_t b)
{
    double score = 0.0;
    double ties_a = 0.0;
    double ties_b = 0.0;
    for (std::size_t s = 0; s < t; ++s)
    {
        for (std::size_t r = s + 1; r < t; ++r)
        {
            const double da = u[s * m + a] - u[r * m + a];
            const double db = u[s * m + b] - u[r * m + b];
            score += static_cast<double>((da * db > 0.0) - (da * db < 0.0));
            ties_a += da == 0.0 ? 1.0 : 0.0;
            ties_b += db == 0.0 ? 1.0 : 0.0;
        }
    }
    const double pairs = 0.5 * static_cast<double>(t) * static_cast<double>(t - 1);
    return score / std::sqrt((pairs - ties_a) * (pairs - ties_b));
}

// kendall_correlation of u equals the sin(pi tau_b / 2) of the pair count to the bit, or both are not positive
// definite.
void expect_kendall_matches_pair_count(const std::vector<double>& u, std::size_t t, std::size_t m)
{
    std::vector<double> expected(m * m, 1.0);
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < i; ++j)
        {
            expected[i * m + j] = std::sin(0.5 * std::numbers::pi * pair_count_tau_b(u, t, m, i, j));
            expected[j * m + i] = expected[i * m + j];
        }
    }
    std::vector<double> factor = expected;
    const bool positive_definite = ve::linalg::cholesky_factor(std::span<double>(factor), m);
    const std::optional<std::vector<double>> actual = ve::dcc::copula::kendall_correlation(u, t, m);
    ASSERT_EQ(actual.has_value(), positive_definite);
    if (positive_definite)
    {
        EXPECT_EQ(*actual, expected);
    }
}

TEST(CopulaStatic, KendallMatchesPairCount)
{
    std::mt19937_64 engine(11);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    auto panel = [&](std::size_t t, std::size_t m, double grid_on_columns)
    {
        std::vector<double> u(t * m);
        for (std::size_t row = 0; row < t; ++row)
        {
            for (std::size_t column = 0; column < m; ++column)
            {
                double value = uniform(engine);
                // Columns below grid_on_columns are rounded to 1/16, so they tie.
                if (static_cast<double>(column) < grid_on_columns)
                {
                    value = std::round(value * 16.0) / 16.0;
                }
                u[row * m + column] = value;
            }
        }
        return u;
    };
    SCOPED_TRACE("random");
    expect_kendall_matches_pair_count(panel(257, 4, 0.0), 257, 4);
    SCOPED_TRACE("ties in one column");
    expect_kendall_matches_pair_count(panel(257, 4, 1.0), 257, 4);
    SCOPED_TRACE("ties in both columns of a pair");
    expect_kendall_matches_pair_count(panel(257, 4, 4.0), 257, 4);
    SCOPED_TRACE("two rows: every tau is +-1, so R is singular");
    expect_kendall_matches_pair_count(panel(2, 3, 0.0), 2, 3);
    SCOPED_TRACE("equal columns: tau 1, so R is singular");
    std::vector<double> equal = panel(100, 3, 0.0);
    for (std::size_t row = 0; row < 100; ++row)
    {
        equal[row * 3 + 1] = equal[row * 3];
    }
    expect_kendall_matches_pair_count(equal, 100, 3);
}

// rmgarch's static cases on copula3 (plan 4b notes, section 7).
struct Oracle
{
    const char* id;
    DccDist dist;
    DccCopula copula;
    StaticMethod method;
    std::vector<double> theta;
    double llh;
    std::array<double, 3> r;
};

const std::vector<Oracle>& oracles()
{
    static const std::vector<Oracle> cases{
        {"static_mvnorm_kendall_param", DccDist::MvNorm, DccCopula::Parametric, StaticMethod::Kendall, {},
         -8117.1869165044, {0.513710062071992, 0.398508626536416, 0.406936683559792}},
        {"static_mvnorm_ml_param", DccDist::MvNorm, DccCopula::Parametric, StaticMethod::Ml,
         {0.598672396279854, 0.449366756278866, 0.265832497450360}, -8117.1868564310,
         {0.513657953074682, 0.398341381350193, 0.406795476599054}},
        {"static_mvnorm_ml_emp", DccDist::MvNorm, DccCopula::Empirical, StaticMethod::Ml,
         {0.597226254277687, 0.450164982005494, 0.267833429564139}, -8115.7480726458,
         {0.512743670621971, 0.398769267011434, 0.408159448768339}},
        {"static_mvt_kendall_param", DccDist::MvT, DccCopula::Parametric, StaticMethod::Kendall,
         {4.7793578808066}, -8012.1841333659, {0.510975878847646, 0.389184431763653, 0.402305107376601}},
        {"static_mvt_kendall_emp", DccDist::MvT, DccCopula::Empirical, StaticMethod::Kendall,
         {4.95613482263086}, -8015.1511339642, {0.510975878847646, 0.389184431763653, 0.402305107376601}},
        {"static_mvt_ml_param", DccDist::MvT, DccCopula::Parametric, StaticMethod::Ml,
         {0.594217920400736, 0.436936698871691, 0.277810026573495, 4.790827028666584}, -8012.0736645395,
         {0.510836135301799, 0.388010185709091, 0.410293857644785}},
    };
    return cases;
}

TEST(CopulaStatic, MatchesRmgarch)
{
    const std::vector<FitResult> fits = ve::testing::copula3_fits();
    for (const Oracle& oracle : oracles())
    {
        SCOPED_TRACE(oracle.id);
        const DccSpec spec = static_spec(oracle.dist, oracle.copula, oracle.method, Convention::Rmgarch);
        const CopulaInputs inputs = inputs_of(fits, {spec});
        std::vector<double> r(ve::testing::kCopula3Rows * 9);
        const double llh = ve::dcc::copula::copula_loglik(inputs, spec, oracle.theta, r);
        EXPECT_NEAR(llh, oracle.llh, 1e-10 * std::abs(oracle.llh));
        EXPECT_NEAR(r[3], oracle.r[0], 1e-10);
        EXPECT_NEAR(r[6], oracle.r[1], 1e-10);
        EXPECT_NEAR(r[7], oracle.r[2], 1e-10);

        const DccFitResult fit = ve::dcc::copula::fit_copula(inputs, spec, FitConfig{});
        ASSERT_EQ(fit.estimate.size(), oracle.theta.size());
        EXPECT_EQ(fit.status, FitStatus::Converged);
        for (std::size_t k = 0; k < oracle.theta.size(); ++k)
        {
            EXPECT_NEAR(fit.estimate[k], oracle.theta[k], 1e-3 * std::abs(oracle.theta[k])) << fit.names[k];
        }
        EXPECT_NEAR(fit.loglik, oracle.llh, 1e-7 * std::abs(oracle.llh));
    }
}

TEST(CopulaStatic, NonPositiveDefiniteKendallFails)
{
    // Review Focus 3: four series whose Kendall sine matrix has a negative eigenvalue (-0.44).
    const std::vector<std::array<double, 4>> rows{{5, 2, 4, 2}, {0, 5, 0, 0}, {1, 4, 5, 5},
                                                  {3, 0, 2, 4}, {4, 3, 3, 1}, {2, 1, 1, 3}};
    auto panel = std::make_shared<StandardizedPanel>();
    panel->t = 6;
    panel->m = 4;
    for (const auto& row : rows)
    {
        panel->z.insert(panel->z.end(), row.begin(), row.end());
    }
    panel->sigma.assign(24, 1.0);
    ve::dcc::refresh_moments(*panel);
    for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT})
    {
        const DccSpec spec = static_spec(dist, DccCopula::Empirical, StaticMethod::Kendall);
        CopulaInputs inputs;
        inputs.univariate = panel;
        inputs.u[2] = ve::dcc::copula::uniforms(std::span<const ve::dcc::copula::Margin>{}, *panel, DccCopula::Empirical, Convention::Standard);
        inputs.kendall[2] = ve::dcc::copula::kendall_correlation(inputs.u[2], 6, 4);
        ASSERT_FALSE(inputs.kendall[2].has_value());
        const DccFitResult fit = ve::dcc::copula::fit_copula(inputs, spec, FitConfig{});
        EXPECT_EQ(fit.status, FitStatus::Failed);
        EXPECT_TRUE(std::isnan(fit.loglik));
        for (double value : fit.estimate)
        {
            EXPECT_TRUE(std::isnan(value));
        }
        EXPECT_EQ(fit.r, nullptr);
    }
}

// m independent sGARCH(1,1)-std series, the shape of the benchmark panel.
std::vector<FitResult> independent_fits(std::size_t m, std::size_t rows)
{
    ve::testing::SimulatedModel model;
    model.dist = ve::Dist::Std;
    model.shape = 8.0;
    ve::Spec spec;
    spec.dist = ve::Dist::Std;
    std::vector<FitResult> fits;
    for (std::size_t i = 0; i < m; ++i)
    {
        fits.push_back(ve::fit_one(ve::testing::simulate_arma_garch(model, i + 1, rows), spec, FitConfig{}));
    }
    return fits;
}

TEST(CopulaStatic, StudentFitsAtManySeries)
{
    // The static Student copula (Kendall and ML) used to fail at m = 21 with no finite point.
    const std::vector<FitResult> fits = independent_fits(21, 1000);
    const DccSpec kendall = static_spec(DccDist::MvT, DccCopula::Parametric, StaticMethod::Kendall);
    const DccSpec ml = static_spec(DccDist::MvT, DccCopula::Parametric, StaticMethod::Ml);
    const CopulaInputs inputs = inputs_of(fits, {kendall, ml});
    double kendall_loglik = 0.0;
    for (const DccSpec& spec : {kendall, ml})
    {
        SCOPED_TRACE(spec.static_method == StaticMethod::Ml ? "ml" : "kendall");
        const DccFitResult fit = ve::dcc::copula::fit_copula(inputs, spec, FitConfig{});
        EXPECT_EQ(fit.status, FitStatus::Converged);
        ASSERT_TRUE(std::isfinite(fit.loglik));
        const double nu = fit.estimate.back();
        // Independent series: the profile rises to the ceiling, so nu is 50 itself.
        EXPECT_GE(nu, 4.0);
        EXPECT_LE(nu, 50.0);
        if (spec.static_method == StaticMethod::Kendall)
        {
            kendall_loglik = fit.loglik;
            continue;
        }
        EXPECT_GE(fit.loglik, kendall_loglik - 1e-6 * std::abs(kendall_loglik));
    }
}

TEST(CopulaStatic, RepeatedR)
{
    const std::vector<FitResult> fits = ve::testing::copula3_fits();
    const DccSpec spec = static_spec(DccDist::MvNorm, DccCopula::Parametric, StaticMethod::Ml);
    const DccFitResult fit = ve::dcc::copula::fit_copula(inputs_of(fits, {spec}), spec, FitConfig{});
    ASSERT_EQ(fit.status, FitStatus::Converged);
    ASSERT_NE(fit.r, nullptr);
    const std::vector<double>& r = *fit.r;
    ASSERT_EQ(r.size(), ve::testing::kCopula3Rows * 9);
    for (std::size_t t = 1; t < ve::testing::kCopula3Rows; ++t)
    {
        for (std::size_t k = 0; k < 9; ++k)
        {
            ASSERT_EQ(r[t * 9 + k], r[k]);
        }
    }
    const std::vector<double> h = ve::dcc::covariance_path(fit);
    const std::vector<double>& sigma = fit.panel->sigma;
    EXPECT_NEAR(h[5 * 9 + 1], sigma[5 * 3] * r[1] * sigma[5 * 3 + 1], 1e-15);
}

}
