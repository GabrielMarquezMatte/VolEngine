#include "dcc/copula/scores.hpp"

#include "dcc/panel.hpp"
#include "dcc/spec.hpp"

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <cmath>
#include <numbers>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace
{

using ve::dcc::DccDist;
using ve::dcc::StandardizedPanel;
using ve::dcc::copula::CopulaPanel;
using ve::dcc::copula::copula_panel;

StandardizedPanel univariate_of(std::size_t t, std::size_t m)
{
    StandardizedPanel panel;
    panel.t = t;
    panel.m = m;
    panel.z.assign(t * m, 0.0);
    for (std::size_t k = 0; k < t * m; ++k)
    {
        panel.sigma.push_back(1.0 + 0.1 * static_cast<double>(k));
    }
    return panel;
}

// u of 3 rows x 2 series; column 0 is (0.025, 0.5, 0.975).
const std::vector<double> kU{0.025, 0.3, 0.5, 0.9, 0.975, 0.1};

TEST(CopulaScores, NormalQuantilesAndMarginal)
{
    const StandardizedPanel univariate = univariate_of(3, 2);
    const CopulaPanel scores = copula_panel(kU, univariate, DccDist::MvNorm, 0.0);
    const StandardizedPanel& panel = *scores.panel;
    EXPECT_NEAR(panel.z[0], -1.959963984540054, 1e-14);
    EXPECT_NEAR(panel.z[2], 0.0, 1e-15);
    EXPECT_NEAR(panel.z[4], 1.959963984540054, 1e-14);
    double marginal = 0.0;
    for (double e : panel.z)
    {
        marginal += -0.5 * std::log(2.0 * std::numbers::pi) - 0.5 * e * e;
    }
    EXPECT_NEAR(scores.marginal_log_density, marginal, 1e-13);
    EXPECT_EQ(panel.sigma, univariate.sigma);
    // qbar is cov(epsilon) with divisor T - 1.
    const std::vector<double> expected = ve::dcc::second_moment(panel.z, 3, 2, true, 2.0);
    for (std::size_t k = 0; k < 4; ++k)
    {
        EXPECT_NEAR(panel.qbar[k], expected[k], 1e-15);
    }
}

TEST(CopulaScores, StandardizedStudentQuantilesAndMarginal)
{
    const double nu = 5.0;
    const double scale = std::sqrt((nu - 2.0) / nu);
    const StandardizedPanel univariate = univariate_of(3, 2);
    const CopulaPanel scores = copula_panel(kU, univariate, DccDist::MvT, nu);
    const boost::math::students_t student(nu);
    double marginal = 0.0;
    for (std::size_t k = 0; k < kU.size(); ++k)
    {
        const double expected = boost::math::quantile(student, kU[k]) * scale;
        EXPECT_NEAR(scores.panel->z[k], expected, 1e-14) << k;
        marginal += std::log(boost::math::pdf(student, expected / scale) / scale);
    }
    EXPECT_NEAR(scores.marginal_log_density, marginal, 1e-13);
}

}
