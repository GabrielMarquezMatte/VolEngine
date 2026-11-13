#include "dcc/copula/margins.hpp"

#include "copula3_panel.hpp"
#include "core/spec.hpp"
#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "dist/cdf.hpp"
#include "fit/fit_result.hpp"
#include "model/layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <vector>

namespace
{

using ve::dcc::Convention;
using ve::dcc::DccCopula;
using ve::dcc::StandardizedPanel;
using ve::dcc::copula::kUniformFloor;
using ve::dcc::copula::uniforms;

StandardizedPanel panel_of(std::vector<double> z, std::size_t t, std::size_t m)
{
    StandardizedPanel panel;
    panel.t = t;
    panel.m = m;
    panel.z = std::move(z);
    panel.sigma.assign(t * m, 1.0);
    return panel;
}

TEST(CopulaMargins, ParametricIsTheFitCdf)
{
    const std::vector<ve::FitResult> fits = ve::testing::copula3_fits();
    const std::vector<const ve::FitResult*> pointers{&fits[0], &fits[1], &fits[2]};
    const StandardizedPanel panel = ve::dcc::standardize(pointers);
    const std::vector<double> u = uniforms(pointers, panel, DccCopula::Parametric, Convention::Standard);
    ASSERT_EQ(u.size(), panel.t * panel.m);
    for (std::size_t i = 0; i < panel.m; ++i)
    {
        const auto layout = ve::model::layout_of(fits[i].spec);
        const auto shape = ve::model::shape_parameters_of<double>(fits[i].estimate, layout);
        for (std::size_t t = 0; t < panel.t; t += 97)
        {
            const double expected =
                std::clamp(ve::dist::cdf(layout.dist, shape, panel.z[t * panel.m + i]), kUniformFloor, 1.0 - kUniformFloor);
            EXPECT_EQ(u[t * panel.m + i], expected) << "t " << t << " series " << i;
        }
    }
}

TEST(CopulaMargins, EmpiricalRanks)
{
    // Two series, five rows; series 0 has a tie at t = 1 and t = 3, broken by the order of t.
    const StandardizedPanel panel = panel_of({0.5, -1.0, 0.2, 2.0, -0.3, 0.1, 0.2, 0.4, 1.0, -2.0}, 5, 2);
    const std::vector<double> u = uniforms(std::span<const ve::dcc::copula::Margin>{}, panel, DccCopula::Empirical, Convention::Standard);
    // Series 0 is 0.5, 0.2, -0.3, 0.2, 1.0; series 1 is -1.0, 2.0, 0.1, 0.4, -2.0.
    const std::vector<double> ranks0{4, 2, 1, 3, 5};
    const std::vector<double> ranks1{2, 5, 3, 4, 1};
    for (std::size_t t = 0; t < 5; ++t)
    {
        EXPECT_EQ(u[t * 2], ranks0[t] / 6.0) << t;
        EXPECT_EQ(u[t * 2 + 1], ranks1[t] / 6.0) << t;
    }
}

TEST(CopulaMargins, RmgarchEmpiricalAndCut)
{
    // ecdf: the share of the series <= z (a tie takes the larger rank), then u > 0.99999 becomes 0.99999 (plan 4b
    // notes, section 2).
    const StandardizedPanel panel = panel_of({0.5, -1.0, 0.2, 2.0, -0.3, 0.1, 0.2, 0.4, 1.0, -2.0}, 5, 2);
    const std::vector<double> u = uniforms(std::span<const ve::dcc::copula::Margin>{}, panel, DccCopula::Empirical, Convention::Rmgarch);
    EXPECT_DOUBLE_EQ(u[0], 4.0 / 5.0);
    EXPECT_DOUBLE_EQ(u[2], 3.0 / 5.0);
    EXPECT_DOUBLE_EQ(u[6], 3.0 / 5.0);
    EXPECT_DOUBLE_EQ(u[8], 0.99999);
    EXPECT_DOUBLE_EQ(u[9], 1.0 / 5.0);
}

TEST(CopulaMargins, ExtremeShockStaysFinite)
{
    // Review Focus 1: a 40-sigma shock on a normal margin has cdf 1.0 exactly.
    ve::FitResult fit;
    fit.spec.dist = ve::Dist::Norm;
    fit.estimate = {0.0, 0.1, 0.1, 0.8};
    const StandardizedPanel panel = panel_of({40.0, -40.0, 0.0, 0.0}, 2, 2);
    const std::vector<const ve::FitResult*> pointers{&fit, &fit};
    const std::vector<double> u = uniforms(pointers, panel, DccCopula::Parametric, Convention::Standard);
    EXPECT_EQ(u[0], 1.0 - kUniformFloor);
    EXPECT_EQ(u[1], kUniformFloor);
    const std::vector<double> cut = uniforms(pointers, panel, DccCopula::Parametric, Convention::Rmgarch);
    EXPECT_EQ(cut[0], 0.99999);
    EXPECT_EQ(cut[1], 0x1p-52);
}

}
