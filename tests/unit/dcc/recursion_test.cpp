#include "dcc/recursion.hpp"

#include "dcc/likelihood.hpp"
#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "dcc3_panel.hpp"
#include "linalg/cholesky.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::dcc::Convention;
using ve::dcc::DccDist;
using ve::dcc::DccModel;
using ve::dcc::DccSpec;
using ve::testing::kDcc3Rows;
using ve::testing::kDcc3Series;

struct EdgeCase
{
    std::string name;
    DccModel model;
    std::vector<int> groups;
    std::vector<double> theta;
};

// The number of observations whose R_t is not a correlation matrix: unit diagonal, exactly symmetric, positive
// definite.
std::size_t invalid_correlations(const ve::dcc::StandardizedPanel& panel, const DccSpec& spec,
                                 std::span<const double> theta, ve::dcc::Stage stage)
{
    const std::size_t m = panel.m;
    std::size_t invalid = 0;
    std::size_t visited = 0;
    const auto check = [&](std::size_t, std::span<const double> r) {
        ++visited;
        std::vector<double> factor(r.begin(), r.end());
        bool valid = ve::linalg::cholesky_factor(factor, m);
        for (std::size_t i = 0; i < m; ++i)
        {
            valid = valid && r[i * m + i] == 1.0;
            for (std::size_t j = 0; j < i; ++j)
            {
                valid = valid && r[i * m + j] == r[j * m + i] && std::fabs(r[i * m + j]) < 1.0;
            }
        }
        invalid += valid ? 0 : 1;
    };
    const auto parameters = ve::dcc::split_parameters(spec, theta);
    EXPECT_TRUE(ve::dcc::correlation_path<double>(panel, spec, parameters, stage, check));
    EXPECT_EQ(visited, kDcc3Rows);
    return invalid;
}

TEST(DccRecursion, CorrelationsAreValid)
{
    const ve::dcc::StandardizedPanel panel = ve::testing::dcc3_panel();
    // Persistence 0.999 and the mvt shape at the edge of (4, 50).
    const std::vector<EdgeCase> cases{
        {"DCC", DccModel::Dcc, {}, {0.05, 0.949, 49.999}},
        {"aDCC", DccModel::ADcc, {}, {0.04, 0.949, 0.002, 49.999}},
        {"FDCC", DccModel::FDcc, {1, 1, 2}, {0.22, 0.23, 0.973, 0.972, 49.999}},
    };
    for (const EdgeCase& c : cases)
    {
        for (const Convention convention : {Convention::Standard, Convention::Rmgarch})
        {
            DccSpec spec;
            spec.model = c.model;
            spec.dist = DccDist::MvT;
            spec.groups = c.groups;
            spec.convention = convention;
            for (const ve::dcc::Stage stage : {ve::dcc::Stage::Objective, ve::dcc::Stage::Reported})
            {
                EXPECT_EQ(invalid_correlations(panel, spec, c.theta, stage), 0U)
                    << c.name << ", convention " << int(convention) << ", stage " << int(stage);
            }
            EXPECT_TRUE(std::isfinite(ve::dcc::reported_loglik(panel, spec, c.theta))) << c.name;
        }
    }
}

}
