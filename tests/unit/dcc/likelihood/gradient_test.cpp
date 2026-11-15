#include "dcc/likelihood/gradient.hpp"

#include "ad/dual.hpp"
#include "dcc/copula/copula3_panel.hpp"
#include "dcc/copula/fit.hpp"
#include "dcc/copula/scores.hpp"
#include "dcc/likelihood.hpp"
#include "dcc/panel.hpp"
#include "dcc/spec.hpp"
#include "dcc/dcc3_panel.hpp"
#include "synthetic_panel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::dcc::Convention;
using ve::dcc::DccCopula;
using ve::dcc::DccDist;
using ve::dcc::DccModel;
using ve::dcc::DccSpec;
using ve::dcc::StandardizedPanel;

constexpr std::size_t kLanes = 8;
// The gradient of the tangents against the Dual of the same objective, relative to the largest component. The two
// differ by the rounding of m^3 / 6 products per observation and of the sums over t.
constexpr double kGradientTolerance = 1e-9;

DccSpec spec_of(DccModel model, DccDist dist, Convention convention, std::vector<int> groups)
{
    DccSpec spec;
    spec.model = model;
    spec.dist = dist;
    spec.convention = convention;
    spec.groups = std::move(groups);
    return spec;
}

// A point of the natural parameters: a (b) the persistence in mind, g only on the aDCC, the shape only on the mvt.
std::vector<double> theta_of(const DccSpec& spec, std::size_t group_count, double persistence, double shape)
{
    std::vector<double> theta;
    if (spec.model == DccModel::FDcc)
    {
        for (std::size_t g = 0; g < group_count; ++g)
        {
            theta.push_back(0.06 - 0.01 * static_cast<double>(g));
        }
        for (std::size_t g = 0; g < group_count; ++g)
        {
            theta.push_back(persistence - 0.005 * static_cast<double>(g));
        }
    }
    if (spec.model != DccModel::FDcc)
    {
        theta = {0.04, persistence - 0.04};
    }
    if (spec.model == DccModel::ADcc)
    {
        theta[1] -= 0.02;
        theta.push_back(0.02);
    }
    if (spec.dist == DccDist::MvT)
    {
        theta.push_back(shape);
    }
    return theta;
}

std::vector<double> dual_gradient(const StandardizedPanel& panel, const DccSpec& spec, std::span<const double> theta)
{
    using Number = ve::ad::Dual<kLanes>;
    std::vector<Number> variables;
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        variables.push_back(ve::ad::variable<kLanes>(theta[i], i));
    }
    const Number value = ve::dcc::objective<Number>(panel, spec, variables);
    return {value.grad.begin(), value.grad.begin() + static_cast<std::ptrdiff_t>(theta.size())};
}

struct Shape
{
    std::string name;
    StandardizedPanel panel;
    std::size_t group_count;
};

std::vector<Shape> shapes()
{
    return {{"dcc3 m=3 T=1500", ve::testing::dcc3_panel(), 2},
            {"m=2", ve::testing::synthetic_panel(2, 90), 2},
            {"m=21, 3 groups of 7", ve::testing::synthetic_panel(21, 45), 3},
            {"m=21, groups of 10 and 11", ve::testing::synthetic_panel(21, 45), 2}};
}

TEST(DccGradient, MatchesDual)
{
    for (const Shape& shape : shapes())
    {
        for (const Convention convention : {Convention::Standard, Convention::Rmgarch})
        {
            for (const DccModel model : {DccModel::Dcc, DccModel::ADcc, DccModel::FDcc})
            {
                for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT, DccDist::MvLaplace})
                {
                    const std::vector<int> groups =
                        model == DccModel::FDcc ? ve::testing::groups_of(shape.panel.m, shape.group_count)
                                                : std::vector<int>{};
                    const DccSpec spec = spec_of(model, dist, convention, groups);
                    for (const double persistence : {0.93, 0.995})
                    {
                        for (const double nu : {4.5, 30.0})
                        {
                            if (dist != DccDist::MvT && nu > 4.5)
                            {
                                continue;
                            }
                            const std::vector<double> theta = theta_of(spec, shape.group_count, persistence, nu);
                            SCOPED_TRACE(shape.name + ", model " + std::to_string(static_cast<int>(model)) + ", dist " +
                                         std::to_string(static_cast<int>(dist)) + ", convention " +
                                         std::to_string(static_cast<int>(convention)) + ", persistence " +
                                         std::to_string(persistence) + ", shape " + std::to_string(nu));
                            const double expected_value = ve::dcc::objective<double>(shape.panel, spec, theta);
                            ASSERT_TRUE(std::isfinite(expected_value)) << "the case must be inside the domain";
                            std::vector<double> gradient(theta.size(), -99.0);
                            ve::dcc::ObjectiveGradient evaluator;
                            const double value = evaluator.evaluate(shape.panel, spec, theta, gradient);
                            EXPECT_EQ(value, expected_value) << "the value is the one of objective<double>, to the bit";
                            const std::vector<double> expected = dual_gradient(shape.panel, spec, theta);
                            double scale = 1.0;
                            for (const double g : expected)
                            {
                                scale = std::max(scale, std::fabs(g));
                            }
                            for (std::size_t i = 0; i < theta.size(); ++i)
                            {
                                EXPECT_NEAR(gradient[i], expected[i], kGradientTolerance * scale) << "parameter " << i;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Orders above (1, 1): the lags of the news and of the earlier Q read the ring of dQ_(t - k) and the presample moments
// of every lag, which a single lag never does.
TEST(DccGradient, MatchesDualForHigherOrders)
{
    struct Orders
    {
        DccModel model;
        int p;
        int q;
    };
    const std::vector<Orders> cases{{DccModel::Dcc, 2, 1}, {DccModel::Dcc, 1, 2}, {DccModel::Dcc, 3, 2},
                                    {DccModel::ADcc, 2, 2}, {DccModel::ADcc, 2, 1}};
    for (const StandardizedPanel& panel : {ve::testing::dcc3_panel(), ve::testing::synthetic_panel(3, 60)})
    {
        for (const Orders& orders : cases)
        {
            for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT})
            {
                DccSpec spec = spec_of(orders.model, dist, Convention::Standard, {});
                spec.p = orders.p;
                spec.q = orders.q;
                std::vector<double> theta;
                for (int j = 0; j < orders.p; ++j)
                {
                    theta.push_back(0.03 / (j + 1));
                }
                for (int k = 0; k < orders.q; ++k)
                {
                    theta.push_back(0.85 / (k + 1) / orders.q + 0.02 * k);
                }
                for (int j = 0; orders.model == DccModel::ADcc && j < orders.p; ++j)
                {
                    theta.push_back(0.01 / (j + 1));
                }
                if (dist == DccDist::MvT)
                {
                    theta.push_back(7.0);
                }
                SCOPED_TRACE("model " + std::to_string(static_cast<int>(orders.model)) + ", p " +
                             std::to_string(orders.p) + ", q " + std::to_string(orders.q) + ", dist " +
                             std::to_string(static_cast<int>(dist)));
                const double expected_value = ve::dcc::objective<double>(panel, spec, theta);
                ASSERT_TRUE(std::isfinite(expected_value));
                std::vector<double> gradient(theta.size());
                ve::dcc::ObjectiveGradient evaluator;
                EXPECT_EQ(evaluator.evaluate(panel, spec, theta, gradient), expected_value);
                const std::vector<double> expected = dual_gradient(panel, spec, theta);
                double scale = 1.0;
                for (const double g : expected)
                {
                    scale = std::max(scale, std::fabs(g));
                }
                for (std::size_t i = 0; i < theta.size(); ++i)
                {
                    EXPECT_NEAR(gradient[i], expected[i], kGradientTolerance * scale) << "parameter " << i;
                }
            }
        }
    }
}

// An evaluator keeps its buffers between calls: the second call at another point is not contaminated by the first.
TEST(DccGradient, ReusingTheEvaluatorGivesTheSameResults)
{
    const StandardizedPanel panel = ve::testing::synthetic_panel(5, 70);
    const DccSpec spec = spec_of(DccModel::ADcc, DccDist::MvT, Convention::Standard, {});
    const std::vector<double> first = theta_of(spec, 2, 0.93, 6.0);
    const std::vector<double> second = theta_of(spec, 2, 0.99, 9.0);
    ve::dcc::ObjectiveGradient reused;
    std::vector<double> scratch(first.size());
    reused.evaluate(panel, spec, first, scratch);
    std::vector<double> after(second.size());
    const double reused_value = reused.evaluate(panel, spec, second, after);
    ve::dcc::ObjectiveGradient fresh;
    std::vector<double> alone(second.size());
    const double fresh_value = fresh.evaluate(panel, spec, second, alone);
    EXPECT_EQ(reused_value, fresh_value);
    EXPECT_EQ(after, alone);
}

// A recursion that leaves the positive definite matrices gives NaN terms: +infinity, as objective<double>, and
// a zero gradient.
TEST(DccGradient, OutsideTheDomainTheValueIsInfinity)
{
    const StandardizedPanel panel = ve::testing::synthetic_panel(5, 200);
    const DccSpec spec = spec_of(DccModel::Dcc, DccDist::MvNorm, Convention::Standard, {});
    const std::vector<std::vector<double>> candidates{{0.5, -1.0}, {1.5, -0.9}, {-0.8, 0.5}, {2.0, 0.4}};
    bool found = false;
    for (const std::vector<double>& theta : candidates)
    {
        const double expected = ve::dcc::objective<double>(panel, spec, theta);
        std::vector<double> gradient(theta.size(), -99.0);
        ve::dcc::ObjectiveGradient evaluator;
        const double value = evaluator.evaluate(panel, spec, theta, gradient);
        EXPECT_EQ(value, expected);
        if (std::isfinite(expected))
        {
            continue;
        }
        found = true;
        EXPECT_EQ(value, std::numeric_limits<double>::infinity());
        EXPECT_EQ(gradient, (std::vector<double>(theta.size(), 0.0)));
    }
    EXPECT_TRUE(found) << "no candidate left the domain";
}

// The tangent gradient of spec on panel against the Dual one, and the value to the bit.
void expect_gradient_matches_dual(const StandardizedPanel& panel, const DccSpec& spec, std::span<const double> theta)
{
    const double expected_value = ve::dcc::objective<double>(panel, spec, theta);
    ASSERT_TRUE(std::isfinite(expected_value));
    std::vector<double> gradient(theta.size(), -99.0);
    ve::dcc::ObjectiveGradient evaluator;
    EXPECT_EQ(evaluator.evaluate(panel, spec, theta, gradient), expected_value);
    const std::vector<double> expected = dual_gradient(panel, spec, theta);
    double scale = 1.0;
    for (const double g : expected)
    {
        scale = std::max(scale, std::fabs(g));
    }
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        EXPECT_NEAR(gradient[i], expected[i], kGradientTolerance * scale) << "parameter " << i;
    }
}

// The copula specs of the test: parametric and empirical, in both conventions, DCC and aDCC, mvnorm and mvt.
std::vector<DccSpec> copula_specs()
{
    std::vector<DccSpec> specs;
    for (const DccCopula copula : {DccCopula::Parametric, DccCopula::Empirical})
    {
        for (const Convention convention : {Convention::Standard, Convention::Rmgarch})
        {
            for (const DccModel model : {DccModel::Dcc, DccModel::ADcc})
            {
                for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT})
                {
                    DccSpec spec = spec_of(model, dist, convention, {});
                    spec.copula = copula;
                    specs.push_back(spec);
                }
            }
        }
    }
    return specs;
}

// The score panel of a copula is an ordinary DCC panel, so the tangent evaluator agrees with the Dual on it.
TEST(DccGradient, MatchesDualOnCopulaScores)
{
    const std::vector<ve::FitResult> fits = ve::testing::copula3_fits();
    std::vector<const ve::FitResult*> pointers;
    for (const ve::FitResult& fit : fits)
    {
        pointers.push_back(&fit);
    }
    for (const DccSpec& spec : copula_specs())
    {
        SCOPED_TRACE("copula " + std::to_string(static_cast<int>(spec.copula)) + ", convention " +
                     std::to_string(static_cast<int>(spec.convention)) + ", model " +
                     std::to_string(static_cast<int>(spec.model)) + ", dist " +
                     std::to_string(static_cast<int>(spec.dist)));
        const std::vector<DccSpec> specs{spec};
        const ve::dcc::copula::CopulaInputs inputs = ve::dcc::copula::copula_inputs(pointers, specs);
        const ve::dcc::copula::CopulaPanel scores =
            ve::dcc::copula::copula_panel(inputs.uniforms_of(spec), *inputs.univariate, spec.dist, 6.0);
        expect_gradient_matches_dual(*scores.panel, spec, theta_of(spec, 2, 0.93, 6.0));
    }
}

TEST(DccGradient, CoversListedSpecs)
{
    for (const DccModel model : {DccModel::Dcc, DccModel::ADcc, DccModel::FDcc})
    {
        for (const DccDist dist : {DccDist::MvNorm, DccDist::MvT, DccDist::MvLaplace})
        {
            const DccSpec spec = spec_of(model, dist, Convention::Standard, model == DccModel::FDcc ? std::vector<int>{1, 2} : std::vector<int>{});
            EXPECT_TRUE(ve::dcc::has_fast_gradient(spec)) << static_cast<int>(model) << "/" << static_cast<int>(dist);
        }
    }
    DccSpec copula = spec_of(DccModel::Dcc, DccDist::MvT, Convention::Standard, {});
    copula.copula = DccCopula::Parametric;
    EXPECT_TRUE(ve::dcc::has_fast_gradient(copula));
    EXPECT_FALSE(ve::dcc::has_fast_gradient(spec_of(DccModel::Static, DccDist::MvNorm, Convention::Standard, {})));
}

}
