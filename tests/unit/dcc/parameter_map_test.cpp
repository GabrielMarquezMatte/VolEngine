#include "dcc/parameter_map.hpp"

#include "core/spec.hpp"
#include "dcc/spec.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <vector>

namespace
{

using ve::dcc::DccDist;
using ve::dcc::DccLayout;
using ve::dcc::DccModel;
using ve::dcc::DccSpec;

constexpr double kCeiling = 1.0 - 1e-8;
constexpr double kDelta = 0.4;

DccSpec spec_of(DccModel model, int p, int q, DccDist dist)
{
    DccSpec spec;
    spec.model = model;
    spec.p = p;
    spec.q = q;
    spec.dist = dist;
    return spec;
}

DccSpec fdcc_spec(std::vector<int> groups, DccDist dist)
{
    DccSpec spec = spec_of(DccModel::FDcc, 1, 1, dist);
    spec.groups = std::move(groups);
    return spec;
}

// The rows of the RoundTripAndConstraints table: a spec and its number of series.
struct Case
{
    DccSpec spec;
    std::size_t m;
};

std::vector<Case> round_trip_cases()
{
    std::vector<Case> cases{{spec_of(DccModel::Dcc, 1, 1, DccDist::MvT), 3},
                            {spec_of(DccModel::Dcc, 2, 3, DccDist::MvNorm), 3},
                            {spec_of(DccModel::ADcc, 2, 1, DccDist::MvT), 3},
                            {spec_of(DccModel::ADcc, 1, 1, DccDist::MvLaplace), 3},
                            {fdcc_spec({1, 1, 2}, DccDist::MvNorm), 3},
                            {fdcc_spec({1, 2, 3, 1}, DccDist::MvT), 4}};
    DccSpec fixed_beta = spec_of(DccModel::ADcc, 2, 2, DccDist::MvNorm);
    fixed_beta.fixed = {{"dccb1", 0.5}, {"dccg2", 0.1}};
    cases.push_back({fixed_beta, 3});
    DccSpec fixed_group = fdcc_spec({1, 2, 2}, DccDist::MvT);
    fixed_group.fixed = {{"fdcca1", 0.2}, {"fdccb2", 0.9}, {"mshape", 7.0}};
    cases.push_back({fixed_group, 3});
    return cases;
}

double term_sum(std::span<const double> theta, const DccLayout& layout)
{
    double sum = 0.0;
    for (std::size_t i = 0; i < layout.shape(); ++i)
    {
        double scale = i >= layout.g_begin() ? kDelta : 1.0;
        sum += scale * theta[i];
    }
    return sum;
}

void expect_constraints(std::span<const double> theta, const DccLayout& layout)
{
    ASSERT_TRUE(ve::dcc::is_valid(theta, layout, kDelta));
    if (layout.has_shape)
    {
        EXPECT_GT(theta[layout.shape()], 4.0);
        EXPECT_LT(theta[layout.shape()], 50.0);
    }
    if (layout.model != DccModel::FDcc)
    {
        for (std::size_t i = 0; i < layout.shape(); ++i)
        {
            EXPECT_GE(theta[i], 0.0) << i;
        }
        EXPECT_LT(term_sum(theta, layout), kCeiling);
        return;
    }
    for (std::size_t g = 0; g < layout.a_count; ++g)
    {
        const double a = theta[g];
        const double b = theta[layout.b_begin() + g];
        EXPECT_GT(a, 0.0) << g;
        EXPECT_GT(b, 0.0) << g;
        EXPECT_LT(std::hypot(a, b), kCeiling) << g;
    }
}

void expect_fixed(std::span<const double> theta, const DccSpec& spec, const DccLayout& layout)
{
    const std::vector<std::string> names = ve::dcc::parameter_names(layout);
    for (const ve::ParameterValue& entry : spec.fixed)
    {
        const auto found = std::ranges::find(names, entry.name);
        ASSERT_NE(found, names.end()) << entry.name;
        EXPECT_EQ(theta[static_cast<std::size_t>(found - names.begin())], entry.value) << entry.name;
    }
}

// Uniform in (-3, 3) at the estimated positions, 0 at the fixed ones.
std::vector<double> random_u(const DccLayout& layout, std::mt19937_64& engine)
{
    std::uniform_real_distribution<double> draw(-3.0, 3.0);
    std::vector<double> u(layout.count(), 0.0);
    for (std::size_t i = 0; i < u.size(); ++i)
    {
        if (layout.is_estimated(i))
        {
            u[i] = draw(engine);
        }
    }
    return u;
}

TEST(DccParameterMap, RoundTripAndConstraints)
{
    std::mt19937_64 engine(20261113);
    for (const Case& c : round_trip_cases())
    {
        ASSERT_FALSE(ve::dcc::spec_problem(c.spec, c.m).has_value());
        const DccLayout layout = ve::dcc::layout_of(c.spec, c.m);
        for (int repeat = 0; repeat < 50; ++repeat)
        {
            std::vector<double> u = random_u(layout, engine);
            std::vector<double> theta(layout.count());
            ve::dcc::to_natural<double>(u, theta, layout, kDelta);
            expect_constraints(theta, layout);
            expect_fixed(theta, c.spec, layout);
            std::vector<double> back(layout.count(), 1.0);
            ve::dcc::to_unconstrained(theta, back, layout, kDelta);
            for (std::size_t i = 0; i < u.size(); ++i)
            {
                EXPECT_NEAR(back[i], u[i], 1e-12) << "parameter " << i << " repeat " << repeat;
            }
        }
    }
}

TEST(DccParameterMap, RejectsPointsOutsideTheDomain)
{
    const DccLayout dcc = ve::dcc::layout_of(spec_of(DccModel::ADcc, 1, 1, DccDist::MvT), 3);
    EXPECT_TRUE(ve::dcc::is_valid(std::vector<double>{0.05, 0.9, 0.05, 8.0}, dcc, kDelta));
    EXPECT_FALSE(ve::dcc::is_valid(std::vector<double>{0.05, 0.9, 0.2, 8.0}, dcc, kDelta));
    EXPECT_FALSE(ve::dcc::is_valid(std::vector<double>{-0.01, 0.9, 0.05, 8.0}, dcc, kDelta));
    EXPECT_FALSE(ve::dcc::is_valid(std::vector<double>{0.05, 0.9, 0.05, 3.0}, dcc, kDelta));
    const DccLayout fdcc = ve::dcc::layout_of(fdcc_spec({1, 1, 2}, DccDist::MvNorm), 3);
    EXPECT_TRUE(ve::dcc::is_valid(std::vector<double>{0.19, 0.24, 0.96, 0.94}, fdcc, kDelta));
    EXPECT_FALSE(ve::dcc::is_valid(std::vector<double>{0.3, 0.24, 0.96, 0.94}, fdcc, kDelta));
}

TEST(DccParameterMap, StartValues)
{
    DccSpec adcc = spec_of(DccModel::ADcc, 2, 1, DccDist::MvT);
    adcc.start = {{"dccb1", 0.8}};
    adcc.fixed = {{"dccg2", 0.01}};
    const std::vector<double> start = ve::dcc::start_values(ve::dcc::layout_of(adcc, 3));
    EXPECT_EQ(start, (std::vector<double>{0.025, 0.025, 0.8, 0.025, 0.01, 5.0}));
    const std::vector<double> fdcc = ve::dcc::start_values(ve::dcc::layout_of(fdcc_spec({1, 2}, DccDist::MvNorm), 2));
    EXPECT_EQ(fdcc, (std::vector<double>{0.01, 0.01, 0.9, 0.9}));
}

std::vector<std::string> names_of(const DccSpec& spec, std::size_t m)
{
    return ve::dcc::parameter_names(ve::dcc::layout_of(spec, m));
}

bool has_problem(const DccSpec& spec, std::size_t m, const std::string& part)
{
    const std::optional<std::string> problem = ve::dcc::spec_problem(spec, m);
    return problem.has_value() && problem->find(part) != std::string::npos;
}

TEST(DccParameterMap, NamesAndProblems)
{
    using Names = std::vector<std::string>;
    EXPECT_EQ(names_of(spec_of(DccModel::Dcc, 1, 1, DccDist::MvT), 3), (Names{"dcca1", "dccb1", "mshape"}));
    EXPECT_EQ(names_of(spec_of(DccModel::ADcc, 2, 1, DccDist::MvNorm), 3),
              (Names{"dcca1", "dcca2", "dccb1", "dccg1", "dccg2"}));
    EXPECT_EQ(names_of(fdcc_spec({1, 1, 2}, DccDist::MvNorm), 3), (Names{"fdcca1", "fdcca2", "fdccb1", "fdccb2"}));
    DccSpec fixed = spec_of(DccModel::Dcc, 1, 1, DccDist::MvT);
    fixed.fixed = {{"mshape", 8.0}};
    const DccLayout layout = ve::dcc::layout_of(fixed, 3);
    EXPECT_EQ(layout.estimated_count(), 2U);
    EXPECT_EQ(ve::dcc::estimated_names(layout), (Names{"dcca1", "dccb1"}));

    EXPECT_TRUE(has_problem(spec_of(DccModel::FDcc, 2, 1, DccDist::MvNorm), 3, "(1, 1)"));
    EXPECT_TRUE(has_problem(fdcc_spec({1, 2}, DccDist::MvNorm), 3, "one group per series"));
    EXPECT_TRUE(has_problem(fdcc_spec({1, 3, 3}, DccDist::MvNorm), 3, "group 2 has no series"));
    EXPECT_TRUE(has_problem(fdcc_spec({0, 1, 1}, DccDist::MvNorm), 3, "groups"));
    EXPECT_TRUE(has_problem(fdcc_spec({}, DccDist::MvNorm), 3, "one group per series"));
    EXPECT_TRUE(has_problem(spec_of(DccModel::Dcc, 7, 1, DccDist::MvNorm), 3, "between 1 and 6"));
    EXPECT_TRUE(has_problem(spec_of(DccModel::Dcc, 1, 0, DccDist::MvNorm), 3, "between 1 and 6"));
    DccSpec grouped = spec_of(DccModel::Dcc, 1, 1, DccDist::MvNorm);
    grouped.groups = {1, 1, 2};
    EXPECT_TRUE(has_problem(grouped, 3, "fdcc"));
    DccSpec unknown = spec_of(DccModel::Dcc, 1, 1, DccDist::MvNorm);
    unknown.start = {{"mshape", 8.0}};
    EXPECT_TRUE(has_problem(unknown, 3, "unknown parameter mshape; the spec has: dcca1 dccb1"));
    DccSpec bad_model = spec_of(static_cast<DccModel>(3), 1, 1, DccDist::MvNorm);
    EXPECT_TRUE(has_problem(bad_model, 3, "model"));
    DccSpec bad_dist = spec_of(DccModel::Dcc, 1, 1, static_cast<DccDist>(-1));
    EXPECT_TRUE(has_problem(bad_dist, 3, "dist"));
    EXPECT_FALSE(ve::dcc::spec_problem(fdcc_spec({1, 1, 2}, DccDist::MvT), 3).has_value());
}

TEST(DccParameterMap, RejectsMoreParametersThanTheWidestDual)
{
    const std::size_t widest = ve::kMaxParameterCount;
    const std::size_t groups = widest / 2 + 1;
    std::vector<int> labels(groups);
    for (std::size_t i = 0; i < groups; ++i)
    {
        labels[i] = static_cast<int>(i + 1);
    }
    DccSpec wide = fdcc_spec(labels, DccDist::MvNorm);
    EXPECT_TRUE(has_problem(wide, groups, std::to_string(widest)));
    wide.fixed = {{"fdcca1", 0.1}, {"fdccb1", 0.9}};
    EXPECT_FALSE(ve::dcc::spec_problem(wide, groups).has_value());
}

}
