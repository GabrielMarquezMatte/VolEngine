#include "capi/handles.hpp"
#include "risk/measures.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <volengine/volengine.h>

namespace
{

constexpr std::size_t kRows = 50;
constexpr std::size_t kSeries = 2;
constexpr double kUntouched = 7.0;
constexpr double kNan = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

std::vector<double> panel_data()
{
    std::vector<double> data(kRows * kSeries);
    for (std::size_t k = 0; k < data.size(); ++k)
    {
        data[k] = 0.02 * std::sin(0.9 * static_cast<double>(k));
    }
    return data;
}

struct Panel
{
    std::vector<double> data = panel_data();
    ve_panel* handle = nullptr;

    explicit Panel(std::size_t t = kRows, std::size_t n = kSeries)
    {
        EXPECT_EQ(ve_panel_create(data.data(), t, n, &handle), VE_OK);
    }
    ~Panel()
    {
        ve_panel_free(handle);
    }
    Panel(const Panel&) = delete;
    Panel& operator=(const Panel&) = delete;
};

const std::vector<double> kWeights{0.7, 0.3};

void expect_empirical_refused(const ve_panel* panel, const double* weights, int32_t kind, double confidence)
{
    double var = kUntouched;
    double es = kUntouched;
    EXPECT_EQ(ve_risk_empirical(panel, weights, kind, confidence, &var, &es), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()), "");
    EXPECT_EQ(var, kUntouched);
    EXPECT_EQ(es, kUntouched);
}

struct ParametricCall
{
    int32_t dist = VE_DIST_STD;
    double skew = kNan;
    double shape = 6.0;
    double ghlambda = kNan;
    double mean = 0.001;
    double sigma = 0.02;
    double confidence = 0.99;
};

ve_status call(const ParametricCall& c, double* var, double* es)
{
    return ve_risk_parametric(c.dist, c.skew, c.shape, c.ghlambda, c.mean, c.sigma, c.confidence, var, es);
}

void expect_parametric_refused(const ParametricCall& c)
{
    double var = kUntouched;
    double es = kUntouched;
    EXPECT_EQ(call(c, &var, &es), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()), "");
    EXPECT_EQ(var, kUntouched);
    EXPECT_EQ(es, kUntouched);
}

}

TEST(CapiRisk, EmpiricalMatchesCore)
{
    Panel panel;
    double var = 0.0;
    double es = 0.0;
    ASSERT_EQ(ve_risk_empirical(panel.handle, kWeights.data(), VE_RETURNS_LOG, 0.95, &var, &es), VE_OK);
    const auto expected = ve::risk::empirical(panel.handle->view, kWeights, ve::risk::ReturnKind::log, 0.95);
    EXPECT_EQ(var, expected.var);
    EXPECT_EQ(es, expected.es);
}

TEST(CapiRisk, ParametricMatchesCore)
{
    double var = 0.0;
    double es = 0.0;
    ASSERT_EQ(ve_risk_parametric(VE_DIST_NIG, -0.3, 1.5, kNan, 0.001, 0.02, 0.99, &var, &es), VE_OK);
    const auto expected = ve::risk::parametric(ve::Dist::Nig, {.skew = -0.3, .shape = 1.5},
                                               ve::risk::Location{.mean = 0.001, .sigma = 0.02}, 0.99);
    EXPECT_EQ(var, expected.var);
    EXPECT_EQ(es, expected.es);
}

TEST(CapiRisk, IgnoredParametersMayBeNan)
{
    double var = 0.0;
    double es = 0.0;
    EXPECT_EQ(ve_risk_parametric(VE_DIST_NORM, kNan, kNan, kNan, 0.0, 1.0, 0.99, &var, &es), VE_OK);
    EXPECT_NEAR(var, 2.3263478740408408, 1e-14);
}

TEST(CapiRisk, EmpiricalNullPanel)
{
    double var = kUntouched;
    double es = kUntouched;
    EXPECT_EQ(ve_risk_empirical(nullptr, kWeights.data(), VE_RETURNS_LOG, 0.99, &var, &es), VE_INVALID_HANDLE);
    EXPECT_EQ(var, kUntouched);
}

TEST(CapiRisk, EmpiricalNullOutputs)
{
    Panel panel;
    double value = 0.0;
    EXPECT_EQ(ve_risk_empirical(panel.handle, kWeights.data(), VE_RETURNS_LOG, 0.99, nullptr, &value),
              VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_risk_empirical(panel.handle, kWeights.data(), VE_RETURNS_LOG, 0.99, &value, nullptr),
              VE_INVALID_ARGUMENT);
}

TEST(CapiRisk, EmpiricalRefusals)
{
    Panel panel;
    expect_empirical_refused(panel.handle, nullptr, VE_RETURNS_LOG, 0.99);
    expect_empirical_refused(panel.handle, kWeights.data(), 2, 0.99);
    expect_empirical_refused(panel.handle, kWeights.data(), -1, 0.99);
    for (double confidence : {0.0, 1.0, -0.5, 1.5, kNan})
    {
        expect_empirical_refused(panel.handle, kWeights.data(), VE_RETURNS_LOG, confidence);
    }
    const std::vector<double> bad_weights{0.5, kInf};
    expect_empirical_refused(panel.handle, bad_weights.data(), VE_RETURNS_LOG, 0.99);
}

TEST(CapiRisk, EmpiricalPanelRefusals)
{
    Panel empty_rows(0, kSeries);
    expect_empirical_refused(empty_rows.handle, kWeights.data(), VE_RETURNS_LOG, 0.99);
    Panel ranged;
    const std::vector<std::size_t> starts(kSeries, 0);
    const std::vector<std::size_t> lengths(kSeries, kRows);
    ASSERT_EQ(ve_panel_set_ranges(ranged.handle, starts.data(), lengths.data()), VE_OK);
    expect_empirical_refused(ranged.handle, kWeights.data(), VE_RETURNS_LOG, 0.99);
    Panel with_nan;
    with_nan.data.back() = kNan;
    expect_empirical_refused(with_nan.handle, kWeights.data(), VE_RETURNS_LOG, 0.99);
}

TEST(CapiRisk, NonFinitePortfolioRefused)
{
    const std::vector<double> data{710.0, 0.0, 0.0, 0.0};
    ve_panel* panel = nullptr;
    ASSERT_EQ(ve_panel_create(data.data(), 2, 2, &panel), VE_OK);
    const std::vector<double> weights{0.0, 1.0};
    expect_empirical_refused(panel, weights.data(), VE_RETURNS_LOG, 0.99);
    ve_panel_free(panel);
}

TEST(CapiRisk, EmpiricalNoSeries)
{
    ve_panel* panel = nullptr;
    ASSERT_EQ(ve_panel_create(nullptr, kRows, 0, &panel), VE_OK);
    expect_empirical_refused(panel, kWeights.data(), VE_RETURNS_LOG, 0.99);
    ve_panel_free(panel);
}

TEST(CapiRisk, ParametricRefusals)
{
    expect_parametric_refused(ParametricCall{.dist = -1});
    expect_parametric_refused(ParametricCall{.dist = 8});
    expect_parametric_refused(ParametricCall{.shape = 2.0});
    expect_parametric_refused(ParametricCall{.shape = kNan});
    expect_parametric_refused(ParametricCall{.dist = VE_DIST_SSTD, .skew = -1.0});
    expect_parametric_refused(ParametricCall{.mean = kNan});
    expect_parametric_refused(ParametricCall{.sigma = -0.01});
    expect_parametric_refused(ParametricCall{.sigma = kInf});
    for (double confidence : {0.0, 1.0, kNan})
    {
        expect_parametric_refused(ParametricCall{.confidence = confidence});
    }
}

TEST(CapiRisk, ParametricNullOutputs)
{
    double value = 0.0;
    EXPECT_EQ(call(ParametricCall{}, nullptr, &value), VE_INVALID_ARGUMENT);
    EXPECT_EQ(call(ParametricCall{}, &value, nullptr), VE_INVALID_ARGUMENT);
}

TEST(CapiRisk, EmpiricalTailSumOverflowIsFinite)
{
    const std::vector<double> data{-1.5e308, -1.5e308, 0.0, 0.0, 0.0};
    ve_panel* panel = nullptr;
    ASSERT_EQ(ve_panel_create(data.data(), 5, 1, &panel), VE_OK);
    const std::vector<double> weights{1.0};
    double var = 0.0;
    double es = 0.0;
    EXPECT_EQ(ve_risk_empirical(panel, weights.data(), VE_RETURNS_SIMPLE, 0.6, &var, &es), VE_OK);
    EXPECT_DOUBLE_EQ(var, 6e307);
    EXPECT_EQ(es, 1.5e308);
    ve_panel_free(panel);
}

TEST(CapiRisk, EmpiricalInterpolationAcrossTheRange)
{
    const std::vector<double> data{-1e308, 1e308};
    ve_panel* panel = nullptr;
    ASSERT_EQ(ve_panel_create(data.data(), 2, 1, &panel), VE_OK);
    const std::vector<double> weights{1.0};
    double var = kUntouched;
    double es = kUntouched;
    EXPECT_EQ(ve_risk_empirical(panel, weights.data(), VE_RETURNS_SIMPLE, 0.5, &var, &es), VE_OK);
    EXPECT_EQ(var, 0.0);
    EXPECT_EQ(es, 1e308);
    ve_panel_free(panel);
}

// 1 - confidence rounds to 1 below confidence ~1.1e-16, which would put the quantile at +inf.
TEST(CapiRisk, ConfidenceWhoseTailRoundsToOneRefused)
{
    for (double confidence : {1e-17, 1e-300, std::numeric_limits<double>::denorm_min()})
    {
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_NORM, .confidence = confidence});
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_GED, .shape = 1.5, .confidence = confidence});
        EXPECT_EQ(std::string(ve_last_error()), "confidence must be in (0, 1)");
        Panel panel;
        expect_empirical_refused(panel.handle, kWeights.data(), VE_RETURNS_LOG, confidence);
    }
}

// A parameter the density reads must be finite; one it ignores is not looked at.
TEST(CapiRisk, NonFiniteReadParameterRefused)
{
    for (double value : {kInf, -kInf, kNan})
    {
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_STD, .shape = value});
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_GED, .shape = value});
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_SSTD, .skew = 1.2, .shape = value});
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_SSTD, .skew = value, .shape = 5.0});
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_SGED, .skew = 1.2, .shape = value});
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_SGED, .skew = value, .shape = 1.5});
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_NIG, .skew = value, .shape = 1.5});
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_NIG, .skew = 0.3, .shape = value});
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_GHYP, .skew = 0.3, .shape = 1.5, .ghlambda = value});
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_JSU, .skew = 0.3, .shape = value});
        expect_parametric_refused(ParametricCall{.dist = VE_DIST_JSU, .skew = value, .shape = 1.5});
        EXPECT_EQ(std::string(ve_last_error()), "the distribution parameters are outside the density's domain");
    }
    double var = 0.0;
    double es = 0.0;
    EXPECT_EQ(ve_risk_parametric(VE_DIST_STD, kInf, 6.0, kInf, 0.0, 1.0, 0.99, &var, &es), VE_OK);
    EXPECT_EQ(ve_risk_parametric(VE_DIST_NIG, 0.3, 1.5, kInf, 0.0, 1.0, 0.99, &var, &es), VE_OK);
}

// The Johnson SU past the overflow of its variance factor, where the measure is still a double.
TEST(CapiRisk, JsuPastOverflowMatchesCore)
{
    for (const auto& [skew, shape] : {std::pair{400.0, 1.0}, std::pair{-400.0, 1.0}, std::pair{0.5, 0.05}})
    {
        double var = 0.0;
        double es = 0.0;
        ASSERT_EQ(ve_risk_parametric(VE_DIST_JSU, skew, shape, kNan, 0.0, 1.0, 0.99, &var, &es), VE_OK);
        const auto expected = ve::risk::parametric(ve::Dist::Jsu, {.skew = skew, .shape = shape},
                                                   ve::risk::Location{.mean = 0.0, .sigma = 1.0}, 0.99);
        EXPECT_EQ(var, expected.var);
        EXPECT_EQ(es, expected.es);
        EXPECT_TRUE(std::isfinite(var) && std::isfinite(es) && var > 0.0 && es > var) << skew << " " << shape;
    }
}

// A measure that is not a double (a Johnson SU scale below the smallest normal, or sigma q past DBL_MAX) is refused.
TEST(CapiRisk, NonFiniteMeasureRefused)
{
    expect_parametric_refused(ParametricCall{.dist = VE_DIST_JSU, .skew = 0.5, .shape = 0.02});
    expect_parametric_refused(ParametricCall{.dist = VE_DIST_JSU, .skew = 1000.0, .shape = 1.0});
    EXPECT_EQ(std::string(ve_last_error()), "the risk measure is not finite (an input overflows)");
    expect_parametric_refused(ParametricCall{.dist = VE_DIST_NORM, .mean = 0.0, .sigma = 1e308});
    EXPECT_EQ(std::string(ve_last_error()), "the risk measure is not finite (an input overflows)");
    expect_parametric_refused(ParametricCall{.dist = VE_DIST_NORM, .mean = -1.7e308, .sigma = 1e308});
}

// Extreme parameters inside every density's domain: the measure is a finite double, or the call is refused as an
// argument error with its outputs untouched. A Boost failure never escapes as VE_INTERNAL_ERROR.
TEST(CapiRisk, ExtremeParametersAreFiniteOrRefused)
{
    std::vector<ParametricCall> calls;
    const std::vector<double> skews{1e-300, 1e-155, 1e-150, 1e-20, 0.05, 20.0, 1e20, 1e150, 1e155, 1e300};
    for (double shape : {2.0000000001, 2.05, 30.0, 1e8, 1e300})
    {
        calls.push_back({.dist = VE_DIST_STD, .shape = shape});
        for (double skew : skews)
        {
            calls.push_back({.dist = VE_DIST_SSTD, .skew = skew, .shape = shape});
        }
    }
    for (double shape : {0.1, 0.2, 2.0, 100.0, 1e8, 1e300})
    {
        calls.push_back({.dist = VE_DIST_GED, .shape = shape});
        for (double skew : skews)
        {
            calls.push_back({.dist = VE_DIST_SGED, .skew = skew, .shape = shape});
        }
    }
    for (double rho : {-0.999999, -0.5, 0.0, 0.99, 0.999999})
    {
        for (double zeta : {1e-6, 0.01, 2.0, 1e4, 1e6})
        {
            calls.push_back({.dist = VE_DIST_NIG, .skew = rho, .shape = zeta});
            for (double lambda : {-50.0, -0.5, 1.0, 50.0})
            {
                calls.push_back({.dist = VE_DIST_GHYP, .skew = rho, .shape = zeta, .ghlambda = lambda});
            }
        }
    }
    for (double nu : {-1000.0, -5.0, 0.0, 300.0})
    {
        for (double tau : {0.01, 0.04, 1.0, 1e5, 1e200})
        {
            calls.push_back({.dist = VE_DIST_JSU, .skew = nu, .shape = tau});
        }
    }
    for (ParametricCall c : calls)
    {
        c.mean = 0.0;
        c.sigma = 1.0;
        for (double confidence : {0.5, 0.99})
        {
            c.confidence = confidence;
            double var = kUntouched;
            double es = kUntouched;
            const ve_status status = call(c, &var, &es);
            const std::string message = ve_last_error();
            SCOPED_TRACE(::testing::Message() << "dist " << c.dist << " skew " << c.skew << " shape " << c.shape
                                              << " ghlambda " << c.ghlambda << " confidence " << confidence);
            EXPECT_TRUE(status == VE_OK || status == VE_INVALID_ARGUMENT) << status << " " << message;
            if (status == VE_OK)
            {
                EXPECT_TRUE(std::isfinite(var) && std::isfinite(es));
            }
            if (status != VE_OK)
            {
                EXPECT_EQ(var, kUntouched);
                EXPECT_EQ(es, kUntouched);
            }
        }
    }
}
