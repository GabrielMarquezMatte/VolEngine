#include "capi/handles.hpp"
#include "risk/measures.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
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
