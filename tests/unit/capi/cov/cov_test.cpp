#include "capi/handles.hpp"
#include "cov/estimators.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>
#include <volengine/volengine.h>

namespace
{

constexpr std::size_t kRows = 40;
constexpr std::size_t kSeries = 3;
constexpr double kUntouched = 7.0;

std::vector<double> panel_data()
{
    std::vector<double> data(kRows * kSeries);
    for (std::size_t k = 0; k < data.size(); ++k)
    {
        data[k] = std::sin(0.7 * static_cast<double>(k)) + 0.1 * static_cast<double>(k % kSeries);
    }
    return data;
}

ve_panel* make_panel(const std::vector<double>& data, std::size_t t, std::size_t n)
{
    ve_panel* panel = nullptr;
    EXPECT_EQ(ve_panel_create(data.data(), t, n, &panel), VE_OK);
    return panel;
}

// Both estimators refuse the panel with VE_INVALID_ARGUMENT, a message, and out and shrinkage left as they were.
void expect_refused(const ve_panel* panel)
{
    std::vector<double> out(kSeries * kSeries, kUntouched);
    double shrinkage = kUntouched;
    EXPECT_EQ(ve_cov_sample(panel, out.data()), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()), "");
    EXPECT_EQ(ve_cov_ledoit_wolf(panel, out.data(), &shrinkage), VE_INVALID_ARGUMENT);
    EXPECT_NE(std::string(ve_last_error()), "");
    EXPECT_EQ(out, std::vector<double>(kSeries * kSeries, kUntouched));
    EXPECT_EQ(shrinkage, kUntouched);
}

}

TEST(CapiCov, SampleMatchesCore)
{
    const auto data = panel_data();
    ve_panel* panel = make_panel(data, kRows, kSeries);
    std::vector<double> out(kSeries * kSeries);
    ASSERT_EQ(ve_cov_sample(panel, out.data()), VE_OK);
    EXPECT_EQ(out, ve::cov::sample(panel->view));
    ve_panel_free(panel);
}

TEST(CapiCov, LedoitWolfMatchesCore)
{
    const auto data = panel_data();
    ve_panel* panel = make_panel(data, kRows, kSeries);
    std::vector<double> out(kSeries * kSeries);
    double shrinkage = -1.0;
    ASSERT_EQ(ve_cov_ledoit_wolf(panel, out.data(), &shrinkage), VE_OK);
    const auto expected = ve::cov::ledoit_wolf(panel->view);
    EXPECT_EQ(out, expected.covariance);
    EXPECT_EQ(shrinkage, expected.shrinkage);
    ve_panel_free(panel);
}

TEST(CapiCov, NullPanel)
{
    std::vector<double> out(1);
    double shrinkage = 0.0;
    EXPECT_EQ(ve_cov_sample(nullptr, out.data()), VE_INVALID_HANDLE);
    EXPECT_EQ(ve_cov_ledoit_wolf(nullptr, out.data(), &shrinkage), VE_INVALID_HANDLE);
}

TEST(CapiCov, NullOut)
{
    const auto data = panel_data();
    ve_panel* panel = make_panel(data, kRows, kSeries);
    double shrinkage = 0.0;
    EXPECT_EQ(ve_cov_sample(panel, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_cov_ledoit_wolf(panel, nullptr, &shrinkage), VE_INVALID_ARGUMENT);
    ve_panel_free(panel);
}

TEST(CapiCov, NullShrinkage)
{
    const auto data = panel_data();
    ve_panel* panel = make_panel(data, kRows, kSeries);
    std::vector<double> out(kSeries * kSeries, kUntouched);
    EXPECT_EQ(ve_cov_ledoit_wolf(panel, out.data(), nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(out, std::vector<double>(kSeries * kSeries, kUntouched));
    ve_panel_free(panel);
}

TEST(CapiCov, RangesRefused)
{
    const auto data = panel_data();
    ve_panel* panel = make_panel(data, kRows, kSeries);
    const std::vector<std::size_t> starts(kSeries, 0);
    const std::vector<std::size_t> lengths(kSeries, kRows);
    ASSERT_EQ(ve_panel_set_ranges(panel, starts.data(), lengths.data()), VE_OK);
    expect_refused(panel);
    ve_panel_free(panel);
}

TEST(CapiCov, TooFewRows)
{
    const auto data = panel_data();
    ve_panel* panel = make_panel(data, 1, kSeries);
    expect_refused(panel);
    ve_panel_free(panel);
}

TEST(CapiCov, NoSeries)
{
    ve_panel* panel = nullptr;
    ASSERT_EQ(ve_panel_create(nullptr, kRows, 0, &panel), VE_OK);
    expect_refused(panel);
    ve_panel_free(panel);
}

TEST(CapiCov, NanRefused)
{
    auto data = panel_data();
    data.back() = std::numeric_limits<double>::quiet_NaN();
    ve_panel* panel = make_panel(data, kRows, kSeries);
    expect_refused(panel);
    ve_panel_free(panel);
}

TEST(CapiCov, InfRefused)
{
    auto data = panel_data();
    data[kSeries + 1] = -std::numeric_limits<double>::infinity();
    ve_panel* panel = make_panel(data, kRows, kSeries);
    expect_refused(panel);
    ve_panel_free(panel);
}

// Every value is finite, but the products overflow: refused, out and shrinkage left as they were.
TEST(CapiCov, OverflowRefused)
{
    const std::vector<double> data{1e200, -1e200, 3e199, 2.0, -1e200, 5.0};
    ve_panel* panel = make_panel(data, 3, 2);
    std::vector<double> out(4, kUntouched);
    double shrinkage = kUntouched;
    EXPECT_EQ(ve_cov_sample(panel, out.data()), VE_INVALID_ARGUMENT);
    EXPECT_EQ(std::string(ve_last_error()), "a covariance is not finite (an input overflows)");
    EXPECT_EQ(ve_cov_ledoit_wolf(panel, out.data(), &shrinkage), VE_INVALID_ARGUMENT);
    EXPECT_EQ(std::string(ve_last_error()), "a covariance is not finite (an input overflows)");
    EXPECT_EQ(out, std::vector<double>(4, kUntouched));
    EXPECT_EQ(shrinkage, kUntouched);
    ve_panel_free(panel);
}
