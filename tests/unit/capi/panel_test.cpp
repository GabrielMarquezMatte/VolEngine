#include "capi/handles.hpp"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <utility>
#include <vector>
#include <volengine/volengine.h>

namespace
{

constexpr size_t kMax = std::numeric_limits<size_t>::max();
constexpr ptrdiff_t kPtrMax = std::numeric_limits<ptrdiff_t>::max();

ve_panel* make_panel(const std::vector<double>& data, size_t t, size_t n)
{
    ve_panel* panel = nullptr;
    EXPECT_EQ(ve_panel_create(data.data(), t, n, &panel), VE_OK);
    return panel;
}

}

TEST(CapiPanel, CreateUsesCOrderDefaults)
{
    const std::vector<double> data(20, 1.0);
    ve_panel* panel = make_panel(data, 10, 2);
    ASSERT_NE(panel, nullptr);
    EXPECT_EQ(panel->view.data, data.data());
    EXPECT_EQ(panel->view.stride_t, 2);
    EXPECT_EQ(panel->view.stride_n, 1);
    EXPECT_FALSE(panel->view.has_ranges());
    ve_panel_free(panel);
}

TEST(CapiPanel, NullDataRejected)
{
    ve_panel* panel = reinterpret_cast<ve_panel*>(0x1);
    EXPECT_EQ(ve_panel_create(nullptr, 10, 2, &panel), VE_INVALID_ARGUMENT);
    EXPECT_EQ(panel, nullptr);
    EXPECT_NE(std::string(ve_last_error()).find("data"), std::string::npos);
}

TEST(CapiPanel, NullDataAllowedWhenEmpty)
{
    ve_panel* panel = nullptr;
    EXPECT_EQ(ve_panel_create(nullptr, 0, 3, &panel), VE_OK);
    ve_panel_free(panel);
    panel = nullptr;
    EXPECT_EQ(ve_panel_create(nullptr, 5, 0, &panel), VE_OK);
    ve_panel_free(panel);
}

TEST(CapiPanel, NullOutRejected)
{
    const std::vector<double> data(4, 0.0);
    EXPECT_EQ(ve_panel_create(data.data(), 2, 2, nullptr), VE_INVALID_ARGUMENT);
}

TEST(CapiPanel, SizeOverflowRejectedAndOutCleared)
{
    const double value = 0.0;
    ve_panel* panel = reinterpret_cast<ve_panel*>(0x1);
    EXPECT_EQ(ve_panel_create(&value, kMax / 2 + 1, 2, &panel), VE_INVALID_ARGUMENT);
    EXPECT_EQ(panel, nullptr);
    EXPECT_NE(std::string(ve_last_error()).find("overflow"), std::string::npos);
}

TEST(CapiPanel, ZeroStrideRejected)
{
    const std::vector<double> data(20, 0.0);
    ve_panel* panel = make_panel(data, 10, 2);
    EXPECT_EQ(ve_panel_set_strides(panel, 0, 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_panel_set_strides(panel, 2, 0), VE_INVALID_ARGUMENT);
    EXPECT_EQ(panel->view.stride_t, 2);
    EXPECT_EQ(panel->view.stride_n, 1);
    ve_panel_free(panel);
}

TEST(CapiPanel, ZeroStrideAllowedForUnitDimension)
{
    const std::vector<double> data(10, 0.0);
    ve_panel* panel = make_panel(data, 10, 1);
    EXPECT_EQ(ve_panel_set_strides(panel, 1, 0), VE_OK);
    ve_panel_free(panel);
}

TEST(CapiPanel, StrideOverflowRejected)
{
    const std::vector<double> data(20, 0.0);
    ve_panel* panel = make_panel(data, 10, 2);
    EXPECT_EQ(ve_panel_set_strides(panel, kPtrMax, 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_panel_set_strides(panel, kPtrMax / 5, 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_panel_set_strides(panel, -kPtrMax, 1), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_panel_set_strides(panel, 1, kPtrMax), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_panel_set_strides(panel, std::numeric_limits<ptrdiff_t>::min(), 1),
        VE_INVALID_ARGUMENT);
    EXPECT_EQ(panel->view.stride_t, 2);
    ve_panel_free(panel);
}

TEST(CapiPanel, ColumnMajorStridesAccepted)
{
    const std::vector<double> data = {1, 2, 3, 10, 20, 30};
    ve_panel* panel = make_panel(data, 3, 2);
    ASSERT_EQ(ve_panel_set_strides(panel, 1, 3), VE_OK);
    EXPECT_EQ(panel->view.at(2, 1), 30.0);
    EXPECT_EQ(panel->view.at(1, 0), 2.0);
    ve_panel_free(panel);
}

TEST(CapiPanel, NegativeStrideReadsBackwardsFromElementZero)
{
    const std::vector<double> data = {0, 1, 2, 3, 4, 5};
    ve_panel* panel = nullptr;
    ASSERT_EQ(ve_panel_create(data.data() + 5, 3, 2, &panel), VE_OK);
    ASSERT_EQ(ve_panel_set_strides(panel, -2, -1), VE_OK);
    EXPECT_EQ(panel->view.at(0, 0), 5.0);
    EXPECT_EQ(panel->view.at(0, 1), 4.0);
    EXPECT_EQ(panel->view.at(2, 1), 0.0);
    ve_panel_free(panel);
}

TEST(CapiPanel, RangesOutOfBoundsRejected)
{
    const std::vector<double> data(10, 0.0);
    ve_panel* panel = make_panel(data, 10, 1);
    const size_t starts[] = {5};
    const size_t lengths[] = {6};
    EXPECT_EQ(ve_panel_set_ranges(panel, starts, lengths), VE_INVALID_ARGUMENT);
    EXPECT_FALSE(panel->view.has_ranges());
    const size_t huge_lengths[] = {kMax};
    EXPECT_EQ(ve_panel_set_ranges(panel, starts, huge_lengths), VE_INVALID_ARGUMENT);
    ve_panel_free(panel);
}

TEST(CapiPanel, RangesErrorLeavesPreviousRangesUnchanged)
{
    const std::vector<double> data(20, 0.0);
    ve_panel* panel = make_panel(data, 10, 2);
    const size_t starts[] = {0, 2};
    const size_t lengths[] = {10, 8};
    ASSERT_EQ(ve_panel_set_ranges(panel, starts, lengths), VE_OK);
    const size_t bad_lengths[] = {10, 9};
    EXPECT_EQ(ve_panel_set_ranges(panel, starts, bad_lengths), VE_INVALID_ARGUMENT);
    EXPECT_EQ(panel->view.series_range(1), std::make_pair(size_t{2}, size_t{8}));
    ve_panel_free(panel);
}

TEST(CapiPanel, RangesCopiedAndClearable)
{
    const std::vector<double> data(20, 0.0);
    ve_panel* panel = make_panel(data, 10, 2);
    EXPECT_EQ(panel->view.series_range(1), std::make_pair(size_t{0}, size_t{10}));
    std::vector<size_t> starts = {1, 3};
    std::vector<size_t> lengths = {9, 7};
    ASSERT_EQ(ve_panel_set_ranges(panel, starts.data(), lengths.data()), VE_OK);
    starts[0] = 99;
    EXPECT_EQ(panel->view.series_range(0), std::make_pair(size_t{1}, size_t{9}));
    EXPECT_EQ(panel->view.series_range(1), std::make_pair(size_t{3}, size_t{7}));
    ASSERT_EQ(ve_panel_set_ranges(panel, nullptr, nullptr), VE_OK);
    EXPECT_FALSE(panel->view.has_ranges());
    ve_panel_free(panel);
}

TEST(CapiPanel, RangesWithOneNullRejected)
{
    const std::vector<double> data(10, 0.0);
    ve_panel* panel = make_panel(data, 10, 1);
    const size_t values[] = {0};
    EXPECT_EQ(ve_panel_set_ranges(panel, values, nullptr), VE_INVALID_ARGUMENT);
    EXPECT_EQ(ve_panel_set_ranges(panel, nullptr, values), VE_INVALID_ARGUMENT);
    ve_panel_free(panel);
}
