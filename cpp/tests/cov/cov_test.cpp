#include "../fixture_data.hpp"

#include <cstddef>
#include <gtest/gtest.h>
#include <vector>
#include <volengine/volengine.hpp>

namespace
{

// copula3.csv as a row-major T x 3 panel.
std::vector<double> copula3_rows()
{
    const auto columns = ve::testing::read_csv_columns("copula3.csv");
    std::vector<double> rows;
    for (std::size_t row = 0; row < columns[0].size(); ++row)
    {
        for (const auto& column : columns)
        {
            rows.push_back(column[row]);
        }
    }
    return rows;
}

TEST(Cov, SampleMatchesCapi)
{
    const auto data = copula3_rows();
    const auto panel = ve::Panel::create(data, data.size() / 3, 3).value();
    std::vector<double> expected(9);
    ASSERT_EQ(ve_cov_sample(panel.native(), expected.data()), VE_OK);
    EXPECT_EQ(ve::cov::sample(panel).value(), expected);
}

TEST(Cov, LedoitWolfMatchesCapi)
{
    const auto data = copula3_rows();
    const auto panel = ve::Panel::create(data, data.size() / 3, 3).value();
    std::vector<double> expected(9);
    double shrinkage = 0.0;
    ASSERT_EQ(ve_cov_ledoit_wolf(panel.native(), expected.data(), &shrinkage), VE_OK);
    const auto result = ve::cov::ledoit_wolf(panel).value();
    EXPECT_EQ(result.covariance, expected);
    EXPECT_EQ(result.shrinkage, shrinkage);
}

TEST(Cov, RangesAreAnError)
{
    const auto data = copula3_rows();
    auto panel = ve::Panel::create(data, data.size() / 3, 3).value();
    const std::vector<std::size_t> starts(3, 0);
    const std::vector<std::size_t> lengths(3, 10);
    ASSERT_TRUE(panel.set_ranges(starts, lengths).has_value());
    const auto sample = ve::cov::sample(panel);
    ASSERT_FALSE(sample.has_value());
    EXPECT_EQ(sample.error().status, ve::Status::invalid_argument);
    EXPECT_FALSE(sample.error().message.empty());
    EXPECT_FALSE(ve::cov::ledoit_wolf(panel).has_value());
}

}
