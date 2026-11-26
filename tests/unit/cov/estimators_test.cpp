#include "cov/estimators.hpp"
#include "core/panel.hpp"
#include "simd/fixtures.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace
{

// A row-major t x n panel over `data`.
ve::PanelView row_major(const std::vector<double>& data, std::size_t t, std::size_t n)
{
    ve::PanelView view;
    view.data = data.data();
    view.t = t;
    view.n = n;
    view.stride_t = static_cast<std::ptrdiff_t>(n);
    view.stride_n = 1;
    return view;
}

// t x n correlated values: series j mixes the shocks of series 0..j.
std::vector<double> correlated(std::size_t t, std::size_t n, unsigned seed)
{
    ve::test::Stream stream(seed);
    const auto shocks = stream.vector(t * n);
    std::vector<double> data(t * n, 0.0);
    for (std::size_t row = 0; row < t; ++row)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            for (std::size_t k = 0; k <= j; ++k)
            {
                data[row * n + j] += shocks[row * n + k] / static_cast<double>(k + 1);
            }
        }
    }
    return data;
}

double max_abs(const std::vector<double>& values)
{
    double largest = 0.0;
    for (double value : values)
    {
        largest = std::max(largest, std::abs(value));
    }
    return largest;
}

void expect_symmetric(const std::vector<double>& matrix, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            EXPECT_EQ(matrix[i * n + j], matrix[j * n + i]) << i << "," << j;
        }
    }
}

void center(std::vector<double>& z, std::size_t t, std::size_t n)
{
    for (std::size_t j = 0; j < n; ++j)
    {
        double mean = 0.0;
        for (std::size_t row = 0; row < t; ++row)
        {
            mean += z[row * n + j];
        }
        mean /= static_cast<double>(t);
        for (std::size_t row = 0; row < t; ++row)
        {
            z[row * n + j] -= mean;
        }
    }
}

// Ledoit-Wolf written straight from its definition (row-major t x n panel).
ve::cov::Shrunk ledoit_wolf_by_hand(std::vector<double> z, std::size_t t, std::size_t n)
{
    center(z, t, n);
    std::vector<double> e(n * n, 0.0);
    double fourth = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            double cross = 0.0;
            double squares = 0.0;
            for (std::size_t row = 0; row < t; ++row)
            {
                cross += z[row * n + i] * z[row * n + j];
                squares += z[row * n + i] * z[row * n + i] * z[row * n + j] * z[row * n + j];
            }
            e[i * n + j] = cross / static_cast<double>(t);
            fourth += squares;
        }
    }
    double trace = 0.0;
    double frobenius = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        trace += e[i * n + i];
    }
    for (double value : e)
    {
        frobenius += value * value;
    }
    const double mu = trace / static_cast<double>(n);
    const double nt = static_cast<double>(n * t);
    const double beta = (fourth / static_cast<double>(t) - frobenius) / nt;
    double delta = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            const double difference = e[i * n + j] - (i == j ? mu : 0.0);
            delta += difference * difference;
        }
    }
    delta /= static_cast<double>(n);
    ve::cov::Shrunk result;
    result.shrinkage = std::min(beta, delta) / delta;
    result.covariance = e;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            const double target = i == j ? mu : 0.0;
            result.covariance[i * n + j] = (1.0 - result.shrinkage) * e[i * n + j] + result.shrinkage * target;
        }
    }
    return result;
}

TEST(CovSample, HandValues)
{
    const std::vector<double> data{1.0, 2.0, 3.0, 5.0, 5.0, 11.0};
    const auto covariance = ve::cov::sample(row_major(data, 3, 2));
    EXPECT_EQ(covariance, (std::vector<double>{4.0, 9.0, 9.0, 21.0}));
}

TEST(CovSample, ExactlySymmetric)
{
    const auto data = correlated(200, 7, 3);
    expect_symmetric(ve::cov::sample(row_major(data, 200, 7)), 7);
}

TEST(CovSample, StridedEqualsContiguous)
{
    constexpr std::size_t t = 120;
    constexpr std::size_t n = 5;
    const auto data = correlated(t, n, 4);
    std::vector<double> columns(t * n);
    for (std::size_t row = 0; row < t; ++row)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            columns[j * t + row] = data[row * n + j];
        }
    }
    ve::PanelView column_major = row_major(columns, t, n);
    column_major.stride_t = 1;
    column_major.stride_n = static_cast<std::ptrdiff_t>(t);
    EXPECT_EQ(ve::cov::sample(column_major), ve::cov::sample(row_major(data, t, n)));
}

TEST(CovSample, LargeOffset)
{
    constexpr std::size_t t = 500;
    constexpr std::size_t n = 3;
    auto data = correlated(t, n, 5);
    for (double& value : data)
    {
        value *= 1e-2;
    }
    const auto plain = ve::cov::sample(row_major(data, t, n));
    for (std::size_t row = 0; row < t; ++row)
    {
        data[row * n + 1] += 1e6;
    }
    const auto shifted = ve::cov::sample(row_major(data, t, n));
    const double scale = max_abs(plain);
    for (std::size_t k = 0; k < plain.size(); ++k)
    {
        EXPECT_NEAR(shifted[k], plain[k], 1e-9 * scale) << k;
    }
}

TEST(CovLedoitWolf, SingleSeries)
{
    const std::vector<double> data{1.0, 4.0, 2.0, 7.0};
    const auto result = ve::cov::ledoit_wolf(row_major(data, 4, 1));
    EXPECT_EQ(result.shrinkage, 0.0);
    // mean 3.5, squared deviations 6.25 + 0.25 + 2.25 + 12.25 = 21, divided by t = 4
    ASSERT_EQ(result.covariance.size(), 1U);
    EXPECT_DOUBLE_EQ(result.covariance[0], 5.25);
}

TEST(CovLedoitWolf, ConstantSeries)
{
    const std::vector<double> data(30 * 3, 2.5);
    const auto result = ve::cov::ledoit_wolf(row_major(data, 30, 3));
    EXPECT_EQ(result.shrinkage, 0.0);
    EXPECT_EQ(result.covariance, std::vector<double>(9, 0.0));
}

TEST(CovLedoitWolf, MatchesFormula)
{
    constexpr std::size_t t = 50;
    constexpr std::size_t n = 4;
    const auto data = correlated(t, n, 6);
    const auto result = ve::cov::ledoit_wolf(row_major(data, t, n));
    const auto expected = ledoit_wolf_by_hand(data, t, n);
    EXPECT_GE(result.shrinkage, 0.0);
    EXPECT_LE(result.shrinkage, 1.0);
    EXPECT_NEAR(result.shrinkage, expected.shrinkage, 1e-14 * expected.shrinkage);
    const double scale = max_abs(expected.covariance);
    ASSERT_EQ(result.covariance.size(), n * n);
    for (std::size_t k = 0; k < n * n; ++k)
    {
        EXPECT_NEAR(result.covariance[k], expected.covariance[k], 1e-14 * scale) << k;
    }
}

TEST(CovLedoitWolf, ExactlySymmetric)
{
    const auto data = correlated(80, 9, 7);
    expect_symmetric(ve::cov::ledoit_wolf(row_major(data, 80, 9)).covariance, 9);
}

}
