#include "linalg/least_squares.hpp"

#include <cmath>
#include <cstddef>
#include <gtest/gtest.h>
#include <vector>

namespace
{

constexpr std::size_t kRows = 12;

double noise(std::size_t t)
{
    return 0.05 * std::sin(2.3 * static_cast<double>(t) + 0.4);
}

std::vector<double> design_of_ones_and_x(std::size_t copies_of_ones)
{
    std::vector<double> a;
    for (std::size_t t = 0; t < kRows; ++t)
    {
        for (std::size_t c = 0; c < copies_of_ones; ++c)
        {
            a.push_back(1.0);
        }
        a.push_back(0.3 * static_cast<double>(t) - 1.0 + 0.2 * std::cos(static_cast<double>(t)));
    }
    return a;
}

std::vector<double> response_of(const std::vector<double>& a, std::size_t columns)
{
    std::vector<double> y(kRows);
    for (std::size_t t = 0; t < kRows; ++t)
    {
        y[t] = 1.0 + 2.0 * a[t * columns + columns - 1] + noise(t);
    }
    return y;
}

TEST(LeastSquares, MatchesClosedForm)
{
    std::vector<double> a = design_of_ones_and_x(1);
    std::vector<double> y = response_of(a, 2);
    double sx = 0.0;
    double sy = 0.0;
    double sxx = 0.0;
    double sxy = 0.0;
    for (std::size_t t = 0; t < kRows; ++t)
    {
        double x = a[t * 2 + 1];
        sx += x;
        sy += y[t];
        sxx += x * x;
        sxy += x * y[t];
    }
    double n = static_cast<double>(kRows);
    double det = n * sxx - sx * sx;
    std::vector<double> b = ve::linalg::least_squares(a, 2, y);
    ASSERT_EQ(b.size(), 2u);
    EXPECT_NEAR(b[0], (sxx * sy - sx * sxy) / det, 1e-12);
    EXPECT_NEAR(b[1], (n * sxy - sx * sy) / det, 1e-12);
}

TEST(LeastSquares, MinimumNormWhenCollinear)
{
    std::vector<double> regular = design_of_ones_and_x(1);
    std::vector<double> collinear = design_of_ones_and_x(2);
    std::vector<double> y = response_of(regular, 2);
    std::vector<double> reference = ve::linalg::least_squares(regular, 2, y);
    std::vector<double> b = ve::linalg::least_squares(collinear, 3, y);
    ASSERT_EQ(b.size(), 3u);
    for (double value : b)
    {
        ASSERT_TRUE(std::isfinite(value));
    }
    EXPECT_NEAR(b[0], b[1], 1e-12);
    for (std::size_t t = 0; t < kRows; ++t)
    {
        double fitted = b[0] + b[1] + b[2] * collinear[t * 3 + 2];
        EXPECT_NEAR(fitted, reference[0] + reference[1] * regular[t * 2 + 1], 1e-10);
    }
}

TEST(LeastSquares, ZeroColumnGetsZeroCoefficient)
{
    std::vector<double> a;
    std::vector<double> y;
    for (std::size_t t = 0; t < kRows; ++t)
    {
        double x = static_cast<double>(t);
        a.insert(a.end(), {0.0, x});
        y.push_back(3.0 * x);
    }
    std::vector<double> b = ve::linalg::least_squares(a, 2, y);
    EXPECT_EQ(b[0], 0.0);
    EXPECT_NEAR(b[1], 3.0, 1e-12);
}

}
