#include "diagnostics/sign_bias.hpp"

#include "diagnostics/tails.hpp"
#include "linalg/cholesky.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace ve::diagnostics
{

namespace
{

constexpr std::size_t kColumns = 4;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

double finite_or_zero(double value)
{
    if (std::isfinite(value))
    {
        return value;
    }
    return 0.0;
}

std::array<double, kColumns> regressors(double residual)
{
    if (residual < 0.0)
    {
        return {1.0, 1.0, residual, 0.0};
    }
    return {1.0, 0.0, 0.0, residual};
}

struct Regression
{
    std::array<double, 16> gram{};
    std::array<double, 4> rhs{};
};

Regression normal_equations(std::span<const double> z, std::span<const double> residuals)
{
    Regression regression;
    for (std::size_t t = 1; t < z.size(); ++t)
    {
        auto x = regressors(residuals[t - 1]);
        double response = finite_or_zero(z[t]);
        response *= response;
        for (std::size_t i = 0; i < kColumns; ++i)
        {
            regression.rhs[i] += x[i] * response;
            for (std::size_t j = 0; j < kColumns; ++j)
            {
                regression.gram[i * kColumns + j] += x[i] * x[j];
            }
        }
    }
    return regression;
}

double residual_variance(std::span<const double> z, std::span<const double> residuals,
                         const std::array<double, 4>& coefficients)
{
    double rss = 0.0;
    for (std::size_t t = 1; t < z.size(); ++t)
    {
        auto x = regressors(residuals[t - 1]);
        double value = finite_or_zero(z[t]);
        double error = value * value - std::inner_product(x.begin(), x.end(), coefficients.begin(), 0.0);
        rss += error * error;
    }
    return rss / static_cast<double>(z.size() - 5);
}

TestResult coefficient_test(const std::array<double, 16>& factor, std::size_t index,
                            double coefficient, double variance, double degrees)
{
    std::array<double, 4> unit{};
    unit[index] = 1.0;
    linalg::cholesky_solve(factor, kColumns, unit);
    double statistic = std::abs(coefficient) / std::sqrt(variance * unit[index]);
    return {statistic, student_two_sided(statistic, degrees)};
}

TestResult joint_test(const Regression& regression, const std::array<double, 4>& coefficients, double variance)
{
    double statistic = 0.0;
    for (std::size_t i = 1; i < kColumns; ++i)
    {
        for (std::size_t j = 1; j < kColumns; ++j)
        {
            double schur = regression.gram[i * kColumns + j] -
                           regression.gram[i * kColumns] * regression.gram[j] / regression.gram[0];
            statistic += coefficients[i] * schur * coefficients[j];
        }
    }
    statistic /= variance;
    return {statistic, chi_squared_upper(statistic, 3.0)};
}

}

SignBias sign_bias(std::span<const double> z, std::span<const double> residuals)
{
    Regression regression = normal_equations(z, residuals);
    std::array<double, 16> factor = regression.gram;
    if (!linalg::cholesky_factor(factor, kColumns))
    {
        return {{kNaN, kNaN}, {kNaN, kNaN}, {kNaN, kNaN}, {kNaN, kNaN}};
    }
    std::array<double, 4> coefficients = regression.rhs;
    linalg::cholesky_solve(factor, kColumns, coefficients);
    double variance = residual_variance(z, residuals, coefficients);
    double degrees = static_cast<double>(z.size() - 5);
    return {coefficient_test(factor, 1, coefficients[1], variance, degrees),
            coefficient_test(factor, 2, coefficients[2], variance, degrees),
            coefficient_test(factor, 3, coefficients[3], variance, degrees),
            joint_test(regression, coefficients, variance)};
}

}
