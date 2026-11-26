#include "cov/estimators.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <span>

namespace ve::cov
{
namespace
{

// The panel centered, series j contiguous at [j * t, (j + 1) * t).
std::vector<double> centered_columns(const PanelView& panel)
{
    std::vector<double> columns(panel.t * panel.n);
    for (std::size_t j = 0; j < panel.n; ++j)
    {
        double sum = 0.0;
        for (std::size_t row = 0; row < panel.t; ++row)
        {
            columns[j * panel.t + row] = panel.at(row, j);
            sum += columns[j * panel.t + row];
        }
        const double mean = sum / static_cast<double>(panel.t);
        for (std::size_t row = 0; row < panel.t; ++row)
        {
            columns[j * panel.t + row] -= mean;
        }
    }
    return columns;
}

// (columns' columns) / divisor for n columns of length t: the upper triangle computed and mirrored.
std::vector<double> cross_products(std::span<const double> columns, std::size_t t, std::size_t n, double divisor)
{
    std::vector<double> products(n * n);
    for (std::size_t i = 0; i < n; ++i)
    {
        const auto left = columns.subspan(i * t, t);
        for (std::size_t j = i; j < n; ++j)
        {
            const auto right = columns.subspan(j * t, t);
            const double value = std::inner_product(left.begin(), left.end(), right.begin(), 0.0) / divisor;
            products[i * n + j] = value;
            products[j * n + i] = value;
        }
    }
    return products;
}

double trace(std::span<const double> matrix, std::size_t n)
{
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        total += matrix[i * n + i];
    }
    return total;
}

// ||E - mu I||^2 / n, a sum of squares, so never below zero and exactly zero for one series.
double distance_to_target(std::span<const double> empirical, std::size_t n, double mu)
{
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            const double target = i == j ? mu : 0.0;
            const double difference = empirical[i * n + j] - target;
            total += difference * difference;
        }
    }
    return total / static_cast<double>(n);
}

double intensity(std::span<const double> columns, std::span<const double> empirical, std::size_t t, std::size_t n)
{
    const double delta = distance_to_target(empirical, n, trace(empirical, n) / static_cast<double>(n));
    if (delta <= 0.0)
    {
        return 0.0;
    }
    std::vector<double> squared(columns.begin(), columns.end());
    for (double& value : squared)
    {
        value *= value;
    }
    const auto fourth = cross_products(squared, t, n, 1.0);
    const double beta_prime = std::accumulate(fourth.begin(), fourth.end(), 0.0);
    const double delta_prime = std::inner_product(empirical.begin(), empirical.end(), empirical.begin(), 0.0);
    const double beta = (beta_prime / static_cast<double>(t) - delta_prime) / static_cast<double>(n * t);
    return std::min(beta, delta) / delta;
}

}

std::vector<double> sample(const PanelView& panel)
{
    const auto columns = centered_columns(panel);
    return cross_products(columns, panel.t, panel.n, static_cast<double>(panel.t - 1));
}

Shrunk ledoit_wolf(const PanelView& panel)
{
    const auto columns = centered_columns(panel);
    Shrunk result;
    result.covariance = cross_products(columns, panel.t, panel.n, static_cast<double>(panel.t));
    result.shrinkage = intensity(columns, result.covariance, panel.t, panel.n);
    const double mu = trace(result.covariance, panel.n) / static_cast<double>(panel.n);
    const double keep = 1.0 - result.shrinkage;
    for (std::size_t i = 0; i < panel.n; ++i)
    {
        for (std::size_t j = 0; j < panel.n; ++j)
        {
            const double target = i == j ? mu : 0.0;
            result.covariance[i * panel.n + j] = keep * result.covariance[i * panel.n + j] + result.shrinkage * target;
        }
    }
    return result;
}

}
