#include "fit/nyblom.hpp"

#include "linalg/cholesky.hpp"

#include <algorithm>
#include <cassert>

namespace ve::fit
{

namespace
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct ScoreProducts
{
    std::vector<double> cumulative;
    std::vector<double> a;
    std::vector<double> b;
};

ScoreProducts score_products(std::span<const double> scores, std::size_t k,
                             std::span<const std::size_t> columns)
{
    const std::size_t m = columns.size();
    ScoreProducts products{std::vector<double>(m), std::vector<double>(m * m), std::vector<double>(m * m)};
    for (std::size_t offset = 0; offset < scores.size(); offset += k)
    {
        for (std::size_t i = 0; i < m; ++i)
        {
            products.cumulative[i] += scores[offset + columns[i]];
        }
        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j <= i; ++j)
            {
                products.a[i * m + j] += products.cumulative[i] * products.cumulative[j];
                products.b[i * m + j] += scores[offset + columns[i]] * scores[offset + columns[j]];
            }
        }
    }
    return products;
}

double joint_statistic(ScoreProducts& products, std::size_t m, double n)
{
    std::vector<double> diagonal(m);
    for (std::size_t i = 0; i < m; ++i)
    {
        diagonal[i] = products.b[i * m + i];
    }
    if (!linalg::cholesky_factor(products.b, m))
    {
        return kNaN;
    }
    for (std::size_t i = 0; i < m; ++i)
    {
        const double pivot = products.b[i * m + i];
        const double tolerance = std::numeric_limits<double>::epsilon() * static_cast<double>(m) * diagonal[i];
        if (pivot * pivot <= tolerance)
        {
            return kNaN;
        }
    }
    double trace = 0.0;
    std::vector<double> column(m);
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < m; ++j)
        {
            const std::size_t row = std::max(i, j);
            const std::size_t col = std::min(i, j);
            column[j] = products.a[row * m + col];
        }
        linalg::cholesky_solve(products.b, m, column);
        trace += column[i];
    }
    return trace / n;
}

}

Nyblom nyblom_statistics(std::span<const double> scores, std::size_t k, std::span<const bool> estimated)
{
    assert(estimated.size() == k);
    assert(k == 0 || scores.size() % k == 0);
    Nyblom result{std::vector<double>(k, kNaN)};
    std::vector<std::size_t> columns;
    for (std::size_t i = 0; i < k; ++i)
    {
        if (estimated[i])
        {
            columns.push_back(i);
        }
    }
    result.estimated = columns.size();
    if (columns.empty() || scores.empty())
    {
        return result;
    }
    const double n = static_cast<double>(scores.size() / k);
    ScoreProducts products = score_products(scores, k, columns);
    const std::size_t m = columns.size();
    for (std::size_t i = 0; i < m; ++i)
    {
        if (products.b[i * m + i] > 0.0)
        {
            result.individual[columns[i]] = products.a[i * m + i] / (n * products.b[i * m + i]);
        }
    }
    result.joint = joint_statistic(products, m, n);
    return result;
}

std::optional<std::array<double, 3>> nyblom_critical(std::size_t parameters)
{
    // Hansen's asymptotic 10%, 5%, and 1% critical values.
    static constexpr std::array<std::array<double, 3>, 20> values{{
        {0.353, 0.470, 0.748}, {0.610, 0.749, 1.07}, {0.846, 1.01, 1.35}, {1.07, 1.24, 1.60},
        {1.28, 1.47, 1.88}, {1.49, 1.68, 2.12}, {1.69, 1.90, 2.35}, {1.89, 2.11, 2.59},
        {2.10, 2.32, 2.82}, {2.29, 2.54, 3.05}, {2.49, 2.75, 3.27}, {2.69, 2.96, 3.51},
        {2.89, 3.15, 3.69}, {3.08, 3.34, 3.90}, {3.26, 3.54, 4.07}, {3.46, 3.75, 4.30},
        {3.64, 3.95, 4.51}, {3.83, 4.14, 4.73}, {4.03, 4.33, 4.92}, {4.22, 4.52, 5.13}}};
    if (parameters == 0 || parameters > values.size())
    {
        return std::nullopt;
    }
    return values[parameters - 1];
}

}
