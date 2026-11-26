#include "risk/measures.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <vector>

namespace ve::risk
{
namespace
{

std::vector<double> portfolio_returns(const PanelView& returns, std::span<const double> weights, ReturnKind kind)
{
    std::vector<double> portfolio(returns.t, 0.0);
    for (std::size_t row = 0; row < returns.t; ++row)
    {
        for (std::size_t series = 0; series < returns.n; ++series)
        {
            const double value = returns.at(row, series);
            const double simple = kind == ReturnKind::log ? std::expm1(value) : value;
            portfolio[row] += weights[series] * simple;
        }
    }
    return portfolio;
}

// numpy.quantile's linear interpolation (numpy's _lerp), digit for digit; where b - a overflows, the weighted sum
// of the two, which lies between them.
double interpolate(double a, double b, double t)
{
    const double difference = b - a;
    if (!std::isfinite(difference))
    {
        return (1.0 - t) * a + t * b;
    }
    if (t >= 0.5)
    {
        return b - difference * (1.0 - t);
    }
    return a + difference * t;
}

// The mean of the sorted returns at or below the quantile; where their sum overflows, the sum of each over the
// count, whose magnitude is at most the largest of them.
double mean_at_or_below(std::span<const double> sorted, double quantile)
{
    std::size_t count = 0;
    double tail = 0.0;
    for (; count < sorted.size() && sorted[count] <= quantile; ++count)
    {
        tail += sorted[count];
    }
    if (std::isfinite(tail))
    {
        return tail / static_cast<double>(count);
    }
    double mean = 0.0;
    for (std::size_t k = 0; k < count; ++k)
    {
        mean += sorted[k] / static_cast<double>(count);
    }
    return mean;
}

}

Measure empirical(const PanelView& returns, std::span<const double> weights, ReturnKind kind, double confidence)
{
    std::vector<double> sorted = portfolio_returns(returns, weights, kind);
    // Finite inputs can still overflow (exp(710), 1e308 * 10); NaN would break the sort's ordering.
    if (!std::ranges::all_of(sorted, [](double value) { return std::isfinite(value); }))
    {
        return Measure{.var = std::numeric_limits<double>::quiet_NaN(), .es = std::numeric_limits<double>::quiet_NaN()};
    }
    std::ranges::sort(sorted);
    const double position = static_cast<double>(sorted.size() - 1) * (1.0 - confidence);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double quantile = interpolate(sorted[lower], sorted[upper], position - static_cast<double>(lower));
    return Measure{.var = -quantile, .es = -mean_at_or_below(sorted, quantile)};
}

}
