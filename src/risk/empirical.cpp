#include "risk/measures.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
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

}

Measure empirical(const PanelView& returns, std::span<const double> weights, ReturnKind kind, double confidence)
{
    std::vector<double> sorted = portfolio_returns(returns, weights, kind);
    std::ranges::sort(sorted);
    const double position = static_cast<double>(sorted.size() - 1) * (1.0 - confidence);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const std::size_t upper = std::min(lower + 1, sorted.size() - 1);
    const double quantile = sorted[lower] + (position - static_cast<double>(lower)) * (sorted[upper] - sorted[lower]);
    double tail = 0.0;
    std::size_t count = 0;
    for (; count < sorted.size() && sorted[count] <= quantile; ++count)
    {
        tail += sorted[count];
    }
    return Measure{.var = -quantile, .es = -tail / static_cast<double>(count)};
}

}
