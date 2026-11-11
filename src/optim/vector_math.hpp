#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <span>

namespace ve::optim::detail
{

inline double dot(std::span<const double> a, std::span<const double> b)
{
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

inline double max_abs(std::span<const double> values)
{
    double result = 0.0;
    for (double value : values)
    {
        result = std::max(result, std::abs(value));
    }
    return result;
}

inline bool all_finite(std::span<const double> values)
{
    return std::ranges::all_of(values, [](double value) { return std::isfinite(value); });
}

}
