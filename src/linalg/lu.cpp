#include "linalg/lu.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ve::linalg
{

namespace
{

std::size_t pivot_row(std::span<const double> a, std::size_t n, std::size_t column)
{
    std::size_t best = column;
    for (std::size_t row = column + 1; row < n; ++row)
    {
        if (std::abs(a[row * n + column]) > std::abs(a[best * n + column]))
        {
            best = row;
        }
    }
    return best;
}

void eliminate_below(std::span<double> a, std::size_t n, std::size_t column)
{
    const double pivot = a[column * n + column];
    for (std::size_t row = column + 1; row < n; ++row)
    {
        const double multiplier = a[row * n + column] / pivot;
        a[row * n + column] = multiplier;
        for (std::size_t k = column + 1; k < n; ++k)
        {
            a[row * n + k] -= multiplier * a[column * n + k];
        }
    }
}

}

bool lu_factor(std::span<double> a, std::span<std::size_t> pivots, std::size_t n)
{
    for (std::size_t column = 0; column < n; ++column)
    {
        const std::size_t pivot = pivot_row(a, n, column);
        pivots[column] = pivot;
        const double value = a[pivot * n + column];
        if (value == 0.0 || !std::isfinite(value))
        {
            return false;
        }
        std::swap_ranges(a.begin() + static_cast<std::ptrdiff_t>(column * n),
                         a.begin() + static_cast<std::ptrdiff_t>((column + 1) * n),
                         a.begin() + static_cast<std::ptrdiff_t>(pivot * n));
        eliminate_below(a, n, column);
    }
    return true;
}

void lu_solve(std::span<const double> factor, std::span<const std::size_t> pivots, std::size_t n,
              std::span<double> b)
{
    for (std::size_t row = 0; row < n; ++row)
    {
        std::swap(b[row], b[pivots[row]]);
        for (std::size_t k = 0; k < row; ++k)
        {
            b[row] -= factor[row * n + k] * b[k];
        }
    }
    for (std::size_t row = n; row-- > 0;)
    {
        for (std::size_t k = row + 1; k < n; ++k)
        {
            b[row] -= factor[row * n + k] * b[k];
        }
        b[row] /= factor[row * n + row];
    }
}

}
