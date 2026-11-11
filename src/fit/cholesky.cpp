#include "fit/cholesky.hpp"

#include <cassert>
#include <cmath>

namespace ve::fit
{

namespace
{

double factor_entry(std::span<const double> a, std::size_t n, std::size_t row, std::size_t col)
{
    double sum = a[row * n + col];
    for (std::size_t m = 0; m < col; ++m)
    {
        sum -= a[row * n + m] * a[col * n + m];
    }
    return sum;
}

}

bool cholesky_factor(std::span<double> a, std::size_t n)
{
    assert(a.size() == n * n);
    for (std::size_t col = 0; col < n; ++col)
    {
        double pivot = factor_entry(a, n, col, col);
        if (!(pivot > 0.0) || !std::isfinite(pivot))
        {
            return false;
        }
        a[col * n + col] = std::sqrt(pivot);
        for (std::size_t row = col + 1; row < n; ++row)
        {
            a[row * n + col] = factor_entry(a, n, row, col) / a[col * n + col];
        }
    }
    return true;
}

void cholesky_solve(std::span<const double> factor, std::size_t n, std::span<double> b)
{
    assert(factor.size() == n * n && b.size() == n);
    for (std::size_t row = 0; row < n; ++row)
    {
        double sum = b[row];
        for (std::size_t m = 0; m < row; ++m)
        {
            sum -= factor[row * n + m] * b[m];
        }
        b[row] = sum / factor[row * n + row];
    }
    for (std::size_t row = n; row-- > 0;)
    {
        double sum = b[row];
        for (std::size_t m = row + 1; m < n; ++m)
        {
            sum -= factor[m * n + row] * b[m];
        }
        b[row] = sum / factor[row * n + row];
    }
}

}
