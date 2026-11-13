#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>

namespace ve::linalg
{

// Row-major n x n symmetric matrix; reads and overwrites the lower triangle with L, where A = L L^T.
// Returns false, leaving the contents unspecified, unless A is positive definite.
bool cholesky_factor(std::span<double> a, std::size_t n);

// Solves A x = b in place given the factor from cholesky_factor.
void cholesky_solve(std::span<const double> factor, std::size_t n, std::span<double> b);

// The same three operations for T = double or ad::Dual, which needs only arithmetic, comparisons, sqrt and log
// found by argument-dependent lookup. On double they are the functions above, to the bit.
namespace detail
{

template <class T>
T factor_entry(std::span<const T> a, std::size_t n, std::size_t row, std::size_t col)
{
    T sum = a[row * n + col];
    for (std::size_t m = 0; m < col; ++m)
    {
        sum = sum - a[row * n + m] * a[col * n + m];
    }
    return sum;
}

}

template <class T>
bool cholesky_factor(std::span<T> a, std::size_t n)
{
    using std::sqrt;
    assert(a.size() == n * n);
    const T zero{0.0};
    const T infinity{std::numeric_limits<double>::infinity()};
    for (std::size_t col = 0; col < n; ++col)
    {
        const T pivot = detail::factor_entry<T>(a, n, col, col);
        if (!(pivot > zero && pivot < infinity))
        {
            return false;
        }
        a[col * n + col] = sqrt(pivot);
        for (std::size_t row = col + 1; row < n; ++row)
        {
            a[row * n + col] = detail::factor_entry<T>(a, n, row, col) / a[col * n + col];
        }
    }
    return true;
}

template <class T>
void cholesky_solve(std::span<const T> factor, std::size_t n, std::span<T> b)
{
    assert(factor.size() == n * n && b.size() == n);
    for (std::size_t row = 0; row < n; ++row)
    {
        T sum = b[row];
        for (std::size_t m = 0; m < row; ++m)
        {
            sum = sum - factor[row * n + m] * b[m];
        }
        b[row] = sum / factor[row * n + row];
    }
    for (std::size_t row = n; row-- > 0;)
    {
        T sum = b[row];
        for (std::size_t m = row + 1; m < n; ++m)
        {
            sum = sum - factor[m * n + row] * b[m];
        }
        b[row] = sum / factor[row * n + row];
    }
}

// log |A| = 2 sum log L_ii, given the factor from cholesky_factor.
template <class T>
T cholesky_log_determinant(std::span<const T> factor, std::size_t n)
{
    using std::log;
    T sum{0.0};
    for (std::size_t i = 0; i < n; ++i)
    {
        sum = sum + log(factor[i * n + i]);
    }
    return 2.0 * sum;
}

}
