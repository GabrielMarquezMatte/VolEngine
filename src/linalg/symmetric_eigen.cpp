#include "linalg/symmetric_eigen.hpp"

#include "linalg/cholesky.hpp"

#include <cassert>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace ve::linalg
{

namespace
{

constexpr double kTolerance = 1e-14;
constexpr int kMaxIterations = 100000;

// Overwrites the row-major b with L^-1 b, L the lower factor.
void forward_solve_columns(std::span<const double> factor, std::span<double> b, std::size_t size)
{
    for (std::size_t row = 0; row < size; ++row)
    {
        for (std::size_t k = 0; k < row; ++k)
        {
            for (std::size_t col = 0; col < size; ++col)
            {
                b[row * size + col] -= factor[row * size + k] * b[k * size + col];
            }
        }
        for (std::size_t col = 0; col < size; ++col)
        {
            b[row * size + col] /= factor[row * size + row];
        }
    }
}

void transpose_in_place(std::span<double> a, std::size_t size)
{
    for (std::size_t i = 0; i < size; ++i)
    {
        for (std::size_t j = i + 1; j < size; ++j)
        {
            std::swap(a[i * size + j], a[j * size + i]);
        }
    }
}

std::vector<double> multiply(std::span<const double> a, std::span<const double> x, std::size_t size)
{
    std::vector<double> y(size, 0.0);
    for (std::size_t i = 0; i < size; ++i)
    {
        for (std::size_t j = 0; j < size; ++j)
        {
            y[i] += a[i * size + j] * x[j];
        }
    }
    return y;
}

double norm_of(std::span<const double> x)
{
    double sum = 0.0;
    for (const double v : x)
    {
        sum += v * v;
    }
    return std::sqrt(sum);
}

double power_iteration(std::span<const double> m, std::size_t size)
{
    std::vector<double> x(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        x[i] = 1.0 + 0.37 * static_cast<double>(i);
    }
    const double start = norm_of(x);
    for (double& v : x)
    {
        v /= start;
    }
    double lambda = 0.0;
    for (int iteration = 0; iteration < kMaxIterations; ++iteration)
    {
        const std::vector<double> y = multiply(m, x, size);
        const double next = norm_of(y);
        if (next == 0.0)
        {
            return 0.0;
        }
        for (std::size_t i = 0; i < size; ++i)
        {
            x[i] = y[i] / next;
        }
        const bool done = std::abs(next - lambda) <= kTolerance * next;
        lambda = next;
        if (done)
        {
            break;
        }
    }
    return lambda;
}

}

double largest_relative_eigenvalue(std::span<const double> q, std::span<const double> n, std::size_t size)
{
    assert(q.size() == size * size && n.size() == size * size);
    std::vector<double> factor(q.begin(), q.end());
    if (!cholesky_factor(factor, size))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> m(n.begin(), n.end());
    forward_solve_columns(factor, m, size);
    transpose_in_place(m, size);
    forward_solve_columns(factor, m, size);
    return power_iteration(m, size);
}

}
