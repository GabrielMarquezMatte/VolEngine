#include "linalg/least_squares.hpp"

#include "linalg/cholesky.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

namespace ve::linalg
{

namespace
{

// A pivot this small against its column's own diagonal of A'A is rounding noise of a collinear column.
constexpr double kPivotTolerance = 1e-12;
constexpr double kEigenvalueCut = 1e-12;
constexpr int kMaxSweeps = 60;
constexpr double kOffDiagonalTolerance = 1e-32;

struct NormalEquations
{
    std::vector<double> gram;
    std::vector<double> rhs;
};

NormalEquations normal_equations(std::span<const double> a, std::size_t columns, std::span<const double> y)
{
    NormalEquations n{std::vector<double>(columns * columns), std::vector<double>(columns)};
    for (std::size_t t = 0; t < y.size(); ++t)
    {
        const double* row = a.data() + t * columns;
        for (std::size_t i = 0; i < columns; ++i)
        {
            n.rhs[i] += row[i] * y[t];
            for (std::size_t j = 0; j <= i; ++j)
            {
                n.gram[i * columns + j] += row[i] * row[j];
            }
        }
    }
    for (std::size_t i = 0; i < columns; ++i)
    {
        for (std::size_t j = i + 1; j < columns; ++j)
        {
            n.gram[i * columns + j] = n.gram[j * columns + i];
        }
    }
    return n;
}

bool cholesky_solution(const NormalEquations& n, std::size_t columns, std::vector<double>& b)
{
    std::vector<double> factor = n.gram;
    if (!cholesky_factor(factor, columns))
    {
        return false;
    }
    for (std::size_t k = 0; k < columns; ++k)
    {
        double pivot = factor[k * columns + k];
        if (pivot * pivot <= kPivotTolerance * n.gram[k * columns + k])
        {
            return false;
        }
    }
    b = n.rhs;
    cholesky_solve(factor, columns, b);
    return true;
}

// Rotates the plane (p, q) of the symmetric matrix m to zero m[p][q]; v accumulates the same rotation.
void rotate(std::vector<double>& m, std::vector<double>& v, std::size_t n, std::size_t p, std::size_t q)
{
    double apq = m[p * n + q];
    if (apq == 0.0)
    {
        return;
    }
    double theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
    double t = std::copysign(1.0, theta) / (std::abs(theta) + std::hypot(theta, 1.0));
    double c = 1.0 / std::hypot(t, 1.0);
    double s = t * c;
    for (std::size_t k = 0; k < n; ++k)
    {
        double kp = m[k * n + p];
        double kq = m[k * n + q];
        m[k * n + p] = c * kp - s * kq;
        m[k * n + q] = s * kp + c * kq;
    }
    for (std::size_t k = 0; k < n; ++k)
    {
        double pk = m[p * n + k];
        double qk = m[q * n + k];
        m[p * n + k] = c * pk - s * qk;
        m[q * n + k] = s * pk + c * qk;
        double vp = v[k * n + p];
        double vq = v[k * n + q];
        v[k * n + p] = c * vp - s * vq;
        v[k * n + q] = s * vp + c * vq;
    }
}

bool diagonalized(const std::vector<double>& m, std::size_t n)
{
    double off = 0.0;
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            double square = m[i * n + j] * m[i * n + j];
            total += square;
            off += i == j ? 0.0 : square;
        }
    }
    return off <= kOffDiagonalTolerance * total;
}

// Cyclic Jacobi: the diagonal of m ends as the eigenvalues, the columns of v as the eigenvectors.
void jacobi_eigen(std::vector<double>& m, std::vector<double>& v, std::size_t n)
{
    v.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        v[i * n + i] = 1.0;
    }
    for (int sweep = 0; sweep < kMaxSweeps && !diagonalized(m, n); ++sweep)
    {
        for (std::size_t p = 0; p + 1 < n; ++p)
        {
            for (std::size_t q = p + 1; q < n; ++q)
            {
                rotate(m, v, n, p, q);
            }
        }
    }
}

std::vector<double> pseudo_inverse_solution(NormalEquations n, std::size_t columns)
{
    std::vector<double> vectors;
    jacobi_eigen(n.gram, vectors, columns);
    double largest = 0.0;
    for (std::size_t k = 0; k < columns; ++k)
    {
        largest = std::max(largest, n.gram[k * columns + k]);
    }
    std::vector<double> b(columns);
    for (std::size_t k = 0; k < columns; ++k)
    {
        double eigenvalue = n.gram[k * columns + k];
        if (eigenvalue <= kEigenvalueCut * largest)
        {
            continue;
        }
        double projection = 0.0;
        for (std::size_t j = 0; j < columns; ++j)
        {
            projection += vectors[j * columns + k] * n.rhs[j];
        }
        for (std::size_t i = 0; i < columns; ++i)
        {
            b[i] += vectors[i * columns + k] * projection / eigenvalue;
        }
    }
    return b;
}

}

std::vector<double> least_squares(std::span<const double> a, std::size_t columns, std::span<const double> y)
{
    assert(a.size() == y.size() * columns);
    NormalEquations n = normal_equations(a, columns, y);
    std::vector<double> b;
    if (!cholesky_solution(n, columns, b))
    {
        b = pseudo_inverse_solution(std::move(n), columns);
    }
    return b;
}

}
