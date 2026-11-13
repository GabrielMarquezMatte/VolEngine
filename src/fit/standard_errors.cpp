#include "fit/standard_errors.hpp"

#include "linalg/cholesky.hpp"
#include "model/parameter_map.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

namespace ve::fit
{

namespace
{

constexpr double kRelativeStep = 1e-5;
constexpr double kWeightStepFloor = 1e-6;
constexpr double kRealStepFloor = 1e-3;
constexpr int kMaxStepHalvings = 30;

using Matrix = std::vector<double>;
using Indices = std::vector<std::size_t>;

struct Problem
{
    const GradientFn& gradient;
    const model::Layout& layout;
    const model::Standardization& standardization;
};

bool gradient_at(const Problem& problem, std::span<const double> theta, std::span<double> grad)
{
    return problem.gradient(theta, grad) && std::ranges::all_of(grad,
                                                                [](double v)
                                                                {
                                                                    return std::isfinite(v);
                                                                });
}

bool difference_column(const Problem& problem, std::span<const double> theta, std::size_t index, double step,
                       std::span<double> column)
{
    std::size_t k = theta.size();
    std::vector<double> up(theta.begin(), theta.end());
    std::vector<double> down(theta.begin(), theta.end());
    up[index] += step;
    down[index] -= step;
    std::vector<double> grad_up(k);
    std::vector<double> grad_down(k);
    if (!gradient_at(problem, up, grad_up) || !gradient_at(problem, down, grad_down))
    {
        return false;
    }
    for (std::size_t j = 0; j < k; ++j)
    {
        column[j] = (grad_up[j] - grad_down[j]) / (2.0 * step);
    }
    return true;
}

// Steps proportional to each parameter's own scale keep the Hessian equivariant to the units of r and X. Parameters
// that may sit at ~0 get an absolute floor.
double initial_step(const Problem& problem, std::span<const double> theta, std::size_t index)
{
    double value = theta[index];
    switch (problem.layout.kind(index))
    {
        case model::ParameterKind::Location:
            return kRelativeStep *
                   std::max(std::abs(value),
                            model::natural_scale(problem.layout, problem.standardization, theta, index));
        case model::ParameterKind::Real:
            return kRelativeStep * std::max(std::abs(value), kRealStepFloor);
        case model::ParameterKind::Positive:
            return kRelativeStep * value;
        case model::ParameterKind::Weight:
            return kRelativeStep * std::max(value, kWeightStepFloor);
    }
    return kRelativeStep * value;
}

bool shrinking_difference_column(const Problem& problem, std::span<const double> theta, std::size_t index,
                                 std::span<double> column)
{
    double step = initial_step(problem, theta, index);
    for (int attempt = 0; attempt <= kMaxStepHalvings; ++attempt)
    {
        if (difference_column(problem, theta, index, step, column))
        {
            return true;
        }
        step *= 0.5;
    }
    return false;
}

void symmetrize(Matrix& matrix, std::size_t m)
{
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = i + 1; j < m; ++j)
        {
            double mean = 0.5 * (matrix[i * m + j] + matrix[j * m + i]);
            matrix[i * m + j] = mean;
            matrix[j * m + i] = mean;
        }
    }
}

// The Hessian block over `interior`, m x m row-major.
std::optional<Matrix> hessian_block(const Problem& problem, std::span<const double> theta,
                                    std::span<const std::size_t> interior)
{
    std::size_t m = interior.size();
    std::optional<Matrix> hessian(std::in_place, m * m);
    std::vector<double> column(theta.size());
    for (std::size_t a = 0; a < m; ++a)
    {
        if (!shrinking_difference_column(problem, theta, interior[a], column))
        {
            hessian.reset();
            return hessian;
        }
        for (std::size_t b = 0; b < m; ++b)
        {
            (*hessian)[b * m + a] = column[interior[b]];
        }
    }
    symmetrize(*hessian, m);
    return hessian;
}

Matrix score_cross_product(std::span<const double> scores, std::size_t k, const Indices& interior)
{
    std::size_t m = interior.size();
    Matrix product(m * m, 0.0);
    std::size_t count = scores.size() / k;
    for (std::size_t t = 0; t < count; ++t)
    {
        std::span<const double> row = scores.subspan(t * k, k);
        for (std::size_t a = 0; a < m; ++a)
        {
            for (std::size_t b = 0; b < m; ++b)
            {
                product[a * m + b] += row[interior[a]] * row[interior[b]];
            }
        }
    }
    return product;
}

// Entry (index, index) of A^-1 R^T, where A = L L^T is given by `factor` and R is row-major k x k.
double solved_diagonal_entry(std::span<const double> factor, std::span<const double> rows, std::size_t k,
                             std::size_t index)
{
    std::vector<double> x(rows.begin() + static_cast<std::ptrdiff_t>(index * k),
                          rows.begin() + static_cast<std::ptrdiff_t>((index + 1) * k));
    linalg::cholesky_solve(factor, k, x);
    return x[index];
}

// A^-1 M for symmetric M, one column at a time.
Matrix solve_columns(std::span<const double> factor, std::span<const double> symmetric, std::size_t k)
{
    Matrix result(k * k);
    for (std::size_t j = 0; j < k; ++j)
    {
        std::vector<double> x(symmetric.begin() + static_cast<std::ptrdiff_t>(j * k),
                              symmetric.begin() + static_cast<std::ptrdiff_t>((j + 1) * k));
        linalg::cholesky_solve(factor, k, x);
        for (std::size_t i = 0; i < k; ++i)
        {
            result[i * k + j] = x[i];
        }
    }
    return result;
}

StandardErrors unavailable(std::size_t k)
{
    double nan = std::numeric_limits<double>::quiet_NaN();
    return {std::vector<double>(k, nan), std::vector<double>(k, nan)};
}

// `factor` is the Cholesky factor of the negated Hessian.
StandardErrors from_factor(const Matrix& factor, const Matrix& middle, std::size_t k)
{
    Matrix identity(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i)
    {
        identity[i * k + i] = 1.0;
    }
    Matrix inverse_times_middle = solve_columns(factor, middle, k);
    StandardErrors errors{std::vector<double>(k), std::vector<double>(k)};
    for (std::size_t i = 0; i < k; ++i)
    {
        errors.hessian[i] = std::sqrt(solved_diagonal_entry(factor, identity, k, i));
        errors.robust[i] = std::sqrt(solved_diagonal_entry(factor, inverse_times_middle, k, i));
    }
    return errors;
}

StandardErrors from_hessian(const Matrix& hessian, const Matrix& middle, std::size_t k)
{
    Matrix factor(k * k);
    std::ranges::transform(hessian, factor.begin(), [](double value) { return -value; });
    if (!linalg::cholesky_factor(factor, k))
    {
        return unavailable(k);
    }
    return from_factor(factor, middle, k);
}

}

std::vector<std::size_t> interior_indices(const model::Layout& layout, std::span<const double> theta)
{
    std::vector<std::size_t> interior;
    for (std::size_t i = 0; i < theta.size(); ++i)
    {
        if (layout.is_estimated(i) && !model::on_boundary(layout, theta, i))
        {
            interior.push_back(i);
        }
    }
    return interior;
}

std::optional<std::vector<double>> numerical_hessian(const GradientFn& gradient, std::span<const double> theta,
                                                     std::span<const std::size_t> indices,
                                                     const model::Layout& layout,
                                                     const model::Standardization& standardization)
{
    return hessian_block(Problem{gradient, layout, standardization}, theta, indices);
}

StandardErrors standard_errors(const GradientFn& gradient, std::span<const double> theta,
                               std::span<const double> scores, const model::Layout& layout,
                               const model::Standardization& standardization)
{
    std::size_t k = theta.size();
    Problem problem{gradient, layout, standardization};
    Indices interior = interior_indices(layout, theta);
    std::optional<Matrix> hessian = hessian_block(problem, theta, interior);
    StandardErrors errors = unavailable(k);
    if (!hessian)
    {
        return errors;
    }
    StandardErrors block = from_hessian(*hessian, score_cross_product(scores, k, interior), interior.size());
    for (std::size_t a = 0; a < interior.size(); ++a)
    {
        errors.hessian[interior[a]] = block.hessian[a];
        errors.robust[interior[a]] = block.robust[a];
    }
    return errors;
}

}
