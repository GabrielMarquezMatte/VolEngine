#include "fit/standard_errors.hpp"

#include "fit/cholesky.hpp"

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
constexpr double kStepScaleFloor = 1e-3;
constexpr int kMaxStepHalvings = 30;

using Matrix = std::vector<double>;

struct Problem
{
    std::span<const double> r;
    const garch::SGarchConfig& config;
};

std::size_t mean_offset(const garch::SGarchConfig& config)
{
    return config.include_mean ? 1 : 0;
}

bool in_valid_region(std::span<const double> theta, const garch::SGarchConfig& config)
{
    std::size_t offset = mean_offset(config);
    double omega = theta[offset];
    double alpha = theta[offset + 1];
    double beta = theta[offset + 2];
    return omega > 0.0 && alpha >= 0.0 && beta >= 0.0 && alpha + beta < 1.0;
}

bool gradient_at(const Problem& problem, std::span<const double> theta, std::span<double> grad)
{
    garch::SGarchOutputs out;
    out.grad = grad;
    double loglik = garch::sgarch_norm_loglik(problem.r, theta, problem.config, out);
    return std::isfinite(loglik) && std::ranges::all_of(grad, [](double v) { return std::isfinite(v); });
}

bool difference_column(const Problem& problem, std::span<const double> theta, std::size_t index, double step,
                       std::span<double> column)
{
    std::size_t k = theta.size();
    std::vector<double> up(theta.begin(), theta.end());
    std::vector<double> down(theta.begin(), theta.end());
    up[index] += step;
    down[index] -= step;
    if (!in_valid_region(up, problem.config) || !in_valid_region(down, problem.config))
    {
        return false;
    }
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

bool shrinking_difference_column(const Problem& problem, std::span<const double> theta, std::size_t index,
                                 std::span<double> column)
{
    double step = kRelativeStep * std::max(std::abs(theta[index]), kStepScaleFloor);
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

std::optional<Matrix> numerical_hessian(const Problem& problem, std::span<const double> theta)
{
    std::size_t k = theta.size();
    Matrix hessian(k * k);
    std::vector<double> column(k);
    for (std::size_t i = 0; i < k; ++i)
    {
        if (!shrinking_difference_column(problem, theta, i, column))
        {
            return std::nullopt;
        }
        for (std::size_t j = 0; j < k; ++j)
        {
            hessian[j * k + i] = column[j];
        }
    }
    for (std::size_t i = 0; i < k; ++i)
    {
        for (std::size_t j = i + 1; j < k; ++j)
        {
            double mean = 0.5 * (hessian[i * k + j] + hessian[j * k + i]);
            hessian[i * k + j] = mean;
            hessian[j * k + i] = mean;
        }
    }
    return hessian;
}

Matrix score_cross_product(std::span<const double> scores, std::size_t k)
{
    Matrix product(k * k, 0.0);
    std::size_t count = scores.size() / k;
    for (std::size_t t = 0; t < count; ++t)
    {
        for (std::size_t i = 0; i < k; ++i)
        {
            for (std::size_t j = 0; j < k; ++j)
            {
                product[i * k + j] += scores[t * k + i] * scores[t * k + j];
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
    cholesky_solve(factor, k, x);
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
        cholesky_solve(factor, k, x);
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

StandardErrors from_hessian(const Matrix& hessian, std::span<const double> scores, std::size_t k)
{
    Matrix factor(k * k);
    std::ranges::transform(hessian, factor.begin(), [](double value) { return -value; });
    if (!cholesky_factor(factor, k))
    {
        return unavailable(k);
    }
    Matrix identity(k * k, 0.0);
    for (std::size_t i = 0; i < k; ++i)
    {
        identity[i * k + i] = 1.0;
    }
    Matrix middle = score_cross_product(scores, k);
    Matrix inverse_times_middle = solve_columns(factor, middle, k);
    StandardErrors errors{std::vector<double>(k), std::vector<double>(k)};
    for (std::size_t i = 0; i < k; ++i)
    {
        errors.hessian[i] = std::sqrt(solved_diagonal_entry(factor, identity, k, i));
        errors.robust[i] = std::sqrt(solved_diagonal_entry(factor, inverse_times_middle, k, i));
    }
    return errors;
}

}

StandardErrors sgarch_standard_errors(std::span<const double> r, std::span<const double> theta,
                                      const garch::SGarchConfig& config, std::span<const double> scores)
{
    std::size_t k = theta.size();
    Problem problem{r, config};
    std::optional<Matrix> hessian = numerical_hessian(problem, theta);
    if (!hessian)
    {
        return unavailable(k);
    }
    return from_hessian(*hessian, scores, k);
}

}
