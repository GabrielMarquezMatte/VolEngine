#include "fit/standard_errors.hpp"

#include "ad/dual.hpp"
#include "fit/cholesky.hpp"
#include "model/likelihood.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>

namespace ve::fit
{

namespace
{

constexpr double kRelativeStep = 1e-5;
constexpr double kPersistenceStepFloor = 1e-6;
constexpr int kMaxStepHalvings = 30;

constexpr std::size_t kMaxParameters = 8;

using Matrix = std::vector<double>;
using Gradient = ad::Dual<kMaxParameters>;

struct Problem
{
    const LikelihoodProblem& likelihood;
    double series_scale;
};

bool in_valid_region(std::span<const double> theta, const model::Layout& layout)
{
    double omega = theta[layout.omega()];
    double alpha = theta[layout.alpha_begin()];
    double beta = theta[layout.beta_begin()];
    return omega > 0.0 && alpha >= 0.0 && beta >= 0.0 && alpha + beta < 1.0;
}

bool gradient_at(const Problem& problem, std::span<const double> theta, std::span<double> grad)
{
    double loglik = loglik_with_gradient(problem.likelihood, theta, grad, {});
    return std::isfinite(loglik) && std::ranges::all_of(grad, [](double v) { return std::isfinite(v); });
}

void store_scores(std::span<const Gradient> terms, std::size_t k, std::span<double> scores)
{
    for (std::size_t t = 0; t < terms.size(); ++t)
    {
        std::copy_n(terms[t].grad.begin(), k, scores.begin() + static_cast<std::ptrdiff_t>(t * k));
    }
}

bool difference_column(const Problem& problem, std::span<const double> theta, std::size_t index, double step,
                       std::span<double> column)
{
    std::size_t k = theta.size();
    std::vector<double> up(theta.begin(), theta.end());
    std::vector<double> down(theta.begin(), theta.end());
    up[index] += step;
    down[index] -= step;
    const model::Layout& layout = problem.likelihood.layout;
    if (!in_valid_region(up, layout) || !in_valid_region(down, layout))
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

// Steps proportional to each parameter's own scale keep the Hessian equivariant to the units of r. alpha and
// beta may sit at ~0, so only their steps have an absolute floor.
double initial_step(const Problem& problem, std::span<const double> theta, std::size_t index)
{
    std::size_t omega = problem.likelihood.layout.omega();
    if (index < omega)
    {
        return kRelativeStep * std::max(std::abs(theta[index]), problem.series_scale);
    }
    if (index == omega)
    {
        return kRelativeStep * theta[index];
    }
    return kRelativeStep * std::max(theta[index], kPersistenceStepFloor);
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

double loglik_with_gradient(const LikelihoodProblem& problem, std::span<const double> theta, std::span<double> grad,
                            std::span<double> scores)
{
    std::size_t k = theta.size();
    assert(k <= kMaxParameters && grad.size() == k);
    std::vector<Gradient> seeds(k);
    for (std::size_t i = 0; i < k; ++i)
    {
        seeds[i] = ad::variable<kMaxParameters>(theta[i], i);
    }
    std::vector<Gradient> terms;
    if (!scores.empty())
    {
        terms.resize(problem.r.size());
    }
    Gradient loglik = model::loglik<Gradient>(problem.r, seeds, problem.layout, problem.var_init, {.terms = terms});
    std::copy_n(loglik.grad.begin(), k, grad.begin());
    store_scores(terms, k, scores);
    return loglik.value;
}

StandardErrors sgarch_standard_errors(const LikelihoodProblem& likelihood, std::span<const double> theta,
                                      std::span<const double> scores, double series_scale)
{
    std::size_t k = theta.size();
    Problem problem{likelihood, series_scale};
    std::optional<Matrix> hessian = numerical_hessian(problem, theta);
    if (!hessian)
    {
        return unavailable(k);
    }
    return from_hessian(*hessian, scores, k);
}

}
