#include "dcc/standard_errors.hpp"

#include "dcc/copula/fit.hpp"
#include "dcc/copula/margins.hpp"
#include "dcc/likelihood.hpp"
#include "dcc/panel.hpp"
#include "dcc/parameter_map.hpp"
#include "fit/fit.hpp"
#include "fit/standard_errors.hpp"
#include "linalg/cholesky.hpp"
#include "linalg/lu.hpp"
#include "model/likelihood.hpp"
#include "model/parameter_map.hpp"
#include "model/start_values.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace ve::dcc
{

namespace
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

using Matrix = std::vector<double>;

// rmgarch's finite-difference steps scale with the cube root of the machine epsilon.
double step_scale()
{
    return std::cbrt(std::numeric_limits<double>::epsilon());
}

// rmgarch's Hessian step, eps^(1/3) max(|x|, 1), as the difference it makes to x in floating point.
double rmgarch_step(double x)
{
    const double step = step_scale() * std::max(std::abs(x), 1.0);
    return (x + step) - x;
}

double score_step(double x)
{
    return step_scale() * std::max(std::abs(x) / 2.0, 0.01);
}

std::vector<double> moved(std::span<const double> theta, std::size_t index, double step)
{
    std::vector<double> out(theta.begin(), theta.end());
    out[index] += step;
    return out;
}

bool same_bits(std::span<const double> a, std::span<const double> b)
{
    const auto bits = [](double x) { return std::bit_cast<std::uint64_t>(x); };
    return std::ranges::equal(a, b, {}, bits, bits);
}

// A univariate series of theta: its likelihood rebuilt from the fit (the problem views returns), the fit's
// estimate and the entries of it that theta holds.
struct Series
{
    const Spec* spec = nullptr;
    std::vector<double> returns;
    fit::LikelihoodProblem problem;
    std::vector<double> estimate;
    std::vector<std::size_t> free;
    model::Standardization standardization;
};

// Moving the vector keeps each returns buffer in place, so the problems stay valid. The fits passed
// standard_errors_problem.
std::vector<Series> series_of(std::span<const FitResult* const> fits)
{
    std::vector<Series> series(fits.size());
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        const std::optional<fit::LikelihoodProblem> problem = fit::problem_of(*fits[i], series[i].returns);
        assert(problem.has_value());
        series[i].spec = &fits[i]->spec;
        series[i].problem = *problem;
        series[i].estimate = fits[i]->estimate;
        series[i].free = fit::interior_indices(problem->layout, series[i].estimate);
        series[i].standardization = model::standardization_of(problem->data);
    }
    return series;
}

// The panel at one univariate point, with sum_t log sigma_it and the estimate of each series, and for a copula
// spec the copula inputs there.
struct Point
{
    StandardizedPanel panel;
    std::vector<double> log_sigma;
    std::vector<std::vector<double>> estimates;
    std::shared_ptr<const copula::CopulaInputs> copula;
};

// Column `column` of z and sigma, NaN where the filter fails, and its log-sigma sum, from series filtered at theta.
// The caller refreshes the moments.
void filter_into(const Series& series, std::span<const double> theta, std::size_t column, Point& point)
{
    const std::size_t t = point.panel.t;
    const std::size_t m = point.panel.m;
    std::vector<double> residuals(t);
    std::vector<double> sigma2(t);
    const fit::LikelihoodProblem& problem = series.problem;
    const double loglik = model::loglik<double>(problem.data, theta, problem.layout, problem.var_init,
                                                {.residuals = residuals, .sigma2 = sigma2});
    if (!std::isfinite(loglik))
    {
        std::ranges::fill(sigma2, kNaN);
    }
    double log_sigma = 0.0;
    for (std::size_t row = 0; row < t; ++row)
    {
        const double sigma = std::sqrt(sigma2[row]);
        point.panel.sigma[row * m + column] = sigma;
        point.panel.z[row * m + column] = residuals[row] / sigma;
        log_sigma += std::log(sigma);
    }
    point.log_sigma[column] = log_sigma;
}

Point base_point(std::span<const Series> series, std::size_t t)
{
    const std::size_t m = series.size();
    Point point;
    point.panel.t = t;
    point.panel.m = m;
    point.panel.z.resize(t * m);
    point.panel.sigma.resize(t * m);
    point.log_sigma.resize(m);
    for (std::size_t i = 0; i < m; ++i)
    {
        filter_into(series[i], series[i].estimate, i, point);
        point.estimates.push_back(series[i].estimate);
    }
    refresh_moments(point.panel);
    return point;
}

// The copula inputs of spec at point, each series' margin at the point's estimate.
std::shared_ptr<const copula::CopulaInputs> copula_at(std::span<const Series> series, const Point& point,
                                                      const DccSpec& spec, const copula::CopulaInputs& base)
{
    std::vector<copula::Margin> margins;
    for (std::size_t i = 0; i < series.size(); ++i)
    {
        margins.push_back({series[i].spec, point.estimates[i]});
    }
    auto univariate = std::make_shared<const StandardizedPanel>(point.panel);
    return std::make_shared<const copula::CopulaInputs>(copula::inputs_at(std::move(univariate), margins, spec, base));
}

// The per-observation terms of the reported loglik that depend on the second stage: log f(z_t; R_t) of the DCC, the
// copula's c_t of a copula spec. What they leave out (sum_i log sigma_it, the univariate loglik) cancels in every
// difference of the DCC rows.
std::vector<double> dcc_terms(const Point& point, const DccSpec& spec, std::span<const double> theta)
{
    std::vector<double> terms(point.panel.t);
    correlation_terms<double>(point.panel, spec, split_parameters(spec, theta), Stage::Reported, {terms});
    return terms;
}

std::vector<double> reported_terms(const Point& point, const DccSpec& spec, std::span<const double> theta)
{
    if (point.copula != nullptr)
    {
        return copula::copula_terms(*point.copula, spec, theta);
    }
    return dcc_terms(point, spec, theta);
}

// The reported loglik less the terms that cancel: sum_t log f(z_t; R_t) - sum_i sum_t log sigma_it (DCC), sum_t c_t
// (copula: subtracting sum log sigma there would only add roundoff, which keeps the empirical cross block off 0).
double reported_total(const Point& point, const DccSpec& spec, std::span<const double> theta)
{
    double total = 0.0;
    for (const double term : reported_terms(point, spec, theta))
    {
        total += term;
    }
    if (point.copula != nullptr)
    {
        return total;
    }
    for (const double log_sigma : point.log_sigma)
    {
        total -= log_sigma;
    }
    return total;
}

// theta: the full DCC parameters (fixed ones included), free: the positions of the estimated ones; univariate: the
// (series, entry of its estimate) of each univariate parameter of the two-stage theta, in order.
struct TwoStage
{
    const DccSpec& spec;
    DccLayout layout;
    std::vector<double> theta;
    std::vector<std::size_t> free;
    std::vector<Series> series;
    std::vector<std::pair<std::size_t, std::size_t>> univariate;
    Point base;
    // The copula inputs of the fits, for a copula spec.
    std::optional<copula::CopulaInputs> copula;

    std::size_t parameters() const { return univariate.size() + free.size(); }
};

TwoStage two_stage_of(const DccFitResult& fit, std::span<const FitResult* const> fits)
{
    TwoStage stage{fit.spec, layout_of(fit.spec, fit.series), {}, {}, series_of(fits), {}, {}, std::nullopt};
    stage.theta = full_parameters(stage.layout, fit.estimate);
    stage.free = estimated_positions(stage.layout);
    for (std::size_t i = 0; i < stage.series.size(); ++i)
    {
        for (const std::size_t entry : stage.series[i].free)
        {
            stage.univariate.emplace_back(i, entry);
        }
    }
    stage.base = base_point(stage.series, fit.nobs);
    if (fit.spec.copula != DccCopula::None)
    {
        const DccSpec spec[] = {fit.spec};
        stage.copula = copula::copula_inputs(fits, spec);
        stage.base.copula = copula_at(stage.series, stage.base, fit.spec, *stage.copula);
    }
    return stage;
}

// Block (offset, offset) of A, the Hessian of -l_i, and the matching columns of the scores, those of -l_it; NaN
// where they are not finite.
void univariate_block(const Series& series, std::size_t offset, std::size_t n, Matrix& a, Matrix& scores)
{
    const std::size_t t = series.problem.data.r.size();
    const std::size_t k = series.estimate.size();
    const std::size_t count = series.free.size();
    const Matrix hessian = fit::loglik_hessian(series.problem, series.estimate, series.free)
                               .value_or(Matrix(count * count, kNaN));
    for (std::size_t p = 0; p < count; ++p)
    {
        for (std::size_t q = 0; q < count; ++q)
        {
            a[(offset + p) * n + offset + q] = -hessian[p * count + q];
        }
    }
    std::vector<double> residuals(t);
    std::vector<double> sigma2(t);
    std::vector<double> all(t * k);
    if (!std::isfinite(fit::loglik_with_scores(series.problem, series.estimate, {residuals, sigma2, all})))
    {
        std::ranges::fill(all, kNaN);
    }
    for (std::size_t row = 0; row < t; ++row)
    {
        for (std::size_t p = 0; p < count; ++p)
        {
            scores[row * n + offset + p] = -all[row * k + series.free[p]];
        }
    }
}

// f(theta) and f(theta + h_j e_j) for each estimated DCC parameter j, with h_j, on the base point.
struct DccDifferences
{
    double f0 = 0.0;
    std::vector<double> steps;
    std::vector<double> shifted;
};

DccDifferences dcc_differences(const TwoStage& stage)
{
    DccDifferences differences{reported_total(stage.base, stage.spec, stage.theta), {}, {}};
    for (const std::size_t position : stage.free)
    {
        const double step = rmgarch_step(stage.theta[position]);
        differences.steps.push_back(step);
        differences.shifted.push_back(reported_total(stage.base, stage.spec, moved(stage.theta, position, step)));
    }
    return differences;
}

// H_jk = [f(theta + h_j e_j + h_k e_k) - f(theta + h_j e_j) - f(theta + h_k e_k) + f(theta)] / (h_j h_k).
void dcc_block(const TwoStage& stage, const DccDifferences& differences, Matrix& a)
{
    const std::size_t n = stage.parameters();
    const std::size_t u = stage.univariate.size();
    for (std::size_t j = 0; j < stage.free.size(); ++j)
    {
        for (std::size_t k = 0; k <= j; ++k)
        {
            const double h_j = differences.steps[j];
            const double h_k = differences.steps[k];
            const std::vector<double> theta = moved(moved(stage.theta, stage.free[j], h_j), stage.free[k], h_k);
            const double both = reported_total(stage.base, stage.spec, theta);
            const double difference = both - differences.shifted[j] - differences.shifted[k] + differences.f0;
            const double value = difference / (h_j * h_k);
            a[(u + j) * n + u + k] = value;
            a[(u + k) * n + u + j] = value;
        }
    }
}

// The step of a univariate parameter in the DCC rows. Rmgarch: rmgarch's, absolute below 1. Standard: the univariate
// Hessian's, relative to the parameter's scale, so that the cross rows do not depend on the units of the returns.
double univariate_step(const TwoStage& stage, const Series& series, std::size_t entry)
{
    const double x = series.estimate[entry];
    if (stage.spec.convention == Convention::Rmgarch)
    {
        return rmgarch_step(x);
    }
    const double step = fit::hessian_step(series.problem.layout, series.standardization, series.estimate, entry);
    return (x + step) - x;
}

// Column `column` of the DCC rows, a univariate parameter: its series is refiltered once, the others are kept.
void cross_column(const TwoStage& stage, const DccDifferences& differences, std::size_t column, Matrix& a)
{
    const auto [index, entry] = stage.univariate[column];
    const Series& series = stage.series[index];
    std::vector<double> estimate = series.estimate;
    const double step = univariate_step(stage, series, entry);
    estimate[entry] += step;
    Point point = stage.base;
    filter_into(series, estimate, index, point);
    refresh_moments(point.panel);
    point.estimates[index] = estimate;
    if (stage.copula)
    {
        point.copula = copula_at(stage.series, point, stage.spec, *stage.copula);
    }
    const double shifted = reported_total(point, stage.spec, stage.theta);
    const std::size_t n = stage.parameters();
    const std::size_t u = stage.univariate.size();
    for (std::size_t j = 0; j < stage.free.size(); ++j)
    {
        const double h_j = differences.steps[j];
        const double both = reported_total(point, stage.spec, moved(stage.theta, stage.free[j], h_j));
        a[(u + j) * n + column] = (both - shifted - differences.shifted[j] + differences.f0) / (step * h_j);
    }
}

// The DCC columns of the scores: central differences of l_t, whose - sum_i log sigma_it does not move.
void dcc_scores(const TwoStage& stage, Matrix& scores)
{
    const std::size_t n = stage.parameters();
    const std::size_t u = stage.univariate.size();
    for (std::size_t j = 0; j < stage.free.size(); ++j)
    {
        const std::size_t position = stage.free[j];
        const double step = score_step(stage.theta[position]);
        const std::vector<double> up = reported_terms(stage.base, stage.spec, moved(stage.theta, position, step));
        const std::vector<double> down = reported_terms(stage.base, stage.spec, moved(stage.theta, position, -step));
        for (std::size_t row = 0; row < up.size(); ++row)
        {
            scores[row * n + u + j] = (up[row] - down[row]) / (2.0 * step);
        }
    }
}

struct Lu
{
    Matrix factor;
    std::vector<std::size_t> pivots;
};

std::optional<Lu> lu_of(Matrix a, std::size_t n)
{
    std::optional<Lu> lu(std::in_place, std::move(a), std::vector<std::size_t>(n));
    if (!linalg::lu_factor(lu->factor, lu->pivots, n))
    {
        lu.reset();
    }
    return lu;
}

Matrix transpose(std::span<const double> a, std::size_t n)
{
    Matrix out(n * n);
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            out[j * n + i] = a[i * n + j];
        }
    }
    return out;
}

// Row i of the result is A^-1 times row i of rows.
Matrix solve_rows(const Lu& lu, Matrix rows, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        linalg::lu_solve(lu.factor, lu.pivots, n, std::span<double>(rows).subspan(i * n, n));
    }
    return rows;
}

// sqrt(diag((-H_DD)^-1)) of the last dcc rows and columns of A.
std::vector<double> hessian_errors(const TwoStagePieces& pieces, std::size_t dcc)
{
    const std::size_t n = pieces.parameters;
    const std::size_t u = n - dcc;
    Matrix factor(dcc * dcc);
    for (std::size_t i = 0; i < dcc * dcc; ++i)
    {
        factor[i] = -pieces.a[(u + i / dcc) * n + u + i % dcc];
    }
    std::vector<double> errors(dcc, kNaN);
    if (!linalg::cholesky_factor(factor, dcc))
    {
        return errors;
    }
    for (std::size_t i = 0; i < dcc; ++i)
    {
        std::vector<double> unit(dcc, 0.0);
        unit[i] = 1.0;
        linalg::cholesky_solve(factor, dcc, unit);
        errors[i] = std::sqrt(unit[i]);
    }
    return errors;
}

}

std::optional<std::string> standard_errors_problem(std::span<const FitResult* const> fits)
{
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        if (fits[i]->spec.mxreg != 0 || fits[i]->spec.vxreg != 0)
        {
            return "series " + std::to_string(i + 1) +
                   " has regressors, which its fit does not keep, so the two-stage standard errors cannot refilter it";
        }
    }
    return std::nullopt;
}

std::optional<std::string> fits_problem(const DccFitResult& fit, std::span<const FitResult* const> fits)
{
    if (fits.size() != fit.series)
    {
        return "the DCC fit has " + std::to_string(fit.series) + " series, got " + std::to_string(fits.size()) +
               " univariate fits";
    }
    std::vector<double> estimates;
    for (std::size_t i = 0; i < fits.size(); ++i)
    {
        if (fits[i] == nullptr || fits[i]->sigma.size() != fit.nobs)
        {
            return "univariate fit " + std::to_string(i + 1) + " does not have the " + std::to_string(fit.nobs) +
                   " observations of the DCC fit";
        }
        estimates.insert(estimates.end(), fits[i]->estimate.begin(), fits[i]->estimate.end());
    }
    if (!same_bits(estimates, fit.univariate_estimates))
    {
        return std::string("the univariate fits are not those of the DCC fit: their estimates differ");
    }
    return std::nullopt;
}

std::expected<TwoStagePieces, std::string> two_stage_pieces(const DccFitResult& fit,
                                                            std::span<const FitResult* const> fits)
{
    if (std::optional<std::string> problem = fits_problem(fit, fits))
    {
        return std::unexpected(std::move(*problem));
    }
    if (!std::ranges::all_of(fit.estimate, [](double v) { return std::isfinite(v); }))
    {
        return std::unexpected(std::string("the DCC fit has no finite estimate"));
    }
    if (std::optional<std::string> problem = standard_errors_problem(fits))
    {
        return std::unexpected(std::move(*problem));
    }
    const TwoStage stage = two_stage_of(fit, fits);
    const std::size_t n = stage.parameters();
    TwoStagePieces pieces{n, Matrix(n * n, 0.0), Matrix(fit.nobs * n, 0.0)};
    std::size_t offset = 0;
    for (const Series& one : stage.series)
    {
        univariate_block(one, offset, n, pieces.a, pieces.scores);
        offset += one.free.size();
    }
    const DccDifferences differences = dcc_differences(stage);
    dcc_block(stage, differences, pieces.a);
    for (std::size_t column = 0; column < stage.univariate.size(); ++column)
    {
        cross_column(stage, differences, column, pieces.a);
    }
    dcc_scores(stage, pieces.scores);
    return pieces;
}

std::vector<double> sandwich(const TwoStagePieces& pieces, Convention convention)
{
    const std::size_t n = pieces.parameters;
    Matrix v(n * n, kNaN);
    if (n == 0)
    {
        return v;
    }
    const std::size_t t = pieces.scores.size() / n;
    const std::optional<Lu> a = lu_of(pieces.a, n);
    std::optional<Lu> right = a;
    if (convention == Convention::Rmgarch)
    {
        right = lu_of(transpose(pieces.a, n), n);
    }
    if (!a || !right)
    {
        return v;
    }
    const auto count = static_cast<double>(t);
    const Matrix b = second_moment(pieces.scores, t, n, true, count - 1.0);
    // B is symmetric, so the rows of solve_rows(A, B) are the columns of A^-1 B.
    const Matrix inverse_times_b = transpose(solve_rows(*a, b, n), n);
    v = solve_rows(*right, inverse_times_b, n);
    for (double& entry : v)
    {
        entry *= count;
    }
    if (convention == Convention::Standard)
    {
        const Matrix flipped = transpose(v, n);
        std::ranges::transform(v, flipped, v.begin(), [](double x, double y) { return 0.5 * (x + y); });
    }
    return v;
}

std::expected<DccStandardErrors, std::string> dcc_standard_errors(const DccFitResult& fit,
                                                                  std::span<const FitResult* const> fits)
{
    const std::expected<TwoStagePieces, std::string> pieces = two_stage_pieces(fit, fits);
    if (!pieces)
    {
        return std::unexpected(pieces.error());
    }
    const std::size_t dcc = fit.estimate.size();
    const std::size_t n = pieces->parameters;
    const std::size_t u = n - dcc;
    const Matrix v = sandwich(*pieces, fit.spec.convention);
    DccStandardErrors errors{hessian_errors(*pieces, dcc), std::vector<double>(dcc)};
    for (std::size_t i = 0; i < dcc; ++i)
    {
        errors.robust[i] = std::sqrt(std::abs(v[(u + i) * n + u + i]));
    }
    return errors;
}

}
