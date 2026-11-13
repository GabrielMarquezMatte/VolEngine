#include "dcc/copula/static_fit.hpp"

#include "dcc/likelihood.hpp"
#include "fit/optimizer.hpp"
#include "linalg/cholesky.hpp"
#include "optim/lbfgs.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ve::dcc::copula
{

namespace
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();

// Kendall's tau-b of columns a and b: (C - D) / sqrt((n0 - n1)(n0 - n2)), ties in either column counting neither.
double tau_b(std::span<const double> u, std::size_t t, std::size_t m, std::size_t a, std::size_t b)
{
    double score = 0.0;
    double ties_a = 0.0;
    double ties_b = 0.0;
    for (std::size_t s = 0; s < t; ++s)
    {
        for (std::size_t r = s + 1; r < t; ++r)
        {
            const double da = u[s * m + a] - u[r * m + a];
            const double db = u[s * m + b] - u[r * m + b];
            score += static_cast<double>((da * db > 0.0) - (da * db < 0.0));
            ties_a += da == 0.0 ? 1.0 : 0.0;
            ties_b += db == 0.0 ? 1.0 : 0.0;
        }
    }
    const double pairs = 0.5 * static_cast<double>(t) * static_cast<double>(t - 1);
    return score / std::sqrt((pairs - ties_a) * (pairs - ties_b));
}

bool positive_definite(std::span<const double> r, std::size_t m)
{
    std::vector<double> factor(r.begin(), r.end());
    return linalg::cholesky_factor(std::span<double>(factor), m);
}

// The unit lower-triangular A of x, row-major m x m.
std::vector<double> unit_lower(std::span<const double> x, std::size_t m)
{
    assert(x.size() == m * (m - 1) / 2);
    std::vector<double> a(m * m, 0.0);
    std::size_t next = 0;
    for (std::size_t j = 0; j < m; ++j)
    {
        a[j * m + j] = 1.0;
        for (std::size_t i = j + 1; i < m; ++i)
        {
            a[i * m + j] = x[next++];
        }
    }
    return a;
}

// Divides each row of a by its norm, returning the norms.
std::vector<double> normalize_rows(std::vector<double>& a, std::size_t m)
{
    std::vector<double> norms(m);
    for (std::size_t i = 0; i < m; ++i)
    {
        double sum = 0.0;
        for (std::size_t j = 0; j <= i; ++j)
        {
            sum += a[i * m + j] * a[i * m + j];
        }
        norms[i] = std::sqrt(sum);
        for (std::size_t j = 0; j <= i; ++j)
        {
            a[i * m + j] /= norms[i];
        }
    }
    return norms;
}

// R^-1 from the Cholesky factor, m x m.
std::vector<double> inverse_of(std::span<const double> factor, std::size_t m)
{
    std::vector<double> inverse(m * m);
    std::vector<double> column(m);
    for (std::size_t j = 0; j < m; ++j)
    {
        std::ranges::fill(column, 0.0);
        column[j] = 1.0;
        linalg::cholesky_solve(factor, m, std::span<double>(column));
        for (std::size_t i = 0; i < m; ++i)
        {
            inverse[i * m + j] = column[i];
        }
    }
    return inverse;
}

// gradient = -(T/2) R^-1 + 1/2 weighted, both m x m.
void finish_gradient(std::span<const double> factor, std::span<const double> weighted, std::size_t t, std::size_t m,
                     std::span<double> gradient)
{
    const std::vector<double> inverse = inverse_of(factor, m);
    for (std::size_t k = 0; k < m * m; ++k)
    {
        gradient[k] = -0.5 * static_cast<double>(t) * inverse[k] + 0.5 * weighted[k];
    }
}

// q_t = epsilon_t' R^-1 epsilon_t from the Cholesky factor, leaving R^-1 epsilon_t in y.
double solved_quadratic(std::span<const double> factor, const StandardizedPanel& scores, std::size_t t,
                        std::span<double> y)
{
    const std::size_t m = scores.m;
    const std::span<const double> epsilon(scores.z.data() + t * m, m);
    std::ranges::copy(epsilon, y.begin());
    linalg::cholesky_solve(factor, m, y);
    double quadratic = 0.0;
    for (std::size_t i = 0; i < m; ++i)
    {
        quadratic += epsilon[i] * y[i];
    }
    return quadratic;
}

// -static_terms / T over the free entries of x, with its gradient by free_gradient.
class StaticObjective
{
public:
    StaticObjective(const StandardizedPanel& scores, DccDist dist, double shape, std::vector<double> x,
                    std::vector<std::size_t> free)
        : scores_(scores), dist_(dist), shape_(shape), x_(std::move(x)), free_(std::move(free)),
          dl_dr_(scores.m * scores.m), dl_dx_(x_.size())
    {
    }

    double operator()(std::span<const double> values, std::span<double> grad)
    {
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            x_[free_[i]] = values[i];
        }
        const std::vector<double> r = correlation_from_free(x_, scores_.m);
        const double loglik = static_terms(scores_, r, dist_, shape_, dl_dr_);
        if (!std::isfinite(loglik))
        {
            std::ranges::fill(grad, 0.0);
            return kInfinity;
        }
        free_gradient(x_, dl_dr_, dl_dx_, scores_.m);
        const auto count = static_cast<double>(scores_.t);
        for (std::size_t i = 0; i < grad.size(); ++i)
        {
            grad[i] = -dl_dx_[free_[i]] / count;
        }
        return -loglik / count;
    }

    const std::vector<double>& x() const { return x_; }

private:
    const StandardizedPanel& scores_;
    DccDist dist_;
    double shape_;
    std::vector<double> x_;
    std::vector<std::size_t> free_;
    std::vector<double> dl_dr_;
    std::vector<double> dl_dx_;
};

}

std::optional<std::vector<double>> kendall_correlation(std::span<const double> u, std::size_t t, std::size_t m)
{
    assert(u.size() == t * m);
    std::vector<double> r(m * m, 1.0);
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j < i; ++j)
        {
            r[i * m + j] = std::sin(0.5 * std::numbers::pi * tau_b(u, t, m, i, j));
            r[j * m + i] = r[i * m + j];
        }
    }
    if (!positive_definite(r, m))
    {
        return std::nullopt;
    }
    return r;
}

std::vector<double> correlation_from_free(std::span<const double> x, std::size_t m)
{
    std::vector<double> l = unit_lower(x, m);
    normalize_rows(l, m);
    std::vector<double> r(m * m);
    for (std::size_t i = 0; i < m; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k <= j; ++k)
            {
                sum += l[i * m + k] * l[j * m + k];
            }
            r[i * m + j] = i == j ? 1.0 : sum;
            r[j * m + i] = r[i * m + j];
        }
    }
    return r;
}

// dl/dL = 2 G L (G symmetric); row i of L is a_i / n_i, so dl/da_i = (g_i - (g_i . l_i) l_i) / n_i.
void free_gradient(std::span<const double> x, std::span<const double> dl_dr, std::span<double> dl_dx, std::size_t m)
{
    std::vector<double> l = unit_lower(x, m);
    const std::vector<double> norms = normalize_rows(l, m);
    std::vector<double> dl_da(m * m, 0.0);
    for (std::size_t i = 0; i < m; ++i)
    {
        std::vector<double> g(i + 1, 0.0);
        for (std::size_t k = 0; k <= i; ++k)
        {
            for (std::size_t j = 0; j < m; ++j)
            {
                g[k] += 2.0 * dl_dr[i * m + j] * l[j * m + k];
            }
        }
        double projection = 0.0;
        for (std::size_t k = 0; k <= i; ++k)
        {
            projection += g[k] * l[i * m + k];
        }
        for (std::size_t k = 0; k <= i; ++k)
        {
            dl_da[i * m + k] = (g[k] - projection * l[i * m + k]) / norms[i];
        }
    }
    std::size_t next = 0;
    for (std::size_t j = 0; j < m; ++j)
    {
        for (std::size_t i = j + 1; i < m; ++i)
        {
            dl_dx[next++] = dl_da[i * m + j];
        }
    }
}

double static_terms(const StandardizedPanel& scores, std::span<const double> r, DccDist dist, double shape,
                    std::span<double> gradient)
{
    const std::size_t m = scores.m;
    std::vector<double> factor(r.begin(), r.end());
    if (!linalg::cholesky_factor(std::span<double>(factor), m))
    {
        return kNaN;
    }
    const double log_det = linalg::cholesky_log_determinant(std::span<const double>(factor), m);
    std::vector<double> weighted(gradient.empty() ? 0 : m * m, 0.0);
    std::vector<double> y(m);
    double sum = 0.0;
    for (std::size_t t = 0; t < scores.t; ++t)
    {
        const double quadratic = solved_quadratic(factor, scores, t, y);
        sum += log_density_at(dist, m, shape, log_det, quadratic);
        if (gradient.empty())
        {
            continue;
        }
        const double w = dist == DccDist::MvT ? (shape + static_cast<double>(m)) / (shape - 2.0 + quadratic) : 1.0;
        for (std::size_t k = 0; k < m * m; ++k)
        {
            weighted[k] += w * y[k / m] * y[k % m];
        }
    }
    if (!gradient.empty())
    {
        finish_gradient(factor, weighted, scores.t, m, gradient);
    }
    return sum;
}

std::vector<double> static_observations(const StandardizedPanel& scores, std::span<const double> r, DccDist dist,
                                        double shape)
{
    const std::size_t m = scores.m;
    std::vector<double> terms(scores.t, kNaN);
    std::vector<double> factor(r.begin(), r.end());
    if (!linalg::cholesky_factor(std::span<double>(factor), m))
    {
        return terms;
    }
    const double log_det = linalg::cholesky_log_determinant(std::span<const double>(factor), m);
    std::vector<double> y(m);
    for (std::size_t t = 0; t < scores.t; ++t)
    {
        terms[t] = log_density_at(dist, m, shape, log_det, solved_quadratic(factor, scores, t, y));
    }
    return terms;
}

StaticCorrelation static_ml(const StandardizedPanel& scores, const DccLayout& layout, double shape,
                            const FitConfig& config)
{
    const DccDist dist = layout.has_shape ? DccDist::MvT : DccDist::MvNorm;
    std::vector<double> x(layout.c_count);
    std::vector<std::size_t> free;
    for (std::size_t k = 0; k < layout.c_count; ++k)
    {
        const std::size_t index = layout.c_begin() + k;
        x[k] = layout.given[index];
        if (layout.is_estimated(index))
        {
            free.push_back(k);
        }
    }
    std::vector<double> values(free.size());
    for (std::size_t i = 0; i < free.size(); ++i)
    {
        values[i] = x[free[i]];
    }
    StaticObjective objective(scores, dist, shape, x, free);
    optim::LbfgsOptions options;
    options.max_iter = config.max_iter;
    options.gtol = config.tol;
    optim::Stop stop = optim::Stop::Converged;
    if (!free.empty())
    {
        stop = optim::lbfgs_minimize(objective, values, options).stop;
    }
    std::vector<double> gradient(values.size());
    objective(values, gradient);
    StaticCorrelation out{fit::status_from(stop), {}, objective.x()};
    if (stop == optim::Stop::NonFinite)
    {
        out.status = FitStatus::Failed;
    }
    out.r = correlation_from_free(out.x, scores.m);
    return out;
}

}
