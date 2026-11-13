#include "dcc/likelihood.hpp"

#include "ad/dual.hpp"
#include "core/spec.hpp"
#include "dist/bessel.hpp"
#include "linalg/cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

namespace ve::dcc
{

namespace
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();

// The terms of log f(z; R) that depend on neither R nor z.
template <class T>
struct Density
{
    DccDist dist;
    double m;
    T shape;
    T constant;
};

template <class T>
T density_constant(DccDist dist, double m, const T& shape)
{
    using std::lgamma;
    using std::log;
    const double log_two_pi = std::log(2.0 * std::numbers::pi);
    if (dist == DccDist::MvT)
    {
        const T log_gamma_ratio = lgamma(0.5 * (shape + m)) - lgamma(0.5 * shape);
        return log_gamma_ratio - 0.5 * m * log(std::numbers::pi * (shape - 2.0));
    }
    if (dist == DccDist::MvLaplace)
    {
        return T{std::numbers::ln2 - 0.5 * m * log_two_pi};
    }
    return T{-0.5 * m * log_two_pi};
}

// log f(z; R) from L = log |R| and q = z' R^-1 z.
template <class T>
T log_density(const Density<T>& density, const T& log_det, const T& quadratic)
{
    using std::log;
    using std::log1p;
    using std::sqrt;
    const T half_log_det = 0.5 * log_det;
    if (density.dist == DccDist::MvT)
    {
        const T tail = 0.5 * (density.shape + density.m) * log1p(quadratic / (density.shape - 2.0));
        return density.constant - half_log_det - tail;
    }
    if (density.dist == DccDist::MvLaplace)
    {
        const double order = 1.0 - 0.5 * density.m;
        const T power = 0.5 * order * log(0.5 * quadratic);
        const T bessel = dist::log_bessel_k(order, sqrt(2.0 * quadratic));
        return density.constant - half_log_det + power + bessel;
    }
    return density.constant - 0.5 * (log_det + quadratic);
}

template <class T>
struct Workspace
{
    std::vector<T> factor;
    std::vector<T> solution;
};

// R^-1 and log |R| from one Cholesky factor of R; NaN unless R is positive definite.
template <class T>
T observation_term(std::span<const T> r, std::span<const double> z, const Density<T>& density, Workspace<T>& work)
{
    const std::size_t m = z.size();
    std::ranges::copy(r, work.factor.begin());
    if (!linalg::cholesky_factor(std::span<T>(work.factor), m))
    {
        return T{kNaN};
    }
    for (std::size_t i = 0; i < m; ++i)
    {
        work.solution[i] = T{z[i]};
    }
    const std::span<const T> factor(work.factor);
    linalg::cholesky_solve(factor, m, std::span<T>(work.solution));
    T quadratic{0.0};
    for (std::size_t i = 0; i < m; ++i)
    {
        quadratic = quadratic + z[i] * work.solution[i];
    }
    return log_density(density, linalg::cholesky_log_determinant(factor, m), quadratic);
}

template <class T>
void store_correlation(std::span<const T> r_t, std::size_t t, std::span<double> r)
{
    if (r.empty())
    {
        return;
    }
    const std::size_t cells = r_t.size();
    for (std::size_t k = 0; k < cells; ++k)
    {
        r[t * cells + k] = ad::value_of(r_t[k]);
    }
}

}

template <class T>
void correlation_terms(const StandardizedPanel& panel, const DccSpec& spec, const DccParameters<T>& theta,
                       Stage stage, std::span<T> terms, std::span<double> r)
{
    const std::size_t m = panel.m;
    const auto dimension = static_cast<double>(m);
    const Density<T> density{spec.dist, dimension, theta.shape, density_constant(spec.dist, dimension, theta.shape)};
    Workspace<T> work{std::vector<T>(m * m), std::vector<T>(m)};
    const auto visit = [&](std::size_t t, std::span<const T> r_t) {
        const std::span<const double> z(panel.z.data() + t * m, m);
        terms[t] = observation_term(r_t, z, density, work);
        store_correlation(r_t, t, r);
    };
    if (correlation_path(panel, spec, theta, stage, visit))
    {
        return;
    }
    std::ranges::fill(terms, T{kNaN});
    std::ranges::fill(r, kNaN);
}

double reported_loglik(const StandardizedPanel& panel, const DccSpec& spec, std::span<const double> theta,
                       std::span<double> r)
{
    std::vector<double> terms(panel.t);
    correlation_terms<double>(panel, spec, split_parameters(spec, theta), Stage::Reported, terms, r);
    double loglik = 0.0;
    for (std::size_t t = 0; t < panel.t; ++t)
    {
        double log_sigma = 0.0;
        for (std::size_t i = 0; i < panel.m; ++i)
        {
            log_sigma += std::log(panel.sigma[t * panel.m + i]);
        }
        loglik += terms[t] - log_sigma;
    }
    return loglik;
}

template <class T>
T objective(const StandardizedPanel& panel, const DccSpec& spec, std::span<const T> theta)
{
    std::vector<T> terms(panel.t);
    correlation_terms<T>(panel, spec, split_parameters(spec, theta), Stage::Objective, terms);
    T total{0.0};
    if (spec.convention == Convention::Rmgarch && spec.dist == DccDist::MvNorm)
    {
        const double log_two_pi = std::log(2.0 * std::numbers::pi);
        total = T{-0.5 * static_cast<double>(panel.m * panel.t) * log_two_pi};
    }
    for (const T& term : terms)
    {
        total = total - term;
    }
    if (!std::isfinite(ad::value_of(total)))
    {
        total = T{kInfinity};
    }
    return total;
}

template void correlation_terms<double>(const StandardizedPanel&, const DccSpec&, const DccParameters<double>&, Stage,
                                        std::span<double>, std::span<double>);
template void correlation_terms<ad::Dual<8>>(const StandardizedPanel&, const DccSpec&,
                                             const DccParameters<ad::Dual<8>>&, Stage, std::span<ad::Dual<8>>,
                                             std::span<double>);
template void correlation_terms<ad::Dual<16>>(const StandardizedPanel&, const DccSpec&,
                                              const DccParameters<ad::Dual<16>>&, Stage, std::span<ad::Dual<16>>,
                                              std::span<double>);
template void correlation_terms<ad::Dual<32>>(const StandardizedPanel&, const DccSpec&,
                                              const DccParameters<ad::Dual<32>>&, Stage, std::span<ad::Dual<32>>,
                                              std::span<double>);
template void correlation_terms<ad::Dual<kMaxParameterCount>>(const StandardizedPanel&, const DccSpec&,
                                                              const DccParameters<ad::Dual<kMaxParameterCount>>&,
                                                              Stage, std::span<ad::Dual<kMaxParameterCount>>,
                                                              std::span<double>);

template double objective<double>(const StandardizedPanel&, const DccSpec&, std::span<const double>);
template ad::Dual<8> objective<ad::Dual<8>>(const StandardizedPanel&, const DccSpec&, std::span<const ad::Dual<8>>);
template ad::Dual<16> objective<ad::Dual<16>>(const StandardizedPanel&, const DccSpec&,
                                              std::span<const ad::Dual<16>>);
template ad::Dual<32> objective<ad::Dual<32>>(const StandardizedPanel&, const DccSpec&,
                                              std::span<const ad::Dual<32>>);
template ad::Dual<kMaxParameterCount> objective<ad::Dual<kMaxParameterCount>>(
    const StandardizedPanel&, const DccSpec&, std::span<const ad::Dual<kMaxParameterCount>>);

}
