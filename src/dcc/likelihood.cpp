#include "dcc/likelihood.hpp"

#include "ad/dual.hpp"
#include "core/spec.hpp"
#include "dcc/density.hpp"
#include "linalg/cholesky.hpp"
#include "simd/cholesky_kernels.hpp"
#include "simd/dcc_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <type_traits>
#include <vector>

namespace ve::dcc
{

namespace
{

using detail::Density;
using detail::density_constant;
using detail::log_density;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();

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

// Matrices per call of the batched Cholesky kernel: a whole vector of t at every target (2, 4 or 8 lanes).
constexpr std::size_t kBlock = 8;

// The double instantiation: the packed Q_t of kBlock consecutive t go to the vector kernel together, which
// normalizes them and gives log |R_t| and z_t' R_t^-1 z_t; the density per t stays scalar.
class BlockedTerms
{
public:
    BlockedTerms(const StandardizedPanel& panel, const Density<double>& density, const CorrelationOutput<double>& out)
        : panel_(panel), density_(density), out_(out), cells_(panel.m * (panel.m + 1) / 2), q_(kBlock * cells_),
          scale_(out.r.empty() ? 0 : panel.m)
    {
    }

    void add(std::size_t t, std::span<const double> q_t)
    {
        std::ranges::copy(q_t, q_.begin() + static_cast<std::ptrdiff_t>((t - first_) * cells_));
        if (!out_.r.empty())
        {
            const std::size_t square = panel_.m * panel_.m;
            simd::correlation_from_q(q_t, panel_.m, scale_, out_.r.subspan(t * square, square));
        }
        if (t + 1 - first_ == kBlock || t + 1 == panel_.t)
        {
            flush(t + 1);
        }
    }

private:
    void flush(std::size_t end)
    {
        const std::size_t m = panel_.m;
        const std::size_t count = end - first_;
        std::array<double, kBlock> log_det;
        std::array<double, kBlock> quadratic;
        simd::log_det_and_quadratic(std::span<const double>(q_).first(count * cells_),
                                    std::span<const double>(panel_.z).subspan(first_ * m, count * m), m,
                                    std::span<double>(log_det).first(count),
                                    std::span<double>(quadratic).first(count));
        for (std::size_t lane = 0; lane < count; ++lane)
        {
            out_.terms[first_ + lane] =
                std::isnan(log_det[lane]) ? kNaN : log_density(density_, log_det[lane], quadratic[lane]);
        }
        first_ = end;
    }

    const StandardizedPanel& panel_;
    const Density<double>& density_;
    const CorrelationOutput<double>& out_;
    std::size_t cells_;
    std::vector<double> q_;
    std::vector<double> scale_;
    std::size_t first_ = 0;
};

template <class T>
Density<T> density_of(const StandardizedPanel& panel, const DccSpec& spec, const DccParameters<T>& theta)
{
    const auto dimension = static_cast<double>(panel.m);
    return {spec.dist, dimension, theta.shape, density_constant(spec.dist, dimension, theta.shape)};
}

// True when theta is inside the domain of correlation_path and out has been filled.
template <class T>
bool fill_terms(const StandardizedPanel& panel, const DccSpec& spec, const DccParameters<T>& theta, Stage stage,
                const CorrelationOutput<T>& out)
{
    const Density<T> density = density_of(panel, spec, theta);
    const std::size_t m = panel.m;
    Workspace<T> work{std::vector<T>(m * m), std::vector<T>(m)};
    return correlation_path(panel, spec, theta, stage, [&](std::size_t t, std::span<const T> r_t) {
        const std::span<const double> z(panel.z.data() + t * m, m);
        out.terms[t] = observation_term(r_t, z, density, work);
        store_correlation(r_t, t, out.r);
    });
}

bool fill_terms(const StandardizedPanel& panel, const DccSpec& spec, const DccParameters<double>& theta, Stage stage,
                const CorrelationOutput<double>& out)
{
    const Density<double> density = density_of(panel, spec, theta);
    BlockedTerms blocked(panel, density, out);
    return q_path(panel, spec, theta, stage, [&](std::size_t t, std::span<const double> q_t) { blocked.add(t, q_t); });
}

}

template <class T>
void correlation_terms(const StandardizedPanel& panel, const DccSpec& spec, const DccParameters<T>& theta,
                       Stage stage, const CorrelationOutput<T>& out)
{
    if (fill_terms(panel, spec, theta, stage, out))
    {
        return;
    }
    std::ranges::fill(out.terms, T{kNaN});
    std::ranges::fill(out.r, kNaN);
}

double log_density_at(DccDist dist, std::size_t m, double shape, double log_det, double quadratic)
{
    const auto dimension = static_cast<double>(m);
    const Density<double> density{dist, dimension, shape, density_constant<double>(dist, dimension, shape)};
    return log_density<double>(density, log_det, quadratic);
}

double reported_loglik(const StandardizedPanel& panel, const DccSpec& spec, std::span<const double> theta,
                       std::span<double> r)
{
    std::vector<double> terms(panel.t);
    correlation_terms<double>(panel, spec, split_parameters(spec, theta), Stage::Reported, {terms, r});
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
    correlation_terms<T>(panel, spec, split_parameters(spec, theta), Stage::Objective, {terms});
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
                                        const CorrelationOutput<double>&);
template void correlation_terms<ad::Dual<8>>(const StandardizedPanel&, const DccSpec&,
                                             const DccParameters<ad::Dual<8>>&, Stage,
                                             const CorrelationOutput<ad::Dual<8>>&);
template void correlation_terms<ad::Dual<12>>(const StandardizedPanel&, const DccSpec&,
                                              const DccParameters<ad::Dual<12>>&, Stage,
                                              const CorrelationOutput<ad::Dual<12>>&);
template void correlation_terms<ad::Dual<16>>(const StandardizedPanel&, const DccSpec&,
                                              const DccParameters<ad::Dual<16>>&, Stage,
                                              const CorrelationOutput<ad::Dual<16>>&);
template void correlation_terms<ad::Dual<32>>(const StandardizedPanel&, const DccSpec&,
                                              const DccParameters<ad::Dual<32>>&, Stage,
                                              const CorrelationOutput<ad::Dual<32>>&);
template void correlation_terms<ad::Dual<kMaxParameterCount>>(const StandardizedPanel&, const DccSpec&,
                                                              const DccParameters<ad::Dual<kMaxParameterCount>>&,
                                                              Stage,
                                                              const CorrelationOutput<ad::Dual<kMaxParameterCount>>&);

template double objective<double>(const StandardizedPanel&, const DccSpec&, std::span<const double>);
template ad::Dual<8> objective<ad::Dual<8>>(const StandardizedPanel&, const DccSpec&, std::span<const ad::Dual<8>>);
template ad::Dual<12> objective<ad::Dual<12>>(const StandardizedPanel&, const DccSpec&,
                                              std::span<const ad::Dual<12>>);
template ad::Dual<16> objective<ad::Dual<16>>(const StandardizedPanel&, const DccSpec&,
                                              std::span<const ad::Dual<16>>);
template ad::Dual<32> objective<ad::Dual<32>>(const StandardizedPanel&, const DccSpec&,
                                              std::span<const ad::Dual<32>>);
template ad::Dual<kMaxParameterCount> objective<ad::Dual<kMaxParameterCount>>(
    const StandardizedPanel&, const DccSpec&, std::span<const ad::Dual<kMaxParameterCount>>);

}
