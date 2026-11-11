#include "model/parameter_map.hpp"

#include "dist/densities.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>

namespace ve::model
{

namespace
{

constexpr double kPersistenceCeiling = 1.0 - 1e-8;

template <class T>
T logistic(const T& x)
{
    using std::exp;
    if (ad::value_of(x) >= 0.0)
    {
        return 1.0 / (1.0 + exp(-x));
    }
    T exponential = exp(x);
    return exponential / (1.0 + exponential);
}

double log_ratio(double numerator, double denominator)
{
    constexpr double kFloor = std::numeric_limits<double>::min();
    return std::log(std::max(numerator, kFloor)) - std::log(std::max(denominator, kFloor));
}

// u holds v followed by the n - 1 logits; weights has n entries.
template <class T>
void persistence_to_natural(std::span<const T> u, std::span<T> weights)
{
    using std::exp;
    std::size_t count = weights.size();
    double peak = 0.0;
    for (std::size_t i = 1; i < count; ++i)
    {
        peak = std::max(peak, ad::value_of(u[i]));
    }
    T total{};
    for (std::size_t i = 0; i < count; ++i)
    {
        T logit = i + 1 < count ? u[i + 1] : T{};
        weights[i] = exp(logit - peak);
        total = total + weights[i];
    }
    T persistence = logistic(u[0]) * kPersistenceCeiling;
    for (T& weight : weights)
    {
        weight = persistence * (weight / total);
    }
}

template <class T>
T shape_to_natural(const T& x, Dist dist)
{
    using std::exp;
    return dist == Dist::Std ? 2.0 + exp(x) : exp(x);
}

double shape_to_unconstrained(double shape, Dist dist)
{
    return std::log(dist == Dist::Std ? shape - 2.0 : shape);
}

std::size_t weight_count_of(const Layout& layout)
{
    return static_cast<std::size_t>(layout.arch + layout.garch);
}

}

template <class T>
void to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout,
                const Standardization& standardization)
{
    using std::exp;
    assert(u.size() == layout.count() && theta.size() == layout.count());
    assert(standardization.scale > 0.0);
    double scale = standardization.scale;
    if (layout.include_mean)
    {
        theta[layout.mu()] = standardization.location + scale * u[layout.mu()];
    }
    std::copy(u.begin() + static_cast<std::ptrdiff_t>(layout.ar_begin()),
              u.begin() + static_cast<std::ptrdiff_t>(layout.omega()),
              theta.begin() + static_cast<std::ptrdiff_t>(layout.ar_begin()));
    theta[layout.omega()] = (scale * scale) * exp(u[layout.omega()]);
    std::size_t weight_count = weight_count_of(layout);
    persistence_to_natural<T>(u.subspan(layout.omega() + 1, weight_count),
                              theta.subspan(layout.alpha_begin(), weight_count));
    if (layout.has_shape())
    {
        theta[layout.shape()] = shape_to_natural(u[layout.shape()], layout.dist);
    }
}

template void to_natural<double>(std::span<const double>, std::span<double>, const Layout&, const Standardization&);
template void to_natural<ad::Dual<8>>(std::span<const ad::Dual<8>>, std::span<ad::Dual<8>>, const Layout&,
                                      const Standardization&);
template void to_natural<ad::Dual<16>>(std::span<const ad::Dual<16>>, std::span<ad::Dual<16>>, const Layout&,
                                       const Standardization&);
template void to_natural<ad::Dual<32>>(std::span<const ad::Dual<32>>, std::span<ad::Dual<32>>, const Layout&,
                                       const Standardization&);

void to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout,
                      const Standardization& standardization)
{
    assert(is_valid(theta, layout));
    assert(u.size() == layout.count() && standardization.scale > 0.0);
    double scale = standardization.scale;
    if (layout.include_mean)
    {
        u[layout.mu()] = (theta[layout.mu()] - standardization.location) / scale;
    }
    std::copy(theta.begin() + static_cast<std::ptrdiff_t>(layout.ar_begin()),
              theta.begin() + static_cast<std::ptrdiff_t>(layout.omega()),
              u.begin() + static_cast<std::ptrdiff_t>(layout.ar_begin()));
    u[layout.omega()] = std::log(theta[layout.omega()] / (scale * scale));
    std::size_t weight_count = weight_count_of(layout);
    std::span<const double> weights = theta.subspan(layout.alpha_begin(), weight_count);
    double persistence = std::accumulate(weights.begin(), weights.end(), 0.0);
    u[layout.omega() + 1] = log_ratio(persistence, kPersistenceCeiling - persistence);
    for (std::size_t i = 0; i + 1 < weight_count; ++i)
    {
        u[layout.omega() + 2 + i] = log_ratio(weights[i], weights[weight_count - 1]);
    }
    if (layout.has_shape())
    {
        u[layout.shape()] = shape_to_unconstrained(theta[layout.shape()], layout.dist);
    }
}

bool is_valid(std::span<const double> theta, const Layout& layout)
{
    assert(theta.size() == layout.count());
    if (!std::ranges::all_of(theta, [](double value) { return std::isfinite(value); }))
    {
        return false;
    }
    std::span<const double> weights = theta.subspan(layout.alpha_begin(), weight_count_of(layout));
    double persistence = std::accumulate(weights.begin(), weights.end(), 0.0);
    bool weights_valid = std::ranges::all_of(weights, [](double value) { return value >= 0.0; }) &&
                         persistence < kPersistenceCeiling;
    bool shape_valid = !layout.has_shape() || dist::in_domain(layout.dist, theta[layout.shape()]);
    return theta[layout.omega()] > 0.0 && weights_valid && shape_valid;
}

}
