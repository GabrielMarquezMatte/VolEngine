#pragma once

#include "ad/dual.hpp"
#include "core/spec.hpp"
#include "dist/densities.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>

namespace ve::dist
{

namespace detail
{

constexpr std::size_t kGaussLegendreNodes = 64;

struct QuadratureNode
{
    double t;
    double weight;
};

// The Gauss-Legendre rule on [0, 1].
const std::array<QuadratureNode, kGaussLegendreNodes>& gauss_legendre();

template <class T>
bool infinite_moment(Dist dist, const T& shape, const T& power)
{
    return dist == Dist::Std && ad::value_of(shape) <= ad::value_of(power);
}

template <class T>
T normal_abs_moment(const T& power)
{
    using std::exp;
    using std::lgamma;
    T log_moment = 0.5 * power * std::numbers::ln2 + lgamma(0.5 * (power + 1.0));
    return exp(log_moment) / std::sqrt(std::numbers::pi);
}

// log of (nu - 2)^(-b) Gamma(a + b) / Gamma(a), with a = nu / 2 and b = -power / 2. Its lgamma terms grow as
// a log(a) while it tends to -b log(2), so for large a the difference is taken from the Stirling expansions:
//   (a + b - 1/2) log1p(b / a) - b - b log(2) - b log1p(-1 / a) + S(a + b) - S(a).
template <class T>
T student_log_gamma_ratio(const T& nu, const T& power)
{
    using std::lgamma;
    using std::log;
    using std::log1p;
    T a = nu / 2.0;
    T b = -power / 2.0;
    if (ad::value_of(a + b) < kStirlingThreshold)
    {
        return lgamma(a + b) - lgamma(a) - b * log(nu - 2.0);
    }
    T expansion = (a + b - 0.5) * log1p(b / a) - b - b * std::numbers::ln2 - b * log1p(-1.0 / a);
    return expansion + (stirling_series(a + b) - stirling_series(a));
}

// (nu - 2)^(power / 2) Gamma((power + 1) / 2) Gamma((nu - power) / 2) / (sqrt(pi) Gamma(nu / 2)), for nu > power.
template <class T>
T student_abs_moment(const T& nu, const T& power)
{
    using std::exp;
    using std::lgamma;
    T log_moment = student_log_gamma_ratio(nu, power) + lgamma(0.5 * (power + 1.0));
    return exp(log_moment) / std::sqrt(std::numbers::pi);
}

// lambda^power 2^(power / nu) Gamma((power + 1) / nu) / Gamma(1 / nu), with the lambda of ged_log_density.
template <class T>
T ged_abs_moment(const T& nu, const T& power)
{
    using std::exp;
    using std::lgamma;
    using std::log;
    T log_lambda = log(ged_log_density(nu).scale);
    T log_gamma_ratio = lgamma((power + 1.0) / nu) - lgamma(1.0 / nu);
    return exp(power * log_lambda + (power / nu) * std::numbers::ln2 + log_gamma_ratio);
}

// (|z - eta2| - eta1 (z - eta2))^power f(z).
template <class T>
struct ShockPowerIntegrand
{
    LogDensity<T> density;
    T eta1;
    T eta2;
    T power;

    T operator()(const T& z) const
    {
        using std::abs;
        using std::exp;
        using std::pow;
        T shift = z - eta2;
        return pow(abs(shift) - eta1 * shift, power) * exp(log_density(density, z));
    }
};

// int over [origin, inf) (direction 1) or (-inf, origin] (direction -1), through z = origin + direction t / (1 - t).
template <class T>
T tail_integral(const ShockPowerIntegrand<T>& integrand, const T& origin, double direction)
{
    T sum{0.0};
    for (const QuadratureNode& node : gauss_legendre())
    {
        double complement = 1.0 - node.t;
        double jacobian = 1.0 / (complement * complement);
        sum = sum + (node.weight * jacobian) * integrand(origin + direction * (node.t / complement));
    }
    return sum;
}

template <class T>
T segment_integral(const ShockPowerIntegrand<T>& integrand, const T& lower, const T& upper)
{
    T width = upper - lower;
    T sum{0.0};
    for (const QuadratureNode& node : gauss_legendre())
    {
        sum = sum + node.weight * integrand(lower + width * node.t);
    }
    return width * sum;
}

// The integrand has a cusp at z = eta2, and the ged density is not smooth at z = 0: the quadrature is split at
// both, so that no piece holds a kink.
template <class T>
T shock_power_quadrature(const ShockPowerIntegrand<T>& integrand)
{
    T zero{0.0};
    T lower = std::min(zero, integrand.eta2);
    T upper = std::max(zero, integrand.eta2);
    T inner = segment_integral(integrand, lower, upper);
    return tail_integral(integrand, lower, -1.0) + inner + tail_integral(integrand, upper, 1.0);
}

}

// E|z|^power; +inf for std with shape <= power.
template <class T>
T abs_moment(Dist dist, const T& shape, const T& power)
{
    if (detail::infinite_moment(dist, shape, power))
    {
        return T{std::numeric_limits<double>::infinity()};
    }
    if (dist == Dist::Std)
    {
        return detail::student_abs_moment(shape, power);
    }
    if (dist == Dist::Ged)
    {
        return detail::ged_abs_moment(shape, power);
    }
    return detail::normal_abs_moment(power);
}

// E(|z| - eta1 z)^power = ((1 - eta1)^power + (1 + eta1)^power) / 2 E|z|^power, for a symmetric density.
template <class T>
T asymmetric_moment(Dist dist, const T& shape, const T& eta1, const T& power)
{
    using std::pow;
    T asymmetry = 0.5 * (pow(1.0 - eta1, power) + pow(1.0 + eta1, power));
    return asymmetry * abs_moment(dist, shape, power);
}

// E(|z - eta2| - eta1 (z - eta2))^power, by quadrature; +inf for std with shape <= power.
template <class T>
T shifted_moment(Dist dist, const T& shape, const T& eta1, const T& eta2, const T& power)
{
    if (detail::infinite_moment(dist, shape, power))
    {
        return T{std::numeric_limits<double>::infinity()};
    }
    detail::ShockPowerIntegrand<T> integrand{make_log_density(dist, shape), eta1, eta2, power};
    return detail::shock_power_quadrature(integrand);
}

}
