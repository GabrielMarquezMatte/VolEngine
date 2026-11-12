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

// int over [edge + 1, inf) (direction 1) or (-inf, edge - 1] (direction -1), through
// z = edge + direction (1 - t)^-exponent.
template <class T>
T tail_integral(const ShockPowerIntegrand<T>& integrand, const T& edge, double direction, double exponent)
{
    T sum{0.0};
    for (const QuadratureNode& node : gauss_legendre())
    {
        double complement = 1.0 - node.t;
        double distance = std::pow(complement, -exponent);
        double jacobian = exponent * distance / complement;
        sum = sum + (node.weight * jacobian) * integrand(edge + direction * distance);
    }
    return sum;
}

// A zero-width segment is skipped: its nodes would all sit on the cusp, where the slope of a power below 1 is
// infinite.
template <class T>
T segment_integral(const ShockPowerIntegrand<T>& integrand, const T& lower, const T& upper)
{
    T width = upper - lower;
    if (ad::value_of(width) == 0.0)
    {
        return T{0.0};
    }
    T sum{0.0};
    for (const QuadratureNode& node : gauss_legendre())
    {
        sum = sum + node.weight * integrand(lower + width * node.t);
    }
    return width * sum;
}

// A std integrand decays as |z|^(power - nu - 1), which z = edge + (1 - t)^-k turns into (1 - t)^(k (nu - power) - 1):
// smooth enough at t = 1 once k (nu - power) >= 2.
template <class T>
double tail_exponent(Dist dist, const T& shape, const T& power)
{
    if (dist != Dist::Std)
    {
        return 1.0;
    }
    return std::max(1.0, 2.0 / (ad::value_of(shape) - ad::value_of(power)));
}

// The integrand has a cusp at z = eta2, and the ged density is not smooth at z = 0: the quadrature is split at
// both, so that no piece holds a kink. At eta2 = 0 both split points follow eta2, so that no piece differentiates
// the cusp of a power below 1.
template <class T>
T shock_power_quadrature(const ShockPowerIntegrand<T>& integrand, double exponent)
{
    T lower = integrand.eta2;
    T upper = integrand.eta2;
    if (ad::value_of(integrand.eta2) > 0.0)
    {
        lower = T{0.0};
    }
    if (ad::value_of(integrand.eta2) < 0.0)
    {
        upper = T{0.0};
    }
    T near = segment_integral(integrand, lower - 1.0, lower) + segment_integral(integrand, lower, upper) +
             segment_integral(integrand, upper, upper + 1.0);
    return tail_integral(integrand, lower, -1.0, exponent) + near + tail_integral(integrand, upper, 1.0, exponent);
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
    return detail::shock_power_quadrature(integrand, detail::tail_exponent(dist, shape, power));
}

}
