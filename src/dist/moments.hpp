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

// norm, std and ged keep the closed forms of plan 2b.
inline bool is_symmetric(Dist dist)
{
    return dist == Dist::Norm || dist == Dist::Std || dist == Dist::Ged;
}

// The sstd has the tail of the t.
inline bool has_student_tail(Dist dist)
{
    return dist == Dist::Std || dist == Dist::Sstd;
}

template <class T>
bool infinite_moment(Dist dist, const ShapeParameters<T>& parameters, const T& power)
{
    return has_student_tail(dist) && ad::value_of(parameters.shape) <= ad::value_of(power);
}

// (|z - eta2| - eta1 (z - eta2))^power f(z).
template <class T>
struct ShockPowerIntegrand
{
    Density<T> density;
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

// f(z).
template <class T>
struct DensityIntegrand
{
    Density<T> density;

    T operator()(const T& z) const
    {
        using std::exp;
        return exp(log_density(density, z));
    }
};

// int over [edge + 1, inf) (direction 1) or (-inf, edge - 1] (direction -1), through
// z = edge + direction (1 - t)^-exponent.
template <class T, class Integrand>
T tail_integral(const Integrand& integrand, const T& edge, double direction, double exponent)
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
template <class T, class Integrand>
T segment_integral(const Integrand& integrand, const T& lower, const T& upper)
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

// A std or sstd integrand decays as |z|^(power - nu - 1), which z = edge + (1 - t)^-k turns into
// (1 - t)^(k (nu - power) - 1): smooth enough at t = 1 once k (nu - power) >= 2.
template <class T>
double tail_exponent(Dist dist, const ShapeParameters<T>& parameters, const T& power)
{
    if (!has_student_tail(dist))
    {
        return 1.0;
    }
    return std::max(1.0, 2.0 / (ad::value_of(parameters.shape) - ad::value_of(power)));
}

// The integrand has a cusp at z = eta2, and the density is not smooth at its kink (0 for ged, z* for sstd
// and sged): the quadrature is split at both, so that no piece holds a kink. With eta2 on the kink both split
// points follow eta2, so that no piece differentiates the cusp of a power below 1.
template <class T>
T shock_power_quadrature(const ShockPowerIntegrand<T>& integrand, double exponent)
{
    T kink = density_kink(integrand.density);
    T lower = integrand.eta2;
    T upper = integrand.eta2;
    if (ad::value_of(integrand.eta2) > ad::value_of(kink))
    {
        lower = kink;
    }
    if (ad::value_of(integrand.eta2) < ad::value_of(kink))
    {
        upper = kink;
    }
    T near = segment_integral(integrand, lower - 1.0, lower) + segment_integral(integrand, lower, upper) +
             segment_integral(integrand, upper, upper + 1.0);
    return tail_integral(integrand, lower, -1.0, exponent) + near + tail_integral(integrand, upper, 1.0, exponent);
}

// int over (-inf, 0), split at the kink when it lies below 0.
template <class T>
T below_zero_quadrature(const DensityIntegrand<T>& integrand, double exponent)
{
    T kink = density_kink(integrand.density);
    T lower{0.0};
    if (ad::value_of(kink) < 0.0)
    {
        lower = kink;
    }
    T near = segment_integral(integrand, lower - 1.0, lower) + segment_integral(integrand, lower, T{0.0});
    return tail_integral(integrand, lower, -1.0, exponent) + near;
}

}

// E(|z - eta2| - eta1 (z - eta2))^power, by quadrature; +inf for std and sstd with shape <= power.
template <class T>
T shifted_moment(Dist dist, const ShapeParameters<T>& parameters, const T& eta1, const T& eta2,
                 const T& power)
{
    if (detail::infinite_moment(dist, parameters, power))
    {
        return T{std::numeric_limits<double>::infinity()};
    }
    detail::ShockPowerIntegrand<T> integrand{make_density(dist, parameters), eta1, eta2, power};
    return detail::shock_power_quadrature(integrand, detail::tail_exponent(dist, parameters, power));
}

// E|z|^power: the closed form for norm, std and ged, the quadrature for the others; +inf for std and sstd
// with shape <= power.
template <class T>
T abs_moment(Dist dist, const ShapeParameters<T>& parameters, const T& power)
{
    if (detail::infinite_moment(dist, parameters, power))
    {
        return T{std::numeric_limits<double>::infinity()};
    }
    if (!detail::is_symmetric(dist))
    {
        return shifted_moment(dist, parameters, T{0.0}, T{0.0}, power);
    }
    if (dist == Dist::Std)
    {
        return detail::student_abs_moment(parameters.shape, power);
    }
    if (dist == Dist::Ged)
    {
        return detail::ged_abs_moment(parameters.shape, power);
    }
    return detail::normal_abs_moment(power);
}

// E(|z| - eta1 z)^power, which is ((1 - eta1)^power + (1 + eta1)^power) / 2 E|z|^power for a symmetric
// density.
template <class T>
T asymmetric_moment(Dist dist, const ShapeParameters<T>& parameters, const T& eta1, const T& power)
{
    using std::pow;
    if (!detail::is_symmetric(dist))
    {
        return shifted_moment(dist, parameters, eta1, T{0.0}, power);
    }
    T asymmetry = 0.5 * (pow(1.0 - eta1, power) + pow(1.0 + eta1, power));
    return asymmetry * abs_moment(dist, parameters, power);
}

// P(z < 0): exactly 0.5 for norm, std and ged, the quadrature of f over (-inf, 0) for the others.
template <class T>
T probability_below_zero(Dist dist, const ShapeParameters<T>& parameters)
{
    if (detail::is_symmetric(dist))
    {
        return T{0.5};
    }
    detail::DensityIntegrand<T> integrand{make_density(dist, parameters)};
    return detail::below_zero_quadrature(integrand, detail::tail_exponent(dist, parameters, T{0.0}));
}

}
