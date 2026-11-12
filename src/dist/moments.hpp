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
#include <span>
#include <variant>

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

// The narrow side of a Fernandez-Steel density holds its mass within a few z-scales of the kink, and that
// scale falls as 1 / xi^2 with the skew. A side whose scale is below kRefinedBelow gets split points at
// scale 4^k from the kink, out to kRefinedBelow; at most kMaxRefinements of them, so a scale below
// 0.5 4^-12 (xi ~ 6000) starts with a first piece wider than the scale.
constexpr std::size_t kMaxRefinements = 12;
constexpr double kRefinedBelow = 0.5;
constexpr double kRefinementRatio = 4.0;

// The z-scale of the density on each side of its kink: 1 / (xi sigma) below and xi / sigma above for sstd and
// sged, 1 for the others.
struct SideScales
{
    double below;
    double above;
};

template <class T>
SideScales side_scales(const Density<T>& density)
{
    const SkewedLogDensity<T>* skewed = std::get_if<SkewedLogDensity<T>>(&density);
    if (skewed == nullptr)
    {
        return {1.0, 1.0};
    }
    double skew = ad::value_of(skewed->skew);
    double stretch = ad::value_of(skewed->stretch);
    return {1.0 / (skew * stretch), skew / stretch};
}

// The split points of one quadrature.
template <class T>
struct Breakpoints
{
    std::array<T, 2 * kMaxRefinements + 2> points{};
    std::size_t count = 0;

    void add(const T& point)
    {
        points[count] = point;
        ++count;
    }

    void sort()
    {
        std::sort(points.begin(), points.begin() + static_cast<std::ptrdiff_t>(count),
                  [](const T& a, const T& b) { return ad::value_of(a) < ad::value_of(b); });
    }

    std::span<const T> view() const
    {
        return {points.data(), count};
    }
};

template <class T>
void add_refinements(Breakpoints<T>& breakpoints, const T& kink, double direction, double scale)
{
    double smallest = kRefinedBelow * std::pow(kRefinementRatio, -static_cast<double>(kMaxRefinements));
    for (double distance = std::max(scale, smallest); distance < kRefinedBelow; distance *= kRefinementRatio)
    {
        breakpoints.add(kink + direction * distance);
    }
}

template <class T>
void add_side_refinements(Breakpoints<T>& breakpoints, const Density<T>& density, const T& kink)
{
    SideScales scales = side_scales(density);
    add_refinements(breakpoints, kink, -1.0, scales.below);
    add_refinements(breakpoints, kink, 1.0, scales.above);
}

// int over [points[0] - 1, points.back()], one piece between each pair of neighbouring points.
template <class T, class Integrand>
T breakpoint_sum(const Integrand& integrand, std::span<const T> points)
{
    T sum = segment_integral(integrand, points[0] - 1.0, points[0]);
    for (std::size_t i = 1; i < points.size(); ++i)
    {
        sum = sum + segment_integral(integrand, points[i - 1], points[i]);
    }
    return sum;
}

// The integrand has a cusp at z = eta2, and the density is not smooth at its kink (0 for ged, z* for sstd
// and sged): the quadrature is split at both, so that no piece holds a kink. With eta2 on the kink both split
// points follow eta2, so that no piece differentiates the cusp of a power below 1.
template <class T>
Breakpoints<T> shock_breakpoints(const ShockPowerIntegrand<T>& integrand)
{
    T kink = density_kink(integrand.density);
    if (ad::value_of(integrand.eta2) == ad::value_of(kink))
    {
        kink = integrand.eta2;
    }
    Breakpoints<T> breakpoints;
    breakpoints.add(kink);
    breakpoints.add(integrand.eta2);
    add_side_refinements(breakpoints, integrand.density, kink);
    breakpoints.sort();
    return breakpoints;
}

template <class T>
T shock_power_quadrature(const ShockPowerIntegrand<T>& integrand, double exponent)
{
    Breakpoints<T> breakpoints = shock_breakpoints(integrand);
    std::span<const T> points = breakpoints.view();
    T last = points.back();
    T near = breakpoint_sum(integrand, points) + segment_integral(integrand, last, last + 1.0);
    return tail_integral(integrand, points.front(), -1.0, exponent) + near +
           tail_integral(integrand, last, 1.0, exponent);
}

// The kink when it lies below 0, 0, and the refinements below 0.
template <class T>
Breakpoints<T> below_zero_breakpoints(const Density<T>& density)
{
    T kink = density_kink(density);
    T lower{0.0};
    if (ad::value_of(kink) < 0.0)
    {
        lower = kink;
    }
    Breakpoints<T> breakpoints;
    breakpoints.add(lower);
    breakpoints.add(T{0.0});
    add_side_refinements(breakpoints, density, kink);
    breakpoints.sort();
    while (ad::value_of(breakpoints.points[breakpoints.count - 1]) > 0.0)
    {
        --breakpoints.count;
    }
    return breakpoints;
}

// int over (-inf, 0).
template <class T>
T below_zero_quadrature(const DensityIntegrand<T>& integrand, double exponent)
{
    Breakpoints<T> breakpoints = below_zero_breakpoints(integrand.density);
    std::span<const T> points = breakpoints.view();
    return tail_integral(integrand, points.front(), -1.0, exponent) + breakpoint_sum(integrand, points);
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
