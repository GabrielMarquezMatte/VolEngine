#pragma once

#include "ad/dual.hpp"

#include <cmath>
#include <limits>
#include <numbers>

namespace ve::dist
{

// Johnson SU in the parametrization of Rigby and Stasinopoulos, with mean 0 and variance 1:
//   log f(z) = log_constant - log(1 + x^2) / 2 - r^2 / 2,
// x = (z - center) / scale, r = -skew + shape asinh(x), scale = c, center = c sqrt(w) sinh(omega) and
// log_constant = log(shape) - log(c) - log(2 pi) / 2.
template <class T>
struct JohnsonLogDensity
{
    T log_constant;
    T center;
    T scale;
    T skew;
    T shape;
};

namespace detail
{

// sinh and cosh overflow past |x| ~ 710; beyond this limit their logs are |x| - log 2 to rounding.
constexpr double kJohnsonLogLimit = 700.0;

template <class T>
T log_cosh(const T& x)
{
    using std::abs;
    using std::cosh;
    using std::log;
    if (std::fabs(ad::value_of(x)) > kJohnsonLogLimit)
    {
        return abs(x) - std::numbers::ln2;
    }
    return log(cosh(x));
}

// log sinh(x) for x > 0.
template <class T>
T log_sinh(const T& x)
{
    using std::log;
    using std::sinh;
    if (ad::value_of(x) > kJohnsonLogLimit)
    {
        return x - std::numbers::ln2;
    }
    return log(sinh(x));
}

// e^log_factor sinh(x), where sinh(x) alone may overflow.
template <class T>
T scaled_sinh(const T& log_factor, const T& x)
{
    using std::abs;
    using std::exp;
    using std::sinh;
    if (std::fabs(ad::value_of(x)) <= kJohnsonLogLimit)
    {
        return exp(log_factor) * sinh(x);
    }
    double sign = ad::value_of(x) < 0.0 ? -1.0 : 1.0;
    return sign * exp(log_factor + abs(x) - std::numbers::ln2);
}

// The standardisation in logs, for where the variance factor V overflows:
//   log V = log sinh(h) + h + log(e^A + 1), A = 2h + log cosh(2 omega),
// log(e^A + 1) = A + log(1 + e^-A) with A >= 0. A scale c = V^(-1/2) below the smallest normal double is not
// representable, and every field is then NaN.
template <class T>
JohnsonLogDensity<T> johnson_log_density_in_logs(const T& nu, const T& tau)
{
    using std::exp;
    using std::log;
    using std::log1p;
    T half_inverse = 0.5 / (tau * tau);
    T omega = -nu / tau;
    T exponent = 2.0 * half_inverse + log_cosh(2.0 * omega);
    T log_variance_factor = log_sinh(half_inverse) + half_inverse + exponent + log1p(exp(-exponent));
    T log_scale = -0.5 * log_variance_factor;
    T scale = exp(log_scale);
    double scale_value = ad::value_of(scale);
    if (!(scale_value >= std::numeric_limits<double>::min() && scale_value <= std::numeric_limits<double>::max()))
    {
        T not_representable{std::numeric_limits<double>::quiet_NaN()};
        return {not_representable, not_representable, not_representable, nu, tau};
    }
    T log_constant = log(tau) + 0.5 * log_variance_factor - 0.5 * std::log(2.0 * std::numbers::pi);
    return {log_constant, scaled_sinh(log_scale + half_inverse, omega), scale, nu, tau};
}

}

// w = exp(1 / tau^2), omega = -nu / tau, c = [(w - 1) (w cosh(2 omega) + 1) / 2]^(-1/2). w - 1 is taken as
// 2 sinh(h) e^h with h = 1 / (2 tau^2), which keeps its digits as tau grows, where exp(1 / tau^2) - 1
// cancels. Where the variance factor overflows (tau below ~0.053, |nu| / tau above ~354) it is taken in logs.
template <class T>
JohnsonLogDensity<T> johnson_log_density(const T& nu, const T& tau)
{
    using std::cosh;
    using std::exp;
    using std::log;
    using std::sinh;
    using std::sqrt;
    T half_inverse = 0.5 / (tau * tau);
    T root_w = exp(half_inverse);
    T omega = -nu / tau;
    T variance_factor = sinh(half_inverse) * root_w * (root_w * root_w * cosh(2.0 * omega) + 1.0);
    double factor_value = ad::value_of(variance_factor);
    if (!(factor_value > 0.0 && std::isfinite(factor_value)))
    {
        return detail::johnson_log_density_in_logs(nu, tau);
    }
    T scale = 1.0 / sqrt(variance_factor);
    T log_constant = log(tau) + 0.5 * log(variance_factor) - 0.5 * std::log(2.0 * std::numbers::pi);
    return {log_constant, scale * root_w * sinh(omega), scale, nu, tau};
}

template <class T>
T log_density(const JohnsonLogDensity<T>& density, const T& z)
{
    using std::asinh;
    using std::log1p;
    T x = (z - density.center) / density.scale;
    T r = density.shape * asinh(x) - density.skew;
    return density.log_constant - 0.5 * log1p(x * x) - 0.5 * (r * r);
}

}
