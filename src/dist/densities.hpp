#pragma once

#include "ad/dual.hpp"
#include "core/spec.hpp"

#include <cmath>
#include <numbers>

namespace ve::dist
{

// log f(z) of a density with mean 0 and variance 1, as rugarch and arch standardize them:
//   norm: log_constant - z^2 / 2
//   std:  log_constant - power * log1p(z^2 / scale), scale = nu - 2, power = (nu + 1) / 2
//   ged:  log_constant - |z / scale|^power / 2,       scale = lambda,  power = nu
template <class T>
struct LogDensity
{
    Dist dist;
    T log_constant;
    T scale;
    T power;
};

inline bool in_domain(Dist dist, double shape)
{
    if (dist == Dist::Std)
    {
        return shape > 2.0;
    }
    if (dist == Dist::Ged)
    {
        return shape > 0.0;
    }
    return true;
}

namespace detail
{

// The Stirling series of lgamma(x) through x^-11 is accurate to ~1e-15 only from x = 10.
constexpr double kStirlingThreshold = 10.0;

// lgamma(x) - ((x - 1/2) log(x) - x + log(2 pi) / 2).
template <class T>
T stirling_series(const T& x)
{
    T inverse = 1.0 / x;
    T inverse2 = inverse * inverse;
    return inverse *
           (1.0 / 12.0 +
            inverse2 *
                (-1.0 / 360.0 +
                 inverse2 * (1.0 / 1260.0 +
                             inverse2 * (-1.0 / 1680.0 +
                                         inverse2 * (1.0 / 1188.0 + inverse2 * (-691.0 / 360360.0))))));
}

// lgamma((nu + 1) / 2) - lgamma(nu / 2) - log(pi (nu - 2)) / 2, with a = nu / 2. Its terms grow as a log(a)
// while it tends to -log(2 pi) / 2, so subtracting them leaves an error of ulp(a log(a)), on a likelihood
// whose slope in nu vanishes as nu grows. For large a the cancellation is done analytically: log(a) / 2 - log(pi (2 a - 2)) / 2 = -log(2 pi) / 2 - log1p(-1 / a) / 2,
// and the lgamma difference is the difference of the Stirling expansions of both terms.
template <class T>
T student_log_constant(const T& nu)
{
    using std::lgamma;
    using std::log;
    using std::log1p;
    T a = nu / 2.0;
    if (ad::value_of(a) < kStirlingThreshold)
    {
        return (lgamma(a + 0.5) - lgamma(a)) - 0.5 * log(std::numbers::pi * (nu - 2.0));
    }
    T expansion = (a * log1p(0.5 / a) - 0.5) + (stirling_series(a + 0.5) - stirling_series(a));
    return expansion - 0.5 * log1p(-1.0 / a) - 0.5 * std::log(2.0 * std::numbers::pi);
}

}

template <class T>
LogDensity<T> student_log_density(const T& nu)
{
    T scale = nu - 2.0;
    T power = (nu + 1.0) / 2.0;
    T log_constant = detail::student_log_constant(nu);
    return {Dist::Std, log_constant, scale, power};
}

// log lambda, with lambda = sqrt(2^(-2/nu) Gamma(1/nu) / Gamma(3/nu)). lambda itself underflows to 0 for nu below
// ~0.003, its logarithm stays finite.
template <class T>
T ged_log_scale(const T& nu)
{
    using std::lgamma;
    T inverse = 1.0 / nu;
    return 0.5 * (-2.0 * inverse * std::numbers::ln2 + lgamma(inverse) - lgamma(3.0 * inverse));
}

template <class T>
LogDensity<T> ged_log_density(const T& nu)
{
    using std::exp;
    using std::lgamma;
    using std::log;
    T inverse = 1.0 / nu;
    T log_lambda = ged_log_scale(nu);
    T log_constant = log(nu) - log_lambda - (1.0 + inverse) * std::numbers::ln2 - lgamma(inverse);
    return {Dist::Ged, log_constant, exp(log_lambda), nu};
}

// shape is read only for std and ged.
template <class T>
LogDensity<T> make_log_density(Dist dist, const T& shape)
{
    if (dist == Dist::Std)
    {
        return student_log_density(shape);
    }
    if (dist == Dist::Ged)
    {
        return ged_log_density(shape);
    }
    double log_two_pi = std::log(2.0 * std::numbers::pi);
    return {Dist::Norm, T{-0.5 * log_two_pi}, T{1.0}, T{1.0}};
}

template <class T>
T log_density(const LogDensity<T>& density, const T& z)
{
    using std::abs;
    using std::log1p;
    using std::pow;
    if (density.dist == Dist::Std)
    {
        return density.log_constant - density.power * log1p(z * z / density.scale);
    }
    if (density.dist == Dist::Ged)
    {
        return density.log_constant - 0.5 * pow(abs(z / density.scale), density.power);
    }
    return density.log_constant - 0.5 * (z * z);
}

}
