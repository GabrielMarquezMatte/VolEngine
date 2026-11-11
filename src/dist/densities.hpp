#pragma once

#include "core/spec.hpp"

#include <cmath>
#include <numbers>

namespace ve::dist
{

// log f(z) of a density with mean 0 and variance 1, as rugarch and arch standardize them:
//   norm: log_constant - z^2 / 2
//   std:  log_constant - power * log(1 + z^2 / scale), scale = nu - 2, power = (nu + 1) / 2
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

template <class T>
LogDensity<T> student_log_density(const T& nu)
{
    using std::lgamma;
    using std::log;
    T scale = nu - 2.0;
    T power = (nu + 1.0) / 2.0;
    T log_normalizer = lgamma(power) - lgamma(nu / 2.0);
    T log_constant = log_normalizer - 0.5 * log(std::numbers::pi * scale);
    return {Dist::Std, log_constant, scale, power};
}

// lambda = sqrt(2^(-2/nu) Gamma(1/nu) / Gamma(3/nu)), taken through its logarithm.
template <class T>
LogDensity<T> ged_log_density(const T& nu)
{
    using std::exp;
    using std::lgamma;
    using std::log;
    T inverse = 1.0 / nu;
    T log_lambda = 0.5 * (-2.0 * inverse * std::numbers::ln2 + lgamma(inverse) - lgamma(3.0 * inverse));
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
    using std::log;
    using std::pow;
    if (density.dist == Dist::Std)
    {
        return density.log_constant - density.power * log(1.0 + z * z / density.scale);
    }
    if (density.dist == Dist::Ged)
    {
        return density.log_constant - 0.5 * pow(abs(z / density.scale), density.power);
    }
    return density.log_constant - 0.5 * (z * z);
}

}
