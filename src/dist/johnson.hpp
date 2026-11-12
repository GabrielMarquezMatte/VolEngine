#pragma once

#include "ad/dual.hpp"

#include <cmath>
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

// w = exp(1 / tau^2), omega = -nu / tau, c = [(w - 1) (w cosh(2 omega) + 1) / 2]^(-1/2). w - 1 is taken as
// 2 sinh(h) e^h with h = 1 / (2 tau^2), which keeps its digits as tau grows, where exp(1 / tau^2) - 1
// cancels.
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
