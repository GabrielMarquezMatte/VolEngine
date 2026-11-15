#pragma once

#include "ad/dual.hpp"
#include "dcc/spec.hpp"
#include "dist/bessel.hpp"

#include <cmath>
#include <numbers>

namespace ve::dcc::detail
{

// The terms of log f(z; R) that depend on neither R nor z.
template <class T>
struct Density
{
    DccDist dist;
    double m;
    T shape;
    T constant;
};

template <class T>
T density_constant(DccDist dist, double m, const T& shape)
{
    using ve::ad::log_gamma;
    using std::log;
    const double log_two_pi = std::log(2.0 * std::numbers::pi);
    if (dist == DccDist::MvT)
    {
        const T log_gamma_ratio = log_gamma(0.5 * (shape + m)) - log_gamma(0.5 * shape);
        return log_gamma_ratio - 0.5 * m * log(std::numbers::pi * (shape - 2.0));
    }
    if (dist == DccDist::MvLaplace)
    {
        return T{std::numbers::ln2 - 0.5 * m * log_two_pi};
    }
    return T{-0.5 * m * log_two_pi};
}

// log f(z; R) from L = log |R| and q = z' R^-1 z.
template <class T>
T log_density(const Density<T>& density, const T& log_det, const T& quadratic)
{
    using std::log;
    using std::log1p;
    using std::sqrt;
    const T half_log_det = 0.5 * log_det;
    if (density.dist == DccDist::MvT)
    {
        const T tail = 0.5 * (density.shape + density.m) * log1p(quadratic / (density.shape - 2.0));
        return density.constant - half_log_det - tail;
    }
    if (density.dist == DccDist::MvLaplace)
    {
        const double order = 1.0 - 0.5 * density.m;
        const T power = 0.5 * order * log(0.5 * quadratic);
        const T bessel = dist::log_bessel_k(order, sqrt(2.0 * quadratic));
        return density.constant - half_log_det + power + bessel;
    }
    return density.constant - 0.5 * (log_det + quadratic);
}

}
