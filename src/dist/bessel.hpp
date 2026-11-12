#pragma once

#include "ad/dual.hpp"

#include <algorithm>
#include <cmath>

namespace ve::dist
{

// log K_order(x), K the modified Bessel function of the second kind. Never throws: an input outside the domain
// gives NaN, an overflow +inf and an underflow -inf.
double log_bessel_k(double order, double x);

namespace detail
{

// d log K_v / dx = -K_(v-1) / K_v - v / x, the ratio taken as a difference of logs.
inline double log_bessel_k_slope(double order, double x)
{
    double ratio = std::exp(log_bessel_k(order - 1.0, x) - log_bessel_k(order, x));
    return -ratio - order / x;
}

}

// Derivative in x only.
template <class T>
T log_bessel_k(double order, const T& x)
{
    double v = log_bessel_k(order, ad::value_of(x));
    return ad::chain(x, v, detail::log_bessel_k_slope(order, ad::value_of(x)));
}

// Derivative in order by central difference of log K (the one approximate derivative of the engine), in x as above.
template <class T>
T log_bessel_k(const T& order, const T& x)
{
    double nu = ad::value_of(order);
    double xv = ad::value_of(x);
    double step = 1e-5 * std::max(1.0, std::fabs(nu));
    double d_order = (log_bessel_k(nu + step, xv) - log_bessel_k(nu - step, xv)) / (2.0 * step);
    return ad::chain(order, x, log_bessel_k(nu, xv), d_order, detail::log_bessel_k_slope(nu, xv));
}

}
