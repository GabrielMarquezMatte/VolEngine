#pragma once

#include "ad/dual.hpp"

#include <algorithm>
#include <cmath>

namespace ve::dist
{

// log K_order(x), K the modified Bessel function of the second kind. Never throws: an input outside the domain
// gives NaN, an overflow +inf and an underflow -inf.
double log_bessel_k(double order, double x);

// log(e^x K_order(x)): log K_order(x) + x, without the rounding of a large x that log_bessel_k carries, so
// that differences across orders keep their digits. Never throws, as log_bessel_k.
double log_bessel_k_scaled(double order, double x);

namespace detail
{

// d log K_v / dx = -K_(v-1) / K_v - v / x, the ratio taken as a difference of logs.
inline double log_bessel_k_slope(double order, double x)
{
    double ratio = std::exp(log_bessel_k(order - 1.0, x) - log_bessel_k(order, x));
    return -ratio - order / x;
}

// d log(e^x K_v) / dx = 1 - K_(v-1) / K_v - v / x, the ratio taken from the scaled logs.
inline double log_bessel_k_scaled_slope(double order, double x)
{
    double ratio = std::exp(log_bessel_k_scaled(order - 1.0, x) - log_bessel_k_scaled(order, x));
    return 1.0 - ratio - order / x;
}

// d f / d order by Richardson extrapolation of central differences at step and 2 step, accurate to O(step^4).
// log K is analytic in the order, so the step can be large. Boost computes K in double where long double is
// double (Apple Silicon, MSVC), and a step of 1e-5 turned its rounding into 1e-7 of a likelihood gradient.
template <class F>
double order_slope(F f, double order, double x)
{
    double step = 1e-3 * std::max(1.0, std::fabs(order));
    double fine = (f(order + step, x) - f(order - step, x)) / (2.0 * step);
    double coarse = (f(order + 2.0 * step, x) - f(order - 2.0 * step, x)) / (4.0 * step);
    return (4.0 * fine - coarse) / 3.0;
}


struct LogBesselSlopes
{
    double order;
    double x;
};

// d log K_order(x) / d order and / d x (the unscaled K: the slope in x of log(e^x K) is this + 1).
LogBesselSlopes log_bessel_k_slopes(double order, double x);

}

// Derivative in x only.
template <class T>
T log_bessel_k(double order, const T& x)
{
    double v = log_bessel_k(order, ad::value_of(x));
    return ad::chain(x, v, detail::log_bessel_k_slope(order, ad::value_of(x)));
}

// Derivatives in order and in x from the trapezoid sum of detail::log_bessel_k_slopes; the value is the double one.
template <class T>
T log_bessel_k(const T& order, const T& x)
{
    double nu = ad::value_of(order);
    double xv = ad::value_of(x);
    detail::LogBesselSlopes slopes = detail::log_bessel_k_slopes(nu, xv);
    return ad::chain(order, x, log_bessel_k(nu, xv), slopes.order, slopes.x);
}

// The scaled log of each overload above, with the same derivatives plus 1 in x.
template <class T>
T log_bessel_k_scaled(double order, const T& x)
{
    double v = log_bessel_k_scaled(order, ad::value_of(x));
    return ad::chain(x, v, detail::log_bessel_k_scaled_slope(order, ad::value_of(x)));
}

template <class T>
T log_bessel_k_scaled(const T& order, const T& x)
{
    double nu = ad::value_of(order);
    double xv = ad::value_of(x);
    detail::LogBesselSlopes slopes = detail::log_bessel_k_slopes(nu, xv);
    return ad::chain(order, x, log_bessel_k_scaled(nu, xv), slopes.order, slopes.x + 1.0);
}

}
