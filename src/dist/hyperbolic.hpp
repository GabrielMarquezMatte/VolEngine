#pragma once

#include "ad/dual.hpp"
#include "dist/bessel.hpp"

#include <cmath>
#include <numbers>

namespace ve::dist
{

// The generalized hyperbolic density in the (rho, zeta) parametrization, with mean 0 and variance 1:
//   log f(z) = log_constant + ((order - 1/2) / 2) log(q) + log K_(order - 1/2)(alpha sqrt(q))
//              + beta (z - location),
// q = delta2 + (z - location)^2. nig is order = -1/2 with free_order false: its Bessel orders are doubles and
// carry no derivative.
template <class T>
struct HyperbolicLogDensity
{
    T log_constant;
    T alpha;
    T beta;
    T delta2;
    T location;
    T order;
    bool free_order;
};

namespace detail
{

template <class T>
T hyperbolic_log_bessel(const T& order, bool free_order, const T& x)
{
    if (free_order)
    {
        return log_bessel_k(order, x);
    }
    return log_bessel_k(ad::value_of(order), x);
}

}

// With R = K_(order + 1)(zeta) / K_order(zeta) and Q = K_(order + 2)(zeta) / K_order(zeta):
//   delta = [R / zeta + rho^2 / (1 - rho^2) (Q - R^2)]^(-1/2), alpha = zeta / (delta sqrt(1 - rho^2)),
//   beta = rho alpha, location = -beta delta^2 R / zeta,
// and log a = (order / 2) log(alpha^2 - beta^2) - log(2 pi) / 2 - (order - 1/2) log(alpha) - order log(delta)
// - log K_order(zeta), taken as (order / 2) log(1 - rho^2) + log(alpha) / 2 - ... since alpha^2 - beta^2 =
// alpha^2 (1 - rho^2).
template <class T>
HyperbolicLogDensity<T> hyperbolic_log_density(const T& rho, const T& zeta, const T& order, bool free_order)
{
    using std::exp;
    using std::log;
    using std::sqrt;
    T log_k = detail::hyperbolic_log_bessel(order, free_order, zeta);
    T r = exp(detail::hyperbolic_log_bessel(order + 1.0, free_order, zeta) - log_k);
    T q = exp(detail::hyperbolic_log_bessel(order + 2.0, free_order, zeta) - log_k);
    T one_minus_rho2 = 1.0 - rho * rho;
    T delta2 = 1.0 / (r / zeta + (rho * rho / one_minus_rho2) * (q - r * r));
    T alpha = zeta / sqrt(delta2 * one_minus_rho2);
    T beta = rho * alpha;
    T location = -(beta * delta2 * r / zeta);
    T log_constant = 0.5 * order * log(one_minus_rho2) + 0.5 * log(alpha) -
                     0.5 * std::log(2.0 * std::numbers::pi) - 0.5 * order * log(delta2) - log_k;
    return {log_constant, alpha, beta, delta2, location, order, free_order};
}

template <class T>
T log_density(const HyperbolicLogDensity<T>& density, const T& z)
{
    using std::log;
    using std::sqrt;
    T offset = z - density.location;
    T q = density.delta2 + offset * offset;
    T order = density.order - 0.5;
    T log_k = detail::hyperbolic_log_bessel(order, density.free_order, density.alpha * sqrt(q));
    return density.log_constant + 0.5 * order * log(q) + log_k + density.beta * offset;
}

}
