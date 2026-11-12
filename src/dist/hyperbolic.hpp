#pragma once

#include "ad/dual.hpp"
#include "dist/bessel.hpp"

#include <cmath>
#include <numbers>

namespace ve::dist
{

// The generalized hyperbolic density in the (rho, zeta) parametrization, with mean 0 and variance 1:
//   log f(z) = log_constant + ((order - 1/2) / 2) log(q) + log(e^y K_(order - 1/2)(y)) - alpha g(x),
// x = z - location, q = delta^2 + x^2, y = alpha sqrt(q), cosine = sqrt(1 - rho^2) and
//   g(x) = sqrt(q) - rho x - delta cosine = (delta rho - x cosine)^2 / (sqrt(q) + rho x + delta cosine).
// The usual exponent beta x - y + zeta (zeta = alpha delta cosine, beta = rho alpha) is -alpha g(x): as one
// product it keeps its digits where y and zeta are large and close. nig is order = -1/2 with free_order
// false: its Bessel orders are doubles and carry no derivative.
template <class T>
struct HyperbolicLogDensity
{
    T log_constant;
    T alpha;
    T rho;
    T cosine;
    T delta;
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
        return log_bessel_k_scaled(order, x);
    }
    return log_bessel_k_scaled(ad::value_of(order), x);
}

}

// With R = K_(order + 1)(zeta) / K_order(zeta) and Q = K_(order + 2)(zeta) / K_order(zeta), both from scaled
// logs (Q - R^2 ~ 1 / zeta at large zeta):
//   delta = [R / zeta + rho^2 / (1 - rho^2) (Q - R^2)]^(-1/2), alpha = zeta / (delta sqrt(1 - rho^2)),
//   location = -rho alpha delta^2 R / zeta = -rho delta R / sqrt(1 - rho^2),
// and log a = (order / 2) log(alpha^2 - beta^2) - log(2 pi) / 2 - (order - 1/2) log(alpha) - order log(delta)
// - log K_order(zeta), taken as (order / 2) log(1 - rho^2) + log(alpha) / 2 - ... since alpha^2 - beta^2 =
// alpha^2 (1 - rho^2); its + zeta goes to the exponent, as in g.
template <class T>
HyperbolicLogDensity<T> hyperbolic_log_density(const T& rho, const T& zeta, const T& order, bool free_order)
{
    using std::exp;
    using std::log;
    using std::sqrt;
    T log_k = detail::hyperbolic_log_bessel(order, free_order, zeta);
    T r = exp(detail::hyperbolic_log_bessel(order + 1.0, free_order, zeta) - log_k);
    T q = exp(detail::hyperbolic_log_bessel(order + 2.0, free_order, zeta) - log_k);
    T cosine2 = (1.0 - rho) * (1.0 + rho);
    T cosine = sqrt(cosine2);
    T delta = 1.0 / sqrt(r / zeta + (rho * rho / cosine2) * (q - r * r));
    T alpha = zeta / (delta * cosine);
    T location = -(rho * delta * r / cosine);
    T log_constant = 0.5 * order * log(cosine2) + 0.5 * log(alpha) - 0.5 * std::log(2.0 * std::numbers::pi) -
                     order * log(delta) - log_k;
    return {log_constant, alpha, rho, cosine, delta, location, order, free_order};
}

template <class T>
T log_density(const HyperbolicLogDensity<T>& density, const T& z)
{
    using std::log;
    using std::sqrt;
    T offset = z - density.location;
    T q = density.delta * density.delta + offset * offset;
    T root = sqrt(q);
    T edge = density.delta * density.cosine;
    T gap = density.delta * density.rho - offset * density.cosine;
    T excess = gap * gap / (root + density.rho * offset + edge);
    T order = density.order - 0.5;
    T log_k = detail::hyperbolic_log_bessel(order, density.free_order, density.alpha * root);
    return density.log_constant + 0.5 * order * log(q) + log_k - density.alpha * excess;
}

// Past this zeta a nig or ghyp is not computable. Q - R^2 ~ 1 / zeta is the difference of Bessel ratios near
// 1 whose rounding (~5e-15 against long-precision values) costs it a relative ~5e-15 zeta: at most 6e-10 up
// to 1e6, 3e-7 at 1e8 and 1e-4 at 1e11, and the variance and the normalization inherit it. The density is
// by then a normal to an excess kurtosis of order 1 / zeta (3 (1 + 4 rho^2) / zeta for nig).
constexpr double kMaxHyperbolicShape = 1e6;

// Whether double precision standardizes the density: |rho| < 1, 0 < zeta <= kMaxHyperbolicShape and every
// constant finite, which a tiny zeta or a large |order| breaks by overflowing the Bessel ratios.
inline bool hyperbolic_computable(double rho, double zeta, double order)
{
    bool bounded = std::fabs(rho) < 1.0 && zeta > 0.0 && zeta <= kMaxHyperbolicShape && std::isfinite(order);
    if (!bounded)
    {
        return false;
    }
    HyperbolicLogDensity<double> density = hyperbolic_log_density(rho, zeta, order, false);
    bool positive = density.alpha > 0.0 && density.delta > 0.0;
    return positive && std::isfinite(density.alpha) && std::isfinite(density.delta) &&
           std::isfinite(density.location) && std::isfinite(density.log_constant);
}

}
