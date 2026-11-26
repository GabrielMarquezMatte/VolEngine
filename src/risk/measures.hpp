#pragma once

#include "core/panel.hpp"
#include "core/spec.hpp"
#include "dist/densities.hpp"

#include <span>

namespace ve::risk
{

// VaR and ES at a confidence in (0, 1) (0.99: the 1% tail), both positive as losses, in units of the return.
struct Measure
{
    double var = 0.0;
    double es = 0.0;
};

enum class ReturnKind
{
    log,
    simple,
};

// Row t of the panel is the portfolio's simple return sum_j weights[j] s_tj (s = exp(r) - 1 for log returns). VaR is
// minus its quantile at 1 - confidence with numpy.quantile's linear interpolation, ES minus the mean of the returns at
// or below it. The caller guarantees a rectangular panel with t >= 1, finite values, weights.size() == n and a
// confidence in (0, 1). A portfolio return that overflows to a non-finite value gives a NaN measure.
Measure empirical(const PanelView& returns, std::span<const double> weights, ReturnKind kind, double confidence);

// E[z | z <= q_p] of the standardized density, q_p its quantile at p in (0, 1); parameters in the density's domain.
// Closed form for norm, std and jsu, the quadrature below for the others.
double tail_mean(Dist dist, const dist::ShapeParameters<double>& parameters, double p);

// (1 / p) times the integral of z f(z) up to q_p, by double-exponential quadrature cut at 0 and at the density's kink.
double tail_mean_by_quadrature(Dist dist, const dist::ShapeParameters<double>& parameters, double p);

struct Location
{
    double mean = 0.0;
    double sigma = 0.0;
};

// The return mean + sigma z: VaR = -(mean + sigma q_p), ES = -(mean + sigma tail_mean), p = 1 - confidence. NaN where
// the density's standardisation is not a double (a jsu scale below the smallest normal); +-inf where sigma q_p
// overflows.
Measure parametric(Dist dist, const dist::ShapeParameters<double>& parameters, Location location, double confidence);

}
