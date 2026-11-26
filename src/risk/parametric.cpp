#include "dist/cdf.hpp"
#include "dist/johnson.hpp"
#include "risk/measures.hpp"

#include <algorithm>
#include <array>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/math/quadrature/tanh_sinh.hpp>
#include <cmath>
#include <limits>
#include <numbers>

namespace ve::risk
{
namespace
{

// The quadratures stop at this relative change between levels; the error left is far below it for these smooth
// integrands (the closed forms of norm and std are matched to 1e-10 in the tests).
constexpr double kTolerance = 1e-13;

double normal_tail_mean(double q, double p)
{
    return -std::exp(-0.5 * q * q) / (std::sqrt(2.0 * std::numbers::pi) * p);
}

// q is the standardized quantile, so the Student-t quantile is q / s with s = sqrt((nu - 2) / nu).
double student_tail_mean(double nu, double q, double p)
{
    const double scale = std::sqrt((nu - 2.0) / nu);
    const double t = q / scale;
    const double density = boost::math::pdf(boost::math::students_t_distribution<double>(nu), t);
    return -scale * density * (nu + t * t) / ((nu - 1.0) * p);
}

}

namespace
{

// log Phi(x); below -37 erfc underflows and the asymptotic series takes over (its next term is below 1e-11 there).
double log_normal_cdf(double x)
{
    if (x > -37.0)
    {
        return std::log(0.5 * std::erfc(-x / std::numbers::sqrt2));
    }
    const double inverse_square = 1.0 / (x * x);
    return -0.5 * x * x - std::log(-x) - 0.5 * std::log(2.0 * std::numbers::pi)
        + std::log1p(-inverse_square + 3.0 * inverse_square * inverse_square);
}

// The Johnson SU is z = center + scale sinh((Z + skew) / shape), Z standard normal, so with a = Phi^-1(p) and
// E[e^(bZ); Z <= a] = e^(b^2 / 2) Phi(a - b):
//   m_p = center + scale / (2p) [e^(skew/shape + h) Phi(a - 1/shape) - e^(-skew/shape + h) Phi(a + 1/shape)],
// h = 1 / (2 shape^2), each term taken in logs so that a small shape does not overflow.
double johnson_tail_mean(const dist::ShapeParameters<double>& parameters, double p)
{
    const auto density = dist::johnson_log_density(parameters.skew, parameters.shape);
    const double a = dist::quantile(Dist::Norm, {}, p);
    const double inverse = 1.0 / density.shape;
    const double base = std::log(density.scale) + 0.5 * inverse * inverse - std::log(2.0 * p);
    const double rising = std::exp(base + density.skew * inverse + log_normal_cdf(a - inverse));
    const double falling = std::exp(base - density.skew * inverse + log_normal_cdf(a + inverse));
    return density.center + rising - falling;
}

// The integral of z f(z) over (-inf, q], cut at 0 and at the density's kink (ged and the Fernandez-Steel densities)
// where they lie below q: exp-sinh on the half line, tanh-sinh on each finite piece, so no rule straddles a kink.
double tail_integral(const dist::Density<double>& density, double q)
{
    const auto integrand = [&density](double z) { return z * std::exp(dist::log_density(density, z)); };
    std::array<double, 3> cuts{std::min(0.0, dist::density_kink(density)), std::max(0.0, dist::density_kink(density)), q};
    for (double& cut : cuts)
    {
        cut = std::min(cut, q);
    }
    double integral = boost::math::quadrature::exp_sinh<double>().integrate(
        integrand, -std::numeric_limits<double>::infinity(), cuts[0], kTolerance);
    for (std::size_t k = 1; k < cuts.size(); ++k)
    {
        if (cuts[k] > cuts[k - 1])
        {
            integral += boost::math::quadrature::tanh_sinh<double>().integrate(integrand, cuts[k - 1], cuts[k], kTolerance);
        }
    }
    return integral;
}

// The tail mean at a quantile q already solved: nig and ghyp solve it by root finding, so it is solved once.
double tail_mean_at(Dist dist, const dist::ShapeParameters<double>& parameters, double p, double q)
{
    if (dist == Dist::Norm)
    {
        return normal_tail_mean(q, p);
    }
    if (dist == Dist::Std)
    {
        return student_tail_mean(parameters.shape, q, p);
    }
    if (dist == Dist::Jsu)
    {
        return johnson_tail_mean(parameters, p);
    }
    return tail_integral(dist::make_density(dist, parameters), q) / p;
}

}

double tail_mean_by_quadrature(Dist dist, const dist::ShapeParameters<double>& parameters, double p)
{
    return tail_integral(dist::make_density(dist, parameters), dist::quantile(dist, parameters, p)) / p;
}

double tail_mean(Dist dist, const dist::ShapeParameters<double>& parameters, double p)
{
    return tail_mean_at(dist, parameters, p, dist::quantile(dist, parameters, p));
}

namespace
{

// Past a skew of ~1.3e154 (or below its inverse) xi^2 overflows the Fernandez-Steel standardisation. From 1e150 on
// the density already is its half-kernel limit to rounding: the other half holds a mass 1 / (1 + xi^2) below 1e-300
// and the shift and stretch move by a relative xi^-2, so a skew past the edge is measured at the edge.
constexpr double kFernandezSteelSkewEdge = 1e150;

dist::ShapeParameters<double> within_skew_edge(Dist dist, dist::ShapeParameters<double> parameters)
{
    if (dist::is_fernandez_steel(dist))
    {
        parameters.skew = std::clamp(parameters.skew, 1.0 / kFernandezSteelSkewEdge, kFernandezSteelSkewEdge);
    }
    return parameters;
}

}

Measure parametric(Dist dist, const dist::ShapeParameters<double>& shape_parameters, Location location,
                   double confidence)
{
    const dist::ShapeParameters<double> parameters = within_skew_edge(dist, shape_parameters);
    const double p = 1.0 - confidence;
    const double q = dist::quantile(dist, parameters, p);
    return Measure{.var = -(location.mean + location.sigma * q),
                   .es = -(location.mean + location.sigma * tail_mean_at(dist, parameters, p, q))};
}

}
