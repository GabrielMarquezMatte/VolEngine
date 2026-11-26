#include "dist/cdf.hpp"
#include "risk/measures.hpp"

#include <algorithm>
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

double tail_mean_by_quadrature(Dist dist, const dist::ShapeParameters<double>& parameters, double p)
{
    const dist::Density<double> density = dist::make_density(dist, parameters);
    const auto integrand = [&density](double z) { return z * std::exp(dist::log_density(density, z)); };
    const double q = dist::quantile(dist, parameters, p);
    // ponytail: one split at 0, where ged and the skewed densities have their kink in the usual tails; a kink inside
    // (q, 0) for a quantile far from the tail would cost accuracy, split at the mode if that case matters.
    const double edge = std::min(q, 0.0);
    double integral = boost::math::quadrature::exp_sinh<double>().integrate(
        integrand, -std::numeric_limits<double>::infinity(), edge, kTolerance);
    if (q > 0.0)
    {
        integral += boost::math::quadrature::tanh_sinh<double>().integrate(integrand, 0.0, q, kTolerance);
    }
    return integral / p;
}

double tail_mean(Dist dist, const dist::ShapeParameters<double>& parameters, double p)
{
    if (dist == Dist::Norm)
    {
        return normal_tail_mean(dist::quantile(dist, parameters, p), p);
    }
    if (dist == Dist::Std)
    {
        return student_tail_mean(parameters.shape, dist::quantile(dist, parameters, p), p);
    }
    return tail_mean_by_quadrature(dist, parameters, p);
}

Measure parametric(Dist dist, const dist::ShapeParameters<double>& parameters, Location location, double confidence)
{
    const double p = 1.0 - confidence;
    const double q = dist::quantile(dist, parameters, p);
    return Measure{.var = -(location.mean + location.sigma * q),
                   .es = -(location.mean + location.sigma * tail_mean(dist, parameters, p))};
}

}
