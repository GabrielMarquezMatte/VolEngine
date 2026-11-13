#include "dist/cdf.hpp"
#include "dist/moments.hpp"

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <algorithm>
#include <array>
#include <variant>
#include <cmath>
#include <limits>
#include <numbers>

namespace ve::dist
{
namespace
{

double normal_cdf(double z)
{
    return 0.5 * std::erfc(-z / std::numbers::sqrt2);
}

double kernel_cdf(const LogDensity<double>& density, double z)
{
    if (density.dist == Dist::Norm)
    {
        return normal_cdf(z);
    }
    if (density.dist == Dist::Std)
    {
        double nu = 2.0 * density.power - 1.0;
        double tail = 0.5 * boost::math::ibeta(nu / 2.0, 0.5, density.scale / (density.scale + z * z));
        if (z < 0.0)
        {
            return tail;
        }
        return 1.0 - tail;
    }
    double argument = 0.5 * std::pow(std::abs(z) / density.scale, density.power);
    double tail = 0.5 * boost::math::gamma_q(1.0 / density.power, argument);
    if (z < 0.0)
    {
        return tail;
    }
    return 1.0 - tail;
}

double skewed_cdf(const SkewedLogDensity<double>& density, double z)
{
    double y = density.shift + density.stretch * z;
    double skew_squared = density.skew * density.skew;
    double denominator = 1.0 + skew_squared;
    if (y < 0.0)
    {
        return 2.0 / denominator * kernel_cdf(density.kernel, y * density.skew);
    }
    return (1.0 + 2.0 * skew_squared * (kernel_cdf(density.kernel, y / density.skew) - 0.5)) / denominator;
}

// The width over which a nig or ghyp density changes: delta below 1 puts its peak within a few delta of the location.
double density_scale(const detail::DensityIntegrand<double>& integrand)
{
    const auto* hyperbolic = std::get_if<HyperbolicLogDensity<double>>(&integrand.density);
    if (hyperbolic == nullptr)
    {
        return 1.0;
    }
    return std::min(1.0, hyperbolic->delta);
}

// Five-point Gauss-Legendre on [0, 1]: exact to rounding on a piece much narrower than the density scale, which is
// every gap between neighbouring residuals of a fit.
constexpr std::array<detail::QuadratureNode, 5> kShortRule{{
    {0.04691007703066800, 0.11846344252809454},
    {0.23076534494715845, 0.23931433524968324},
    {0.5, 0.28444444444444444},
    {0.76923465505284155, 0.23931433524968324},
    {0.95308992296933200, 0.11846344252809454},
}};
constexpr double kShortPiece = 0.02;

double short_integral(const detail::DensityIntegrand<double>& integrand, double lower, double upper)
{
    double width = upper - lower;
    double sum = 0.0;
    for (const detail::QuadratureNode& node : kShortRule)
    {
        sum += node.weight * integrand(lower + width * node.t);
    }
    return width * sum;
}

// int over [lower, upper] in pieces no wider than the density scale (at most 1024 of them).
double integrate_segment(const detail::DensityIntegrand<double>& integrand, double lower, double upper)
{
    double scale = density_scale(integrand);
    if (!(upper > lower))
    {
        return 0.0;
    }
    if (upper - lower <= kShortPiece * scale)
    {
        return short_integral(integrand, lower, upper);
    }
    double pieces = std::min(1024.0, std::ceil((upper - lower) / scale));
    double width = (upper - lower) / pieces;
    double sum = 0.0;
    for (double i = 0.0; i < pieces; i += 1.0)
    {
        sum += detail::segment_integral(integrand, lower + i * width, lower + (i + 1.0) * width);
    }
    return sum;
}

// The integral from 0, for |z| <= 1.
double central_cdf(const detail::DensityIntegrand<double>& integrand, double z)
{
    double below_zero = detail::below_zero_quadrature(integrand, 1.0);
    if (z < 0.0)
    {
        return below_zero - integrate_segment(integrand, z, 0.0);
    }
    return below_zero + integrate_segment(integrand, 0.0, z);
}

// Beyond |z| = 1 the tail integral, scaled so that it meets the central integral at +-1: the two quadratures differ
// by up to ~1e-8 on a peaked density, which would otherwise make the cdf jump there. The scale keeps the tail going
// to 0 (or 1).
double quadrature_cdf(const detail::DensityIntegrand<double>& integrand, double z)
{
    if (z < -1.0)
    {
        double scale = central_cdf(integrand, -1.0) / detail::tail_integral(integrand, 0.0, -1.0, 1.0);
        return std::clamp(scale * detail::tail_integral(integrand, z + 1.0, -1.0, 1.0), 0.0, 1.0);
    }
    if (z > 1.0)
    {
        double scale = (1.0 - central_cdf(integrand, 1.0)) / detail::tail_integral(integrand, 0.0, 1.0, 1.0);
        return std::clamp(1.0 - scale * detail::tail_integral(integrand, z - 1.0, 1.0, 1.0), 0.0, 1.0);
    }
    return std::clamp(central_cdf(integrand, z), 0.0, 1.0);
}

bool needs_quadrature(Dist dist)
{
    return dist == Dist::Nig || dist == Dist::Ghyp;
}

}

double cdf(Dist dist, const ShapeParameters<double>& parameters, double z)
{
    if (!in_domain(dist, parameters) || std::isnan(z))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (std::isinf(z))
    {
        return z < 0.0 ? 0.0 : 1.0;
    }
    if (detail::is_symmetric(dist))
    {
        return kernel_cdf(make_log_density(dist, parameters.shape), z);
    }
    if (is_fernandez_steel(dist))
    {
        return skewed_cdf(skewed_log_density(dist, parameters.skew, parameters.shape), z);
    }
    if (dist == Dist::Jsu)
    {
        auto density = johnson_log_density(parameters.skew, parameters.shape);
        return normal_cdf(detail::johnson_normal_of(density, z));
    }
    return quadrature_cdf({make_density(dist, parameters)}, z);
}

std::vector<double> sorted_cdf(Dist dist, const ShapeParameters<double>& parameters,
                               std::span<const double> sorted)
{
    std::vector<double> result(sorted.size());
    if (!needs_quadrature(dist) || !in_domain(dist, parameters))
    {
        std::ranges::transform(sorted, result.begin(), [&](double z) { return cdf(dist, parameters, z); });
        return result;
    }
    if (!sorted.empty())
    {
        detail::DensityIntegrand<double> integrand{make_density(dist, parameters)};
        result[0] = cdf(dist, parameters, sorted[0]);
        for (std::size_t i = 1; i < sorted.size(); ++i)
        {
            double probability = result[i - 1] + integrate_segment(integrand, sorted[i - 1], sorted[i]);
            result[i] = std::clamp(probability, 0.0, 1.0);
        }
    }
    return result;
}

}
