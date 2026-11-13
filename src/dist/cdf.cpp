#include "dist/cdf.hpp"
#include "dist/moments.hpp"

#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>

#include <algorithm>
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

double integrate_segment(const detail::DensityIntegrand<double>& integrand, double lower, double upper)
{
    double pieces = std::min(1024.0, std::ceil(upper - lower));
    if (pieces <= 0.0)
    {
        return 0.0;
    }
    double width = (upper - lower) / pieces;
    double sum = 0.0;
    for (double i = 0.0; i < pieces; i += 1.0)
    {
        sum += detail::segment_integral(integrand, lower + i * width, lower + (i + 1.0) * width);
    }
    return sum;
}

double quadrature_cdf(const detail::DensityIntegrand<double>& integrand, double z)
{
    if (z < -1.0)
    {
        return std::clamp(detail::tail_integral(integrand, z + 1.0, -1.0, 1.0), 0.0, 1.0);
    }
    if (z > 1.0)
    {
        return std::clamp(1.0 - detail::tail_integral(integrand, z - 1.0, 1.0, 1.0), 0.0, 1.0);
    }
    double below_zero = detail::below_zero_quadrature(integrand, 1.0);
    if (z < 0.0)
    {
        return std::clamp(below_zero - integrate_segment(integrand, z, 0.0), 0.0, 1.0);
    }
    return std::clamp(below_zero + integrate_segment(integrand, 0.0, z), 0.0, 1.0);
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
