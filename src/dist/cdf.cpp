#include "dist/cdf.hpp"

#include "ad/special.hpp"
#include "dist/moments.hpp"

#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/tools/roots.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <variant>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>
#include <limits>
#include <numbers>

namespace ve::dist
{
namespace
{

// Computing in double, not promoted to long double, keeps the cdf to the bit whatever the thread's x87 precision
// (on MinGW the Python main thread and the pool threads differ, which made a copula's standard errors depend on
// whether they were computed in the batch or afterwards).
using DoublePolicy = boost::math::policies::policy<boost::math::policies::promote_double<false>>;

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
        double tail = 0.5 * boost::math::ibeta(nu / 2.0, 0.5, density.scale / (density.scale + z * z), DoublePolicy{});
        if (z < 0.0)
        {
            return tail;
        }
        return 1.0 - tail;
    }
    double argument = 0.5 * std::pow(std::abs(z) / density.scale, density.power);
    double tail = 0.5 * boost::math::gamma_q(1.0 / density.power, argument, DoublePolicy{});
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

// The integral from 0, for |z| <= 1, with the mass below 0 given.
double central_cdf(const detail::DensityIntegrand<double>& integrand, double below_zero, double z)
{
    if (z < 0.0)
    {
        return below_zero - integrate_segment(integrand, z, 0.0);
    }
    return below_zero + integrate_segment(integrand, 0.0, z);
}

double central_cdf(const detail::DensityIntegrand<double>& integrand, double z)
{
    return central_cdf(integrand, detail::below_zero_quadrature(integrand, 1.0), z);
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

detail::CdfConstants cdf_constants(const detail::DensityIntegrand<double>& integrand)
{
    double below_zero = detail::below_zero_quadrature(integrand, 1.0);
    return {below_zero, central_cdf(integrand, below_zero, -1.0) / detail::tail_integral(integrand, 0.0, -1.0, 1.0),
            (1.0 - central_cdf(integrand, below_zero, 1.0)) / detail::tail_integral(integrand, 0.0, 1.0, 1.0)};
}

// quadrature_cdf with the constants taken from cdf_constants: the same arithmetic, so the same bits.
double quadrature_cdf(const detail::DensityIntegrand<double>& integrand, const detail::CdfConstants& constants,
                      double z)
{
    if (z < -1.0)
    {
        return std::clamp(constants.lower_scale * detail::tail_integral(integrand, z + 1.0, -1.0, 1.0), 0.0, 1.0);
    }
    if (z > 1.0)
    {
        return std::clamp(1.0 - constants.upper_scale * detail::tail_integral(integrand, z - 1.0, 1.0, 1.0), 0.0, 1.0);
    }
    return std::clamp(central_cdf(integrand, constants.below_zero, z), 0.0, 1.0);
}

bool needs_quadrature(Dist dist)
{
    return dist == Dist::Nig || dist == Dist::Ghyp;
}


// Below this gamma_q_inv has underflowed (it returns 0 or a denormal).
constexpr double kTinyArgument = 1e-290;

using NormalDistribution = boost::math::normal_distribution<double, DoublePolicy>;

// A solve that returns its last bracket instead of throwing when it runs out of iterations.
using RootPolicy = boost::math::policies::policy<boost::math::policies::promote_double<false>,
                                                 boost::math::policies::evaluation_error<boost::math::policies::ignore_error>>;

double normal_quantile(double u)
{
    return boost::math::quantile(NormalDistribution{}, u);
}

// The z <= 0 with kernel_cdf(z) == tail, tail in (0, 1/2]: the inverse of the formulas of kernel_cdf.
double kernel_lower_quantile(const LogDensity<double>& density, double tail)
{
    if (density.dist == Dist::Norm)
    {
        return normal_quantile(tail);
    }
    if (density.dist == Dist::Std)
    {
        double nu = 2.0 * density.power - 1.0;
        double complement = 0.0;
        double ratio = boost::math::ibeta_inv(nu / 2.0, 0.5, 2.0 * tail, &complement, DoublePolicy{});
        return -std::sqrt(density.scale * complement / ratio);
    }
    double order = 1.0 / density.power;
    double argument = boost::math::gamma_q_inv(order, 2.0 * tail, DoublePolicy{});
    if (argument > kTinyArgument)
    {
        return -density.scale * std::pow(2.0 * argument, order);
    }
    // A large power puts the argument of the central quantiles below the double range (P(a, x) ~ x^a / Gamma(a + 1)
    // for a = 1/power): the same root, in logs.
    double log_argument = (std::log1p(-2.0 * tail) + ve::ad::log_gamma(order + 1.0)) * density.power;
    return -density.scale * std::exp(order * (std::numbers::ln2 + log_argument));
}

double kernel_quantile(const LogDensity<double>& density, double u)
{
    if (u < 0.5)
    {
        return kernel_lower_quantile(density, u);
    }
    return -kernel_lower_quantile(density, 1.0 - u);
}

// skewed_cdf inverted on each side of y = 0, where P(y < 0) = 1 / (1 + skew^2). The upper side goes through the
// upper tail, 1 - u, so it keeps its digits as u tends to 1.
double skewed_quantile(const SkewedLogDensity<double>& density, double u)
{
    double skew_squared = density.skew * density.skew;
    double denominator = 1.0 + skew_squared;
    if (u * denominator < 1.0)
    {
        double y = kernel_quantile(density.kernel, 0.5 * u * denominator) / density.skew;
        return (y - density.shift) / density.stretch;
    }
    double y = -density.skew * kernel_lower_quantile(density.kernel, 0.5 * (1.0 - u) * denominator / skew_squared);
    return (y - density.shift) / density.stretch;
}

constexpr int kRootIterations = 60;
constexpr int kBracketDoublings = 60;

// ponytail: ~20 quadrature cdfs per call (~110-140 us); QuantileTable (below) answers in ~3 us for a caller with many
// probabilities of one distribution, and falls back to this solve beyond its grid. Add a table to a new caller that
// needs one nig or ghyp quantile at a time only if it shows up in a profile.
// The root of cdf(z) - u by toms748 on a bracket grown from [-1, 1] (mean 0, variance 1) until the sign changes.
double solved_quantile(const detail::DensityIntegrand<double>& integrand, double u)
{
    auto residual = [&](double z) { return quadrature_cdf(integrand, z) - u; };
    double lower = -1.0;
    double upper = 1.0;
    for (int i = 0; i < kBracketDoublings && residual(lower) > 0.0; ++i)
    {
        lower *= 2.0;
    }
    for (int i = 0; i < kBracketDoublings && residual(upper) < 0.0; ++i)
    {
        upper *= 2.0;
    }
    double at_lower = residual(lower);
    double at_upper = residual(upper);
    if (at_lower == 0.0)
    {
        return lower;
    }
    if (at_upper == 0.0)
    {
        return upper;
    }
    if (!(at_lower < 0.0 && at_upper > 0.0))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    auto iterations = static_cast<std::uintmax_t>(kRootIterations);
    auto bracket = boost::math::tools::toms748_solve(residual, lower, upper, at_lower, at_upper,
                                                     boost::math::tools::eps_tolerance<double>(48), iterations,
                                                     RootPolicy{});
    return 0.5 * (bracket.first + bracket.second);
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

double quantile(Dist dist, const ShapeParameters<double>& parameters, double u)
{
    if (!in_domain(dist, parameters) || !(u >= 0.0 && u <= 1.0))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (u == 0.0 || u == 1.0)
    {
        return u == 0.0 ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    }
    if (detail::is_symmetric(dist))
    {
        return kernel_quantile(make_log_density(dist, parameters.shape), u);
    }
    if (is_fernandez_steel(dist))
    {
        return skewed_quantile(skewed_log_density(dist, parameters.skew, parameters.shape), u);
    }
    if (dist == Dist::Jsu)
    {
        auto density = johnson_log_density(parameters.skew, parameters.shape);
        double x = std::sinh((normal_quantile(u) + density.skew) / density.shape);
        return density.center + density.scale * x;
    }
    return solved_quantile({make_density(dist, parameters)}, u);
}

namespace
{

// The grid is z = kTableCore sinh(w), w uniform: fine at the core of the density, coarse in the tails, out to
// +-kTableReach. At this many nodes the Hermite guess is within ~1e-7 of the root, which one Newton step turns into
// rounding; a second step confirms it. A probability beyond the grid's cdf takes the root solve.
constexpr std::size_t kTableNodes = 513;
constexpr double kTableCore = 0.25;
constexpr double kTableReach = 40.0;
constexpr int kPolishSteps = 4;
constexpr double kPolishTolerance = 1e-7;

struct TableGrid
{
    std::vector<double> z;
    std::vector<double> cdf;
    std::vector<double> pdf;
};

bool node_usable(double cdf, double pdf)
{
    return std::isfinite(cdf) && cdf > 0.0 && cdf < 1.0 && std::isfinite(pdf) && pdf > 0.0;
}

// The nodes of the grid whose cdf is strictly increasing outward from the middle one, and in (0, 1).
TableGrid usable_grid(const detail::DensityIntegrand<double>& integrand, const detail::CdfConstants& constants)
{
    const double w_max = std::asinh(kTableReach / kTableCore);
    TableGrid all;
    for (std::size_t i = 0; i < kTableNodes; ++i)
    {
        double w = -w_max + 2.0 * w_max * static_cast<double>(i) / static_cast<double>(kTableNodes - 1);
        double z = kTableCore * std::sinh(w);
        all.z.push_back(z);
        all.cdf.push_back(quadrature_cdf(integrand, constants, z));
        all.pdf.push_back(integrand(z));
    }
    const std::size_t middle = kTableNodes / 2;
    std::size_t first = middle;
    while (first > 0 && node_usable(all.cdf[first - 1], all.pdf[first - 1]) && all.cdf[first - 1] < all.cdf[first])
    {
        --first;
    }
    std::size_t last = middle;
    while (last + 1 < kTableNodes && node_usable(all.cdf[last + 1], all.pdf[last + 1]) &&
           all.cdf[last + 1] > all.cdf[last])
    {
        ++last;
    }
    const auto begin = static_cast<std::ptrdiff_t>(first);
    const auto end = static_cast<std::ptrdiff_t>(last + 1);
    return {{all.z.begin() + begin, all.z.begin() + end},
            {all.cdf.begin() + begin, all.cdf.begin() + end},
            {all.pdf.begin() + begin, all.pdf.begin() + end}};
}

}

QuantileTable::QuantileTable(Dist dist, const ShapeParameters<double>& parameters)
    : dist_(dist), parameters_(parameters), density_()
{
    if (!needs_quadrature(dist) || !in_domain(dist, parameters))
    {
        return;
    }
    density_ = make_density(dist, parameters);
    const detail::DensityIntegrand<double> integrand{density_};
    constants_ = cdf_constants(integrand);
    TableGrid grid = usable_grid(integrand, constants_);
    if (grid.z.size() < 2)
    {
        return;
    }
    z_ = std::move(grid.z);
    cdf_ = std::move(grid.cdf);
    pdf_ = std::move(grid.pdf);
}

// The z in [z_segment, z_(segment + 1)] where the cubic Hermite interpolant of the cdf (values and densities at the
// nodes) equals u, from the linear interpolation by Newton steps in the unit coordinate s.
double QuantileTable::hermite_guess(std::size_t segment, double u) const
{
    const double z0 = z_[segment];
    const double width = z_[segment + 1] - z0;
    const double f0 = cdf_[segment];
    const double f1 = cdf_[segment + 1];
    const double slope0 = width * pdf_[segment];
    const double slope1 = width * pdf_[segment + 1];
    double s = std::clamp((u - f0) / (f1 - f0), 0.0, 1.0);
    for (int step = 0; step < 3; ++step)
    {
        const double s2 = s * s;
        const double s3 = s2 * s;
        const double value = (2.0 * s3 - 3.0 * s2 + 1.0) * f0 + (s3 - 2.0 * s2 + s) * slope0 +
                             (-2.0 * s3 + 3.0 * s2) * f1 + (s3 - s2) * slope1;
        const double derivative = (6.0 * s2 - 6.0 * s) * f0 + (3.0 * s2 - 4.0 * s + 1.0) * slope0 +
                                  (-6.0 * s2 + 6.0 * s) * f1 + (3.0 * s2 - 2.0 * s) * slope1;
        if (!(derivative > 0.0))
        {
            break;
        }
        s = std::clamp(s - (value - u) / derivative, 0.0, 1.0);
    }
    return z0 + s * width;
}

double QuantileTable::operator()(double u) const
{
    if (z_.empty() || !(u > cdf_.front() && u < cdf_.back()))
    {
        return quantile(dist_, parameters_, u);
    }
    const auto above = std::ranges::upper_bound(cdf_, u);
    const auto segment = static_cast<std::size_t>(above - cdf_.begin()) - 1;
    const detail::DensityIntegrand<double> integrand{density_};
    double z = hermite_guess(segment, u);
    for (int step = 0; step < kPolishSteps; ++step)
    {
        const double density = integrand(z);
        const double correction = (quadrature_cdf(integrand, constants_, z) - u) / density;
        if (!(density > 0.0) || !std::isfinite(correction))
        {
            break;
        }
        z -= correction;
        if (std::fabs(correction) <= kPolishTolerance * std::max(1.0, std::fabs(z)))
        {
            return z;
        }
    }
    return quantile(dist_, parameters_, u);
}

}
