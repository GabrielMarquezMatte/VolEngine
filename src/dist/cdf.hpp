#pragma once

#include "dist/densities.hpp"

#include <span>
#include <vector>

namespace ve::dist
{

double cdf(Dist dist, const ShapeParameters<double>& parameters, double z);
std::vector<double> sorted_cdf(Dist dist, const ShapeParameters<double>& parameters,
                               std::span<const double> sorted);

// The z with cdf(dist, parameters, z) == u, u in (0, 1), of the standardized density of make_density: -inf at 0,
// +inf at 1, NaN outside [0, 1] or outside the domain. Internal (plan 4c): the copula simulation's margins; no C or
// Python surface.
double quantile(Dist dist, const ShapeParameters<double>& parameters, double u);

}
