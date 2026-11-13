#pragma once

#include "dist/densities.hpp"

#include <span>
#include <vector>

namespace ve::dist
{

double cdf(Dist dist, const ShapeParameters<double>& parameters, double z);
std::vector<double> sorted_cdf(Dist dist, const ShapeParameters<double>& parameters,
                               std::span<const double> sorted);

}
