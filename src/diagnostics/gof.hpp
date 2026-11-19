#pragma once

#include "diagnostics/portmanteau.hpp"
#include "dist/densities.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace ve::diagnostics
{

// The statistic of the probabilities (the CDF of the standardized residuals) in `groups` equal bins of (0, 1), NaN for
// a non-finite one. The CDF of sorted values is non-decreasing only up to its own rounding, and the empty groups
// between two bins are counted by their difference: probabilities one ulp out of order across a bin edge would wrap
// it, so they are sorted first.
double pearson_statistic(std::vector<double>& probabilities, std::size_t groups);

TestResult pearson_gof(std::span<const double> z, Dist dist, const dist::ShapeParameters<double>& parameters,
                       std::size_t groups);

}
