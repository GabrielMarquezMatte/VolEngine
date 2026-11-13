#pragma once

#include "diagnostics/portmanteau.hpp"
#include "dist/densities.hpp"

#include <cstddef>
#include <span>

namespace ve::diagnostics
{

TestResult pearson_gof(std::span<const double> z, Dist dist, const dist::ShapeParameters<double>& parameters,
                       std::size_t groups);

}
