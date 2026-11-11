#pragma once

#include "model/layout.hpp"
#include "model/parameter_map.hpp"

#include <span>
#include <vector>

namespace ve::model
{

// Natural-space start: mu = mean(r), phi = theta = 0, omega = var(r) (1 - 0.95), alpha_i = 0.05 / p,
// beta_j = 0.90 / q, shape = 8 (std) or 2 (ged). r must not be empty.
std::vector<double> start_values(std::span<const double> r, const Layout& layout);

// Mean and population standard deviation of r.
Standardization standardization_of(std::span<const double> r);

}
