#pragma once

#include "model/layout.hpp"
#include "model/parameter_map.hpp"

#include <span>
#include <vector>

namespace ve::model
{

// Natural-space start: mu = mean(r), ARMA = 0, beta_j = 0.9 / q, shape = 8 (std) or 2 (ged), and by model
//   sGARCH:   omega = 0.05 var, alpha = 0.05 / p
//   eGARCH:   omega = 0.1 log var, alpha = 0, gamma = 0.1 / p
//   gjrGARCH: omega = 0.05 var, alpha = 0.03 / p, gamma = 0.04 / p
//   iGARCH:   omega = 0.01 var, alpha = 0.1 / p (the derived beta_q is 0.9 / q as well)
//   csGARCH:  omega = 0.02 var, alpha = 0.05 / p, rho = 0.98, phi = 0.03
//   apARCH, fGARCH: free power 2, gamma = eta1 = eta2 = 0, alpha_i = 0.05 / (p kappa_i),
//             omega = 0.05 var^(power / 2)
// with var the sample variance of r. r must not be empty.
std::vector<double> start_values(std::span<const double> r, const Layout& layout);

// Mean and population standard deviation of r.
Standardization standardization_of(std::span<const double> r);

}
