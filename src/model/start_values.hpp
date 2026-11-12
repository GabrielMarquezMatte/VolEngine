#pragma once

#include "model/data.hpp"
#include "model/layout.hpp"
#include "model/parameter_map.hpp"

#include <span>
#include <vector>

namespace ve::model
{

// Natural-space start: mu = mean(r), ARMA = 0, arfima = 0.25, beta_j = 0.9 / q, the distribution parameters
//   std: shape 8; ged: shape 2; sstd: (xi, nu) = (1, 8); sged: (1, 2); nig: (rho, zeta) = (0, 2);
//   ghyp: (rho, zeta, lambda) = (0, 2, -0.5); jsu: (nu, tau) = (0, 2),
// and by model
//   sGARCH:   omega = 0.05 var, alpha = 0.05 / p
//   eGARCH:   omega = 0.1 log var, alpha = 0, gamma = 0.1 / p
//   gjrGARCH: omega = 0.05 var, alpha = 0.03 / p, gamma = 0.04 / p
//   iGARCH:   omega = 0.01 var, alpha = 0.1 / p (the derived beta_q is 0.9 / q as well)
//   csGARCH:  omega = 0.02 var, alpha = 0.05 / p, rho = 0.98, phi = min(0.03, 0.5 sum(beta))
//   apARCH, fGARCH: free power 2, gamma = eta1 = eta2 = 0, alpha_i = 0.05 / (p kappa_i),
//             omega = 0.05 var^(power / 2)
// with var the sample variance of r. The user's start values and the fixed values replace these, the power
// first, so that the default apARCH and fGARCH alpha_i use the final kappa_i and omega the final power. When the
// weight block has a fixed or started parameter, the default weights are scaled by s = min(1, 0.95 (c - F - U) / D)
// into the persistence left (weight_terms.hpp; c is the start rho in csGARCH), a default gjrGARCH alpha_i beside a
// fixed gamma_i starts at max(0, -gamma_i) + 0.03 / p, and the iGARCH beta_q is 1 - the other weights.
// With mean regressors, mu and the mxreg coefficients are the least squares of r on [1 (with a mean), X] (the
// minimum norm one when the columns are collinear), the fixed mxreg_j taken out of r first; archm and vxreg
// start at 0. data.r must not be empty.
std::vector<double> start_values(const Data& data, const Layout& layout);
std::vector<double> start_values(std::span<const double> r, const Layout& layout);

// Mean and population standard deviation of r, the population standard deviation of each regressor column, with
// 1 for a constant column, and the mean of each mean regressor column.
Standardization standardization_of(const Data& data);
Standardization standardization_of(std::span<const double> r);

}
