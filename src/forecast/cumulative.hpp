#pragma once

#include "model/layout.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace ve::forecast
{

// a(0..horizon-1), a(n) = psi_0 + ... + psi_n, from the MA(inf) weights of the ARMA mean (psi_j = theta_j +
// sum_i phi_i psi_{j-i}) convolved, with ARFIMA, with the coefficients of (1 - L)^-d. All ones without either.
std::vector<double> cumulative_weights(const model::Layout& layout, std::span<const double> theta,
                                       std::size_t horizon);

// sum_{k=1..h} weights[h-k]^2 variances[k-1]: the variance of the sum of the first h returns.
double cumulative_variance(std::span<const double> weights, std::span<const double> variances, std::size_t h);

}
