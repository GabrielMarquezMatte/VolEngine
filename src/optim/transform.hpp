#pragma once

#include <span>

namespace ve::optim
{

// Location and scale of the series, so that the unconstrained problem does not depend on its units.
// scale must be > 0.
struct Standardization
{
    double location = 0.0;
    double scale = 1.0;
};

// natural: (mu if include_mean, omega, alpha, beta); unconstrained: (m if include_mean, u, v, w).
// mu = location + scale m; omega = scale^2 exp(u); persistence s = logistic(v) * (1 - 1e-8);
// share a = logistic(w); alpha = s a; beta = s (1 - a).
void sgarch_to_natural(std::span<const double> u, std::span<double> theta, bool include_mean,
                       const Standardization& standardization);

// theta must satisfy omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1 - 1e-8.
void sgarch_to_unconstrained(std::span<const double> theta, std::span<double> u, bool include_mean,
                             const Standardization& standardization);

// grad_u = J^T grad_theta, with J = d theta / d u.
void sgarch_chain_gradient(std::span<const double> u, std::span<const double> grad_theta,
                           std::span<double> grad_u, bool include_mean, const Standardization& standardization);

}
