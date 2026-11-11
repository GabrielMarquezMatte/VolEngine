#pragma once

#include "core/var_init.hpp"
#include "model/layout.hpp"

#include <span>
#include <vector>

namespace ve::fit
{

// Both vectors have one entry per parameter, NaN when the Hessian is unavailable or not negative definite.
struct StandardErrors
{
    std::vector<double> hessian;
    std::vector<double> robust;
};

// An sGARCH(1,1)-normal likelihood: layout has at most 8 parameters.
struct LikelihoodProblem
{
    std::span<const double> r;
    model::Layout layout;
    VarInit var_init;
};

// Returns the log-likelihood at the natural-space theta and writes its gradient to grad (k entries) and, when scores
// is not empty, the T x k row-major per-observation scores.
double loglik_with_gradient(const LikelihoodProblem& problem, std::span<const double> theta, std::span<double> grad,
                            std::span<double> scores);

// theta is the natural-space estimate; scores is the T x k row-major per-observation score matrix at theta;
// series_scale (> 0) is the standard deviation of r, the step scale for mu.
// hessian: sqrt(diag((-H)^-1)). robust: sqrt(diag(H^-1 (S^T S) H^-1)).
StandardErrors sgarch_standard_errors(const LikelihoodProblem& problem, std::span<const double> theta,
                                      std::span<const double> scores, double series_scale);

}
