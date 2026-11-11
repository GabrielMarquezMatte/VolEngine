#pragma once

#include "garch/sgarch.hpp"

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

// theta is the natural-space estimate; scores is the T x k row-major per-observation score matrix at theta;
// series_scale (> 0) is the standard deviation of r, the step scale for mu.
// hessian: sqrt(diag((-H)^-1)). robust: sqrt(diag(H^-1 (S^T S) H^-1)).
StandardErrors sgarch_standard_errors(std::span<const double> r, std::span<const double> theta,
                                      const garch::SGarchConfig& config, std::span<const double> scores,
                                      double series_scale);

}
