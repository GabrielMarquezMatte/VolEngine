#pragma once

#include "model/layout.hpp"

#include <functional>
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

// gradient(theta, grad): the natural-space gradient of the log-likelihood at theta; false when theta is not valid or
// the log-likelihood is not finite.
using GradientFn = std::function<bool(std::span<const double>, std::span<double>)>;

// theta is the natural-space estimate; scores is the T x k row-major per-observation score matrix at theta;
// series_scale (> 0) is the standard deviation of r, the step scale for mu.
// hessian: sqrt(diag((-H)^-1)). robust: sqrt(diag(H^-1 (S^T S) H^-1)). alpha and beta entries below 1e-8 sit on the
// boundary: they leave H and S and get NaN, and the others come from the remaining block.
StandardErrors standard_errors(const GradientFn& gradient, std::span<const double> theta,
                               std::span<const double> scores, const model::Layout& layout, double series_scale);

}
