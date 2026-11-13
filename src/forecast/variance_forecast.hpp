#pragma once

#include "forecast/inputs.hpp"

#include <vector>

namespace ve::forecast::detail
{

// sigma2 of steps 1..horizon by the rule of the model ("Variance" of plan 3a): each family advances sigma2,
// log sigma2 or sigma^delta, with the shocks of future steps at their expectation, and csGARCH by rugarch's
// closed form from h = 2 on.
std::vector<double> variance_forecast(const Inputs& in);

Moments moments_of(const model::Layout& layout, std::span<const double> theta);

// csGARCH: q of step 1, omega_1 + rho q_{n-1} + phi (e^2_{n-1} - sigma2_{n-1}).
double next_permanent(const Inputs& in);

}
