#pragma once

#include "forecast/inputs.hpp"

#include <vector>

namespace ve::forecast::detail
{

// sigma2 of steps 1..horizon by the rule of the model ("Variance" of plan 3a): each family advances sigma2,
// log sigma2 or sigma^delta, with the shocks of future steps at their expectation, and csGARCH by rugarch's
// closed form from h = 2 on.
std::vector<double> variance_forecast(const Inputs& in);

}
