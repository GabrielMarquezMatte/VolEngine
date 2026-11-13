#pragma once

#include "forecast/inputs.hpp"

namespace ve::forecast::detail
{

// The forecast of h = 1 from in.state (in.horizon == 1): sigma2 and the mean, the level of the mean (mu + the
// regressors + the ARCH-in-mean term, before the ARMA part) and the csGARCH q of the step (0 in the other models).
// One step of a simulation is this step with a drawn shock.
struct Step
{
    double sigma2;
    double mean;
    double level;
    double permanent;
};

Step first_step(const Inputs& in);

}
