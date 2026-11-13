#pragma once

#include "fit/fit_result.hpp"
#include "forecast/inputs.hpp"
#include "forecast/state.hpp"
#include "model/layout.hpp"

#include <cstddef>
#include <vector>

namespace ve::sim
{

// Where every path starts: the point, the state its first step reads and the moments of the shock. presample > 0
// only from a spec: each path first draws that many shocks and writes presample_sigma z over the residuals of its
// copy of the state.
struct Start
{
    model::Layout layout;
    std::vector<double> theta;
    forecast::ForecastState state;
    forecast::detail::Moments moments;
    std::size_t presample = 0;
    double presample_sigma = 0.0;
};

// The final state of the fit at its estimate; fit has a finite point (the caller checks).
Start start_of_fit(const FitResult& fit);

}
