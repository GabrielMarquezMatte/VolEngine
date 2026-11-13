#pragma once

#include "fit/fit_result.hpp"
#include "model/data.hpp"

#include <cstddef>
#include <vector>

namespace ve::forecast
{

// The regressors of the forecast horizon: horizon rows x the spec's mxreg / vxreg columns, row-major; empty when the
// spec has none.
struct FutureRegressors
{
    model::RegressorMatrix mean{};
    model::RegressorMatrix variance{};
};

// sigma[h - 1] and mean[h - 1] for h = 1..horizon after the last observation: rugarch's sigmaFor and seriesFor
// (ugarchforecast, n.roll = 0).
struct Forecast
{
    std::vector<double> sigma;
    std::vector<double> mean;
};

// fit has a finite point (a non-empty state) and future matches its spec; the caller checks both.
Forecast forecast(const FitResult& fit, std::size_t horizon, const FutureRegressors& future = {});

}
