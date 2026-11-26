#pragma once

#include "fit/fit_result.hpp"
#include "model/data.hpp"

#include <cstddef>
#include <span>
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
// (ugarchforecast, n.roll = 0). cumulative_mean[h - 1] and cumulative_sigma[h - 1] are the forecast of the sum of
// the first h returns: the left-to-right sum of mean, and sqrt(sum_k a(h-k)^2 sigma_k^2) with the weights of
// cumulative_weights (the ARCH-in-mean term is taken at its forecast, as in mean).
struct Forecast
{
    std::vector<double> sigma;
    std::vector<double> mean;
    std::vector<double> cumulative_mean{};
    std::vector<double> cumulative_sigma{};
};

// fit has a finite point (a non-empty state) and future matches its spec; the caller checks both.
Forecast forecast(const FitResult& fit, std::size_t horizon, const FutureRegressors& future = {});

// The forecast of a fit from what it keeps: the FitResult overload calls it with fit.spec, fit.estimate and
// fit.state. The same requirements: a non-empty state and a future that matches the spec.
Forecast forecast(const Spec& spec, std::span<const double> estimate, const ForecastState& state,
                  std::size_t horizon, const FutureRegressors& future);

}
