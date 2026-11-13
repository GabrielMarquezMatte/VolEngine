#pragma once

#include "dcc/fit.hpp"
#include "forecast/forecast.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace ve::dcc::future
{

// One FutureRegressors per series, horizon rows each (empty for a series without regressors).
using SeriesRegressors = std::vector<forecast::FutureRegressors>;

// h = 1..horizon: r and h (horizon x m x m), sigma and mean (horizon x m), row-major.
struct DccForecast
{
    std::size_t horizon = 0;
    std::size_t series = 0;
    std::vector<double> r, h, sigma, mean;
};

// Why fit cannot be forecast (or simulated) with future, or nullopt: no horizon, no finite point ("no finite point";
// a fit whose line search failed but kept its point is fine), no series_end (fit_dcc outside a batch), future with
// another number of series, or a series whose rows do not match its spec ("series i: ..." from 1).
std::optional<std::string> forecast_problem(const DccFitResult& fit, std::size_t horizon,
                                            const SeriesRegressors& future);

// The forecast of rules "Regras do forecast" of plan 4c: sigma and mean of each series from its own kept state,
// R_1 the exact recursion one step after T, R_h (1 - k^(h-1)) Rbar + k^(h-1) R_1 after it (k = sum a + sum b, per
// element for the FDCC), the static copula's R at every h, and H = D R D. The caller checked forecast_problem.
DccForecast forecast_dcc(const DccFitResult& fit, std::size_t horizon, const SeriesRegressors& future);

}
