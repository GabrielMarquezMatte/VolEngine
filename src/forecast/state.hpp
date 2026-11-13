#pragma once

#include <vector>

namespace ve::forecast
{

// What a forecast reads of a fit, at its estimate: the last L = max(m, n, p, q) observations of each series, and
// every deviation with ARFIMA, whose forecast reads the whole history. permanent is empty but in csGARCH. All
// empty when the fit has no finite point.
struct ForecastState
{
    std::vector<double> residuals;
    std::vector<double> sigma2;
    std::vector<double> permanent;
    // d_t, the series the ARMA acts on: r_t - mu - sum_k delta_k x_t,k, less the ARCH-in-mean term.
    std::vector<double> deviations;
};

}
