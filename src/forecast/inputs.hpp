#pragma once

#include "forecast/forecast.hpp"
#include "model/layout.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace ve::forecast::detail
{

// The moments of the shock the rules read, computed once per forecast or simulation (the skewed densities take them
// by quadrature): pi = P(z < 0) in gjrGARCH, E|z| in eGARCH, kappa_i in apARCH and fGARCH; 0 or empty elsewhere.
struct Moments
{
    double below_zero = 0.0;
    double abs_moment = 0.0;
    std::vector<double> shock{};
};

// What every rule reads: the point and its layout, the final state of the fit, the future regressors and the
// moments of the shock.
struct Inputs
{
    const ForecastState& state;
    model::Layout layout;
    std::span<const double> theta;
    const FutureRegressors& future;
    std::size_t horizon;
    const Moments& moments;
};

inline std::span<const double> block(std::span<const double> theta, std::size_t begin, int count)
{
    return theta.subspan(begin, static_cast<std::size_t>(count));
}

// x_{n-1+h-lag}: the forecast when it lies after the sample (lag < h), else the last values of the sample.
inline double lagged(std::span<const double> sample, std::span<const double> forecast, std::size_t h, std::size_t lag)
{
    if (lag < h)
    {
        return forecast[h - 1 - lag];
    }
    return sample[sample.size() - 1 - (lag - h)];
}

}
