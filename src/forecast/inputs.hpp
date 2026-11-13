#pragma once

#include "forecast/forecast.hpp"
#include "model/layout.hpp"

#include <cstddef>
#include <span>

namespace ve::forecast::detail
{

// What every rule reads: the point and its layout, the final state of the fit and the future regressors.
struct Inputs
{
    const ForecastState& state;
    model::Layout layout;
    std::span<const double> theta;
    const FutureRegressors& future;
    std::size_t horizon;
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
