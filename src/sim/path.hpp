#pragma once

#include "forecast/forecast.hpp"
#include "forecast/state.hpp"
#include "sim/philox.hpp"
#include "sim/samplers.hpp"
#include "sim/start.hpp"

#include <cstddef>

namespace ve::sim
{

// The regressors of one step of the horizon.
forecast::FutureRegressors future_row(const forecast::FutureRegressors& future, std::size_t step);

// One path of one series: each step forecasts one step from the state of the path and takes the shock z, so that
// e = sigma z and r = mean + e; e, sigma2, the csGARCH q and r - level join the state. The start outlives the path.
class Path
{
public:
    explicit Path(const Start& start);

    // The presample of a spec start (start.presample residuals = presample_sigma * draws).
    void draw_presample(Stream& stream, const Sampler& sampler);

    struct Draw
    {
        double r;
        double sigma;
    };
    Draw step(const forecast::FutureRegressors& row, double z);

private:
    const Start& start_;
    forecast::ForecastState state_;
    std::size_t keep_;
};

}
