#pragma once

#include "core/context.hpp"
#include "forecast/forecast.hpp"
#include "sim/start.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ve::sim
{

// horizon steps of paths paths under seed; future holds horizon rows of the spec's regressors, the same for every
// path (the caller checks them).
struct Request
{
    std::size_t horizon = 0;
    std::size_t paths = 0;
    std::uint64_t seed = 0;
    forecast::FutureRegressors future{};
};

// paths x horizon, row-major: row p holds path p.
struct Simulation
{
    std::size_t paths = 0;
    std::size_t horizon = 0;
    std::vector<double> returns;
    std::vector<double> sigma;
};

// Each step is the forecast of one step from the state of the path with a drawn shock: e = sigma z, r = mean + e,
// and e, sigma2, the csGARCH q and r - level join the state. Path p reads only Stream(seed, p), so the result is the
// same for any number of threads.
Simulation simulate(Context& context, const Start& start, const Request& request);

}
