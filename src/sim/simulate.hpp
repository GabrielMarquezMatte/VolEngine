#pragma once

#include "core/cancel.hpp"
#include "core/context.hpp"
#include "forecast/forecast.hpp"
#include "sim/start.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ve::sim
{

// horizon steps of paths paths under seed; future holds horizon rows of the spec's regressors, the same for every
// path (the caller checks them). cancel is borrowed and polled once before any work and then at each step of each
// path; null never cancels. checkpoints, when not empty, are strictly increasing steps from 1 that end at horizon
// (the caller checks them).
struct Request
{
    std::size_t horizon = 0;
    std::size_t paths = 0;
    std::uint64_t seed = 0;
    forecast::FutureRegressors future{};
    const CancelFlag* cancel = nullptr;
    std::vector<std::size_t> checkpoints{};
};

// paths x horizon, row-major: row p holds path p. With checkpoints, returns and sigma are empty and cumulative holds
// paths x K: element (p, j) is the left-to-right sum of the first checkpoints[j] returns of path p.
struct Simulation
{
    std::size_t paths = 0;
    std::size_t horizon = 0;
    std::vector<double> returns;
    std::vector<double> sigma;
    std::vector<std::size_t> checkpoints{};
    std::vector<double> cumulative{};
};

// Each step is the forecast of one step from the state of the path with a drawn shock: e = sigma z, r = mean + e,
// and e, sigma2, the csGARCH q and r - level join the state. Path p reads only Stream(seed, p), so the result is the
// same for any number of threads.
Simulation simulate(Context& context, const Start& start, const Request& request);

}
