#pragma once

#include "core/cancel.hpp"
#include "core/context.hpp"
#include "dcc/fit.hpp"
#include "dcc/future/forecast.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ve::dcc::future
{

struct DccRequest
{
    std::size_t horizon = 0;
    std::size_t paths = 0;
    std::uint64_t seed = 0;
    bool correlations = false;
    SeriesRegressors future{};
    // Borrowed, polled once before any work and then at each step of each path; null never cancels.
    const CancelFlag* cancel = nullptr;
    // When not empty: strictly increasing steps from 1 that end at horizon, with correlations off (the caller checks).
    std::vector<std::size_t> checkpoints{};
};

// returns and sigma paths x horizon x m; r and h paths x horizon x m x m, empty without correlations. With
// checkpoints, all four are empty and cumulative holds paths x K x m: element (p, j, i) is the left-to-right sum of
// the first checkpoints[j] returns of series i in path p.
struct DccSimulation
{
    std::size_t paths = 0;
    std::size_t horizon = 0;
    std::size_t series = 0;
    std::vector<double> returns, sigma, r, h;
    std::vector<std::size_t> checkpoints{};
    std::vector<double> cumulative{};
};

// Each step takes R from the recursion, draws a shock of covariance R (a Gaussian scale mixture for the mvt and the
// mvlaplace), turns it into the standardized shock of each series (itself, or through the copula and the margin of
// the series) and steps each series from its own state. Path p reads only Stream(seed, p), so the result is the same
// for any number of threads. The caller checked forecast_problem(fit, request.horizon, request.future).
DccSimulation simulate_dcc(Context& context, const DccFitResult& fit, const DccRequest& request);

}
