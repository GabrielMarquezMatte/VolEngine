#pragma once

#include "core/spec.hpp"
#include "fit/fit_result.hpp"
#include "forecast/forecast.hpp"
#include "forecast/inputs.hpp"
#include "forecast/state.hpp"
#include "model/layout.hpp"

#include <cstddef>
#include <expected>
#include <span>
#include <string>
#include <vector>

namespace ve::sim
{

// Where every path starts: the point, the state its first step reads and the moments of the shock. presample > 0
// only from a spec: each path first draws that many shocks and writes presample_sigma z over the residuals of its
// copy of the state.
struct Start
{
    model::Layout layout;
    std::vector<double> theta;
    forecast::ForecastState state;
    forecast::detail::Moments moments;
    std::size_t presample = 0;
    double presample_sigma = 0.0;
};

// The final state of a fit at its estimate: start_of_fit(fit) == start_of(fit.spec, fit.estimate, fit.state). The
// estimate is finite and the state non-empty (the caller checks).
Start start_of(const Spec& spec, std::span<const double> estimate, const forecast::ForecastState& state);
Start start_of_fit(const FitResult& fit);

// rugarch's ugarchpath: the unconditional state of a spec with every parameter fixed, no variance targeting, not
// iGARCH and stationary, with the level of the first row of the future variance regressors. Its L = max(m, n, p, q,
// 1) entries hold the unconditional sigma2 (and q in csGARCH), deviations 0 and residuals the presample draws. The
// message of VE_INVALID_ARGUMENT otherwise.
std::expected<Start, std::string> start_of_spec(const Spec& spec, const forecast::FutureRegressors& future);

}
