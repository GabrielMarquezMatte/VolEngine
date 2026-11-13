#include "sim/start.hpp"

#include "forecast/variance_forecast.hpp"

#include <cassert>

namespace ve::sim
{

Start start_of_fit(const FitResult& fit)
{
    assert(!fit.state.residuals.empty());
    Start start{model::layout_of(fit.spec), fit.estimate, fit.state, {}};
    start.moments = forecast::detail::moments_of(start.layout, start.theta);
    return start;
}

}
