#include "sim/start.hpp"

#include "dist/densities.hpp"
#include "forecast/variance_forecast.hpp"
#include "model/parameter_map.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <optional>
#include <span>

namespace ve::sim
{

namespace
{

constexpr double kNotStationary = std::numeric_limits<double>::quiet_NaN();

std::optional<std::string> refusal(const Spec& spec, const model::Layout& layout)
{
    if (std::optional<std::string> problem = model::spec_problem(spec))
    {
        return problem;
    }
    if (spec.variance_targeting)
    {
        return "variance targeting needs data: to simulate from a spec, turn it off and fix omega";
    }
    if (spec.model == Model::IGarch)
    {
        return "iGARCH has no unconditional variance to start from: simulate from a fit instead";
    }
    std::vector<std::string> names = model::parameter_names(layout);
    for (std::size_t i = 0; i < layout.count(); ++i)
    {
        if (!layout.is_fixed(i) && !layout.is_derived(i))
        {
            return "simulating from a spec needs every parameter fixed; " + names[i] + " is free";
        }
    }
    return std::nullopt;
}

// The unconditional sigma2 of the rows of the "Origem spec" table of plan 3b, with omega + shift; NaN where the
// spec is not stationary.
double unconditional_variance(const model::Layout& layout, std::span<const double> theta, double shift)
{
    double intercept = theta[layout.omega()] + shift;
    double persistence = model::persistence<double>(theta, layout);
    if (layout.model == Model::EGarch)
    {
        return std::abs(persistence) < 1.0 ? std::exp(intercept / (1.0 - persistence)) : kNotStationary;
    }
    if (!(persistence < 1.0))
    {
        return kNotStationary;
    }
    if (layout.model == Model::CsGarch)
    {
        double rho = theta[layout.eta1_begin()];
        return rho < 1.0 ? intercept / (1.0 - rho) : kNotStationary;
    }
    double level = intercept / (1.0 - persistence);
    if (layout.is_power_family())
    {
        return std::pow(level, 2.0 / model::power_of<double>(theta, layout));
    }
    return level;
}

double first_shift(const model::Layout& layout, std::span<const double> theta,
                   const forecast::FutureRegressors& future)
{
    double shift = 0.0;
    for (std::size_t k = 0; k < future.variance.columns; ++k)
    {
        shift += theta[layout.vxreg_begin() + k] * future.variance.at(0, k);
    }
    return shift;
}

forecast::ForecastState unconditional_state(const model::Layout& layout, double sigma2)
{
    std::size_t count = static_cast<std::size_t>(std::max({layout.ar, layout.ma, layout.arch, layout.garch, 1}));
    forecast::ForecastState state{std::vector<double>(count, 0.0), std::vector<double>(count, sigma2), {},
                                  std::vector<double>(count, 0.0)};
    if (layout.model == Model::CsGarch)
    {
        state.permanent.assign(count, sigma2);
    }
    return state;
}

}

std::expected<Start, std::string> start_of_spec(const Spec& spec, const forecast::FutureRegressors& future)
{
    model::Layout layout = model::layout_of(spec);
    if (std::optional<std::string> reason = refusal(spec, layout))
    {
        return std::unexpected(*reason);
    }
    std::vector<double> theta(layout.given.begin(), layout.given.begin() + static_cast<std::ptrdiff_t>(layout.count()));
    if (!dist::in_domain(layout.dist, model::shape_parameters_of<double>(theta, layout)))
    {
        return std::unexpected(std::string("the distribution parameters of the spec are outside their domain"));
    }
    double persistence = model::persistence<double>(theta, layout);
    bool explosive = std::isfinite(persistence) && persistence >= 1.0;
    if (!explosive && !model::is_valid(theta, layout))
    {
        return std::unexpected(std::string("the parameters of the spec are outside the model's domain (the signs "
                                           "and ceilings of its weights)"));
    }
    double sigma2 = unconditional_variance(layout, theta, first_shift(layout, theta, future));
    if (!(std::isfinite(sigma2) && sigma2 > 0.0))
    {
        return std::unexpected(std::string(
            "the spec is not stationary: its persistence must be below 1 and its unconditional variance positive"));
    }
    Start start{layout, theta, unconditional_state(layout, sigma2), forecast::detail::moments_of(layout, theta)};
    start.presample = start.state.residuals.size();
    start.presample_sigma = std::sqrt(sigma2);
    return start;
}

Start start_of_fit(const FitResult& fit)
{
    assert(!fit.state.residuals.empty());
    Start start{model::layout_of(fit.spec), fit.estimate, fit.state, {}};
    start.moments = forecast::detail::moments_of(start.layout, start.theta);
    return start;
}

}
