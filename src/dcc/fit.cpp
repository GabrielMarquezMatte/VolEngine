#include "dcc/fit.hpp"

#include "ad/dual.hpp"
#include "dcc/likelihood.hpp"
#include "dcc/parameter_map.hpp"
#include "fit/optimizer.hpp"
#include "optim/lbfgs.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace ve::dcc
{

namespace
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();

using fit::status_from;
using fit::with_dual_width;

// What the optimiser needs: the spec on its panel, the layout, the delta of the aDCC constraint and the positions
// of the estimated parameters, which the optimiser sees in order.
struct DccProblem
{
    const StandardizedPanel& panel;
    const DccSpec& spec;
    DccLayout layout;
    double delta;
    std::vector<std::size_t> free;
};

std::vector<std::size_t> estimated_positions(const DccLayout& layout)
{
    std::vector<std::size_t> free;
    for (std::size_t index = 0; index < layout.count(); ++index)
    {
        if (layout.is_estimated(index))
        {
            free.push_back(index);
        }
    }
    return free;
}

// objective / T over the free unconstrained parameters, so that the tolerances do not depend on the sample size.
// The fixed positions of u stay 0: to_natural does not read them.
template <std::size_t N>
class MeanObjective
{
public:
    using Number = ad::Dual<N>;

    explicit MeanObjective(const DccProblem& problem)
        : problem_(problem), u_(problem.layout.count()), theta_(problem.layout.count())
    {
    }

    double operator()(std::span<const double> x, std::span<double> grad)
    {
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            u_[problem_.free[i]] = ad::variable<N>(x[i], i);
        }
        to_natural<Number>(u_, theta_, problem_.layout, problem_.delta);
        const Number value = objective<Number>(problem_.panel, problem_.spec, theta_);
        if (!std::isfinite(value.value))
        {
            std::ranges::fill(grad, 0.0);
            return kInfinity;
        }
        const auto count = static_cast<double>(problem_.panel.t);
        for (std::size_t i = 0; i < grad.size(); ++i)
        {
            grad[i] = value.grad[i] / count;
        }
        return value.value / count;
    }

private:
    const DccProblem& problem_;
    std::vector<Number> u_;
    std::vector<Number> theta_;
};

template <std::size_t N>
optim::LbfgsResult minimize(const DccProblem& problem, const FitConfig& config, std::span<double> x)
{
    optim::LbfgsOptions options;
    options.max_iter = config.max_iter;
    options.gtol = config.tol;
    MeanObjective<N> objective(problem);
    return optim::lbfgs_minimize(objective, x, options);
}

// Moves theta, the natural start, to the estimate. With nothing to estimate the optimiser does not run.
optim::Stop estimate_parameters(const DccProblem& problem, const FitConfig& config, std::vector<double>& theta)
{
    if (problem.free.empty())
    {
        return optim::Stop::Converged;
    }
    std::vector<double> u(theta.size());
    to_unconstrained(theta, u, problem.layout, problem.delta);
    std::vector<double> x(problem.free.size());
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        x[i] = u[problem.free[i]];
    }
    const optim::LbfgsResult outcome =
        with_dual_width(x.size(), [&]<std::size_t N>() { return minimize<N>(problem, config, x); });
    if (outcome.stop == optim::Stop::NonFinite)
    {
        return outcome.stop;
    }
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        u[problem.free[i]] = x[i];
    }
    to_natural<double>(u, theta, problem.layout, problem.delta);
    return outcome.stop;
}

DccFitResult empty_result(std::shared_ptr<const StandardizedPanel> panel, const DccSpec& spec)
{
    DccFitResult result;
    result.spec = spec;
    result.names = estimated_names(layout_of(spec, panel->m));
    result.estimate.assign(result.names.size(), kNaN);
    result.se_hessian.assign(result.names.size(), kNaN);
    result.se_robust.assign(result.names.size(), kNaN);
    result.nobs = panel->t;
    result.series = panel->m;
    result.panel = std::move(panel);
    return result;
}

// Fills result, which arrives as empty_result, and returns its status. Unless the reported loglik at the estimate
// is finite, the estimate stays NaN.
FitStatus estimate(const DccProblem& problem, const FitConfig& config, DccFitResult& result)
{
    std::vector<double> theta = start_values(problem.layout);
    if (!is_valid(theta, problem.layout, problem.delta))
    {
        return FitStatus::Failed;
    }
    const optim::Stop stop = estimate_parameters(problem, config, theta);
    if (stop == optim::Stop::NonFinite)
    {
        return FitStatus::Failed;
    }
    const StandardizedPanel& panel = problem.panel;
    auto r = std::make_shared<std::vector<double>>(panel.t * panel.m * panel.m);
    const double loglik = reported_loglik(panel, problem.spec, theta, *r);
    if (!std::isfinite(loglik))
    {
        return FitStatus::Failed;
    }
    result.estimate = estimated_part(problem.layout, theta);
    result.loglik = loglik;
    result.r = std::move(r);
    return status_from(stop);
}

}

DccFitResult fit_dcc(std::shared_ptr<const StandardizedPanel> panel, const DccSpec& spec, const FitConfig& config)
{
    assert(panel != nullptr && !spec_problem(spec, panel->m));
    DccFitResult result = empty_result(std::move(panel), spec);
    const StandardizedPanel& data = *result.panel;
    DccProblem problem{data, result.spec, layout_of(spec, data.m), constraint_delta(data, spec), {}};
    problem.free = estimated_positions(problem.layout);
    result.status = estimate(problem, config, result);
    return result;
}

}
