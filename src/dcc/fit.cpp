#include "dcc/fit.hpp"

#include "ad/dual.hpp"
#include "core/context.hpp"
#include "dcc/likelihood.hpp"
#include "dcc/parameter_map.hpp"
#include "fit/optimizer.hpp"
#include "model/layout.hpp"
#include "optim/lbfgs.hpp"

#include <BS_thread_pool.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
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

// What the fits of the panel say about the univariate stage, the same for every result of a batch.
struct UnivariateStage
{
    std::size_t parameters = 0;
    std::vector<double> estimates;
};

UnivariateStage univariate_stage(std::span<const FitResult* const> fits)
{
    UnivariateStage stage;
    for (const FitResult* fit : fits)
    {
        stage.parameters += model::layout_of(fit->spec).estimated_count();
        stage.estimates.insert(stage.estimates.end(), fit->estimate.begin(), fit->estimate.end());
    }
    return stage;
}

constexpr std::size_t kBlocksPerThread = 4;

// The specs to fit on one panel, all with one config.
struct SpecBatch
{
    const std::shared_ptr<const StandardizedPanel>& panel;
    std::span<const DccSpec> specs;
    const FitConfig& config;
};

// Every spec writes only its own slot of results.
void run_specs(BS::light_thread_pool& pool, int threads, const SpecBatch& batch, std::vector<DccFitResult>& results)
{
    const std::size_t max_blocks = kBlocksPerThread * static_cast<std::size_t>(threads);
    const std::size_t blocks = std::min(results.size(), max_blocks);
    auto run_block = [&](std::size_t first, std::size_t last)
    {
        for (std::size_t index = first; index < last; ++index)
        {
            results[index] = fit_dcc(batch.panel, batch.specs[index], batch.config);
        }
    };
    try
    {
        auto futures = pool.submit_blocks(std::size_t{0}, results.size(), run_block, blocks);
        futures.wait();
        futures.get();
    }
    catch (...)
    {
        pool.wait();
        throw;
    }
}

double criterion_per_obs(double loglik, double parameters, double observations, Criterion which)
{
    const double fit_term = -2.0 * loglik;
    switch (which)
    {
    case Criterion::Aic:
        return (fit_term + 2.0 * parameters) / observations;
    case Criterion::Bic:
        return (fit_term + parameters * std::log(observations)) / observations;
    case Criterion::Hqic:
        return (fit_term + 2.0 * parameters * std::log(std::log(observations))) / observations;
    case Criterion::Sic:
        break;
    }
    return fit_term / observations + std::log((observations + 2.0 * parameters) / observations);
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

std::optional<std::string> check_dcc_batch(std::span<const FitResult* const> fits, std::span<const DccSpec> specs)
{
    if (std::optional<std::string> problem = panel_problem(fits))
    {
        return problem;
    }
    if (specs.empty())
    {
        return "batch has no specs";
    }
    for (std::size_t index = 0; index < specs.size(); ++index)
    {
        if (std::optional<std::string> problem = spec_problem(specs[index], fits.size()))
        {
            return "spec " + std::to_string(index + 1) + ": " + *problem;
        }
    }
    return std::nullopt;
}

std::vector<DccFitResult> fit_dcc_batch(Context& context, std::span<const FitResult* const> fits,
                                        std::span<const DccSpec> specs, const DccOptions& options)
{
    assert(!check_dcc_batch(fits, specs));
    const auto panel = std::make_shared<const StandardizedPanel>(standardize(fits));
    std::vector<DccFitResult> results(specs.size());
    const SpecBatch batch{panel, specs, options.config};
    context.with_pool([&](BS::light_thread_pool& pool, int threads) { run_specs(pool, threads, batch, results); });
    const UnivariateStage stage = univariate_stage(fits);
    for (DccFitResult& result : results)
    {
        result.univariate_parameters = stage.parameters;
        result.univariate_estimates = stage.estimates;
    }
    return results;
}

double dcc_criterion(const DccFitResult& fit, Criterion which, bool per_obs)
{
    if (!std::isfinite(fit.loglik) || fit.nobs == 0)
    {
        return kNaN;
    }
    const auto m = static_cast<double>(fit.series);
    const double parameters =
        static_cast<double>(fit.univariate_parameters + fit.estimate.size()) + m * (m - 1.0) / 2.0;
    const auto observations = static_cast<double>(fit.nobs);
    const double value = criterion_per_obs(fit.loglik, parameters, observations, which);
    return per_obs ? value : value * observations;
}

std::vector<double> covariance_path(const DccFitResult& fit)
{
    std::vector<double> h;
    if (fit.r == nullptr || fit.panel == nullptr)
    {
        return h;
    }
    const std::vector<double>& r = *fit.r;
    const std::vector<double>& sigma = fit.panel->sigma;
    const std::size_t m = fit.panel->m;
    h.resize(r.size());
    for (std::size_t k = 0; k < h.size(); ++k)
    {
        const std::size_t t = k / (m * m);
        h[k] = sigma[t * m + (k / m) % m] * r[k] * sigma[t * m + k % m];
    }
    return h;
}

}
