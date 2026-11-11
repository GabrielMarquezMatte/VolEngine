#include "fit/fit.hpp"

#include "fit/standard_errors.hpp"
#include "model/layout.hpp"
#include "model/likelihood.hpp"
#include "optim/lbfgs.hpp"
#include "optim/transform.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

namespace ve
{

namespace
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr double kInitialAlpha = 0.05;
constexpr double kInitialBeta = 0.90;
constexpr double kInitialPersistence = kInitialAlpha + kInitialBeta;

constexpr std::string_view kNamesWithMean[] = {"mu", "omega", "alpha1", "beta1"};

// The mean negative log-likelihood of the standardized series r / scale, -loglik / n - log(scale), over the
// unconstrained parameters: tolerances depend neither on the sample size nor on the units of r.
class NegativeMeanLoglik
{
public:
    NegativeMeanLoglik(const fit::LikelihoodProblem& problem, const optim::Standardization& standardization)
        : problem_(problem), standardization_(standardization), theta_(problem.layout.count()),
          grad_theta_(theta_.size())
    {
    }

    double operator()(std::span<const double> u, std::span<double> grad)
    {
        bool include_mean = problem_.layout.include_mean;
        optim::sgarch_to_natural(u, theta_, include_mean, standardization_);
        double loglik = fit::loglik_with_gradient(problem_, theta_, grad_theta_, {});
        if (!std::isfinite(loglik))
        {
            std::ranges::fill(grad, 0.0);
            return kInfinity;
        }
        optim::sgarch_chain_gradient(u, grad_theta_, grad, include_mean, standardization_);
        double count = static_cast<double>(problem_.r.size());
        std::ranges::transform(grad, grad.begin(), [count](double value) { return -value / count; });
        return -loglik / count - std::log(standardization_.scale);
    }

private:
    const fit::LikelihoodProblem& problem_;
    optim::Standardization standardization_;
    std::vector<double> theta_;
    std::vector<double> grad_theta_;
};

std::vector<std::string_view> parameter_names(bool include_mean)
{
    std::size_t skipped = include_mean ? 0 : 1;
    return {std::begin(kNamesWithMean) + skipped, std::end(kNamesWithMean)};
}

FitResult empty_result(const Spec& spec, const FitConfig& config, std::size_t nobs, FitStatus status)
{
    FitResult result;
    result.status = status;
    result.spec = spec;
    result.config = config;
    result.names = parameter_names(spec.include_mean);
    result.estimate.assign(result.names.size(), kNaN);
    result.se_hessian.assign(result.names.size(), kNaN);
    result.se_robust.assign(result.names.size(), kNaN);
    result.nobs = nobs;
    return result;
}

double sample_mean(std::span<const double> r)
{
    return std::accumulate(r.begin(), r.end(), 0.0) / static_cast<double>(r.size());
}

double sample_variance(std::span<const double> r)
{
    double mean = sample_mean(r);
    double sum = 0.0;
    for (double value : r)
    {
        sum += (value - mean) * (value - mean);
    }
    return sum / static_cast<double>(r.size());
}

std::vector<double> initial_theta(std::span<const double> r, const Spec& spec)
{
    std::vector<double> theta;
    if (spec.include_mean)
    {
        theta.push_back(sample_mean(r));
    }
    theta.push_back(sample_variance(r) * (1.0 - kInitialPersistence));
    theta.push_back(kInitialAlpha);
    theta.push_back(kInitialBeta);
    return theta;
}

bool has_valid_start(std::span<const double> r, std::span<const double> theta, const Spec& spec)
{
    if (std::ranges::adjacent_find(r, std::not_equal_to<>{}) == r.end())
    {
        return false;
    }
    double omega = theta[spec.include_mean ? 1 : 0];
    return std::isfinite(omega) && omega > 0.0;
}

FitStatus status_from(optim::Stop stop)
{
    switch (stop)
    {
        case optim::Stop::Converged:
            return FitStatus::Converged;
        case optim::Stop::MaxIter:
            return FitStatus::MaxIter;
        case optim::Stop::LineSearchFailed:
        case optim::Stop::NonFinite:
            return FitStatus::Failed;
    }
    return FitStatus::Failed;
}

void fill_series(FitResult& result, std::span<const double> r, std::span<const double> sigma2)
{
    double mu = 0.0;
    if (result.spec.include_mean)
    {
        mu = result.estimate[0];
    }
    std::size_t count = r.size();
    result.sigma.resize(count);
    result.residuals.resize(count);
    result.std_residuals.resize(count);
    result.fitted.assign(count, mu);
    for (std::size_t t = 0; t < count; ++t)
    {
        result.sigma[t] = std::sqrt(sigma2[t]);
        result.residuals[t] = r[t] - mu;
        result.std_residuals[t] = result.residuals[t] / result.sigma[t];
    }
}

bool fill_at_estimate(FitResult& result, const fit::LikelihoodProblem& problem, double series_scale)
{
    std::span<const double> r = problem.r;
    std::size_t k = result.estimate.size();
    std::vector<double> sigma2(r.size());
    std::vector<double> grad(k);
    std::vector<double> scores(r.size() * k);
    double loglik = model::loglik<double>(r, result.estimate, problem.layout, problem.var_init, {.sigma2 = sigma2});
    if (!std::isfinite(loglik))
    {
        return false;
    }
    fit::loglik_with_gradient(problem, result.estimate, grad, scores);
    result.loglik = loglik;
    fill_series(result, r, sigma2);
    fit::StandardErrors errors = fit::sgarch_standard_errors(problem, result.estimate, scores, series_scale);
    result.se_hessian = std::move(errors.hessian);
    result.se_robust = std::move(errors.robust);
    return true;
}

}

FitResult fit_one(std::span<const double> r, const Spec& spec, const FitConfig& config)
{
    assert(spec.model == Model::SGarch && spec.p == 1 && spec.q == 1 && spec.dist == Dist::Norm);
    fit::LikelihoodProblem problem{r, model::layout_of(spec), config.var_init};
    std::size_t k = problem.layout.count();
    if (r.size() < k + 2)
    {
        return empty_result(spec, config, r.size(), FitStatus::InsufficientData);
    }
    FitResult failed = empty_result(spec, config, r.size(), FitStatus::Failed);
    std::vector<double> theta = initial_theta(r, spec);
    if (!has_valid_start(r, theta, spec))
    {
        return failed;
    }
    optim::Standardization standardization{sample_mean(r), std::sqrt(sample_variance(r))};
    std::vector<double> u(k);
    optim::sgarch_to_unconstrained(theta, u, spec.include_mean, standardization);
    optim::LbfgsOptions options;
    options.max_iter = config.max_iter;
    options.gtol = config.tol;
    NegativeMeanLoglik objective(problem, standardization);
    optim::LbfgsResult outcome = optim::lbfgs_minimize(objective, u, options);
    if (outcome.stop == optim::Stop::NonFinite)
    {
        return failed;
    }
    FitResult result = empty_result(spec, config, r.size(), status_from(outcome.stop));
    optim::sgarch_to_natural(u, result.estimate, spec.include_mean, standardization);
    if (!fill_at_estimate(result, problem, standardization.scale))
    {
        return failed;
    }
    return result;
}

}
