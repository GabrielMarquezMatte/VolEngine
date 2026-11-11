#include "fit/fit.hpp"

#include "fit/standard_errors.hpp"
#include "model/layout.hpp"
#include "model/likelihood.hpp"
#include "model/parameter_map.hpp"
#include "model/start_values.hpp"
#include "optim/lbfgs.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace ve
{

namespace
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr std::size_t kMaxBridgeParams = 8;
using BridgeDual = ad::Dual<kMaxBridgeParams>;

constexpr std::string_view kNamesWithMean[] = {"mu", "omega", "alpha1", "beta1"};

// The mean negative log-likelihood of the standardized series r / scale, -loglik / n - log(scale), over the
// unconstrained parameters: tolerances depend neither on the sample size nor on the units of r.
class NegativeMeanLoglik
{
public:
    NegativeMeanLoglik(const fit::LikelihoodProblem& problem, const model::Standardization& standardization)
        : problem_(problem), standardization_(standardization), theta_(problem.layout.count()),
          grad_theta_(theta_.size())
    {
    }

    double operator()(std::span<const double> u, std::span<double> grad)
    {
        std::vector<BridgeDual> seeds;
        for (std::size_t i = 0; i < u.size(); ++i)
        {
            seeds.push_back(ad::variable<kMaxBridgeParams>(u[i], i));
        }
        std::vector<BridgeDual> natural(u.size());
        model::to_natural<BridgeDual>(seeds, natural, problem_.layout, standardization_);
        std::ranges::transform(natural, theta_.begin(), [](const BridgeDual& x) { return x.value; });
        double loglik = fit::loglik_with_gradient(problem_, theta_, grad_theta_, {});
        if (!std::isfinite(loglik))
        {
            std::ranges::fill(grad, 0.0);
            return kInfinity;
        }
        double count = static_cast<double>(problem_.r.size());
        for (std::size_t i = 0; i < grad.size(); ++i)
        {
            double chained = 0.0;
            for (std::size_t j = 0; j < natural.size(); ++j)
            {
                chained += grad_theta_[j] * natural[j].grad[i];
            }
            grad[i] = -chained / count;
        }
        return -loglik / count - std::log(standardization_.scale);
    }

private:
    const fit::LikelihoodProblem& problem_;
    model::Standardization standardization_;
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

bool has_valid_start(std::span<const double> r, std::span<const double> theta, const model::Layout& layout)
{
    if (std::ranges::adjacent_find(r, std::not_equal_to<>{}) == r.end())
    {
        return false;
    }
    double omega = theta[layout.omega()];
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
    std::vector<double> theta = model::start_values(r, problem.layout);
    if (!has_valid_start(r, theta, problem.layout))
    {
        return failed;
    }
    model::Standardization standardization = model::standardization_of(r);
    std::vector<double> u(k);
    model::to_unconstrained(theta, u, problem.layout, standardization);
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
    model::to_natural<double>(u, result.estimate, problem.layout, standardization);
    if (!fill_at_estimate(result, problem, standardization.scale))
    {
        return failed;
    }
    return result;
}

}
