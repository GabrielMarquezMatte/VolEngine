#include "fit/fit.hpp"

#include "ad/dual.hpp"
#include "fit/standard_errors.hpp"
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
#include <utility>
#include <vector>

namespace ve
{

namespace
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInfinity = std::numeric_limits<double>::infinity();
constexpr std::size_t kWidestDual = 32;

// model::loglik and model::to_natural exist only for Dual<8>, Dual<16> and Dual<32>: body runs with the
// narrowest of them that holds k parameters.
template <class F>
decltype(auto) with_dual_width(std::size_t k, F&& body)
{
    assert(k <= kWidestDual);
    if (k <= 8)
    {
        return body.template operator()<8>();
    }
    if (k <= 16)
    {
        return body.template operator()<16>();
    }
    return body.template operator()<kWidestDual>();
}

template <std::size_t N>
void seed(std::span<const double> values, std::span<ad::Dual<N>> seeds)
{
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        seeds[i] = ad::variable<N>(values[i], i);
    }
}

bool all_finite(std::span<const double> values)
{
    return std::ranges::all_of(values, [](double value) { return std::isfinite(value); });
}

// The mean negative log-likelihood of the standardized series r / scale, -loglik / n - log(scale), over the
// unconstrained parameters: tolerances depend neither on the sample size nor on the units of r. Validity is
// judged by the log-likelihood alone, which is not finite outside its domain: model::is_valid can reject
// points that model::to_natural produces at extreme u.
template <std::size_t N>
class NegativeMeanLoglik
{
public:
    using Number = ad::Dual<N>;

    NegativeMeanLoglik(const fit::LikelihoodProblem& problem, const model::Standardization& standardization)
        : problem_(problem), standardization_(standardization), u_(problem.layout.count()), theta_(u_.size()),
          residuals_(problem.r.size()), sigma2_(problem.r.size())
    {
    }

    double operator()(std::span<const double> u, std::span<double> grad)
    {
        seed<N>(u, u_);
        model::to_natural<Number>(u_, theta_, problem_.layout, standardization_);
        Number loglik = model::loglik<Number>(problem_.r, theta_, problem_.layout, problem_.var_init,
                                              {.residuals = residuals_, .sigma2 = sigma2_});
        if (!std::isfinite(loglik.value))
        {
            std::ranges::fill(grad, 0.0);
            return kInfinity;
        }
        double count = static_cast<double>(problem_.r.size());
        for (std::size_t i = 0; i < grad.size(); ++i)
        {
            grad[i] = -loglik.grad[i] / count;
        }
        return -loglik.value / count - std::log(standardization_.scale);
    }

private:
    const fit::LikelihoodProblem& problem_;
    model::Standardization standardization_;
    std::vector<Number> u_;
    std::vector<Number> theta_;
    std::vector<Number> residuals_;
    std::vector<Number> sigma2_;
};

template <std::size_t N>
optim::LbfgsResult minimize(const fit::LikelihoodProblem& problem,
                            const model::Standardization& standardization, const FitConfig& config,
                            std::span<double> u)
{
    optim::LbfgsOptions options;
    options.max_iter = config.max_iter;
    options.gtol = config.tol;
    NegativeMeanLoglik<N> objective(problem, standardization);
    return optim::lbfgs_minimize(objective, u, options);
}

template <std::size_t N>
bool loglik_gradient_with(const fit::LikelihoodProblem& problem, std::span<const double> theta,
                          std::span<double> grad)
{
    using Number = ad::Dual<N>;
    std::vector<Number> seeds(theta.size());
    seed<N>(theta, seeds);
    Number loglik = model::loglik<Number>(problem.r, seeds, problem.layout, problem.var_init, {});
    std::copy_n(loglik.grad.begin(), theta.size(), grad.begin());
    return std::isfinite(loglik.value) && all_finite(grad);
}

template <std::size_t N>
double loglik_with_scores_with(const fit::LikelihoodProblem& problem, std::span<const double> theta,
                               const fit::LoglikSeries& out)
{
    using Number = ad::Dual<N>;
    std::size_t count = problem.r.size();
    std::size_t k = theta.size();
    std::vector<Number> seeds(k);
    std::vector<Number> terms(count);
    std::vector<Number> residuals(count);
    std::vector<Number> sigma2(count);
    seed<N>(theta, seeds);
    Number loglik =
        model::loglik<Number>(problem.r, seeds, problem.layout, problem.var_init, {terms, residuals, sigma2});
    for (std::size_t t = 0; t < count; ++t)
    {
        out.residuals[t] = residuals[t].value;
        out.sigma2[t] = sigma2[t].value;
        std::copy_n(terms[t].grad.begin(), k, out.scores.begin() + static_cast<std::ptrdiff_t>(t * k));
    }
    return loglik.value;
}

FitResult empty_result(const Spec& spec, const FitConfig& config, std::size_t nobs, FitStatus status)
{
    FitResult result;
    result.status = status;
    result.spec = spec;
    result.config = config;
    result.names = model::parameter_names(model::layout_of(spec));
    result.estimate.assign(result.names.size(), kNaN);
    result.se_hessian.assign(result.names.size(), kNaN);
    result.se_robust.assign(result.names.size(), kNaN);
    result.nobs = nobs;
    return result;
}

bool has_insufficient_data(std::span<const double> r, const Spec& spec, std::size_t k)
{
    std::size_t max_order = static_cast<std::size_t>(std::max({spec.ar, spec.ma, spec.p, spec.q}));
    return r.size() < k + 2 || r.size() <= max_order;
}

bool has_valid_start(std::span<const double> r, std::span<const double> theta, const model::Layout& layout)
{
    if (std::ranges::adjacent_find(r, std::not_equal_to<>{}) == r.end())
    {
        return false;
    }
    return model::is_valid(theta, layout);
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

void fill_series(FitResult& result, std::span<const double> r, std::vector<double> residuals,
                 std::span<const double> sigma2)
{
    std::size_t count = r.size();
    result.sigma.resize(count);
    result.std_residuals.resize(count);
    result.fitted.resize(count);
    for (std::size_t t = 0; t < count; ++t)
    {
        result.sigma[t] = std::sqrt(sigma2[t]);
        result.std_residuals[t] = residuals[t] / result.sigma[t];
        result.fitted[t] = r[t] - residuals[t];
    }
    result.residuals = std::move(residuals);
}

bool fill_at_estimate(FitResult& result, const fit::LikelihoodProblem& problem, double series_scale)
{
    std::span<const double> r = problem.r;
    std::vector<double> residuals(r.size());
    std::vector<double> sigma2(r.size());
    std::vector<double> scores(r.size() * result.estimate.size());
    double loglik = fit::loglik_with_scores(problem, result.estimate, {residuals, sigma2, scores});
    if (!std::isfinite(loglik))
    {
        return false;
    }
    result.loglik = loglik;
    fill_series(result, r, std::move(residuals), sigma2);
    fit::GradientFn gradient = [&problem](std::span<const double> theta, std::span<double> grad)
    { return fit::loglik_gradient(problem, theta, grad); };
    fit::StandardErrors errors =
        fit::standard_errors(gradient, result.estimate, scores, problem.layout, series_scale);
    result.se_hessian = std::move(errors.hessian);
    result.se_robust = std::move(errors.robust);
    return true;
}

}

FitResult fit_one(std::span<const double> r, const Spec& spec, const FitConfig& config)
{
    fit::LikelihoodProblem problem{r, model::layout_of(spec), config.var_init};
    std::size_t k = problem.layout.count();
    if (has_insufficient_data(r, spec, k))
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
    optim::LbfgsResult outcome =
        with_dual_width(k, [&]<std::size_t N>() { return minimize<N>(problem, standardization, config, u); });
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

namespace ve::fit
{

bool loglik_gradient(const LikelihoodProblem& problem, std::span<const double> theta, std::span<double> grad)
{
    assert(theta.size() == problem.layout.count() && grad.size() == theta.size());
    if (!model::is_valid(theta, problem.layout))
    {
        return false;
    }
    return with_dual_width(theta.size(),
                           [&]<std::size_t N>() { return loglik_gradient_with<N>(problem, theta, grad); });
}

double loglik_with_scores(const LikelihoodProblem& problem, std::span<const double> theta,
                          const LoglikSeries& out)
{
    std::size_t k = problem.layout.count();
    assert(theta.size() == k && out.scores.size() == problem.r.size() * k);
    assert(out.residuals.size() == problem.r.size() && out.sigma2.size() == problem.r.size());
    return with_dual_width(k,
                           [&]<std::size_t N>()
                           {
                               return loglik_with_scores_with<N>(problem, theta, out);
                           });
}

}
