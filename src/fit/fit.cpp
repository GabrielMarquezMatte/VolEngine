#include "fit/fit.hpp"

#include "ad/dual.hpp"
#include "core/spec.hpp"
#include "fit/standard_errors.hpp"
#include "model/likelihood.hpp"
#include "model/parameter_map.hpp"
#include "model/start_values.hpp"
#include "optim/lbfgs.hpp"

#include <algorithm>
#include <array>
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
constexpr std::size_t kWidestDual = kMaxParameterCount;
// ad::variable writes grad[index] unchecked.
static_assert(kMaxParameterCount <= kWidestDual, "the widest layout must fit the widest Dual");

// model::loglik and model::to_natural exist only for Dual<8>, 16, 32 and kMaxParameterCount: body runs with the
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
    if (k <= 32)
    {
        return body.template operator()<32>();
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
          residuals_(problem.data.r.size()), sigma2_(problem.data.r.size())
    {
    }

    double operator()(std::span<const double> u, std::span<double> grad)
    {
        seed<N>(u, u_);
        model::to_natural<Number>(u_, theta_, problem_.layout, standardization_);
        Number loglik = model::loglik<Number>(problem_.data, theta_, problem_.layout, problem_.var_init,
                                              {.residuals = residuals_, .sigma2 = sigma2_});
        if (!std::isfinite(loglik.value))
        {
            std::ranges::fill(grad, 0.0);
            return kInfinity;
        }
        double count = static_cast<double>(problem_.data.r.size());
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

// The gradient of loglik over the natural parameters. The buffers live as long as the object: the standard errors
// call it 2k times, and allocating them on every call made each call cost clearly more than an optimizer
// evaluation.
template <std::size_t N>
class LoglikGradient
{
public:
    using Number = ad::Dual<N>;

    explicit LoglikGradient(const fit::LikelihoodProblem& problem)
        : problem_(problem), seeds_(problem.layout.count()), residuals_(problem.data.r.size()),
          sigma2_(problem.data.r.size())
    {
    }

    bool operator()(std::span<const double> theta, std::span<double> grad)
    {
        if (!model::is_valid(theta, problem_.layout))
        {
            return false;
        }
        seed<N>(theta, seeds_);
        Number loglik = model::loglik<Number>(problem_.data, seeds_, problem_.layout, problem_.var_init,
                                              {.residuals = residuals_, .sigma2 = sigma2_});
        std::copy_n(loglik.grad.begin(), theta.size(), grad.begin());
        return std::isfinite(loglik.value) && all_finite(grad);
    }

private:
    const fit::LikelihoodProblem& problem_;
    std::vector<Number> seeds_;
    std::vector<Number> residuals_;
    std::vector<Number> sigma2_;
};

template <std::size_t N>
double loglik_with_scores_with(const fit::LikelihoodProblem& problem, std::span<const double> theta,
                               const fit::LoglikSeries& out)
{
    using Number = ad::Dual<N>;
    std::size_t count = problem.data.r.size();
    std::size_t k = theta.size();
    std::vector<Number> seeds(k);
    std::vector<Number> terms(count);
    std::vector<Number> residuals(count);
    std::vector<Number> sigma2(count);
    std::vector<Number> deviations(out.deviations.size());
    std::vector<Number> permanent(out.permanent.size());
    seed<N>(theta, seeds);
    Number loglik = model::loglik<Number>(problem.data, seeds, problem.layout, problem.var_init,
                                          {terms, residuals, sigma2, deviations, permanent});
    for (std::size_t t = 0; t < count; ++t)
    {
        out.residuals[t] = residuals[t].value;
        out.sigma2[t] = sigma2[t].value;
        std::copy_n(terms[t].grad.begin(), k, out.scores.begin() + static_cast<std::ptrdiff_t>(t * k));
    }
    std::ranges::transform(deviations, out.deviations.begin(), [](const Number& x) { return x.value; });
    std::ranges::transform(permanent, out.permanent.begin(), [](const Number& x) { return x.value; });
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

bool has_insufficient_data(std::span<const double> r, const Spec& spec, const model::Layout& layout)
{
    std::size_t max_order = static_cast<std::size_t>(std::max({spec.ar, spec.ma, spec.p, spec.q}));
    return r.size() < layout.estimated_count() + 2 || r.size() <= max_order;
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

std::vector<double> last(std::span<const double> values, std::size_t count)
{
    std::span<const double> tail = values.last(std::min(count, values.size()));
    return {tail.begin(), tail.end()};
}

// The last L = max(m, n, p, q) entries of each series; every deviation with ARFIMA.
forecast::ForecastState final_state(const model::Layout& layout, const fit::LoglikSeries& series)
{
    std::size_t held = static_cast<std::size_t>(std::max({layout.ar, layout.ma, layout.arch, layout.garch}));
    std::size_t deviations = layout.has_arfima() ? series.deviations.size() : held;
    return {last(series.residuals, held), last(series.sigma2, held), last(series.permanent, held),
            last(series.deviations, deviations)};
}

fit::Nyblom nyblom_at_estimate(std::span<const double> scores, const model::Layout& layout)
{
    std::array<bool, kMaxParameterCount> estimated{};
    for (std::size_t i = 0; i < layout.count(); ++i)
    {
        estimated[i] = layout.is_estimated(i);
    }
    return fit::nyblom_statistics(scores, layout.count(), {estimated.data(), layout.count()});
}

bool fill_at_estimate(FitResult& result, const fit::LikelihoodProblem& problem,
                      const model::Standardization& standardization)
{
    std::span<const double> r = problem.data.r;
    std::vector<double> residuals(r.size());
    std::vector<double> sigma2(r.size());
    std::vector<double> scores(r.size() * result.estimate.size());
    std::vector<double> deviations(r.size());
    std::vector<double> permanent(problem.layout.model == Model::CsGarch ? r.size() : 0);
    fit::LoglikSeries series{residuals, sigma2, scores, deviations, permanent};
    double loglik = fit::loglik_with_scores(problem, result.estimate, series);
    if (!std::isfinite(loglik))
    {
        return false;
    }
    result.loglik = loglik;
    result.state = final_state(problem.layout, series);
    fill_series(result, r, std::move(residuals), sigma2);
    if (problem.layout.variance_targeting)
    {
        result.estimate[problem.layout.omega()] =
            model::targeted_omega_at(problem.data, result.estimate, problem.layout);
    }
    fit::StandardErrors errors = with_dual_width(
        problem.layout.count(),
        [&]<std::size_t N>()
        {
            LoglikGradient<N> loglik_gradient(problem);
            fit::GradientFn gradient = [&loglik_gradient](std::span<const double> theta, std::span<double> grad)
            { return loglik_gradient(theta, grad); };
            return fit::standard_errors(gradient, result.estimate, scores, problem.layout, standardization);
        });
    result.se_hessian = std::move(errors.hessian);
    result.se_robust = std::move(errors.robust);
    result.nyblom = nyblom_at_estimate(scores, problem.layout);
    return true;
}

// Writes the estimate of the free parameters into result.estimate, starting from `theta`. With nothing to
// estimate the optimizer does not run and the model is evaluated at `theta`.
optim::Stop estimate_parameters(const fit::LikelihoodProblem& problem,
                                const model::Standardization& standardization, std::vector<double> theta,
                                FitResult& result)
{
    if (problem.layout.estimated_count() == 0)
    {
        result.estimate = std::move(theta);
        return optim::Stop::Converged;
    }
    std::size_t k = problem.layout.count();
    std::vector<double> u(k);
    model::to_unconstrained(theta, u, problem.layout, standardization);
    optim::LbfgsResult outcome = with_dual_width(
        k, [&]<std::size_t N>() { return minimize<N>(problem, standardization, result.config, u); });
    if (outcome.stop == optim::Stop::NonFinite)
    {
        return outcome.stop;
    }
    result.estimate = std::move(theta);
    model::to_natural<double>(u, result.estimate, problem.layout, standardization);
    return outcome.stop;
}

// Fills `result`, which arrives as empty_result, and returns its status. On every status other than a
// finished optimization the estimate stays NaN.
FitStatus estimate(const model::Data& data, FitResult& result)
{
    fit::LikelihoodProblem problem{data, model::layout_of(result.spec), result.config.var_init};
    if (has_insufficient_data(data.r, result.spec, problem.layout))
    {
        return FitStatus::InsufficientData;
    }
    std::vector<double> theta = model::start_values(data, problem.layout);
    if (!has_valid_start(data.r, theta, problem.layout))
    {
        return FitStatus::Failed;
    }
    model::Standardization standardization = model::standardization_of(data);
    optim::Stop stop = estimate_parameters(problem, standardization, std::move(theta), result);
    if (stop == optim::Stop::NonFinite)
    {
        return FitStatus::Failed;
    }
    if (!fill_at_estimate(result, problem, standardization))
    {
        std::ranges::fill(result.estimate, kNaN);
        return FitStatus::Failed;
    }
    return status_from(stop);
}

}

FitResult fit_one(const model::Data& data, const Spec& spec, const FitConfig& config)
{
    FitResult result = empty_result(spec, config, data.r.size(), FitStatus::Failed);
    result.status = estimate(data, result);
    return result;
}

FitResult fit_one(std::span<const double> r, const Spec& spec, const FitConfig& config)
{
    return fit_one(model::Data{r}, spec, config);
}

}

namespace ve::fit
{

bool loglik_gradient(const LikelihoodProblem& problem, std::span<const double> theta, std::span<double> grad)
{
    assert(theta.size() == problem.layout.count() && grad.size() == theta.size());
    return with_dual_width(theta.size(),
                           [&]<std::size_t N>() { return LoglikGradient<N>(problem)(theta, grad); });
}

double loglik_with_scores(const LikelihoodProblem& problem, std::span<const double> theta,
                          const LoglikSeries& out)
{
    std::size_t k = problem.layout.count();
    assert(theta.size() == k && out.scores.size() == problem.data.r.size() * k);
    assert(out.residuals.size() == problem.data.r.size() && out.sigma2.size() == problem.data.r.size());
    return with_dual_width(k,
                           [&]<std::size_t N>()
                           {
                               return loglik_with_scores_with<N>(problem, theta, out);
                           });
}

}
