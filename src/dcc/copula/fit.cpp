#include "dcc/copula/fit.hpp"

#include "dcc/copula/margins.hpp"
#include "dcc/copula/scores.hpp"
#include "dcc/copula/static_fit.hpp"
#include "dcc/likelihood.hpp"
#include "dcc/parameter_map.hpp"
#include "dcc/recursion.hpp"

#include <boost/math/tools/minima.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace ve::dcc::copula
{

namespace
{

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr int kShapeBits = 30;
constexpr std::uintmax_t kShapeIterations = 100;

std::size_t uniform_index(DccCopula copula, Convention convention)
{
    assert(copula != DccCopula::None);
    return 2 * (static_cast<std::size_t>(copula) - 1) + (convention == Convention::Rmgarch ? 1 : 0);
}

// Rmgarch's static mvnorm "Kendall" is cor(epsilon) (plan 4b notes, section 5); every other Kendall spec reads
// sin(pi tau_b / 2).
bool reads_kendall(const DccSpec& spec)
{
    const bool rmgarch_pearson = spec.convention == Convention::Rmgarch && spec.dist == DccDist::MvNorm;
    return spec.model == DccModel::Static && static_method_of(spec) == StaticMethod::Kendall && !rmgarch_pearson;
}

double shape_of(const DccSpec& spec, std::span<const double> theta, const DccLayout& layout)
{
    if (spec.dist != DccDist::MvT)
    {
        return 0.0;
    }
    return theta[layout.shape()];
}

// log f(epsilon_t; R_t) of the dynamic recursion; Rmgarch reads the zero presample (Stage::Objective) for
// everything (plan 4b notes, section 4).
std::vector<double> dynamic_terms(const StandardizedPanel& scores, const DccSpec& spec, std::span<const double> theta,
                                  std::span<double> r)
{
    std::vector<double> terms(scores.t);
    const Stage stage = spec.convention == Convention::Rmgarch ? Stage::Objective : Stage::Reported;
    correlation_terms<double>(scores, spec, split_parameters<double>(spec, theta), stage, {terms, r});
    return terms;
}

// cov2cor of the scores' Qbar = cov(epsilon).
std::vector<double> pearson(const StandardizedPanel& scores)
{
    const std::size_t m = scores.m;
    std::vector<double> r(m * m);
    for (std::size_t k = 0; k < m * m; ++k)
    {
        r[k] = scores.qbar[k] / std::sqrt(scores.qbar[(k / m) * (m + 1)] * scores.qbar[(k % m) * (m + 1)]);
    }
    return r;
}

// The constant R of a static spec at theta, or nullopt (a Kendall matrix that is not positive definite).
std::optional<std::vector<double>> static_correlation(const CopulaInputs& inputs, const DccSpec& spec,
                                                      std::span<const double> theta, const StandardizedPanel& scores)
{
    const DccLayout layout = layout_of(spec, scores.m);
    if (layout.c_count > 0)
    {
        return correlation_from_free(theta.subspan(layout.c_begin(), layout.c_count), scores.m);
    }
    if (reads_kendall(spec))
    {
        return inputs.kendall[uniform_index(spec.copula, spec.convention)];
    }
    return pearson(scores);
}

// log f(epsilon_t; R) of a static spec, writing R at every t of r.
std::vector<double> static_dependence(const CopulaInputs& inputs, const DccSpec& spec, std::span<const double> theta,
                                      const CopulaPanel& scores, std::span<double> r)
{
    const std::optional<std::vector<double>> correlation = static_correlation(inputs, spec, theta, *scores.panel);
    if (!correlation)
    {
        return std::vector<double>(scores.panel->t, kNaN);
    }
    const std::size_t m2 = correlation->size();
    for (std::size_t offset = 0; offset < r.size(); offset += m2)
    {
        std::ranges::copy(*correlation, r.begin() + static_cast<std::ptrdiff_t>(offset));
    }
    return static_observations(*scores.panel, *correlation, spec.dist,
                               shape_of(spec, theta, layout_of(spec, scores.panel->m)));
}

// One point of the shape profile (or the only point when the shape is fixed or absent): the estimate of the other
// parameters at the shape nu, its status and loglik.
struct ProfilePoint
{
    FitStatus status = FitStatus::Failed;
    std::vector<double> theta;
    double loglik = kNaN;
};

ProfilePoint dynamic_point(const CopulaInputs& inputs, const DccSpec& spec, double nu, const FitConfig& config)
{
    const DccLayout layout = layout_of(spec, inputs.univariate->m);
    DccSpec inner_spec = spec;
    if (layout.has_shape && layout.is_estimated(layout.shape()))
    {
        inner_spec.fixed.push_back({"mshape", nu});
    }
    const CopulaPanel scores = copula_panel(inputs.uniforms_of(spec), *inputs.univariate, spec.dist, nu);
    const DccFitResult inner = fit_dcc(scores.panel, inner_spec, config);
    ProfilePoint point{inner.status, full_parameters(layout_of(inner_spec, scores.panel->m), inner.estimate), kNaN};
    point.loglik = copula_loglik(inputs, spec, point.theta);
    return point;
}

ProfilePoint static_point(const CopulaInputs& inputs, const DccSpec& spec, double nu, const FitConfig& config)
{
    const DccLayout layout = layout_of(spec, inputs.univariate->m);
    ProfilePoint point{FitStatus::Converged, layout.given, kNaN};
    if (layout.has_shape)
    {
        point.theta[layout.shape()] = nu;
    }
    if (layout.c_count > 0)
    {
        const CopulaPanel scores = copula_panel(inputs.uniforms_of(spec), *inputs.univariate, spec.dist, nu);
        const StaticCorrelation correlation = static_ml(*scores.panel, layout, nu, config);
        std::ranges::copy(correlation.x, point.theta.begin() + static_cast<std::ptrdiff_t>(layout.c_begin()));
        point.status = correlation.status;
    }
    point.loglik = copula_loglik(inputs, spec, point.theta);
    return point;
}

ProfilePoint point_at(const CopulaInputs& inputs, const DccSpec& spec, double nu, const FitConfig& config)
{
    if (spec.model == DccModel::Static)
    {
        return static_point(inputs, spec, nu, config);
    }
    return dynamic_point(inputs, spec, nu, config);
}

// The shape of the profile maximum (minimising -loglik, +max where a point fails) and whether Brent ran out.
std::pair<double, bool> profile_shape(const CopulaInputs& inputs, const DccSpec& spec, const FitConfig& config)
{
    auto negative = [&](double nu)
    {
        const double loglik = point_at(inputs, spec, nu, config).loglik;
        return std::isfinite(loglik) ? -loglik : std::numeric_limits<double>::max();
    };
    std::uintmax_t iterations = kShapeIterations;
    const auto best = boost::math::tools::brent_find_minima(negative, kShapeFloor, kShapeCeiling, kShapeBits, iterations);
    return {best.first, iterations >= kShapeIterations};
}

DccFitResult result_of(const CopulaInputs& inputs, const DccSpec& spec, const DccLayout& layout,
                       const ProfilePoint& point)
{
    DccFitResult result;
    result.spec = spec;
    result.names = estimated_names(layout);
    result.estimate.assign(result.names.size(), kNaN);
    result.se_hessian.assign(result.names.size(), kNaN);
    result.se_robust.assign(result.names.size(), kNaN);
    result.nobs = inputs.univariate->t;
    result.series = inputs.univariate->m;
    result.panel = inputs.univariate;
    if (!std::isfinite(point.loglik))
    {
        return result;
    }
    auto r = std::make_shared<std::vector<double>>(result.nobs * result.series * result.series);
    result.loglik = copula_loglik(inputs, spec, point.theta, *r);
    result.r = std::move(r);
    result.estimate = estimated_part(layout, point.theta);
    result.status = point.status;
    return result;
}

}

std::span<const double> CopulaInputs::uniforms_of(const DccSpec& spec) const
{
    const std::vector<double>& out = u[uniform_index(spec.copula, spec.convention)];
    assert(!out.empty());
    return out;
}

CopulaInputs copula_inputs(std::span<const FitResult* const> fits, std::span<const DccSpec> specs)
{
    CopulaInputs inputs;
    if (std::ranges::none_of(specs, [](const DccSpec& spec) { return spec.copula != DccCopula::None; }))
    {
        return inputs;
    }
    inputs.univariate = std::make_shared<const StandardizedPanel>(standardize(fits));
    for (const FitResult* fit : fits)
    {
        inputs.univariate_loglik += fit->loglik;
    }
    const std::size_t t = inputs.univariate->t;
    const std::size_t m = inputs.univariate->m;
    for (const DccSpec& spec : specs)
    {
        if (spec.copula == DccCopula::None)
        {
            continue;
        }
        const std::size_t index = uniform_index(spec.copula, spec.convention);
        if (inputs.u[index].empty())
        {
            inputs.u[index] = uniforms(fits, *inputs.univariate, spec.copula, spec.convention);
        }
        if (reads_kendall(spec) && !inputs.kendall[index])
        {
            inputs.kendall[index] = kendall_correlation(inputs.u[index], t, m);
        }
    }
    return inputs;
}

std::vector<double> copula_terms(const CopulaInputs& inputs, const DccSpec& spec, std::span<const double> theta,
                                 std::span<double> r)
{
    const DccLayout layout = layout_of(spec, inputs.univariate->m);
    assert(theta.size() == layout.count());
    std::vector<double> terms(inputs.univariate->t, kNaN);
    // Like the plan 4a recursion, the terms are evaluated off the constraints too (the standard errors' differences
    // step there around a weight at 0 or a shape at its ceiling); only the scores need nu > 2.
    const double nu = shape_of(spec, theta, layout);
    if (layout.has_shape && !(nu > 2.0))
    {
        return terms;
    }
    const CopulaPanel scores = copula_panel(inputs.uniforms_of(spec), *inputs.univariate, spec.dist, nu);
    terms = spec.model == DccModel::Static ? static_dependence(inputs, spec, theta, scores, r)
                                           : dynamic_terms(*scores.panel, spec, theta, r);
    for (std::size_t t = 0; t < terms.size(); ++t)
    {
        terms[t] -= scores.marginal_terms[t];
    }
    return terms;
}

double copula_loglik(const CopulaInputs& inputs, const DccSpec& spec, std::span<const double> theta,
                     std::span<double> r)
{
    const std::vector<double> terms = copula_terms(inputs, spec, theta, r);
    return std::accumulate(terms.begin(), terms.end(), inputs.univariate_loglik);
}

CopulaInputs inputs_at(std::shared_ptr<const StandardizedPanel> univariate, std::span<const Margin> margins,
                       const DccSpec& spec, const CopulaInputs& base)
{
    const std::size_t index = uniform_index(spec.copula, spec.convention);
    CopulaInputs inputs;
    inputs.univariate = std::move(univariate);
    const bool fixed_ranks = spec.convention == Convention::Standard;
    inputs.u[index] = fixed_ranks && spec.copula == DccCopula::Empirical
                          ? base.u[index]
                          : uniforms(margins, *inputs.univariate, spec.copula, spec.convention);
    inputs.kendall[index] = base.kendall[index];
    if (reads_kendall(spec) && !fixed_ranks)
    {
        inputs.kendall[index] = kendall_correlation(inputs.u[index], inputs.univariate->t, inputs.univariate->m);
    }
    return inputs;
}

DccFitResult fit_copula(const CopulaInputs& inputs, const DccSpec& spec, const FitConfig& config)
{
    assert(spec.copula != DccCopula::None && !spec_problem(spec, inputs.univariate->m));
    const DccLayout layout = layout_of(spec, inputs.univariate->m);
    double nu = layout.has_shape ? layout.given[layout.shape()] : 0.0;
    bool ran_out = false;
    if (layout.has_shape && layout.is_estimated(layout.shape()))
    {
        std::tie(nu, ran_out) = profile_shape(inputs, spec, config);
    }
    DccFitResult result = result_of(inputs, spec, layout, point_at(inputs, spec, nu, config));
    if (ran_out && result.status == FitStatus::Converged)
    {
        result.status = FitStatus::MaxIter;
    }
    return result;
}

}
