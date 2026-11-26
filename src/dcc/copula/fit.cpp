#include "dcc/copula/fit.hpp"

#include "dcc/copula/margins.hpp"
#include "dcc/copula/scores.hpp"
#include "dcc/copula/static_fit.hpp"
#include "dcc/likelihood.hpp"
#include "dcc/parameter_map.hpp"
#include "dcc/recursion.hpp"

#include <boost/math/tools/minima.hpp>

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace ve::dcc::copula
{

using detail::ProfilePoint;

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

// The summands of copula_terms from scores, which are those of theta's shape.
std::vector<double> terms_with(const CopulaInputs& inputs, const DccSpec& spec, std::span<const double> theta,
                               const CopulaPanel& scores, std::span<double> r)
{
    std::vector<double> terms = spec.model == DccModel::Static ? static_dependence(inputs, spec, theta, scores, r)
                                                               : dynamic_terms(*scores.panel, spec, theta, r);
    for (std::size_t t = 0; t < terms.size(); ++t)
    {
        terms[t] -= scores.marginal_terms[t];
    }
    return terms;
}

// copula_loglik at a profile point from the scores its fit used, at the point's shape: building them again was
// as costly as the fit of a Student point.
double point_loglik(const CopulaInputs& inputs, const DccSpec& spec, std::span<const double> theta,
                    const CopulaPanel& scores)
{
    const DccLayout layout = layout_of(spec, inputs.univariate->m);
    if (layout.has_shape && !(shape_of(spec, theta, layout) > 2.0))
    {
        return copula_loglik(inputs, spec, theta);
    }
    const std::vector<double> terms = terms_with(inputs, spec, theta, scores, {});
    return std::accumulate(terms.begin(), terms.end(), inputs.univariate_loglik);
}

// The estimated parameters of layout at theta, as a start list.
std::vector<ParameterValue> start_at(const DccLayout& layout, std::span<const double> theta)
{
    const std::vector<std::string> names = parameter_names(layout);
    std::vector<ParameterValue> start;
    for (const std::size_t index : estimated_positions(layout))
    {
        start.push_back({names[index], theta[index]});
    }
    return start;
}

// The point at nu; warm, when not null, is the neighbour whose estimate starts the inner fit (else the spec's start
// does).
ProfilePoint dynamic_point(const CopulaInputs& inputs, const DccSpec& spec, double nu, const FitConfig& config,
                           const ProfilePoint* warm)
{
    const DccLayout layout = layout_of(spec, inputs.univariate->m);
    DccSpec inner_spec = spec;
    if (layout.has_shape && layout.is_estimated(layout.shape()))
    {
        inner_spec.fixed.push_back({"mshape", nu});
    }
    const DccLayout inner_layout = layout_of(inner_spec, inputs.univariate->m);
    if (warm != nullptr)
    {
        inner_spec.start = start_at(inner_layout, warm->theta);
    }
    const CopulaPanel scores = copula_panel(inputs.uniforms_of(spec), *inputs.univariate, spec.dist, nu);
    const DccFitResult inner = fit_dcc(scores.panel, inner_spec, config);
    ProfilePoint point{inner.status, full_parameters(inner_layout, inner.estimate), kNaN, nu};
    point.loglik = point_loglik(inputs, spec, point.theta, scores);
    return point;
}

ProfilePoint static_point(const CopulaInputs& inputs, const DccSpec& spec, double nu, const FitConfig& config,
                          const ProfilePoint* warm)
{
    const DccLayout layout = layout_of(spec, inputs.univariate->m);
    ProfilePoint point{FitStatus::Converged, layout.given, kNaN, nu};
    if (layout.has_shape)
    {
        point.theta[layout.shape()] = nu;
    }
    if (layout.c_count > 0)
    {
        const CopulaPanel scores = copula_panel(inputs.uniforms_of(spec), *inputs.univariate, spec.dist, nu);
        std::span<const double> start;
        if (warm != nullptr)
        {
            start = std::span<const double>(warm->theta).subspan(layout.c_begin(), layout.c_count);
        }
        const StaticCorrelation correlation = static_ml(*scores.panel, layout, nu, config, start);
        std::ranges::copy(correlation.x, point.theta.begin() + static_cast<std::ptrdiff_t>(layout.c_begin()));
        point.status = correlation.status;
        point.loglik = point_loglik(inputs, spec, point.theta, scores);
        return point;
    }
    point.loglik = copula_loglik(inputs, spec, point.theta);
    return point;
}

ProfilePoint point_at(const CopulaInputs& inputs, const DccSpec& spec, double nu, const FitConfig& config,
                      const ProfilePoint* warm)
{
    if (spec.model == DccModel::Static)
    {
        return static_point(inputs, spec, nu, config, warm);
    }
    return dynamic_point(inputs, spec, nu, config, warm);
}

// The profile maximum (minimising -loglik, +max where a point fails) and whether Brent ran out. The points are
// kept in the order of evaluation, so every fit after the first starts from its nearest neighbour; the maximum is
// the point Brent evaluated (a minimum Boost reports is always an abscissa it called the objective at).
// ponytail: Brent re-fits every point (37 of them, ~0.1 s each at m = 21: 3.47 s against 0.144 s of the plain DCC-t,
// docs/performance/profile-5c.md); Newton on the profile through the envelope theorem if that gap matters.
std::pair<ProfilePoint, bool> profile_shape(const CopulaInputs& inputs, const DccSpec& spec, const FitConfig& config)
{
    std::vector<ProfilePoint> points;
    auto negative = [&](double nu)
    {
        points.push_back(point_at(inputs, spec, nu, config, detail::nearest_finite(points, nu)));
        const double loglik = points.back().loglik;
        return std::isfinite(loglik) ? -loglik : std::numeric_limits<double>::max();
    };
    std::uintmax_t iterations = kShapeIterations;
    const auto best = boost::math::tools::brent_find_minima(negative, kShapeFloor, kShapeCeiling, kShapeBits, iterations);
    const auto at_best = [&](const ProfilePoint& point)
    {
        return std::bit_cast<std::uint64_t>(point.nu) == std::bit_cast<std::uint64_t>(best.first);
    };
    const auto found = std::ranges::find_if(points, at_best);
    assert(found != points.end());
    return {std::move(*found), iterations >= kShapeIterations};
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
    // Like the plan 4a recursion, the terms are evaluated off the constraints too (the standard errors' differences
    // step there around a weight at 0 or a shape at its ceiling); only the scores need nu > 2.
    const double nu = shape_of(spec, theta, layout);
    if (layout.has_shape && !(nu > 2.0))
    {
        return std::vector<double>(inputs.univariate->t, kNaN);
    }
    const bool cached = inputs.scores != nullptr &&
                        (spec.dist != DccDist::MvT || std::bit_cast<std::uint64_t>(nu) ==
                                                          std::bit_cast<std::uint64_t>(inputs.scores_shape));
    std::shared_ptr<const CopulaPanel> fresh;
    if (!cached)
    {
        fresh = std::make_shared<const CopulaPanel>(
            copula_panel(inputs.uniforms_of(spec), *inputs.univariate, spec.dist, nu));
    }
    return terms_with(inputs, spec, theta, cached ? *inputs.scores : *fresh, r);
}

double copula_loglik(const CopulaInputs& inputs, const DccSpec& spec, std::span<const double> theta,
                     std::span<double> r)
{
    const std::vector<double> terms = copula_terms(inputs, spec, theta, r);
    return std::accumulate(terms.begin(), terms.end(), inputs.univariate_loglik);
}

CopulaInputs inputs_at(std::shared_ptr<const StandardizedPanel> univariate, std::span<const Margin> margins,
                       const DccSpec& spec, const CopulaInputs& base, double shape)
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
    if (spec.dist != DccDist::MvT || shape > 2.0)
    {
        inputs.scores = std::make_shared<const CopulaPanel>(
            copula_panel(inputs.u[index], *inputs.univariate, spec.dist, shape));
        inputs.scores_shape = shape;
    }
    return inputs;
}

CopulaInputs inputs_at_column(std::shared_ptr<const StandardizedPanel> univariate, const Margin& margin,
                              std::size_t column, const DccSpec& spec, const CopulaInputs& at_base, double shape)
{
    const std::size_t index = uniform_index(spec.copula, spec.convention);
    CopulaInputs inputs;
    inputs.univariate = std::move(univariate);
    const bool fixed_ranks = spec.convention == Convention::Standard;
    inputs.u[index] = at_base.u[index];
    if (!(fixed_ranks && spec.copula == DccCopula::Empirical))
    {
        uniforms_of_column(margin, *inputs.univariate, column, spec.copula, spec.convention, inputs.u[index]);
    }
    inputs.kendall[index] = at_base.kendall[index];
    if (reads_kendall(spec) && !fixed_ranks)
    {
        inputs.kendall[index] = kendall_correlation(inputs.u[index], inputs.univariate->t, inputs.univariate->m);
    }
    if (spec.dist != DccDist::MvT || shape > 2.0)
    {
        assert(at_base.scores != nullptr && std::bit_cast<std::uint64_t>(at_base.scores_shape) ==
                                                  std::bit_cast<std::uint64_t>(shape));
        inputs.scores = std::make_shared<const CopulaPanel>(copula_panel_with_column(
            *at_base.scores, inputs.u[index], *inputs.univariate, column, spec.dist, shape));
        inputs.scores_shape = shape;
    }
    return inputs;
}

namespace detail
{

const ProfilePoint* nearest_finite(std::span<const ProfilePoint> points, double nu)
{
    const ProfilePoint* nearest = nullptr;
    for (const ProfilePoint& point : points)
    {
        if (!std::isfinite(point.loglik) || point.status == FitStatus::Failed)
        {
            continue;
        }
        const double gap = std::abs(point.nu - nu);
        if (nearest == nullptr || gap < std::abs(nearest->nu - nu) ||
            (gap == std::abs(nearest->nu - nu) && point.nu < nearest->nu))
        {
            nearest = &point;
        }
    }
    return nearest;
}

}

DccFitResult fit_copula(const CopulaInputs& inputs, const DccSpec& spec, const FitConfig& config)
{
    assert(spec.copula != DccCopula::None && !spec_problem(spec, inputs.univariate->m));
    const DccLayout layout = layout_of(spec, inputs.univariate->m);
    const double nu = layout.has_shape ? layout.given[layout.shape()] : 0.0;
    const bool profiled = layout.has_shape && layout.is_estimated(layout.shape());
    ProfilePoint point;
    bool ran_out = false;
    if (profiled)
    {
        // A profiled shape is inside [4, 50] and is not checked again: the static copula's maximum sits on the end
        // when the loglik is monotone there (independent series put it at the ceiling). The dynamic copulas fix the
        // shape in a fit_dcc that wants it inside (4, 50), so an end point has no finite loglik and their maximum
        // stays inside.
        std::tie(point, ran_out) = profile_shape(inputs, spec, config);
    }
    // A fixed shape outside (4, 50) fails the spec, as on the plain DCC, for the static copula too.
    const bool fixed_shape_in_range = !layout.has_shape || (nu > kShapeFloor && nu < kShapeCeiling);
    if (!profiled && fixed_shape_in_range)
    {
        point = point_at(inputs, spec, nu, config, nullptr);
    }
    DccFitResult result = result_of(inputs, spec, layout, point);
    if (ran_out && result.status == FitStatus::Converged)
    {
        result.status = FitStatus::MaxIter;
    }
    return result;
}

}
