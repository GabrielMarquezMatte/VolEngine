#include "dcc/copula/fit.hpp"

#include "dcc/copula/margins.hpp"
#include "dcc/copula/scores.hpp"
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

double shape_of(const DccSpec& spec, std::span<const double> theta, const DccLayout& layout)
{
    if (spec.dist != DccDist::MvT)
    {
        return 0.0;
    }
    return theta[layout.shape()];
}

// sum_t log f(epsilon_t; R_t) of the dynamic recursion; Rmgarch reads the zero presample (Stage::Objective) for
// everything (plan 4b notes, section 4).
double dynamic_terms(const StandardizedPanel& scores, const DccSpec& spec, std::span<const double> theta,
                     std::span<double> r)
{
    std::vector<double> terms(scores.t);
    const Stage stage = spec.convention == Convention::Rmgarch ? Stage::Objective : Stage::Reported;
    correlation_terms<double>(scores, spec, split_parameters<double>(spec, theta), stage, {terms, r});
    return std::accumulate(terms.begin(), terms.end(), 0.0);
}

// spec with the mvt shape fixed at nu: what one profile point fits.
DccSpec with_shape(DccSpec spec, double nu)
{
    spec.fixed.push_back({"mshape", nu});
    return spec;
}

// One profile point: the dynamic fit on the scores of the shape nu (the spec's own fixed shape, or none).
struct ProfilePoint
{
    DccFitResult inner;
    std::vector<double> theta;
    double loglik = kNaN;
};

ProfilePoint profile_point(const CopulaInputs& inputs, const DccSpec& spec, double nu, const FitConfig& config)
{
    const DccLayout layout = layout_of(spec, inputs.univariate->m);
    const bool free_shape = layout.has_shape && layout.is_estimated(layout.shape());
    const DccSpec inner_spec = free_shape ? with_shape(spec, nu) : spec;
    const CopulaPanel scores = copula_panel(inputs.uniforms_of(spec), *inputs.univariate, spec.dist, nu);
    ProfilePoint point{fit_dcc(scores.panel, inner_spec, config), {}, kNaN};
    point.theta = full_parameters(layout_of(inner_spec, inputs.univariate->m), point.inner.estimate);
    point.loglik = copula_loglik(inputs, spec, point.theta);
    return point;
}

// The shape of the profile maximum (minimising -loglik, +max where a point fails) and whether Brent ran out.
std::pair<double, bool> profile_shape(const CopulaInputs& inputs, const DccSpec& spec, const FitConfig& config)
{
    auto negative = [&](double nu)
    {
        const double loglik = profile_point(inputs, spec, nu, config).loglik;
        return std::isfinite(loglik) ? -loglik : std::numeric_limits<double>::max();
    };
    std::uintmax_t iterations = kShapeIterations;
    const auto best = boost::math::tools::brent_find_minima(negative, kShapeFloor, kShapeCeiling, kShapeBits, iterations);
    return {best.first, iterations >= kShapeIterations};
}

DccFitResult dynamic_fit(const CopulaInputs& inputs, const DccSpec& spec, const FitConfig& config)
{
    const DccLayout layout = layout_of(spec, inputs.univariate->m);
    double nu = layout.has_shape ? layout.given[layout.shape()] : 0.0;
    bool ran_out = false;
    if (layout.has_shape && layout.is_estimated(layout.shape()))
    {
        std::tie(nu, ran_out) = profile_shape(inputs, spec, config);
    }
    ProfilePoint point = profile_point(inputs, spec, nu, config);
    DccFitResult result = std::move(point.inner);
    result.spec = spec;
    result.names = estimated_names(layout);
    result.panel = inputs.univariate;
    if (!std::isfinite(point.loglik))
    {
        result.status = FitStatus::Failed;
        result.estimate.assign(result.names.size(), kNaN);
        result.se_hessian.assign(result.names.size(), kNaN);
        result.se_robust.assign(result.names.size(), kNaN);
        result.loglik = kNaN;
        result.r = nullptr;
        return result;
    }
    auto r = std::make_shared<std::vector<double>>(inputs.univariate->t * inputs.univariate->m * inputs.univariate->m);
    result.loglik = copula_loglik(inputs, spec, point.theta, *r);
    result.r = std::move(r);
    result.estimate = estimated_part(layout, point.theta);
    result.se_hessian.assign(result.names.size(), kNaN);
    result.se_robust.assign(result.names.size(), kNaN);
    if (ran_out && result.status == FitStatus::Converged)
    {
        result.status = FitStatus::MaxIter;
    }
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
    for (const DccSpec& spec : specs)
    {
        if (spec.copula == DccCopula::None)
        {
            continue;
        }
        std::vector<double>& slot = inputs.u[uniform_index(spec.copula, spec.convention)];
        if (slot.empty())
        {
            slot = uniforms(fits, *inputs.univariate, spec.copula, spec.convention);
        }
    }
    return inputs;
}

double copula_loglik(const CopulaInputs& inputs, const DccSpec& spec, std::span<const double> theta,
                     std::span<double> r)
{
    const DccLayout layout = layout_of(spec, inputs.univariate->m);
    assert(theta.size() == layout.count());
    const double nu = shape_of(spec, theta, layout);
    if (layout.has_shape && !(nu > kShapeFloor && nu < kShapeCeiling))
    {
        return kNaN;
    }
    const CopulaPanel scores = copula_panel(inputs.uniforms_of(spec), *inputs.univariate, spec.dist, nu);
    if (spec.model == DccModel::Static || !is_valid(theta, layout, constraint_delta(*scores.panel, spec)))
    {
        return kNaN;
    }
    const double dependence = dynamic_terms(*scores.panel, spec, theta, r);
    return inputs.univariate_loglik + dependence - scores.marginal_log_density;
}

DccFitResult fit_copula(const CopulaInputs& inputs, const DccSpec& spec, const FitConfig& config)
{
    assert(spec.copula != DccCopula::None && !spec_problem(spec, inputs.univariate->m));
    return dynamic_fit(inputs, spec, config);
}

}
