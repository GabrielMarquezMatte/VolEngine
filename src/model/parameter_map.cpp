#include "model/parameter_map.hpp"

#include "ad/links.hpp"
#include "dist/densities.hpp"
#include "dist/moments.hpp"
#include "model/regressor_map.hpp"
#include "model/weight_terms.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>

namespace ve::model
{

namespace
{

using ad::log_ratio;
using ad::logistic;
using ad::scaled_softmax;

// rugarch bounds the ARFIMA d to [1e-8, 0.5]; the map reaches the open interval.
constexpr double kArfimaFloor = 1e-8;
constexpr double kArfimaCeiling = 0.5;

template <class T>
T sum_of(std::span<const T> values)
{
    T sum{0.0};
    for (const T& value : values)
    {
        sum = sum + value;
    }
    return sum;
}

// u holds v followed by the n - 1 logits; weights has n entries.
template <class T>
void persistence_to_natural(std::span<const T> u, std::span<T> weights)
{
    T persistence = logistic(u[0]) * kPersistenceCeiling;
    scaled_softmax<T>(u.subspan(1), persistence, weights);
}

bool is_hyperbolic(Dist dist)
{
    return dist == Dist::Nig || dist == Dist::Ghyp;
}

bool is_student(Dist dist)
{
    return dist == Dist::Std || dist == Dist::Sstd;
}

// sstd, sged: xi = exp(x); nig, ghyp: rho = 2 logistic(x) - 1; jsu: nu = x.
template <class T>
T skew_to_natural(const T& x, Dist dist)
{
    using std::exp;
    if (dist::is_fernandez_steel(dist))
    {
        return exp(x);
    }
    if (is_hyperbolic(dist))
    {
        return 2.0 * logistic(x) - 1.0;
    }
    return x;
}

double skew_to_unconstrained(double skew, Dist dist)
{
    if (dist::is_fernandez_steel(dist))
    {
        return std::log(skew);
    }
    if (is_hyperbolic(dist))
    {
        return log_ratio(1.0 + skew, 1.0 - skew);
    }
    return skew;
}

template <class T>
T shape_to_natural(const T& x, Dist dist)
{
    using std::exp;
    if (is_student(dist))
    {
        return 2.0 + exp(x);
    }
    return exp(x);
}

double shape_to_unconstrained(double shape, Dist dist)
{
    if (is_student(dist))
    {
        return std::log(shape - 2.0);
    }
    return std::log(shape);
}

// ghlambda is unconstrained in both spaces.
template <class T>
void distribution_to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout)
{
    if (layout.has_skew())
    {
        theta[layout.skew()] = skew_to_natural(u[layout.skew()], layout.dist);
    }
    if (layout.has_shape())
    {
        theta[layout.shape()] = shape_to_natural(u[layout.shape()], layout.dist);
    }
    if (layout.has_ghlambda())
    {
        theta[layout.ghlambda()] = u[layout.ghlambda()];
    }
}

void distribution_to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout)
{
    if (layout.has_skew())
    {
        u[layout.skew()] = skew_to_unconstrained(theta[layout.skew()], layout.dist);
    }
    if (layout.has_shape())
    {
        u[layout.shape()] = shape_to_unconstrained(theta[layout.shape()], layout.dist);
    }
    if (layout.has_ghlambda())
    {
        u[layout.ghlambda()] = theta[layout.ghlambda()];
    }
}

// eGARCH alpha and gamma are unconstrained in both spaces.
template <class T>
void copy_egarch_reals(std::span<const T> source, std::span<T> target, const Layout& layout)
{
    if (layout.model != Model::EGarch)
    {
        return;
    }
    std::ranges::copy(source.subspan(layout.alpha_begin(), count_of(layout.arch)),
                      target.begin() + static_cast<std::ptrdiff_t>(layout.alpha_begin()));
    std::ranges::copy(source.subspan(layout.gamma_begin(), count_of(layout.gamma)),
                      target.begin() + static_cast<std::ptrdiff_t>(layout.gamma_begin()));
}

template <class T>
void mean_to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout,
                     const Standardization& standardization)
{
    if (layout.include_mean)
    {
        theta[layout.mu()] = standardization.location + standardization.scale * u[layout.mu()];
    }
    std::copy(u.begin() + static_cast<std::ptrdiff_t>(layout.ar_begin()),
              u.begin() + static_cast<std::ptrdiff_t>(layout.arfima_index()),
              theta.begin() + static_cast<std::ptrdiff_t>(layout.ar_begin()));
    if (layout.has_arfima())
    {
        std::size_t index = layout.arfima_index();
        theta[index] = kArfimaFloor + (kArfimaCeiling - kArfimaFloor) * logistic(u[index]);
    }
}

// Needs the natural power, so it follows write_fixed.
template <class T>
void regressors_to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout,
                           const Standardization& standardization)
{
    for (std::size_t index = layout.archm_index(); index < layout.skew(); ++index)
    {
        if (is_regressor(layout, index) && layout.is_estimated(index))
        {
            theta[index] = u[index] * regressor_scale<T>(theta, layout, standardization, index);
        }
    }
}

// mu is estimated beside mean regressors, and its u then sets the mean level at the column centers.
bool mean_is_centered(const Layout& layout)
{
    return layout.include_mean && layout.mxreg > 0 && layout.is_estimated(layout.mu());
}

// sum_k mxreg_k center_k, with the natural mxreg_k.
template <class T>
T regressor_offset(std::span<const T> theta, const Layout& layout, const Standardization& standardization)
{
    T offset{0.0};
    for (std::size_t k = 0; k < static_cast<std::size_t>(layout.mxreg); ++k)
    {
        offset = offset + theta[layout.mxreg_begin() + k] * standardization.mean_regressor_center[k];
    }
    return offset;
}

void regressors_to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout,
                                 const Standardization& standardization)
{
    for (std::size_t index = layout.archm_index(); index < layout.skew(); ++index)
    {
        if (is_regressor(layout, index))
        {
            u[index] = theta[index] / regressor_scale<double>(theta, layout, standardization, index);
        }
    }
}

// The distribution parameters, eta1 or the apARCH gamma, eta2 and the free power: everything kappa and pi
// read.
template <class T>
void shock_parameters_to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout)
{
    using std::exp;
    distribution_to_natural<T>(u, theta, layout);
    std::size_t asymmetry = layout.asymmetry_begin();
    for (std::size_t i = asymmetry; i < asymmetry + layout.asymmetry_count(); ++i)
    {
        theta[i] = 2.0 * logistic(u[i]) - 1.0;
    }
    if (layout.is_power_family())
    {
        std::copy_n(u.begin() + static_cast<std::ptrdiff_t>(layout.eta2_begin()), layout.eta2,
                    theta.begin() + static_cast<std::ptrdiff_t>(layout.eta2_begin()));
    }
    if (layout.free_power)
    {
        theta[layout.power()] = exp(u[layout.power()]);
    }
}

template <class T>
void igarch_weights_to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout)
{
    WeightBlock block = weight_block(layout);
    scaled_softmax<T>(u.subspan(block.begin, block.count - 1), T{1.0}, theta.subspan(block.begin, block.count));
    write_derived_weight<T>(theta, layout);
}

// From the weights ((1 - pi) alpha, pi (alpha + gamma)) in the alpha and gamma positions. The symmetric
// densities have pi = 0.5 exactly, where both divisions are exact doublings: alpha = 2 w_alpha and
// gamma = 2 w_gamma - 2 w_alpha = 2 (w_gamma - w_alpha), bit for bit.
template <class T>
void gjr_weights_to_natural(std::span<T> theta, const Layout& layout)
{
    T below = probability_below_zero_of<T>(theta, layout);
    T above = 1.0 - below;
    for (std::size_t i = 0; i < count_of(layout.arch); ++i)
    {
        T alpha = theta[layout.alpha_begin() + i] / above;
        theta[layout.gamma_begin() + i] = theta[layout.gamma_begin() + i] / below - alpha;
        theta[layout.alpha_begin() + i] = alpha;
    }
}

// Whether the weights share a budget below the ceiling of the plain map: some weight is fixed, or the csGARCH
// rho that caps them is.
bool weights_need_budget(const Layout& layout)
{
    WeightBlock block = weight_block(layout);
    for (std::size_t index = block.begin; index < block.begin + block.count; ++index)
    {
        if (layout.is_fixed(index))
        {
            return true;
        }
    }
    return layout.model == Model::CsGarch && layout.is_fixed(layout.eta1_begin());
}

// The free terms, in the order of the free positions: logistic(v) budget softmax(w, 0), with v the u of the first
// free position and w those of the others; iGARCH: budget softmax(w, 0) over the free positions and the derived
// beta_q, with w all the free u.
template <class T>
WeightArray<T> shared_terms(std::span<const T> u, const FreePositions& free, const T& budget, const Layout& layout)
{
    WeightArray<T> terms{};
    if (!(ad::value_of(budget) > 0.0))
    {
        terms.fill(T{std::numeric_limits<double>::quiet_NaN()});
        return terms;
    }
    WeightArray<T> gathered{};
    for (std::size_t m = 0; m < free.count; ++m)
    {
        gathered[m] = u[free.index[m]];
    }
    std::span<const T> draws = std::span<const T>(gathered).first(free.count);
    if (layout.model == Model::IGarch)
    {
        scaled_softmax<T>(draws, budget, std::span<T>(terms).first(free.count + 1));
        return terms;
    }
    if (free.count > 0)
    {
        scaled_softmax<T>(draws.subspan(1), logistic(draws[0]) * budget, std::span<T>(terms).first(free.count));
    }
    return terms;
}

template <class T>
void budget_weights_to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout)
{
    FreePositions free = free_positions(layout);
    T below = below_zero_of<T>(theta, layout);
    T budget = budget_of<T>(theta, layout, below);
    WeightArray<T> terms = shared_terms<T>(u, free, budget, layout);
    for (std::size_t m = 0; m < free.count; ++m)
    {
        theta[free.index[m]] = natural_at<T>(theta, layout, free.index[m], terms[m], below);
    }
    if (layout.model == Model::IGarch)
    {
        write_derived_weight<T>(theta, layout);
    }
}

// The weights of gjrGARCH are ((1 - pi) alpha, beta, pi (alpha + gamma)); of apARCH and fGARCH
// (alpha kappa, beta).
template <class T>
void weights_to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout)
{
    if (weights_need_budget(layout))
    {
        budget_weights_to_natural<T>(u, theta, layout);
        return;
    }
    if (layout.model == Model::IGarch)
    {
        igarch_weights_to_natural<T>(u, theta, layout);
        return;
    }
    WeightBlock block = weight_block(layout);
    persistence_to_natural<T>(u.subspan(block.begin, block.count), theta.subspan(block.begin, block.count));
    if (layout.model == Model::GjrGarch)
    {
        gjr_weights_to_natural<T>(theta, layout);
    }
    for (std::size_t i = 0; i < count_of(layout.arch) && layout.is_power_family(); ++i)
    {
        T moment = shock_moment<T>(theta, layout, i);
        theta[layout.alpha_begin() + i] = theta[layout.alpha_begin() + i] / moment;
    }
}

template <class T>
void csgarch_to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout)
{
    std::span<const T> natural = theta;
    T persistence = sum_of(natural.subspan(layout.alpha_begin(), count_of(layout.arch + layout.garch)));
    T beta_sum = sum_of(natural.subspan(layout.beta_begin(), count_of(layout.garch)));
    if (layout.is_estimated(layout.eta1_begin()))
    {
        theta[layout.eta1_begin()] =
            persistence + (kPersistenceCeiling - persistence) * logistic(u[layout.eta1_begin()]);
    }
    if (layout.is_estimated(layout.eta2_begin()))
    {
        theta[layout.eta2_begin()] = beta_sum * logistic(u[layout.eta2_begin()]);
    }
}

template <class T>
T omega_to_natural(std::span<const T> u, std::span<const T> theta, const Layout& layout, double scale)
{
    using std::exp;
    T x = u[layout.omega()];
    if (layout.model == Model::EGarch)
    {
        T beta_sum = sum_of(theta.subspan(layout.beta_begin(), count_of(layout.garch)));
        return x + (1.0 - beta_sum) * std::log(scale * scale);
    }
    if (layout.is_power_family())
    {
        return exp(power_of(theta, layout) * std::log(scale)) * exp(x);
    }
    return (scale * scale) * exp(x);
}

void shock_parameters_to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout)
{
    distribution_to_unconstrained(theta, u, layout);
    std::size_t asymmetry = layout.asymmetry_begin();
    for (std::size_t i = asymmetry; i < asymmetry + layout.asymmetry_count(); ++i)
    {
        u[i] = log_ratio(1.0 + theta[i], 1.0 - theta[i]);
    }
    if (layout.is_power_family())
    {
        std::copy_n(theta.begin() + static_cast<std::ptrdiff_t>(layout.eta2_begin()), layout.eta2,
                    u.begin() + static_cast<std::ptrdiff_t>(layout.eta2_begin()));
    }
    if (layout.free_power)
    {
        u[layout.power()] = std::log(theta[layout.power()]);
    }
}

// The inverse of budget_weights_to_natural.
void budget_weights_to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout)
{
    FreePositions free = free_positions(layout);
    double below = below_zero_of<double>(theta, layout);
    double budget = budget_of<double>(theta, layout, below);
    WeightArray<double> gathered{};
    for (std::size_t m = 0; m < free.count; ++m)
    {
        gathered[m] = term_at<double>(theta, layout, free.index[m], below);
    }
    std::span<const double> terms = std::span<const double>(gathered).first(free.count);
    double shared = sum_of(terms);
    if (layout.model == Model::IGarch)
    {
        for (std::size_t m = 0; m < free.count; ++m)
        {
            u[free.index[m]] = log_ratio(terms[m], budget - shared);
        }
        return;
    }
    if (free.count == 0)
    {
        return;
    }
    u[free.index[0]] = log_ratio(shared, budget - shared);
    for (std::size_t m = 1; m < free.count; ++m)
    {
        u[free.index[m]] = log_ratio(terms[m - 1], terms[free.count - 1]);
    }
}

// iGARCH: the reference weight is the derived beta_q, recomputed rather than read.
void weights_to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout)
{
    if (weights_need_budget(layout))
    {
        budget_weights_to_unconstrained(theta, u, layout);
        return;
    }
    WeightBlock block = weight_block(layout);
    WeightArray<double> natural = terms_of<double>(theta, layout);
    std::span<const double> weights = std::span<const double>(natural).first(block.count);
    std::span<double> logits = u.subspan(block.begin + 1, block.count - 1);
    double reference = weights[block.count - 1];
    if (layout.model == Model::IGarch)
    {
        logits = u.subspan(block.begin, block.count - 1);
        reference = 1.0 - sum_of(weights.first(block.count - 1));
    }
    if (layout.model != Model::IGarch)
    {
        double persistence = sum_of(weights);
        u[block.begin] = log_ratio(persistence, kPersistenceCeiling - persistence);
    }
    for (std::size_t i = 0; i + 1 < block.count; ++i)
    {
        logits[i] = log_ratio(weights[i], reference);
    }
}

void csgarch_to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout)
{
    double persistence = model::persistence<double>(theta, layout);
    double beta_sum = sum_of(theta.subspan(layout.beta_begin(), count_of(layout.garch)));
    double rho = theta[layout.eta1_begin()];
    double phi = theta[layout.eta2_begin()];
    u[layout.eta1_begin()] = log_ratio(rho - persistence, kPersistenceCeiling - rho);
    u[layout.eta2_begin()] = log_ratio(phi, beta_sum - phi);
}

double omega_to_unconstrained(std::span<const double> theta, const Layout& layout, double scale)
{
    double omega = theta[layout.omega()];
    if (layout.model == Model::EGarch)
    {
        double beta_sum = sum_of(theta.subspan(layout.beta_begin(), count_of(layout.garch)));
        return omega - (1.0 - beta_sum) * std::log(scale * scale);
    }
    if (layout.is_power_family())
    {
        return std::log(omega) - power_of(theta, layout) * std::log(scale);
    }
    return std::log(omega / (scale * scale));
}

bool shock_parameters_valid(std::span<const double> theta, const Layout& layout)
{
    if (!dist::in_domain(layout.dist, shape_parameters_of(theta, layout)))
    {
        return false;
    }
    std::span<const double> asymmetry = theta.subspan(layout.asymmetry_begin(), layout.asymmetry_count());
    bool asymmetry_valid = std::ranges::all_of(asymmetry, [](double value) { return std::abs(value) < 1.0; });
    return asymmetry_valid && (!layout.free_power || theta[layout.power()] > 0.0);
}

// A kappa that is not a positive finite double, or a gjrGARCH pi outside (0, 1), gives NaN weights (see
// weight_terms.hpp), which fail here.
bool weights_valid(std::span<const double> theta, const Layout& layout)
{
    WeightArray<double> natural = terms_of<double>(theta, layout);
    std::span<const double> weights = std::span<const double>(natural).first(weight_block(layout).count);
    if (layout.model == Model::IGarch)
    {
        weights = weights.first(weights.size() - 1);
    }
    if (!std::ranges::all_of(weights, [](double value) { return value >= 0.0; }))
    {
        return false;
    }
    double sum = sum_of(weights);
    if (layout.model == Model::IGarch)
    {
        return sum <= 1.0;
    }
    return sum < kPersistenceCeiling;
}

bool arfima_valid(double d)
{
    return d > kArfimaFloor && d < kArfimaCeiling;
}

bool csgarch_valid(std::span<const double> theta, const Layout& layout)
{
    double persistence = model::persistence<double>(theta, layout);
    double beta_sum = sum_of(theta.subspan(layout.beta_begin(), count_of(layout.garch)));
    double rho = theta[layout.eta1_begin()];
    double phi = theta[layout.eta2_begin()];
    return persistence < rho && rho < kPersistenceCeiling && phi >= 0.0 && phi <= beta_sum;
}

template <class T>
void write_fixed(std::span<T> theta, const Layout& layout)
{
    for (std::size_t index = 0; index < theta.size(); ++index)
    {
        if (layout.is_fixed(index))
        {
            theta[index] = T{layout.given[index]};
        }
    }
}

bool finite_but_derived(std::span<const double> theta, const Layout& layout)
{
    for (std::size_t index = 0; index < theta.size(); ++index)
    {
        if (!layout.is_derived(index) && !std::isfinite(theta[index]))
        {
            return false;
        }
    }
    return true;
}

bool in_block(std::size_t index, std::size_t begin, std::size_t count)
{
    return index >= begin && index < begin + count;
}

// The std likelihood tends to the normal one as 1/nu -> 0, its boundary. There its curvature in nu falls as
// nu^-4 and its cross terms as nu^-2, below what differences of the gradient resolve, and noise in the shape
// row and column alone can make the whole Hessian indefinite. sstd, nig, ghyp and jsu have their limit at
// 1/shape -> 0 as well, and there the ghyp ghlambda is no longer identified.
bool distribution_on_boundary(const Layout& layout, std::span<const double> theta, std::size_t index)
{
    Dist dist = layout.dist;
    if (layout.has_skew() && index == layout.skew())
    {
        return is_hyperbolic(dist) && std::abs(theta[index]) > 1.0 - kBoundary;
    }
    bool shape_limit = is_student(dist) || is_hyperbolic(dist) || dist == Dist::Jsu;
    return shape_limit && 1.0 / theta[layout.shape()] < kBoundary;
}

}

template <class T>
T shock_moment(std::span<const T> theta, const Layout& layout, std::size_t i)
{
    assert(layout.is_power_family() && i < count_of(layout.arch));
    T eta1{0.0};
    if (layout.asymmetry_count() > 0)
    {
        eta1 = theta[layout.asymmetry_begin() + i];
    }
    dist::ShapeParameters<T> parameters = shape_parameters_of(theta, layout);
    T power = power_of(theta, layout);
    if (layout.eta2 == 0)
    {
        return dist::asymmetric_moment(layout.dist, parameters, eta1, power);
    }
    return dist::shifted_moment(layout.dist, parameters, eta1, theta[layout.eta2_begin() + i], power);
}

template <class T>
void to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout,
                const Standardization& standardization)
{
    assert(u.size() == layout.count() && theta.size() == layout.count());
    assert(standardization.scale > 0.0);
    mean_to_natural<T>(u, theta, layout, standardization);
    copy_egarch_reals<T>(u, theta, layout);
    shock_parameters_to_natural<T>(u, theta, layout);
    write_fixed<T>(theta, layout);
    regressors_to_natural<T>(u, theta, layout, standardization);
    if (mean_is_centered(layout))
    {
        theta[layout.mu()] = theta[layout.mu()] - regressor_offset<T>(theta, layout, standardization);
    }
    weights_to_natural<T>(u, theta, layout);
    if (layout.model == Model::CsGarch)
    {
        csgarch_to_natural<T>(u, theta, layout);
    }
    if (layout.is_estimated(layout.omega()))
    {
        theta[layout.omega()] = omega_to_natural<T>(u, theta, layout, standardization.scale);
    }
}

template <class T>
T persistence(std::span<const T> theta, const Layout& layout)
{
    WeightArray<T> terms = terms_of<T>(theta, layout);
    return sum_of(std::span<const T>(terms).first(weight_block(layout).count));
}

template void to_natural<double>(std::span<const double>, std::span<double>, const Layout&,
                                 const Standardization&);
template double shock_moment<double>(std::span<const double>, const Layout&, std::size_t);
template double persistence<double>(std::span<const double>, const Layout&);
template void to_natural<ad::Dual<8>>(std::span<const ad::Dual<8>>, std::span<ad::Dual<8>>, const Layout&,
                                 const Standardization&);
template ad::Dual<8> shock_moment<ad::Dual<8>>(std::span<const ad::Dual<8>>, const Layout&, std::size_t);
template ad::Dual<8> persistence<ad::Dual<8>>(std::span<const ad::Dual<8>>, const Layout&);
template void to_natural<ad::Dual<16>>(std::span<const ad::Dual<16>>, std::span<ad::Dual<16>>, const Layout&,
                                 const Standardization&);
template ad::Dual<16> shock_moment<ad::Dual<16>>(std::span<const ad::Dual<16>>, const Layout&, std::size_t);
template ad::Dual<16> persistence<ad::Dual<16>>(std::span<const ad::Dual<16>>, const Layout&);
template void to_natural<ad::Dual<32>>(std::span<const ad::Dual<32>>, std::span<ad::Dual<32>>, const Layout&,
                                 const Standardization&);
template ad::Dual<32> shock_moment<ad::Dual<32>>(std::span<const ad::Dual<32>>, const Layout&, std::size_t);
template ad::Dual<32> persistence<ad::Dual<32>>(std::span<const ad::Dual<32>>, const Layout&);
using DualMax = ad::Dual<kMaxParameterCount>;
template void to_natural<DualMax>(std::span<const DualMax>, std::span<DualMax>, const Layout&,
                                  const Standardization&);
template DualMax shock_moment<DualMax>(std::span<const DualMax>, const Layout&, std::size_t);
template DualMax persistence<DualMax>(std::span<const DualMax>, const Layout&);

void to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout,
                      const Standardization& standardization)
{
    assert(is_valid(theta, layout));
    assert(u.size() == layout.count() && standardization.scale > 0.0);
    double scale = standardization.scale;
    if (layout.include_mean)
    {
        double mu = theta[layout.mu()];
        mu = mean_is_centered(layout) ? mu + regressor_offset<double>(theta, layout, standardization) : mu;
        u[layout.mu()] = (mu - standardization.location) / scale;
    }
    std::copy(theta.begin() + static_cast<std::ptrdiff_t>(layout.ar_begin()),
              theta.begin() + static_cast<std::ptrdiff_t>(layout.arfima_index()),
              u.begin() + static_cast<std::ptrdiff_t>(layout.ar_begin()));
    if (layout.has_arfima())
    {
        double d = theta[layout.arfima_index()];
        u[layout.arfima_index()] = log_ratio(d - kArfimaFloor, kArfimaCeiling - d);
    }
    regressors_to_unconstrained(theta, u, layout, standardization);
    u[layout.omega()] = omega_to_unconstrained(theta, layout, scale);
    copy_egarch_reals<double>(theta, u, layout);
    shock_parameters_to_unconstrained(theta, u, layout);
    weights_to_unconstrained(theta, u, layout);
    if (layout.model == Model::CsGarch)
    {
        csgarch_to_unconstrained(theta, u, layout);
    }
    for (std::size_t index = 0; index < u.size(); ++index)
    {
        if (!layout.is_estimated(index))
        {
            u[index] = 0.0;
        }
    }
}

double natural_scale(const Layout& layout, const Standardization& standardization, std::span<const double> theta,
                     std::size_t index)
{
    if (is_regressor(layout, index))
    {
        return regressor_scale<double>(theta, layout, standardization, index);
    }
    assert(layout.include_mean && index == layout.mu());
    return standardization.scale;
}

WeightTerms weight_terms(std::span<const double> theta, const Layout& layout)
{
    return WeightTerms{.terms = terms_of<double>(theta, layout),
                       .count = weight_block(layout).count,
                       .ceiling = weight_ceiling(layout)};
}

void natural_from_terms(const WeightTerms& terms, std::span<double> theta, const Layout& layout)
{
    WeightBlock block = weight_block(layout);
    double below = below_zero_of<double>(theta, layout);
    for (std::size_t j = 0; j < block.count; ++j)
    {
        std::size_t index = block.begin + j;
        if (layout.is_estimated(index))
        {
            theta[index] = natural_at<double>(theta, layout, index, terms.terms[j], below);
        }
    }
    if (layout.model == Model::IGarch)
    {
        write_derived_weight<double>(theta, layout);
    }
}

bool is_valid(std::span<const double> theta, const Layout& layout)
{
    assert(theta.size() == layout.count());
    if (!finite_but_derived(theta, layout))
    {
        return false;
    }
    if (!shock_parameters_valid(theta, layout))
    {
        return false;
    }
    if (layout.has_arfima() && !arfima_valid(theta[layout.arfima_index()]))
    {
        return false;
    }
    bool positive_omega = layout.model != Model::EGarch && !layout.is_derived(layout.omega());
    if (positive_omega && !(theta[layout.omega()] > 0.0))
    {
        return false;
    }
    if (!weights_valid(theta, layout))
    {
        return false;
    }
    return layout.model != Model::CsGarch || csgarch_valid(theta, layout);
}

bool on_boundary(const Layout& layout, std::span<const double> theta, std::size_t index)
{
    double value = theta[index];
    if (layout.has_derived() && index == layout.derived())
    {
        return true;
    }
    if (layout.has_arfima() && index == layout.arfima_index())
    {
        return value - kArfimaFloor < kBoundary || kArfimaCeiling - value < kBoundary;
    }
    if (index >= layout.skew())
    {
        return distribution_on_boundary(layout, theta, index);
    }
    if (layout.model == Model::CsGarch && index == layout.eta1_begin())
    {
        return value - persistence<double>(theta, layout) < kBoundary;
    }
    if (layout.model == Model::GjrGarch && in_block(index, layout.gamma_begin(), count_of(layout.gamma)))
    {
        return theta[layout.alpha_begin() + (index - layout.gamma_begin())] + value < kBoundary;
    }
    if (in_block(index, layout.asymmetry_begin(), layout.asymmetry_count()))
    {
        return std::abs(value) > 1.0 - kBoundary;
    }
    std::size_t i = index - layout.alpha_begin();
    if (in_block(index, layout.alpha_begin(), count_of(layout.arch)) && alpha_carries_gamma(layout, i))
    {
        return value - negative_part(theta[layout.gamma_begin() + i]) < kBoundary;
    }
    return layout.kind(index) == ParameterKind::Weight && value < kBoundary;
}

}
