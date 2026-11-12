#include "model/parameter_map.hpp"

#include "dist/densities.hpp"
#include "dist/moments.hpp"

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

constexpr double kPersistenceCeiling = 1.0 - 1e-8;
constexpr double kBoundary = 1e-8;
constexpr double kInfinity = std::numeric_limits<double>::infinity();
// gjrGARCH: 2 p + q.
constexpr std::size_t kMaxWeightCount = 3 * kMaxOrder;

using Weights = std::array<double, kMaxWeightCount>;

template <class T>
T logistic(const T& x)
{
    using std::exp;
    if (ad::value_of(x) >= 0.0)
    {
        return 1.0 / (1.0 + exp(-x));
    }
    T exponential = exp(x);
    return exponential / (1.0 + exponential);
}

double log_ratio(double numerator, double denominator)
{
    constexpr double kFloor = std::numeric_limits<double>::min();
    return std::log(std::max(numerator, kFloor)) - std::log(std::max(denominator, kFloor));
}

std::size_t count_of(int order)
{
    return static_cast<std::size_t>(order);
}

struct WeightBlock
{
    std::size_t begin;
    std::size_t count;
};

WeightBlock weight_block(const Layout& layout)
{
    std::size_t p = count_of(layout.arch);
    std::size_t q = count_of(layout.garch);
    if (layout.model == Model::EGarch)
    {
        return {layout.beta_begin(), q};
    }
    if (layout.model == Model::GjrGarch)
    {
        return {layout.alpha_begin(), 2 * p + q};
    }
    return {layout.alpha_begin(), p + q};
}

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

// weights[i] = scale softmax(logits, 0)[i]; logits has weights.size() - 1 entries.
template <class T>
void softmax_to_natural(std::span<const T> logits, const T& scale, std::span<T> weights)
{
    using std::exp;
    std::size_t count = weights.size();
    double peak = 0.0;
    for (const T& logit : logits)
    {
        peak = std::max(peak, ad::value_of(logit));
    }
    T total{};
    for (std::size_t i = 0; i < count; ++i)
    {
        T logit = i + 1 < count ? logits[i] : T{};
        weights[i] = exp(logit - peak);
        total = total + weights[i];
    }
    for (T& weight : weights)
    {
        weight = scale * (weight / total);
    }
}

// u holds v followed by the n - 1 logits; weights has n entries.
template <class T>
void persistence_to_natural(std::span<const T> u, std::span<T> weights)
{
    T persistence = logistic(u[0]) * kPersistenceCeiling;
    softmax_to_natural<T>(u.subspan(1), persistence, weights);
}

bool is_fernandez_steel(Dist dist)
{
    return dist == Dist::Sstd || dist == Dist::Sged;
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
    if (is_fernandez_steel(dist))
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
    if (is_fernandez_steel(dist))
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

// pi = P(z < 0) splits the gjrGARCH alpha_i + gamma_i between the negative and the positive shocks.
template <class T>
T probability_below_zero_of(std::span<const T> theta, const Layout& layout)
{
    return dist::probability_below_zero(layout.dist, shape_parameters_of(theta, layout));
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
              u.begin() + static_cast<std::ptrdiff_t>(layout.omega()),
              theta.begin() + static_cast<std::ptrdiff_t>(layout.ar_begin()));
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
    softmax_to_natural<T>(u.subspan(block.begin, block.count - 1), T{1.0}, theta.subspan(block.begin, block.count));
    T derived{1.0};
    for (const T& weight : std::span<const T>(theta).subspan(block.begin, block.count - 1))
    {
        derived = derived - weight;
    }
    theta[layout.derived()] = derived;
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

// The weights of gjrGARCH are ((1 - pi) alpha, beta, pi (alpha + gamma)); of apARCH and fGARCH
// (alpha kappa, beta).
template <class T>
void weights_to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout)
{
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
    theta[layout.eta1_begin()] =
        persistence + (kPersistenceCeiling - persistence) * logistic(u[layout.eta1_begin()]);
    theta[layout.eta2_begin()] = beta_sum * logistic(u[layout.eta2_begin()]);
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

// value where 0 < value < upper, NaN otherwise. alpha_i is read back as w / kappa_i, and in gjrGARCH as
// w / (1 - pi) with gamma_i = w / pi - alpha_i, so a kappa or pi outside its range gives NaN weights.
double nan_unless(double value, double upper)
{
    if (value > 0.0 && value < upper)
    {
        return value;
    }
    return std::numeric_limits<double>::quiet_NaN();
}

// The natural weights of theta (see to_natural), in double.
Weights natural_weights(std::span<const double> theta, const Layout& layout)
{
    WeightBlock block = weight_block(layout);
    Weights weights{};
    std::copy_n(theta.begin() + static_cast<std::ptrdiff_t>(block.begin), block.count, weights.begin());
    std::size_t p = count_of(layout.arch);
    double below = 0.5;
    if (layout.model == Model::GjrGarch)
    {
        below = nan_unless(probability_below_zero_of<double>(theta, layout), 1.0);
    }
    for (std::size_t i = 0; i < p && layout.model == Model::GjrGarch; ++i)
    {
        double alpha = theta[layout.alpha_begin() + i];
        weights[i] = (1.0 - below) * alpha;
        weights[block.count - p + i] = below * (alpha + theta[layout.gamma_begin() + i]);
    }
    for (std::size_t i = 0; i < p && layout.is_power_family(); ++i)
    {
        double moment = nan_unless(shock_moment<double>(theta, layout, i), kInfinity);
        weights[i] = theta[layout.alpha_begin() + i] * moment;
    }
    return weights;
}

double persistence_of(std::span<const double> theta, const Layout& layout)
{
    Weights weights = natural_weights(theta, layout);
    return sum_of(std::span<const double>(weights).first(weight_block(layout).count));
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

// iGARCH: the reference weight is the derived beta_q, recomputed rather than read, and its u is unused.
void weights_to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout)
{
    WeightBlock block = weight_block(layout);
    Weights natural = natural_weights(theta, layout);
    std::span<const double> weights = std::span<const double>(natural).first(block.count);
    std::span<double> logits = u.subspan(block.begin + 1, block.count - 1);
    double reference = weights[block.count - 1];
    if (layout.model == Model::IGarch)
    {
        logits = u.subspan(block.begin, block.count - 1);
        reference = 1.0 - sum_of(weights.first(block.count - 1));
        u[layout.derived()] = 0.0;
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
    double persistence = persistence_of(theta, layout);
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
// nan_unless), which fail here.
bool weights_valid(std::span<const double> theta, const Layout& layout)
{
    Weights natural = natural_weights(theta, layout);
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

bool csgarch_valid(std::span<const double> theta, const Layout& layout)
{
    double persistence = persistence_of(theta, layout);
    double beta_sum = sum_of(theta.subspan(layout.beta_begin(), count_of(layout.garch)));
    double rho = theta[layout.eta1_begin()];
    double phi = theta[layout.eta2_begin()];
    return persistence < rho && rho < kPersistenceCeiling && phi >= 0.0 && phi <= beta_sum;
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
    weights_to_natural<T>(u, theta, layout);
    if (layout.model == Model::CsGarch)
    {
        csgarch_to_natural<T>(u, theta, layout);
    }
    theta[layout.omega()] = omega_to_natural<T>(u, theta, layout, standardization.scale);
}

template void to_natural<double>(std::span<const double>, std::span<double>, const Layout&,
                                 const Standardization&);
template double shock_moment<double>(std::span<const double>, const Layout&, std::size_t);
template void to_natural<ad::Dual<8>>(std::span<const ad::Dual<8>>, std::span<ad::Dual<8>>, const Layout&,
                                 const Standardization&);
template ad::Dual<8> shock_moment<ad::Dual<8>>(std::span<const ad::Dual<8>>, const Layout&, std::size_t);
template void to_natural<ad::Dual<16>>(std::span<const ad::Dual<16>>, std::span<ad::Dual<16>>, const Layout&,
                                 const Standardization&);
template ad::Dual<16> shock_moment<ad::Dual<16>>(std::span<const ad::Dual<16>>, const Layout&, std::size_t);
template void to_natural<ad::Dual<32>>(std::span<const ad::Dual<32>>, std::span<ad::Dual<32>>, const Layout&,
                                 const Standardization&);
template ad::Dual<32> shock_moment<ad::Dual<32>>(std::span<const ad::Dual<32>>, const Layout&, std::size_t);
template void to_natural<ad::Dual<42>>(std::span<const ad::Dual<42>>, std::span<ad::Dual<42>>, const Layout&,
                                 const Standardization&);
template ad::Dual<42> shock_moment<ad::Dual<42>>(std::span<const ad::Dual<42>>, const Layout&, std::size_t);

void to_unconstrained(std::span<const double> theta, std::span<double> u, const Layout& layout,
                      const Standardization& standardization)
{
    assert(is_valid(theta, layout));
    assert(u.size() == layout.count() && standardization.scale > 0.0);
    double scale = standardization.scale;
    if (layout.include_mean)
    {
        u[layout.mu()] = (theta[layout.mu()] - standardization.location) / scale;
    }
    std::copy(theta.begin() + static_cast<std::ptrdiff_t>(layout.ar_begin()),
              theta.begin() + static_cast<std::ptrdiff_t>(layout.omega()),
              u.begin() + static_cast<std::ptrdiff_t>(layout.ar_begin()));
    u[layout.omega()] = omega_to_unconstrained(theta, layout, scale);
    copy_egarch_reals<double>(theta, u, layout);
    shock_parameters_to_unconstrained(theta, u, layout);
    weights_to_unconstrained(theta, u, layout);
    if (layout.model == Model::CsGarch)
    {
        csgarch_to_unconstrained(theta, u, layout);
    }
}

bool is_valid(std::span<const double> theta, const Layout& layout)
{
    assert(theta.size() == layout.count());
    if (!std::ranges::all_of(theta, [](double value) { return std::isfinite(value); }))
    {
        return false;
    }
    if (!shock_parameters_valid(theta, layout))
    {
        return false;
    }
    if (layout.model != Model::EGarch && !(theta[layout.omega()] > 0.0))
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
    if (index >= layout.skew())
    {
        return distribution_on_boundary(layout, theta, index);
    }
    if (layout.model == Model::CsGarch && index == layout.eta1_begin())
    {
        return value - persistence_of(theta, layout) < kBoundary;
    }
    if (layout.model == Model::GjrGarch && in_block(index, layout.gamma_begin(), count_of(layout.gamma)))
    {
        return theta[layout.alpha_begin() + (index - layout.gamma_begin())] + value < kBoundary;
    }
    if (in_block(index, layout.asymmetry_begin(), layout.asymmetry_count()))
    {
        return std::abs(value) > 1.0 - kBoundary;
    }
    return layout.kind(index) == ParameterKind::Weight && value < kBoundary;
}

}
