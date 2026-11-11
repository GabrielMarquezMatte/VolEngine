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

bool is_power_family(Model model)
{
    return model == Model::ApArch || model >= Model::FGarchGarch;
}

// eta1, or the apARCH gamma; csGARCH keeps rho in eta1.
std::size_t asymmetry_begin(const Layout& layout)
{
    if (layout.model == Model::ApArch)
    {
        return layout.gamma_begin();
    }
    return layout.eta1_begin();
}

std::size_t asymmetry_count(const Layout& layout)
{
    if (layout.model == Model::ApArch)
    {
        return count_of(layout.gamma);
    }
    if (layout.model == Model::CsGarch)
    {
        return 0;
    }
    return count_of(layout.eta1);
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

template <class T>
T power_of(std::span<const T> theta, const Layout& layout)
{
    if (!layout.free_power)
    {
        return T{layout.fixed_power};
    }
    return theta[layout.power()];
}

template <class T>
T shape_of(std::span<const T> theta, const Layout& layout)
{
    if (!layout.has_shape())
    {
        return T{0.0};
    }
    return theta[layout.shape()];
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

template <class T>
T shape_to_natural(const T& x, Dist dist)
{
    using std::exp;
    return dist == Dist::Std ? 2.0 + exp(x) : exp(x);
}

double shape_to_unconstrained(double shape, Dist dist)
{
    return std::log(dist == Dist::Std ? shape - 2.0 : shape);
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

// shape, eta1 or the apARCH gamma, eta2 and the free power: everything kappa reads.
template <class T>
void shock_parameters_to_natural(std::span<const T> u, std::span<T> theta, const Layout& layout)
{
    using std::exp;
    if (layout.has_shape())
    {
        theta[layout.shape()] = shape_to_natural(u[layout.shape()], layout.dist);
    }
    std::size_t asymmetry = asymmetry_begin(layout);
    for (std::size_t i = asymmetry; i < asymmetry + asymmetry_count(layout); ++i)
    {
        theta[i] = 2.0 * logistic(u[i]) - 1.0;
    }
    if (is_power_family(layout.model))
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

// The weights of gjrGARCH are (alpha / 2, beta, (alpha + gamma) / 2); of apARCH and fGARCH (alpha kappa, beta).
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
    for (std::size_t i = 0; i < count_of(layout.arch) && layout.model == Model::GjrGarch; ++i)
    {
        T half_alpha = theta[layout.alpha_begin() + i];
        T half_sum = theta[layout.gamma_begin() + i];
        theta[layout.alpha_begin() + i] = 2.0 * half_alpha;
        theta[layout.gamma_begin() + i] = 2.0 * (half_sum - half_alpha);
    }
    for (std::size_t i = 0; i < count_of(layout.arch) && is_power_family(layout.model); ++i)
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
    if (is_power_family(layout.model))
    {
        return exp(power_of(theta, layout) * std::log(scale)) * exp(x);
    }
    return (scale * scale) * exp(x);
}

// The natural weights of theta (see to_natural), in double.
Weights natural_weights(std::span<const double> theta, const Layout& layout)
{
    WeightBlock block = weight_block(layout);
    Weights weights{};
    std::copy_n(theta.begin() + static_cast<std::ptrdiff_t>(block.begin), block.count, weights.begin());
    std::size_t p = count_of(layout.arch);
    for (std::size_t i = 0; i < p && layout.model == Model::GjrGarch; ++i)
    {
        double alpha = theta[layout.alpha_begin() + i];
        weights[i] = 0.5 * alpha;
        weights[block.count - p + i] = 0.5 * (alpha + theta[layout.gamma_begin() + i]);
    }
    for (std::size_t i = 0; i < p && is_power_family(layout.model); ++i)
    {
        weights[i] = theta[layout.alpha_begin() + i] * shock_moment<double>(theta, layout, i);
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
    if (layout.has_shape())
    {
        u[layout.shape()] = shape_to_unconstrained(theta[layout.shape()], layout.dist);
    }
    std::size_t asymmetry = asymmetry_begin(layout);
    for (std::size_t i = asymmetry; i < asymmetry + asymmetry_count(layout); ++i)
    {
        u[i] = log_ratio(1.0 + theta[i], 1.0 - theta[i]);
    }
    if (is_power_family(layout.model))
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
    if (is_power_family(layout.model))
    {
        return std::log(omega) - power_of(theta, layout) * std::log(scale);
    }
    return std::log(omega / (scale * scale));
}

bool shock_parameters_valid(std::span<const double> theta, const Layout& layout)
{
    if (layout.has_shape() && !dist::in_domain(layout.dist, theta[layout.shape()]))
    {
        return false;
    }
    std::span<const double> asymmetry = theta.subspan(asymmetry_begin(layout), asymmetry_count(layout));
    bool asymmetry_valid = std::ranges::all_of(asymmetry, [](double value) { return std::abs(value) < 1.0; });
    return asymmetry_valid && (!layout.free_power || theta[layout.power()] > 0.0);
}

// A kappa that is not finite gives a weight that is NaN or +inf, which fails here.
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

}

template <class T>
T shock_moment(std::span<const T> theta, const Layout& layout, std::size_t i)
{
    assert(is_power_family(layout.model) && i < count_of(layout.arch));
    T eta1{0.0};
    if (asymmetry_count(layout) > 0)
    {
        eta1 = theta[asymmetry_begin(layout) + i];
    }
    T shape = shape_of(theta, layout);
    T power = power_of(theta, layout);
    if (layout.eta2 == 0)
    {
        return dist::asymmetric_moment(layout.dist, shape, eta1, power);
    }
    return dist::shifted_moment(layout.dist, shape, eta1, theta[layout.eta2_begin() + i], power);
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
template void to_natural<ad::Dual<40>>(std::span<const ad::Dual<40>>, std::span<ad::Dual<40>>, const Layout&,
                                 const Standardization&);
template ad::Dual<40> shock_moment<ad::Dual<40>>(std::span<const ad::Dual<40>>, const Layout&, std::size_t);

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

// The std likelihood tends to the normal one as 1/nu -> 0, its boundary. There its curvature in nu falls as
// nu^-4 and its cross terms as nu^-2, below what differences of the gradient resolve, and noise in the shape
// row and column alone can make the whole Hessian indefinite.
bool on_boundary(const Layout& layout, std::span<const double> theta, std::size_t index)
{
    double value = theta[index];
    if (layout.has_derived() && index == layout.derived())
    {
        return true;
    }
    if (layout.has_shape() && index == layout.shape())
    {
        return layout.dist == Dist::Std && 1.0 / value < kBoundary;
    }
    if (layout.model == Model::CsGarch && index == layout.eta1_begin())
    {
        return value - persistence_of(theta, layout) < kBoundary;
    }
    if (layout.model == Model::GjrGarch && in_block(index, layout.gamma_begin(), count_of(layout.gamma)))
    {
        return theta[layout.alpha_begin() + (index - layout.gamma_begin())] + value < kBoundary;
    }
    if (in_block(index, asymmetry_begin(layout), asymmetry_count(layout)))
    {
        return std::abs(value) > 1.0 - kBoundary;
    }
    return layout.kind(index) == ParameterKind::Weight && value < kBoundary;
}

}
