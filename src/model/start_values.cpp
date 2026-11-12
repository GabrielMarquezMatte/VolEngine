#include "model/start_values.hpp"

#include "model/weight_terms.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <numeric>

namespace ve::model
{

namespace
{

constexpr double kInitialArch = 0.05;
constexpr double kInitialGarch = 0.90;
constexpr double kInitialOmegaShare = 1.0 - (kInitialArch + kInitialGarch);
constexpr double kInitialStudentShape = 8.0;
constexpr double kInitialGedShape = 2.0;
constexpr double kInitialHyperbolicShape = 2.0;
constexpr double kInitialJohnsonShape = 2.0;
constexpr double kInitialGhLambda = -0.5;
constexpr double kInitialPower = 2.0;
constexpr double kInitialRho = 0.98;
constexpr double kInitialPhi = 0.03;
constexpr double kRescaleMargin = 0.95;

// omega is a share of var(r), of log var(r) in eGARCH, or of var(r)^(power / 2) in apARCH and fGARCH; alpha
// and gamma are totals split evenly over the p lags.
struct VarianceStart
{
    double omega;
    double alpha;
    double gamma;
};

VarianceStart variance_start(Model model)
{
    switch (model)
    {
        case Model::EGarch:
            return {0.1, 0.0, 0.1};
        case Model::GjrGarch:
            return {kInitialOmegaShare, 0.03, 0.04};
        case Model::IGarch:
            return {0.01, 0.1, 0.0};
        case Model::CsGarch:
            return {0.02, kInitialArch, 0.0};
        case Model::SGarch:
        case Model::ApArch:
        case Model::FGarchGarch:
        case Model::FGarchTGarch:
        case Model::FGarchAvGarch:
        case Model::FGarchNGarch:
        case Model::FGarchNaGarch:
        case Model::FGarchApArch:
        case Model::FGarchGjrGarch:
        case Model::FGarchAllGarch:
            return {kInitialOmegaShare, kInitialArch, 0.0};
    }
    return {kInitialOmegaShare, kInitialArch, 0.0};
}

// The symmetric member of each family (skew xi = 1, rho = 0 or nu = 0); nig and ghyp start with the kurtosis
// of a t with 8 degrees of freedom.
dist::ShapeParameters<double> distribution_start(Dist dist)
{
    switch (dist)
    {
        case Dist::Std:
            return {.shape = kInitialStudentShape};
        case Dist::Ged:
            return {.shape = kInitialGedShape};
        case Dist::Sstd:
            return {.skew = 1.0, .shape = kInitialStudentShape};
        case Dist::Sged:
            return {.skew = 1.0, .shape = kInitialGedShape};
        case Dist::Nig:
            return {.skew = 0.0, .shape = kInitialHyperbolicShape};
        case Dist::Ghyp:
            return {.skew = 0.0, .shape = kInitialHyperbolicShape, .ghlambda = kInitialGhLambda};
        case Dist::Jsu:
            return {.skew = 0.0, .shape = kInitialJohnsonShape};
        case Dist::Norm:
            break;
    }
    return {};
}

void set_distribution_start(std::vector<double>& theta, const Layout& layout)
{
    dist::ShapeParameters<double> start = distribution_start(layout.dist);
    if (layout.has_skew())
    {
        theta[layout.skew()] = start.skew;
    }
    if (layout.has_shape())
    {
        theta[layout.shape()] = start.shape;
    }
    if (layout.has_ghlambda())
    {
        theta[layout.ghlambda()] = start.ghlambda;
    }
}

double sample_variance(std::span<const double> r, double mean)
{
    double sum = 0.0;
    for (double value : r)
    {
        sum += (value - mean) * (value - mean);
    }
    return sum / static_cast<double>(r.size());
}

double sample_mean(std::span<const double> r)
{
    return std::accumulate(r.begin(), r.end(), 0.0) / static_cast<double>(r.size());
}

void fill_block(std::vector<double>& theta, std::size_t begin, int count, double value)
{
    std::fill_n(theta.begin() + static_cast<std::ptrdiff_t>(begin), count, value);
}

// alpha_i = alpha / (p kappa_i), so that the alpha_i kappa_i weights sum to alpha.
void scale_power_alpha(std::vector<double>& theta, const Layout& layout)
{
    for (std::size_t i = 0; i < static_cast<std::size_t>(layout.arch); ++i)
    {
        theta[layout.alpha_begin() + i] /= shock_moment<double>(theta, layout, i);
    }
}

double omega_start(const Layout& layout, double variance, double power)
{
    double share = variance_start(layout.model).omega;
    if (layout.model == Model::EGarch)
    {
        return share * std::log(variance);
    }
    if (layout.is_power_family())
    {
        return share * std::pow(variance, power / 2.0);
    }
    return variance * share;
}

bool has_user_value(const Layout& layout, std::size_t index)
{
    return layout.is_fixed(index) || layout.is_started(index);
}

enum class Part
{
    Weights,
    Omega,
    Rest
};

Part part_of(const Layout& layout, std::size_t index)
{
    WeightBlock block = weight_block(layout);
    if (index == layout.omega())
    {
        return Part::Omega;
    }
    if (index >= block.begin && index < block.begin + block.count)
    {
        return Part::Weights;
    }
    return Part::Rest;
}

void apply_user_values(std::vector<double>& theta, const Layout& layout, Part part)
{
    for (std::size_t index = 0; index < theta.size(); ++index)
    {
        if (has_user_value(layout, index) && part_of(layout, index) == part)
        {
            theta[index] = layout.given[index];
        }
    }
}

bool weights_have_user_value(const Layout& layout)
{
    WeightBlock block = weight_block(layout);
    for (std::size_t index = block.begin; index < block.begin + block.count; ++index)
    {
        if (has_user_value(layout, index))
        {
            return true;
        }
    }
    return false;
}

// A fixed gamma_i < 0 asks alpha_i >= -gamma_i, so a default alpha_i starts above that.
void raise_default_gjr_alpha(std::vector<double>& theta, const Layout& layout)
{
    if (layout.model != Model::GjrGarch)
    {
        return;
    }
    for (std::size_t i = 0; i < static_cast<std::size_t>(layout.arch); ++i)
    {
        std::size_t alpha = layout.alpha_begin() + i;
        std::size_t gamma = layout.gamma_begin() + i;
        if (layout.is_fixed(gamma) && !has_user_value(layout, alpha))
        {
            theta[alpha] += std::max(0.0, -theta[gamma]);
        }
    }
}

// The csGARCH terms must stay below the start rho, whether it is fixed, given or the default.
double start_ceiling(const std::vector<double>& theta, const Layout& layout)
{
    if (layout.model == Model::CsGarch)
    {
        return theta[layout.eta1_begin()];
    }
    return weight_ceiling(layout);
}

enum class Role
{
    Fixed,
    Started,
    Derived,
    Standard
};

Role role_of(const Layout& layout, std::size_t index)
{
    if (layout.is_fixed(index))
    {
        return Role::Fixed;
    }
    if (layout.is_started(index))
    {
        return Role::Started;
    }
    if (layout.is_derived(index))
    {
        return Role::Derived;
    }
    return Role::Standard;
}

double sum_of_role(const WeightTerms& terms, const Layout& layout, Role role)
{
    std::size_t begin = weight_block(layout).begin;
    double sum = 0.0;
    for (std::size_t j = 0; j < terms.count; ++j)
    {
        if (role_of(layout, begin + j) == role)
        {
            sum += terms.terms[j];
        }
    }
    return sum;
}

// The standard terms are scaled by s = min(1, 0.95 (c - F - U) / D), F, U and D being the sums of the fixed,
// user-started and standard terms; nothing changes when D = 0 or c - F - U <= 0 (the start is then invalid).
void rescale_standard_weights(std::vector<double>& theta, const Layout& layout)
{
    WeightTerms terms = weight_terms(theta, layout);
    double room = start_ceiling(theta, layout) - sum_of_role(terms, layout, Role::Fixed) -
                  sum_of_role(terms, layout, Role::Started);
    double standard = sum_of_role(terms, layout, Role::Standard);
    if (!(standard > 0.0 && room > 0.0))
    {
        return;
    }
    double scale = std::min(1.0, kRescaleMargin * room / standard);
    if (scale == 1.0)
    {
        return;
    }
    std::size_t begin = weight_block(layout).begin;
    for (std::size_t j = 0; j < terms.count; ++j)
    {
        if (role_of(layout, begin + j) == Role::Standard)
        {
            terms.terms[j] *= scale;
        }
    }
    std::vector<double> rescaled = theta;
    natural_from_terms(terms, rescaled, layout);
    for (std::size_t j = 0; j < terms.count; ++j)
    {
        if (role_of(layout, begin + j) == Role::Standard)
        {
            theta[begin + j] = rescaled[begin + j];
        }
    }
}

void set_weights(std::vector<double>& theta, const Layout& layout)
{
    if (layout.is_power_family())
    {
        scale_power_alpha(theta, layout);
    }
    if (!weights_have_user_value(layout))
    {
        return;
    }
    apply_user_values(theta, layout, Part::Weights);
    raise_default_gjr_alpha(theta, layout);
    rescale_standard_weights(theta, layout);
    if (layout.model == Model::IGarch)
    {
        write_derived_weight<double>(theta, layout);
    }
}

// phi <= sum(beta) keeps the default start inside the domain when the betas are small.
void set_default_phi(std::vector<double>& theta, const Layout& layout)
{
    if (layout.model != Model::CsGarch || has_user_value(layout, layout.eta2_begin()))
    {
        return;
    }
    auto betas = theta.begin() + static_cast<std::ptrdiff_t>(layout.beta_begin());
    double beta_sum = std::accumulate(betas, betas + layout.garch, 0.0);
    theta[layout.eta2_begin()] = std::min(kInitialPhi, 0.5 * beta_sum);
}

}

Standardization standardization_of(std::span<const double> r)
{
    double mean = sample_mean(r);
    return {mean, std::sqrt(sample_variance(r, mean))};
}

std::vector<double> start_values(std::span<const double> r, const Layout& layout)
{
    assert(!r.empty());
    double mean = sample_mean(r);
    std::vector<double> theta(layout.count(), 0.0);
    if (layout.include_mean)
    {
        theta[layout.mu()] = mean;
    }
    VarianceStart start = variance_start(layout.model);
    fill_block(theta, layout.alpha_begin(), layout.arch, start.alpha / layout.arch);
    fill_block(theta, layout.beta_begin(), layout.garch, kInitialGarch / layout.garch);
    fill_block(theta, layout.gamma_begin(), layout.gamma, start.gamma / layout.arch);
    if (layout.free_power)
    {
        theta[layout.power()] = kInitialPower;
    }
    if (layout.model == Model::CsGarch)
    {
        theta[layout.eta1_begin()] = kInitialRho;
    }
    set_distribution_start(theta, layout);
    apply_user_values(theta, layout, Part::Rest);
    set_weights(theta, layout);
    set_default_phi(theta, layout);
    theta[layout.omega()] = omega_start(layout, sample_variance(r, mean), power_of<double>(theta, layout));
    apply_user_values(theta, layout, Part::Omega);
    return theta;
}

}
