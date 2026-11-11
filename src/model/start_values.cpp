#include "model/start_values.hpp"

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
constexpr double kInitialStdShape = 8.0;
constexpr double kInitialGedShape = 2.0;
constexpr double kInitialPower = 2.0;
constexpr double kInitialRho = 0.98;
constexpr double kInitialPhi = 0.03;

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
    double power = layout.fixed_power;
    if (layout.free_power)
    {
        power = kInitialPower;
        theta[layout.power()] = power;
    }
    if (layout.model == Model::CsGarch)
    {
        theta[layout.eta1_begin()] = kInitialRho;
        theta[layout.eta2_begin()] = kInitialPhi;
    }
    if (layout.has_shape())
    {
        theta[layout.shape()] = layout.dist == Dist::Std ? kInitialStdShape : kInitialGedShape;
    }
    if (layout.is_power_family())
    {
        scale_power_alpha(theta, layout);
    }
    theta[layout.omega()] = omega_start(layout, sample_variance(r, mean), power);
    return theta;
}

}
