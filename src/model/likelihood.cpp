#include "model/likelihood.hpp"

#include "dist/densities.hpp"
#include "garch/sgarch.hpp"
#include "mean/arma.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace ve::model
{

namespace
{

// arch backcast: exponentially weighted mean of the first min(75, T) squared residuals, decay 0.94.
constexpr std::size_t kBackcastWindow = 75;
constexpr double kBackcastDecay = 0.94;

constexpr double kInvalid = -std::numeric_limits<double>::infinity();

double sample_mean(std::span<const double> r)
{
    double sum = 0.0;
    for (double value : r)
    {
        sum += value;
    }
    return sum / static_cast<double>(r.size());
}

// arch computes the backcast once, from the residuals at its starting mean (the sample mean, or zero without a
// mean), and keeps it fixed during the optimization: it depends on no parameter.
double backcast(std::span<const double> r, bool include_mean)
{
    double center = 0.0;
    if (include_mean)
    {
        center = sample_mean(r);
    }
    double weight = 1.0;
    double weight_sum = 0.0;
    double weighted_square_sum = 0.0;
    for (double value : r.first(std::min(kBackcastWindow, r.size())))
    {
        double residual = value - center;
        weight_sum += weight;
        weighted_square_sum += weight * residual * residual;
        weight *= kBackcastDecay;
    }
    return weighted_square_sum / weight_sum;
}

template <class T>
std::span<const T> block(std::span<const T> theta, std::size_t begin, int count)
{
    return theta.subspan(begin, static_cast<std::size_t>(count));
}

template <class T>
mean::ArmaParameters<T> arma_parameters(std::span<const T> theta, const Layout& layout)
{
    T mu{0.0};
    if (layout.include_mean)
    {
        mu = theta[layout.mu()];
    }
    return {mu, block(theta, layout.ar_begin(), layout.ar), block(theta, layout.ma_begin(), layout.ma)};
}

template <class T>
garch::SGarchParameters<T> sgarch_parameters(std::span<const T> theta, const Layout& layout)
{
    return {theta[layout.omega()], block(theta, layout.alpha_begin(), layout.arch),
            block(theta, layout.beta_begin(), layout.garch)};
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

// rugarch holds sigma2_t at the sample mean of the squared residuals, taken at the current mean parameters, for
// t < max(p, q).
template <class T>
garch::Presample<T> presample(std::span<const double> r, std::span<const T> squared_residuals, const Layout& layout,
                              VarInit var_init)
{
    if (var_init == VarInit::Backcast)
    {
        return {T{backcast(r, layout.include_mean)}, 0};
    }
    T sum{0.0};
    for (const T& square : squared_residuals)
    {
        sum = sum + square;
    }
    std::size_t held = static_cast<std::size_t>(std::max(layout.arch, layout.garch));
    return {sum / static_cast<double>(squared_residuals.size()), held};
}

template <class T>
std::span<T> provided_or(std::span<T> provided, std::vector<T>& storage, std::size_t size)
{
    if (!provided.empty())
    {
        return provided;
    }
    storage.resize(size);
    return storage;
}

template <class T>
bool is_valid_variance(const T& sigma2)
{
    double value = ad::value_of(sigma2);
    return std::isfinite(value) && value > 0.0;
}

template <class T>
T sum_of_terms(std::span<const T> residuals, std::span<const T> sigma2, const dist::LogDensity<T>& density,
               std::span<T> terms)
{
    using std::log;
    using std::sqrt;
    T total{0.0};
    for (std::size_t t = 0; t < residuals.size(); ++t)
    {
        if (!is_valid_variance(sigma2[t]))
        {
            return T{kInvalid};
        }
        T z = residuals[t] / sqrt(sigma2[t]);
        T term = dist::log_density(density, z) - 0.5 * log(sigma2[t]);
        if (!terms.empty())
        {
            terms[t] = term;
        }
        total = total + term;
    }
    return total;
}

}

template <class T>
T loglik(std::span<const double> r, std::span<const T> theta, const Layout& layout, VarInit var_init,
         const LoglikOutputs<T>& out)
{
    assert(theta.size() == layout.count());
    assert(out.terms.empty() || out.terms.size() == r.size());
    assert(out.residuals.empty() || out.residuals.size() == r.size());
    assert(out.sigma2.empty() || out.sigma2.size() == r.size());
    T shape = shape_of(theta, layout);
    if (!dist::in_domain(layout.dist, ad::value_of(shape)))
    {
        return T{kInvalid};
    }
    std::vector<T> residual_storage;
    std::vector<T> variance_storage;
    std::span<T> residuals = provided_or(out.residuals, residual_storage, r.size());
    std::span<T> sigma2 = provided_or(out.sigma2, variance_storage, r.size());
    mean::arma_residuals(r, arma_parameters(theta, layout), residuals);
    std::vector<T> squared_residuals(r.size());
    std::ranges::transform(residuals, squared_residuals.begin(), [](const T& e) { return e * e; });
    garch::Presample<T> initial = presample<T>(r, squared_residuals, layout, var_init);
    garch::sgarch_variance<T>(squared_residuals, sgarch_parameters(theta, layout), initial, sigma2);
    return sum_of_terms<T>(residuals, sigma2, dist::make_log_density(layout.dist, shape), out.terms);
}

template double loglik<double>(std::span<const double>, std::span<const double>, const Layout&, VarInit,
                               const LoglikOutputs<double>&);
template ad::Dual<8> loglik<ad::Dual<8>>(std::span<const double>, std::span<const ad::Dual<8>>, const Layout&,
                                         VarInit, const LoglikOutputs<ad::Dual<8>>&);
template ad::Dual<16> loglik<ad::Dual<16>>(std::span<const double>, std::span<const ad::Dual<16>>, const Layout&,
                                           VarInit, const LoglikOutputs<ad::Dual<16>>&);
template ad::Dual<32> loglik<ad::Dual<32>>(std::span<const double>, std::span<const ad::Dual<32>>, const Layout&,
                                           VarInit, const LoglikOutputs<ad::Dual<32>>&);

}
